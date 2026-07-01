import argparse
import copy
import csv
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from rust_pipeline_utils import (
    build_ast,
    assign_flags,
    prune_ast_in_place,
    get_pipeline_string_from_root,
)

from build_pdcat_trainset import (
    build_trainset,
    extract_prog_name_from_source_path,
)


# ==========================================
# Shared Helpers
# ==========================================

def execute_terminal_command(
    command: List[str],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    verbose: bool = False,
) -> bool:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            if verbose:
                print("Command:")
                print(" ".join(shlex.quote(x) for x in command))
                if result.stdout:
                    print("[stdout]")
                    print(result.stdout)
                if result.stderr:
                    print("[stderr]")
                    print(result.stderr)
            return False
        return True
    except Exception as e:
        print("Exec Error:", str(e))
        return False


def run_and_measure_once(
    command: List[str],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> Tuple[bool, float]:
    try:
        start = time.perf_counter()
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        end = time.perf_counter()

        if proc.returncode != 0:
            return False, -1.0

        return True, max(end - start, 0.0001)

    except subprocess.TimeoutExpired:
        return False, -1.0
    except Exception as e:
        print("Run Error:", str(e))
        return False, -1.0


def append_csv_row(csv_file: str, row: List) -> None:
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def cleanup_paths(*paths: str) -> None:
    for path in paths:
        if not path:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def parse_pass_txt(filepath: str) -> Tuple[List[str], str]:
    mir_passes = []
    llvm_pipeline = ""
    mode = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "MIR:":
                mode = "MIR"
                continue
            elif line == "LLVM:":
                mode = "LLVM"
                continue

            if mode == "MIR":
                mir_passes.append(line)
            elif mode == "LLVM":
                llvm_pipeline += line

    return mir_passes, llvm_pipeline


def infer_project_info(source_path: str) -> Tuple[str, str]:
    abs_source = os.path.abspath(source_path)
    file_name = os.path.basename(abs_source)
    parent_dir = os.path.dirname(abs_source)

    if file_name == "main.rs":
        project_dir = os.path.abspath(os.path.join(parent_dir, ".."))
        cargo_toml = os.path.join(project_dir, "Cargo.toml")

        package_name = None
        in_package = False
        try:
            with open(cargo_toml, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("[") and s.endswith("]"):
                        in_package = (s == "[package]")
                        continue
                    if in_package and s.startswith("name") and "=" in s:
                        rhs = s.split("=", 1)[1].strip()
                        if rhs.startswith('"') and rhs.endswith('"'):
                            package_name = rhs.strip('"')
                            break
        except Exception:
            pass

        if package_name is None:
            raise RuntimeError("Cannot infer binary name from [package].name in Cargo.toml.")

        return project_dir, package_name

    if os.path.basename(parent_dir) == "bin":
        project_dir = os.path.abspath(os.path.join(parent_dir, "..", ".."))
        bin_name = os.path.splitext(file_name)[0]
        return project_dir, bin_name

    project_dir = os.path.abspath(os.path.join(parent_dir, ".."))
    bin_name = os.path.splitext(file_name)[0]
    return project_dir, bin_name


def build_binary_path(target_dir: str, bin_name: str) -> str:
    return os.path.join(target_dir, "release", bin_name)


def build_mir_flag_string(mir_passes: List[str], mir_bits: List[int]) -> str:
    items = []
    for pass_name, bit in zip(mir_passes, mir_bits):
        prefix = "+" if bit == 1 else "-"
        items.append(f"{prefix}{pass_name}")
    return ",".join(items)


def build_rustc_extra_args(
    mir_flag_str: str,
    llvm_pipeline_str: str,
    use_unsound_mir_opts: bool = False,
) -> List[str]:
    extra = []

    if mir_flag_str:
        extra.extend(["-Z", f"mir-enable-passes={mir_flag_str}"])

    if use_unsound_mir_opts and mir_flag_str:
        extra.extend(["-Z", "unsound-mir-opts"])

    if llvm_pipeline_str:
        extra.extend(["-C", "no-prepopulate-passes"])
        extra.extend(["-C", f"passes={llvm_pipeline_str}"])

    return extra


def build_cargo_rustc_command(bin_name: str, extra_rustc_args: List[str]) -> List[str]:
    cmd = ["cargo", "rustc", "--release", "--bin", bin_name]
    if extra_rustc_args:
        cmd.append("--")
        cmd.extend(extra_rustc_args)
    return cmd


def make_build_env(target_dir: str, use_rustc_bootstrap: bool) -> dict:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = target_dir
    env["CARGO_INCREMENTAL"] = "0"

    if use_rustc_bootstrap:
        env["RUSTC_BOOTSTRAP"] = "1"

    return env


def build_llvm_pipeline_from_seq(master_root, llvm_flags: List[str], llvm_seq: List[int]) -> str:
    selected_ids = {flag for flag, bit in zip(llvm_flags, llvm_seq) if bit == 1}

    current_root = copy.deepcopy(master_root)
    new_children = []
    for ch in current_root.children:
        if prune_ast_in_place(ch, selected_ids):
            new_children.append(ch)
    current_root.children = new_children

    return "no-op-module" if not current_root.children else get_pipeline_string_from_root(current_root)


def extract_sequences_from_jsonl(
    file_path: str,
    expected_mir_dim: int = None,
    expected_llvm_dim: int = None,
) -> List[Dict[str, List[int]]]:
    extracted_sequences = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            mir_seq = [int(x) for x in obj.get("MIR", [])]
            llvm_seq = [int(x) for x in obj.get("LLVM", [])]

            if expected_mir_dim is not None and len(mir_seq) != expected_mir_dim:
                continue
            if expected_llvm_dim is not None and len(llvm_seq) != expected_llvm_dim:
                continue

            extracted_sequences.append({
                "MIR": mir_seq,
                "LLVM": llvm_seq,
            })

    return extracted_sequences


def parse_constraints(file_path: str):
    empty_constraints = {
        "strong_dependency": [],
        "weak_dependency": [],
        "synergistic_relationship": []
    }

    if not file_path:
        return empty_constraints
    if not os.path.exists(file_path):
        return empty_constraints
    if os.path.getsize(file_path) == 0:
        return empty_constraints

    strong_dependency = []
    weak_dependency = []
    synergistic_relationship = []
    current_category = None

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Strong dependency:"):
            current_category = "strong_dependency"
            continue
        elif line.startswith("Weak dependency:"):
            current_category = "weak_dependency"
            continue
        elif line.startswith("Synergistic relationship:"):
            current_category = "synergistic_relationship"
            continue

        if "->" in line:
            constraints = [item.strip() for item in line.replace("->", ",").split(",") if item.strip()]
        elif "and" in line:
            constraints = [item.strip() for item in line.replace("and", ",").split(",") if item.strip()]
        else:
            continue

        if len(constraints) < 2:
            continue

        pair = constraints[:2]
        if current_category == "strong_dependency":
            strong_dependency.append(pair)
        elif current_category == "weak_dependency":
            weak_dependency.append(pair)
        elif current_category == "synergistic_relationship":
            synergistic_relationship.append(pair)

    return {
        "strong_dependency": strong_dependency,
        "weak_dependency": weak_dependency,
        "synergistic_relationship": synergistic_relationship
    }


# ==========================================
# Evaluator
# ==========================================

@dataclass
class EvalResult:
    speedup: float
    tuned_time: float
    baseline_time: float
    mir_seq: List[int]
    llvm_seq: List[int]
    mir_pipeline: str
    llvm_pipeline: str


class RustcEvaluator:
    def __init__(
        self,
        exec_param: str,
        mir_passes: List[str],
        llvm_flags: List[str],
        master_root,
        project_dir: str,
        bin_name: str,
        target_tuned: str,
        baseline_run_cmd: List[str],
        baseline_time_ref: float,
        use_unsound_mir_opts: bool,
        use_rustc_bootstrap: bool,
        verbose: bool,
        tune_scope: str,
    ):
        self.exec_param = exec_param
        self.mir_passes = mir_passes
        self.llvm_flags = llvm_flags
        self.master_root = master_root
        self.project_dir = project_dir
        self.bin_name = bin_name
        self.target_tuned = target_tuned
        self.baseline_run_cmd = baseline_run_cmd
        self.baseline_time_ref = baseline_time_ref
        self.use_unsound_mir_opts = use_unsound_mir_opts
        self.use_rustc_bootstrap = use_rustc_bootstrap
        self.verbose = verbose
        self.tune_scope = tune_scope

    def evaluate(self, full_seq: List[int]) -> EvalResult:
        if self.tune_scope == "both":
            mir_seq = full_seq[:len(self.mir_passes)]
            llvm_seq = full_seq[len(self.mir_passes):]

            mir_str = build_mir_flag_string(self.mir_passes, mir_seq)
            llvm_str = build_llvm_pipeline_from_seq(self.master_root, self.llvm_flags, llvm_seq)

        elif self.tune_scope == "mir":
            mir_seq = full_seq
            llvm_seq = []

            mir_str = build_mir_flag_string(self.mir_passes, mir_seq)


            llvm_str = ""

        elif self.tune_scope == "llvm":
            mir_seq = []
            llvm_seq = full_seq


            mir_str = ""
            llvm_str = build_llvm_pipeline_from_seq(self.master_root, self.llvm_flags, llvm_seq)

        else:
            raise ValueError(f"Unknown tune_scope: {self.tune_scope}")

        tuned_extra_args = build_rustc_extra_args(
            mir_flag_str=mir_str,
            llvm_pipeline_str=llvm_str,
            use_unsound_mir_opts=self.use_unsound_mir_opts,
        )
        tuned_cmd = build_cargo_rustc_command(self.bin_name, tuned_extra_args)
        tuned_env = make_build_env(self.target_tuned, self.use_rustc_bootstrap)

        if not execute_terminal_command(
            tuned_cmd,
            cwd=self.project_dir,
            env=tuned_env,
            verbose=self.verbose,
        ):
            return EvalResult(0.0001, -1.0, -1.0, mir_seq, llvm_seq, mir_str, llvm_str)

        exe_tuned = build_binary_path(self.target_tuned, self.bin_name)
        if not os.path.exists(exe_tuned):
            return EvalResult(0.0001, -1.0, -1.0, mir_seq, llvm_seq, mir_str, llvm_str)

        dynamic_timeout = max(2.0 * self.baseline_time_ref + 10.0, 10.0)

        tuned_run_cmd = [exe_tuned]
        if self.exec_param.strip():
            tuned_run_cmd.extend(shlex.split(self.exec_param))

        ok_tuned, time_tuned = run_and_measure_once(
            tuned_run_cmd,
            cwd=self.project_dir,
            timeout=dynamic_timeout,
        )
        if not ok_tuned:
            return EvalResult(0.0001, -1.0, -1.0, mir_seq, llvm_seq, mir_str, llvm_str)

        ok_baseline, time_baseline = run_and_measure_once(
            self.baseline_run_cmd,
            cwd=self.project_dir,
            timeout=dynamic_timeout,
        )
        if not ok_baseline:
            return EvalResult(0.0001, time_tuned, -1.0, mir_seq, llvm_seq, mir_str, llvm_str)

        time_tuned = max(time_tuned, 0.0001)
        time_baseline = max(time_baseline, 0.0001)
        speedup = time_baseline / time_tuned

        return EvalResult(speedup, time_tuned, time_baseline, mir_seq, llvm_seq, mir_str, llvm_str)


# ==========================================
# PDCAT Core
# ==========================================

class PDCATCore:

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        all_flags: List[str],
        good_sequences: List[Dict[str, List[int]]],
        constraints: Dict,
        permax: float,
        permin: float,
        tune_scope: str,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.all_flags = all_flags
        self.good_sequences = good_sequences
        self.constraints = constraints
        self.permax = permax
        self.permin = permin
        self.tune_scope = tune_scope

        self.initial_pro = self._obtain_initial_pro()
        self.current_pro = copy.deepcopy(self.initial_pro)
        self.Es: List[float] = []

    def _flatten_sequence(self, record: Dict[str, List[int]]) -> List[int]:
        if self.tune_scope == "both":
            return record["MIR"] + record["LLVM"]
        elif self.tune_scope == "mir":
            return record["MIR"]
        elif self.tune_scope == "llvm":
            return record["LLVM"]
        else:
            raise ValueError(f"Unknown tune_scope: {self.tune_scope}")

    def _obtain_initial_pro(self) -> np.ndarray:
        if not self.good_sequences:
            return np.array([0.5] * len(self.all_flags), dtype=float)

        flat = [self._flatten_sequence(x) for x in self.good_sequences]
        cal = np.array(flat)
        x = cal.sum(axis=0)
        D = cal.shape[0]

        alpha_post = self.a + x
        beta_post = self.b + D - x
        posterior_means = alpha_post / (alpha_post + beta_post)
        return posterior_means.astype(float)

    def constraints_check(self, seq: List[int]) -> bool:
        if not self.constraints:
            return False

        flag_index = {flag: idx for idx, flag in enumerate(self.all_flags)}

        for a, b in self.constraints.get("strong_dependency", []):
            if a in flag_index and b in flag_index:
                if seq[flag_index[a]] == 0 and seq[flag_index[b]] == 1:
                    return True

        for a, b in self.constraints.get("weak_dependency", []):
            if a in flag_index and b in flag_index:
                if seq[flag_index[a]] == 1 and seq[flag_index[b]] == 0:
                    return True

        for a, b in self.constraints.get("synergistic_relationship", []):
            if a in flag_index and b in flag_index:
                if seq[flag_index[a]] + seq[flag_index[b]] == 1:
                    return True

        return False

    def _trans_prob_to_flags(self, prob: np.ndarray) -> List[int]:
        return [1 if random.random() < p else 0 for p in prob]

    def sample_valid_sequence(self, max_retry: int = 1000) -> List[int]:
        seq = self._trans_prob_to_flags(self.current_pro)
        retry_count = 0

        while self.constraints_check(seq):
            seq = self._trans_prob_to_flags(self.current_pro)
            retry_count += 1
            if retry_count > max_retry:
                break

        return seq

    def update_probability(self, seq: List[int], speedup: float, min_thresh: float = 0.1) -> None:
        if abs(self.permax - self.permin) < 1e-12:
            E = 0.0
        else:
            E = (speedup - self.permin) / (self.permax - self.permin)

        self.Es.append(E)
        avg = sum(self.Es) / len(self.Es)
        diff = abs(E - avg)
        old_pro = self.current_pro.copy()

        if E > avg:
            for i in range(len(self.current_pro)):
                if seq[i] == 1:
                    self.current_pro[i] = old_pro[i] + self.c * (1 - old_pro[i]) * diff
                else:
                    self.current_pro[i] = old_pro[i] - self.c * old_pro[i] * diff

                self.current_pro[i] = min(max(self.current_pro[i], 0.0), 1.0)
                if self.current_pro[i] < min_thresh:
                    self.current_pro[i] = self.initial_pro[i]
        else:
            for i in range(len(self.current_pro)):
                if seq[i] == 1:
                    self.current_pro[i] = old_pro[i] - self.c * old_pro[i] * diff
                else:
                    self.current_pro[i] = old_pro[i] + self.c * (1 - old_pro[i]) * diff

                self.current_pro[i] = min(max(self.current_pro[i], 0.0), 1.0)
                if self.current_pro[i] < min_thresh:
                    self.current_pro[i] = self.initial_pro[i]


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preference-Driven Compiler Auto-Tuning for Rustc")

    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save iter.csv and time.csv")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to Rust source file, e.g. src/bin/gemm.rs")
    parser.add_argument("--exec_param", type=str, default="",
                        help="Execution parameter")
    parser.add_argument("--pipeline_file", type=str, default="pass.txt",
                        help="Path to pass.txt")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing data_*.jsonl for building the training set")
    parser.add_argument("--merged_trainset_path", type=str, default="merged_trainset.jsonl",
                        help="Temporary merged training file to generate automatically")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable deduplication when building merged trainset")

    parser.add_argument("--constraints_path", type=str, default="",
                        help="Constraint file; can be empty")
    parser.add_argument("--permax", type=float, required=True,
                        help="Best performance among generated sequences")
    parser.add_argument("--permin", type=float, required=True,
                        help="Lowest performance among generated sequences")
    parser.add_argument("--budget", type=int, default=5000,
                        help="Time budget in seconds")
    parser.add_argument("--bootstrap_timeout", type=float, default=600.0,
                        help="Timeout for baseline warmup run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed command failure info")
    parser.add_argument("--use_unsound_mir_opts", action="store_true",
                        help="Append -Z unsound-mir-opts when necessary")
    parser.add_argument("--use_rustc_bootstrap", action="store_true",
                        help="Set RUSTC_BOOTSTRAP=1 if not using nightly")
    parser.add_argument("--keep_artifacts", action="store_true",
                        help="Keep target directories after tuning")
    parser.add_argument(
        "--tune_scope",
        type=str,
        default="both",
        choices=["both", "mir", "llvm"],
        help="Tuning scope: both = MIR+LLVM, mir = tune MIR only, llvm = tune LLVM only"
    )

    args = parser.parse_args()

    # =========================

    # =========================
    OUT_DIR = os.path.abspath(args.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)

    ITER_CSV_FILE = os.path.join(OUT_DIR, "iter.csv")
    TIME_CSV_FILE = os.path.join(OUT_DIR, "time.csv")

    for f in [ITER_CSV_FILE, TIME_CSV_FILE]:
        if os.path.exists(f):
            os.remove(f)

    append_csv_row(ITER_CSV_FILE, [
        "Iter",
        "Timestamp",
        "Tuned_Time",
        "Baseline_Time",
        "Speedup",
        "Is_New_Best",
        "MIR_Seq",
        "LLVM_Seq",
        "MIR_Pipeline",
        "LLVM_Pipeline"
    ])

    append_csv_row(TIME_CSV_FILE, [
        "Timestamp",
        "Best_Speedup_So_Far",
        "Best_Iter",
        "Best_MIR_Seq",
        "Best_LLVM_Seq"
    ])

    # =========================

    # =========================
    print(f"Loading MIR and LLVM pipelines from {args.pipeline_file}...")
    try:
        mir_passes, llvm_pipeline_text = parse_pass_txt(args.pipeline_file)
    except FileNotFoundError:
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    master_root = build_ast(llvm_pipeline_text)
    llvm_flags = assign_flags(master_root)

    total_mir_passes = len(mir_passes)
    total_llvm_flags = len(llvm_flags)

    if args.tune_scope == "both":
        total_flags = total_mir_passes + total_llvm_flags
    elif args.tune_scope == "mir":
        total_flags = total_mir_passes
    elif args.tune_scope == "llvm":
        total_flags = total_llvm_flags
    else:
        raise ValueError(f"Unknown tune_scope: {args.tune_scope}")

    print(f"Total MIR passes: {total_mir_passes}")
    print(f"Total LLVM flags: {total_llvm_flags}")
    print(f"Tune scope: {args.tune_scope}")
    print(f"Total Search Space Dimension: {total_flags}")

    # =========================

    # =========================
    try:
        project_dir, bin_name = infer_project_info(args.source_path)
    except Exception as e:
        print(f"Error inferring Cargo project info: {e}")
        sys.exit(1)

    print(f"Project dir: {project_dir}")
    print(f"Binary name: {bin_name}")

    # =========================

    # =========================
    target_prog = extract_prog_name_from_source_path(args.source_path)
    merged_trainset_path = os.path.abspath(args.merged_trainset_path)

    try:
        trainset_info = build_trainset(
            data_dir=args.data_dir,
            target_prog=target_prog,
            output_file=merged_trainset_path,
            expected_mir_dim=total_mir_passes,
            expected_llvm_dim=total_llvm_flags,
            dedup=(not args.no_dedup),
        )
    except Exception as e:
        print(f"Error while building merged trainset: {e}")
        sys.exit(1)

    print("====================================================")
    print("Merged training set built automatically")
    print(f"Target program      : {trainset_info['target_prog']}")
    print(f"Skipped file(s)     : {trainset_info['skipped_files']}")
    print(f"Used file count     : {trainset_info['used_file_count']}")
    print(f"Total merged lines  : {trainset_info['total_merged_lines']}")
    print(f"Merged trainset     : {trainset_info['output_file']}")
    print("====================================================")

    good_sequences = extract_sequences_from_jsonl(
        merged_trainset_path,
        expected_mir_dim=total_mir_passes,
        expected_llvm_dim=total_llvm_flags,
    )
    if not good_sequences:
        print("Error: No valid initial sequences found in merged training set.")
        sys.exit(1)

    print(f"Loaded {len(good_sequences)} initial sequences from merged training set.")

    # =========================

    # =========================
    if args.tune_scope == "both":
        all_flags = mir_passes + llvm_flags
    elif args.tune_scope == "mir":
        all_flags = mir_passes
    elif args.tune_scope == "llvm":
        all_flags = llvm_flags
    else:
        raise ValueError(f"Unknown tune_scope: {args.tune_scope}")
    constraints = parse_constraints(args.constraints_path)
    total_constraints = (
        len(constraints["strong_dependency"]) +
        len(constraints["weak_dependency"]) +
        len(constraints["synergistic_relationship"])
    )
    print(f"Loaded constraints: {total_constraints}")

    # =========================

    # =========================
    target_baseline = os.path.join(project_dir, "target_pdcat_baseline")
    target_tuned = os.path.join(project_dir, "target_pdcat_tuned")

    exe_baseline = build_binary_path(target_baseline, bin_name)
    exec_args = shlex.split(args.exec_param) if args.exec_param.strip() else []

    cleanup_paths(target_baseline, target_tuned)

    print("1. Pre-compiling baseline executable ...")
    baseline_cmd = build_cargo_rustc_command(bin_name, extra_rustc_args=[])
    baseline_env = make_build_env(target_baseline, args.use_rustc_bootstrap)

    if not execute_terminal_command(
        baseline_cmd,
        cwd=project_dir,
        env=baseline_env,
        verbose=args.verbose,
    ):
        print("Error compiling baseline executable.")
        sys.exit(1)

    if not os.path.exists(exe_baseline):
        print(f"Error: baseline executable not found: {exe_baseline}")
        sys.exit(1)

    baseline_run_cmd = [exe_baseline] + exec_args

    print("2. Warm-up baseline executable to establish baseline_time_ref ...")
    ok_baseline_warmup, baseline_time_ref = run_and_measure_once(
        baseline_run_cmd,
        cwd=project_dir,
        timeout=args.bootstrap_timeout,
    )
    if not ok_baseline_warmup:
        print("Error: baseline executable failed during warmup.")
        sys.exit(1)

    dynamic_timeout = max(2.0 * baseline_time_ref + 10.0, 10.0)
    print(f"Baseline reference time established: {baseline_time_ref:.6f}s")
    print(f"Dynamic timeout set to: {dynamic_timeout:.2f}s")

    print(f"3. Starting PDCAT Loop. Results will be saved in: {OUT_DIR}")

    evaluator = RustcEvaluator(
        exec_param=args.exec_param,
        mir_passes=mir_passes,
        llvm_flags=llvm_flags,
        master_root=master_root,
        project_dir=project_dir,
        bin_name=bin_name,
        target_tuned=target_tuned,
        baseline_run_cmd=baseline_run_cmd,
        baseline_time_ref=baseline_time_ref,
        use_unsound_mir_opts=args.use_unsound_mir_opts,
        use_rustc_bootstrap=args.use_rustc_bootstrap,
        verbose=args.verbose,
        tune_scope=args.tune_scope,
    )

    core = PDCATCore(
        a=1,
        b=1,
        c=0.5,
        all_flags=all_flags,
        good_sequences=good_sequences,
        constraints=constraints,
        permax=args.permax,
        permin=args.permin,
        tune_scope=args.tune_scope,
    )

    time_zero = time.perf_counter()
    last_log_time = 0.0
    iter_count = 0

    best_speedup_so_far = float("-inf")
    best_mir_seq_so_far: List[int] = []
    best_llvm_seq_so_far: List[int] = []
    best_tuned_time_so_far = -1.0
    best_baseline_time_so_far = -1.0
    best_iter_so_far = -1
    best_timestamp_so_far = -1.0
    best_mir_pipeline_so_far = ""
    best_llvm_pipeline_so_far = ""

    while True:
        elapsed = time.perf_counter() - time_zero
        if elapsed >= args.budget:
            break

        iter_count += 1

        seq = core.sample_valid_sequence()
        result = evaluator.evaluate(seq)
        core.update_probability(seq, result.speedup)

        current_time = time.perf_counter() - time_zero
        is_new_best = result.speedup > best_speedup_so_far

        if is_new_best:
            best_speedup_so_far = result.speedup
            best_mir_seq_so_far = result.mir_seq
            best_llvm_seq_so_far = result.llvm_seq
            best_tuned_time_so_far = result.tuned_time
            best_baseline_time_so_far = result.baseline_time
            best_iter_so_far = iter_count
            best_timestamp_so_far = current_time
            best_mir_pipeline_so_far = result.mir_pipeline
            best_llvm_pipeline_so_far = result.llvm_pipeline
            print(f"[New Record] Speedup: {best_speedup_so_far:.6f} at Iter {iter_count}")

        best_display = best_speedup_so_far if best_speedup_so_far != float("-inf") else result.speedup
        print(
            f"Iter {iter_count:04d} | "
            f"Tuned: {result.tuned_time:.6f}s | "
            f"Baseline: {result.baseline_time:.6f}s | "
            f"Speedup: {result.speedup:.6f} | "
            f"Best So Far: {best_display:.6f}"
        )

        mir_seq_str = ",".join(map(str, result.mir_seq))
        llvm_seq_str = ",".join(map(str, result.llvm_seq))

        append_csv_row(ITER_CSV_FILE, [
            iter_count,
            f"{current_time:.2f}",
            f"{result.tuned_time:.6f}",
            f"{result.baseline_time:.6f}",
            f"{result.speedup:.6f}",
            is_new_best,
            mir_seq_str,
            llvm_seq_str,
            result.mir_pipeline,
            result.llvm_pipeline,
        ])

        if current_time - last_log_time >= 20:
            best_mir_seq_str = ",".join(map(str, best_mir_seq_so_far)) if best_mir_seq_so_far else ""
            best_llvm_seq_str = ",".join(map(str, best_llvm_seq_so_far)) if best_llvm_seq_so_far else ""

            append_csv_row(TIME_CSV_FILE, [
                f"{current_time:.2f}",
                f"{best_speedup_so_far:.6f}",
                best_iter_so_far,
                best_mir_seq_str,
                best_llvm_seq_str,
            ])
            last_log_time = current_time

    total_elapsed = time.perf_counter() - time_zero

    print("\n=== Tuning Finished ===")
    print(f"Total Iterations: {iter_count}")
    print(f"Total Time: {total_elapsed:.2f}s")
    print(f"Best Iter: {best_iter_so_far}")
    print(f"Best Speedup: {best_speedup_so_far:.6f}")
    print(f"Best Tuned Time: {best_tuned_time_so_far:.6f}s")
    print(f"Best Baseline Time: {best_baseline_time_so_far:.6f}s")
    print(f"Best Found At: {best_timestamp_so_far:.2f}s")
    print("Best Seq:")
    print(f"MIR: {','.join(map(str, best_mir_seq_so_far))}")
    print(f"LLVM: {','.join(map(str, best_llvm_seq_so_far))}")
    print(f"Detailed logs saved to:\n  - {ITER_CSV_FILE}\n  - {TIME_CSV_FILE}")

    if not args.keep_artifacts:
        cleanup_paths(target_baseline, target_tuned)
        