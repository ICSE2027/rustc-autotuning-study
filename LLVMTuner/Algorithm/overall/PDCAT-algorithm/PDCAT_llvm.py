import argparse
import copy
import csv
import glob
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from llvm_pipeline_utils import (
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


def cleanup_glob_patterns(*patterns: str) -> None:
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    os.remove(f)
            except Exception:
                pass


def infer_llvm_link_bin(clang_path: str) -> str:
    base = os.path.basename(clang_path)
    if "clang-" in base:
        version_suffix = base.split("clang-")[-1]
        return f"llvm-link-{version_suffix}"
    return "llvm-link"


# ==========================================
# Trainset / Constraints
# ==========================================

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


def parse_sequence_line(line: str) -> List[int]:
    return [int(x) for x in line.replace(" ", "").split(",") if x in {"0", "1"}]


def extract_sequences_from_file(path: str, expected_dim: Optional[int] = None) -> List[List[int]]:
    extracted_sequences = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            seq = parse_sequence_line(line)
            if not seq:
                continue
            if expected_dim is not None and len(seq) != expected_dim:
                continue
            extracted_sequences.append(seq)
    return extracted_sequences


# ==========================================
# Objective Evaluation
# ==========================================

def build_pipeline_from_seq(master_root, all_flags: List[str], seq: List[int]) -> str:
    selected_ids = {flag for flag, bit in zip(all_flags, seq) if bit == 1}

    current_root = copy.deepcopy(master_root)
    new_children = []
    for ch in current_root.children:
        if prune_ast_in_place(ch, selected_ids):
            new_children.append(ch)
    current_root.children = new_children

    return "no-op-module" if not current_root.children else get_pipeline_string_from_root(current_root)


def get_objective_score_llvm(
    independent: List[int],
    all_flags: List[str],
    master_root,
    base_bc: str,
    opt_path: str,
    clang_path: str,
    baseline_run_cmd: List[str],
    baseline_time_ref: float,
    exec_param: str = "",
    verbose: bool = False,
):
    pipeline_str = build_pipeline_from_seq(master_root, all_flags, independent)

    tuned_bc = "tuned.bc"
    exe_tuned = "tuned.out"

    cleanup_paths(tuned_bc, exe_tuned)

    cmd_opt_tuned = [
        opt_path,
        f"-passes={pipeline_str}",
        base_bc,
        "-o",
        tuned_bc,
    ]
    if not execute_terminal_command(cmd_opt_tuned, verbose=verbose):
        return 0.0001, -1.0, -1.0, pipeline_str

    cmd_link_tuned = [
        clang_path,
        tuned_bc,
        "-O0",
        "-o",
        exe_tuned,
        "-lm",
    ]
    if not execute_terminal_command(cmd_link_tuned, verbose=verbose):
        return 0.0001, -1.0, -1.0, pipeline_str

    if not os.path.exists(exe_tuned):
        print("Tuned executable not found after successful build.")
        return 0.0001, -1.0, -1.0, pipeline_str

    dynamic_timeout = max(2.0 * baseline_time_ref + 10.0, 10.0)

    tuned_run_cmd = [os.path.abspath(exe_tuned)]
    if exec_param.strip():
        tuned_run_cmd.extend(shlex.split(exec_param))

    ok_tuned, time_tuned = run_and_measure_once(
        tuned_run_cmd,
        timeout=dynamic_timeout,
    )
    if not ok_tuned:
        return 0.0001, -1.0, -1.0, pipeline_str

    ok_baseline, time_baseline = run_and_measure_once(
        baseline_run_cmd,
        timeout=dynamic_timeout,
    )
    if not ok_baseline:
        return 0.0001, time_tuned, -1.0, pipeline_str

    time_tuned = max(time_tuned, 0.0001)
    time_baseline = max(time_baseline, 0.0001)
    speedup = time_baseline / time_tuned

    return speedup, time_tuned, time_baseline, pipeline_str


# ==========================================
# PDCAT
# ==========================================

class PDCAT:
    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        all_flags: List[str],
        initial_seqs: List[List[int]],
        constraints: dict,
        permax: float,
        permin: float,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.all_flags = all_flags
        self.initial_seqs = initial_seqs
        self.constraints = constraints
        self.permax = permax
        self.permin = permin

        self.initial_pro = self.obtain_initial_pro()

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

    def obtain_initial_pro(self) -> np.ndarray:
        if not self.initial_seqs:
            return np.array([0.5] * len(self.all_flags), dtype=float)

        cal = np.array(self.initial_seqs)
        x = cal.sum(axis=0)
        D = cal.shape[0]

        alpha_post = self.a + x
        beta_post = self.b + D - x
        posterior_means = alpha_post / (alpha_post + beta_post)
        return posterior_means.astype(float)

    def trans_prob_to_flags(self, prob: np.ndarray) -> List[int]:
        return [1 if random.random() < p else 0 for p in prob]

    def run(
        self,
        budget: int,
        iter_csv_file: str,
        time_csv_file: str,
        score_kwargs: dict,
    ) -> None:
        time_zero = time.perf_counter()
        last_log_time = 0.0

        current_pro = copy.deepcopy(self.initial_pro)
        Es: List[float] = []
        min_thresh = 0.1

        iter_count = 0
        best_speedup_so_far = float("-inf")
        best_seq_so_far: List[int] = []
        best_pipeline_so_far = ""
        best_tuned_time_so_far = -1.0
        best_baseline_time_so_far = -1.0
        best_iter_so_far = -1
        best_timestamp_so_far = -1.0

        while True:
            elapsed = time.perf_counter() - time_zero
            if elapsed >= budget:
                break

            iter_count += 1

            seq = self.trans_prob_to_flags(current_pro)
            while self.constraints_check(seq):
                seq = self.trans_prob_to_flags(current_pro)

            speedup, time_tuned, time_baseline, pipeline_str = get_objective_score_llvm(
                independent=seq,
                **score_kwargs,
            )

            if abs(self.permax - self.permin) < 1e-12:
                E = 0.0
            else:
                E = (speedup - self.permin) / (self.permax - self.permin)

            Es.append(E)
            avg = sum(Es) / len(Es)
            diff = abs(E - avg)
            old_pro = current_pro.copy()

            if E > avg:
                for i in range(len(current_pro)):
                    if seq[i] == 1:
                        current_pro[i] = old_pro[i] + self.c * (1 - old_pro[i]) * diff
                    else:
                        current_pro[i] = old_pro[i] - self.c * old_pro[i] * diff

                    current_pro[i] = min(max(current_pro[i], 0.0), 1.0)
                    if current_pro[i] < min_thresh:
                        current_pro[i] = self.initial_pro[i]
            else:
                for i in range(len(current_pro)):
                    if seq[i] == 1:
                        current_pro[i] = old_pro[i] - self.c * old_pro[i] * diff
                    else:
                        current_pro[i] = old_pro[i] + self.c * (1 - old_pro[i]) * diff

                    current_pro[i] = min(max(current_pro[i], 0.0), 1.0)
                    if current_pro[i] < min_thresh:
                        current_pro[i] = self.initial_pro[i]

            current_time = time.perf_counter() - time_zero
            is_new_best = speedup > best_speedup_so_far

            if is_new_best:
                best_speedup_so_far = speedup
                best_seq_so_far = seq
                best_pipeline_so_far = pipeline_str
                best_tuned_time_so_far = time_tuned
                best_baseline_time_so_far = time_baseline
                best_iter_so_far = iter_count
                best_timestamp_so_far = current_time
                print(f"[New Record] Speedup: {best_speedup_so_far:.6f} at Iter {iter_count}")

            best_display = best_speedup_so_far if best_speedup_so_far != float("-inf") else speedup
            print(
                f"Iter {iter_count:04d} | "
                f"Tuned: {time_tuned:.6f}s | "
                f"Baseline: {time_baseline:.6f}s | "
                f"Speedup: {speedup:.6f} | "
                f"Best So Far: {best_display:.6f}"
            )

            seq_str = ",".join(map(str, seq))
            append_csv_row(iter_csv_file, [
                iter_count,
                f"{current_time:.2f}",
                f"{time_tuned:.6f}",
                f"{time_baseline:.6f}",
                f"{speedup:.6f}",
                is_new_best,
                seq_str,
                pipeline_str,
            ])

            if current_time - last_log_time >= 20:
                best_seq_str = ",".join(map(str, best_seq_so_far)) if best_seq_so_far else ""
                append_csv_row(time_csv_file, [
                    f"{current_time:.2f}",
                    f"{best_speedup_so_far:.6f}",
                    best_iter_so_far,
                    best_seq_str,
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
        print(f"Best Seq: {','.join(map(str, best_seq_so_far))}")
        print(f"Detailed logs saved to:\n  - {iter_csv_file}\n  - {time_csv_file}")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preference-Driven Compiler Auto-Tuning for LLVM (CSV logging)"
    )
    parser.add_argument("--include_dir", type=str, default="",
                        help="Optional include directory for compiling C sources")
    parser.add_argument("--extra_c_file", type=str, default="",
                        help="Optional extra C file to compile together, e.g. polybench.c")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save iter.csv and time.csv")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source program directory")
    parser.add_argument("--clang_path", type=str, required=True,
                        help="Path to clang")
    parser.add_argument("--opt_path", type=str, required=True,
                        help="Path to opt")
    parser.add_argument("--exec_param", type=str, default="",
                        help="Execution parameter for the output executable")
    parser.add_argument("--pipeline_file", type=str, default="o3_pipeline.txt",
                        help="LLVM pipeline file")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing data_*.txt for building the training set")
    parser.add_argument("--merged_trainset_path", type=str, default="merged_trainset.txt",
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
    parser.add_argument("--keep_artifacts", action="store_true",
                        help="Keep intermediate artifacts after tuning")

    args = parser.parse_args()

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
        "Seq",
        "Pipeline",
    ])
    append_csv_row(TIME_CSV_FILE, [
        "Timestamp",
        "Best_Speedup_So_Far",
        "Best_Iter",
        "Best_Seq",
    ])

    print(f"Loading pipeline template from {args.pipeline_file}...")
    try:
        with open(args.pipeline_file, "r", encoding="utf-8") as f:
            pipeline_text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    master_root = build_ast(pipeline_text)
    all_flags = assign_flags(master_root)
    print(f"Total tunable passes found in pipeline: {len(all_flags)}")

    # =========================
    # build merged trainset
    # =========================
    target_prog = extract_prog_name_from_source_path(args.source_path)

    if args.merged_trainset_path == "merged_trainset.txt":
        merged_trainset_path = os.path.join(OUT_DIR, "merged_trainset.txt")
    else:
        merged_trainset_path = os.path.abspath(args.merged_trainset_path)

    try:
        trainset_info = build_trainset(
            data_dir=args.data_dir,
            target_prog=target_prog,
            output_file=merged_trainset_path,
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

    good_sequences = extract_sequences_from_file(
        merged_trainset_path,
        expected_dim=len(all_flags)
    )
    if not good_sequences:
        print("Error: No valid initial sequences found in merged training set.")
        sys.exit(1)

    print(f"Loaded {len(good_sequences)} initial sequences from merged training set.")

    constraints = parse_constraints(args.constraints_path)
    total_constraints = (
        len(constraints["strong_dependency"]) +
        len(constraints["weak_dependency"]) +
        len(constraints["synergistic_relationship"])
    )
    print(f"Loaded constraints: {total_constraints}")

    # =========================
    # build base.bc + baseline
    # =========================
    source_dir = os.path.abspath(args.source_path)
    if not os.path.isdir(source_dir):
        print(f"Error: source_path is not a directory: {source_dir}")
        sys.exit(1)

    source_files = sorted(glob.glob(os.path.join(source_dir, "*.c")))
    if not source_files:
        print(f"Error: no .c files found under {source_dir}")
        sys.exit(1)

    base_bc = "base.bc"
    baseline_bc = "baseline.bc"
    tuned_bc = "tuned.bc"
    exe_baseline = "baseline.out"
    exe_tuned = "tuned.out"

    llvm_link_bin = infer_llvm_link_bin(args.clang_path)
    print(f"Using llvm-link: {llvm_link_bin}")

    cleanup_paths(base_bc, baseline_bc, tuned_bc, exe_baseline, exe_tuned)
    cleanup_glob_patterns("*.bc", "*.o", "*.I", "*.s")

    print("1. Compiling source files to individual .bc files ...")

    include_dir = os.path.abspath(args.include_dir) if args.include_dir else ""
    extra_c_file = os.path.abspath(args.extra_c_file) if args.extra_c_file else ""

    if include_dir and not os.path.isdir(include_dir):
        print(f"Error: include_dir is not a directory: {include_dir}")
        sys.exit(1)

    if extra_c_file and not os.path.isfile(extra_c_file):
        print(f"Error: extra_c_file not found: {extra_c_file}")
        sys.exit(1)

    def build_compile_cmd(src_file: str) -> List[str]:
        cmd = [
            args.clang_path,
            "-O0",
            "-emit-llvm",
            "-c",
            src_file,
        ]
        if include_dir:
            cmd.extend(["-I", include_dir])
        return cmd


    for src in source_files:
        cmd_compile_src = build_compile_cmd(src)
        if not execute_terminal_command(cmd_compile_src, verbose=args.verbose):
            print(f"Error compiling source file to bitcode: {src}")
            sys.exit(1)


    if extra_c_file:
        cmd_compile_extra = build_compile_cmd(extra_c_file)
        if not execute_terminal_command(cmd_compile_extra, verbose=args.verbose):
            print(f"Error compiling extra C file to bitcode: {extra_c_file}")
            sys.exit(1)

    generated_bc_files = sorted(glob.glob("*.bc"))
    input_bc_files = [f for f in generated_bc_files if f not in [base_bc, baseline_bc, tuned_bc]]

    if not input_bc_files:
        print("Error: no temporary .bc files produced.")
        sys.exit(1)

    print("2. Linking individual .bc files into base.bc ...")
    cmd_link_bc = [llvm_link_bin] + input_bc_files + ["-o", base_bc]
    if not execute_terminal_command(cmd_link_bc, verbose=args.verbose):
        print("Error linking bitcode files into base.bc.")
        sys.exit(1)

    for f in input_bc_files:
        cleanup_paths(f)

    if not os.path.exists(base_bc):
        print("Error: base.bc was not generated.")
        sys.exit(1)

    print("3. Pre-compiling baseline executable ...")
    cmd_opt_baseline = [
        args.opt_path,
        f"-passes={pipeline_text}",
        base_bc,
        "-o",
        baseline_bc,
    ]
    if not execute_terminal_command(cmd_opt_baseline, verbose=args.verbose):
        print("Error optimizing baseline bitcode.")
        sys.exit(1)

    cmd_link_baseline = [
        args.clang_path,
        baseline_bc,
        "-O0",
        "-o",
        exe_baseline,
        "-lm",
    ]
    if not execute_terminal_command(cmd_link_baseline, verbose=args.verbose):
        print("Error linking baseline executable.")
        sys.exit(1)

    if not os.path.exists(exe_baseline):
        print(f"Error: baseline executable not found: {exe_baseline}")
        sys.exit(1)

    baseline_run_cmd = [os.path.abspath(exe_baseline)]
    if args.exec_param.strip():
        baseline_run_cmd.extend(shlex.split(args.exec_param))

    print("4. Warm-up baseline executable to establish baseline_time_ref ...")
    ok_baseline_warmup, baseline_time_ref = run_and_measure_once(
        baseline_run_cmd,
        timeout=args.bootstrap_timeout,
    )
    if not ok_baseline_warmup:
        print("Error: baseline executable failed during warmup.")
        sys.exit(1)

    print(f"Baseline reference time established: {baseline_time_ref:.6f}s")
    print(f"Dynamic timeout set to: {max(2.0 * baseline_time_ref + 10.0, 10.0):.2f}s")

    print(f"5. Starting PDCAT Loop. Results will be saved in: {OUT_DIR}")

    pd = PDCAT(
        a=1,
        b=1,
        c=0.5,
        all_flags=all_flags,
        initial_seqs=good_sequences,
        constraints=constraints,
        permax=args.permax,
        permin=args.permin,
    )

    score_kwargs = {
        "all_flags": all_flags,
        "master_root": master_root,
        "base_bc": base_bc,
        "opt_path": args.opt_path,
        "clang_path": args.clang_path,
        "baseline_run_cmd": baseline_run_cmd,
        "baseline_time_ref": baseline_time_ref,
        "exec_param": args.exec_param,
        "verbose": args.verbose,
    }

    pd.run(
        budget=args.budget,
        iter_csv_file=ITER_CSV_FILE,
        time_csv_file=TIME_CSV_FILE,
        score_kwargs=score_kwargs,
    )

    if not args.keep_artifacts:
        cleanup_paths(base_bc, baseline_bc, tuned_bc, exe_baseline, exe_tuned)
        cleanup_glob_patterns("*.bc", "*.o", "*.I", "*.s")