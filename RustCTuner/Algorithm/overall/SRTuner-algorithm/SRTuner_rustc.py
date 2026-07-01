import argparse
import copy
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from tuner import FlagInfo, Evaluator, FLOAT_MAX
from tuner import SRTuner

from rust_pipeline_utils import (
    build_ast,
    assign_flags,
    prune_ast_in_place,
    get_pipeline_string_from_root,
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
        prefix = "+" if bit else "-"
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

    if use_unsound_mir_opts:
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


# ==========================================
# Search Space
# ==========================================

class RustcFlagInfo(FlagInfo):
    def __init__(self, name, configs):
        super().__init__(name, configs)


def build_rustc_search_space(
    mir_passes: List[str],
    llvm_flags: List[str],
) -> Tuple[Dict[str, RustcFlagInfo], List[str], List[str]]:
    search_space = {}
    mir_keys = []
    llvm_keys = []

    for pass_name in mir_passes:
        key = f"MIR__{pass_name}"
        search_space[key] = RustcFlagInfo(name=key, configs=[False, True])
        mir_keys.append(key)

    for flag in llvm_flags:
        key = f"LLVM__{flag}"
        search_space[key] = RustcFlagInfo(name=key, configs=[False, True])
        llvm_keys.append(key)

    return search_space, mir_keys, llvm_keys


def build_llvm_pipeline_from_seq(master_root, llvm_flags: List[str], llvm_seq: List[int]) -> str:
    selected_llvm_ids = {flag for flag, bit in zip(llvm_flags, llvm_seq) if bit == 1}

    current_root = copy.deepcopy(master_root)
    new_children = []
    for ch in current_root.children:
        if prune_ast_in_place(ch, selected_llvm_ids):
            new_children.append(ch)
    current_root.children = new_children

    return "no-op-module" if not current_root.children else get_pipeline_string_from_root(current_root)


# ==========================================
# Evaluator
# ==========================================

class RustcEvaluator(Evaluator):

    def __init__(
        self,
        path: str,
        search_space,
        project_dir: str,
        bin_name: str,
        baseline_run_cmd: List[str],
        target_tuned: str,
        mir_passes: List[str],
        mir_keys: List[str],
        master_root,
        llvm_flags: List[str],
        llvm_keys: List[str],
        iter_csv_file: str,
        time_csv_file: str,
        baseline_time_ref: float,
        use_unsound_mir_opts: bool = False,
        use_rustc_bootstrap: bool = False,
        verbose: bool = False,
    ):
        super().__init__(path)
        self.search_space = search_space
        self.project_dir = project_dir
        self.bin_name = bin_name
        self.baseline_run_cmd = baseline_run_cmd
        self.target_tuned = target_tuned

        self.mir_passes = mir_passes
        self.mir_keys = mir_keys
        self.master_root = master_root
        self.llvm_flags = llvm_flags
        self.llvm_keys = llvm_keys

        self.iter_csv_file = iter_csv_file
        self.time_csv_file = time_csv_file
        self.baseline_time_ref = baseline_time_ref
        self.use_unsound_mir_opts = use_unsound_mir_opts
        self.use_rustc_bootstrap = use_rustc_bootstrap
        self.verbose = verbose

        self.dynamic_timeout = max(2.0 * self.baseline_time_ref + 10.0, 10.0)

        self.best_speedup_so_far = float("-inf")
        self.best_relative_perf_so_far = FLOAT_MAX
        self.best_mir_seq_so_far: List[int] = []
        self.best_llvm_seq_so_far: List[int] = []
        self.best_tuned_time_so_far = -1.0
        self.best_baseline_time_so_far = -1.0
        self.best_iter_so_far = -1
        self.best_timestamp_so_far = -1.0
        self.best_mir_pipeline_so_far = ""
        self.best_llvm_pipeline_so_far = ""

        self.iter_count = 0
        self.time_zero = time.perf_counter()
        self.last_log_time = 0.0

    def evaluate(self, opt_setting):
        self.iter_count += 1

        mir_seq = [1 if opt_setting[k] else 0 for k in self.mir_keys]
        llvm_seq = [1 if opt_setting[k] else 0 for k in self.llvm_keys]

        mir_str = build_mir_flag_string(self.mir_passes, mir_seq)
        llvm_str = build_llvm_pipeline_from_seq(self.master_root, self.llvm_flags, llvm_seq)

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
            return FLOAT_MAX

        exe_tuned = build_binary_path(self.target_tuned, self.bin_name)
        if not os.path.exists(exe_tuned):
            print("Tuned executable not found after successful build.")
            return FLOAT_MAX

        tuned_run_cmd = [exe_tuned]
        ok_tuned, time_tuned = run_and_measure_once(
            tuned_run_cmd,
            cwd=self.project_dir,
            timeout=self.dynamic_timeout,
        )
        if not ok_tuned:
            return FLOAT_MAX

        ok_baseline, time_baseline = run_and_measure_once(
            self.baseline_run_cmd,
            cwd=self.project_dir,
            timeout=self.dynamic_timeout,
        )
        if not ok_baseline:
            return FLOAT_MAX

        relative_perf = time_tuned / time_baseline
        current_speedup = time_baseline / time_tuned
        current_time = time.perf_counter() - self.time_zero

        is_new_best = current_speedup > self.best_speedup_so_far
        if is_new_best:
            self.best_speedup_so_far = current_speedup
            self.best_relative_perf_so_far = relative_perf
            self.best_mir_seq_so_far = mir_seq
            self.best_llvm_seq_so_far = llvm_seq
            self.best_tuned_time_so_far = time_tuned
            self.best_baseline_time_so_far = time_baseline
            self.best_iter_so_far = self.iter_count
            self.best_timestamp_so_far = current_time
            self.best_mir_pipeline_so_far = mir_str
            self.best_llvm_pipeline_so_far = llvm_str
            print(f"[New Record] Speedup: {self.best_speedup_so_far:.6f} at Iter {self.iter_count}")

        best_display = self.best_speedup_so_far if self.best_speedup_so_far != float("-inf") else current_speedup
        print(
            f"Iter {self.iter_count:04d} | "
            f"Tuned: {time_tuned:.6f}s | "
            f"Baseline: {time_baseline:.6f}s | "
            f"Speedup: {current_speedup:.6f} | "
            f"Relative Perf(Tuned/Baseline): {relative_perf:.6f} | "
            f"Best So Far: {best_display:.6f}"
        )

        mir_seq_str = ",".join(map(str, mir_seq))
        llvm_seq_str = ",".join(map(str, llvm_seq))

        append_csv_row(self.iter_csv_file, [
            self.iter_count,
            f"{current_time:.2f}",
            f"{time_tuned:.6f}",
            f"{time_baseline:.6f}",
            f"{current_speedup:.6f}",
            is_new_best,
            mir_seq_str,
            llvm_seq_str,
            mir_str,
            llvm_str,
        ])

        if current_time - self.last_log_time >= 20:
            best_mir_seq_str = ",".join(map(str, self.best_mir_seq_so_far)) if self.best_mir_seq_so_far else ""
            best_llvm_seq_str = ",".join(map(str, self.best_llvm_seq_so_far)) if self.best_llvm_seq_so_far else ""

            append_csv_row(self.time_csv_file, [
                f"{current_time:.2f}",
                f"{self.best_speedup_so_far:.6f}",
                self.best_iter_so_far,
                best_mir_seq_str,
                best_llvm_seq_str,
            ])
            self.last_log_time = current_time

        return relative_perf

    def evaluate_default(self):
        return 1.0


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SRTuner adapted for Rustc MIR + LLVM pipeline tuning (unified protocol, CSV logging)"
    )
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save iter.csv and time.csv")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to Rust source file, e.g. src/main.rs or src/bin/foo.rs")
    parser.add_argument("--exec_param", type=str, default="",
                        help="Execution parameter")
    parser.add_argument("--pipeline_file", type=str, default="pass.txt",
                        help="Path to pass.txt")
    parser.add_argument("--budget", type=int, default=5000,
                        help="Budget passed to SRTuner.tune(...)")
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
        "MIR_Seq",
        "LLVM_Seq",
        "MIR_Pipeline",
        "LLVM_Pipeline",
    ])
    append_csv_row(TIME_CSV_FILE, [
        "Timestamp",
        "Best_Speedup_So_Far",
        "Best_Iter",
        "Best_MIR_Seq",
        "Best_LLVM_Seq",
    ])

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
    total_flags = total_mir_passes + total_llvm_flags

    print(f"Total MIR passes: {total_mir_passes}")
    print(f"Total LLVM flags: {total_llvm_flags}")
    print(f"Total Search Space Dimension: {total_flags}")

    search_space, mir_keys, llvm_keys = build_rustc_search_space(mir_passes, llvm_flags)

    try:
        project_dir, bin_name = infer_project_info(args.source_path)
    except Exception as e:
        print(f"Error inferring Cargo project info: {e}")
        sys.exit(1)

    print(f"Project dir: {project_dir}")
    print(f"Binary name: {bin_name}")

    target_baseline = os.path.join(project_dir, "target_srtuner_baseline")
    target_tuned = os.path.join(project_dir, "target_srtuner_tuned")

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

    print(f"Baseline reference time established: {baseline_time_ref:.6f}s")
    print(f"Dynamic timeout set to: {max(2.0 * baseline_time_ref + 10.0, 10.0):.2f}s")

    print(f"3. Starting SRTuner Loop. Results will be saved in: {OUT_DIR}")

    evaluator = RustcEvaluator(
        path=args.source_path,
        search_space=search_space,
        project_dir=project_dir,
        bin_name=bin_name,
        baseline_run_cmd=baseline_run_cmd,
        target_tuned=target_tuned,
        mir_passes=mir_passes,
        mir_keys=mir_keys,
        master_root=master_root,
        llvm_flags=llvm_flags,
        llvm_keys=llvm_keys,
        iter_csv_file=ITER_CSV_FILE,
        time_csv_file=TIME_CSV_FILE,
        baseline_time_ref=baseline_time_ref,
        use_unsound_mir_opts=args.use_unsound_mir_opts,
        use_rustc_bootstrap=args.use_rustc_bootstrap,
        verbose=args.verbose,
    )

    srtuner = SRTuner(search_space, evaluator, log_file=None)
    best_opt_setting, best_perf = srtuner.tune(args.budget)

    total_elapsed = time.perf_counter() - evaluator.time_zero

    print("\n=== Tuning Finished ===")
    print(f"Best performance metric (Tuned/Baseline): {best_perf}")
    print(f"Best speedup recorded by evaluator: {evaluator.best_speedup_so_far:.6f}")
    print(f"Total Iterations: {evaluator.iter_count}")
    print(f"Total Time: {total_elapsed:.2f}s")
    print(f"Best Iter: {evaluator.best_iter_so_far}")
    print(f"Best Tuned Time: {evaluator.best_tuned_time_so_far:.6f}s")
    print(f"Best Baseline Time: {evaluator.best_baseline_time_so_far:.6f}s")
    print("Best Seq:")
    print(f"MIR: {evaluator.best_mir_seq_so_far}")
    print(f"LLVM: {evaluator.best_llvm_seq_so_far}")
    print(f"Detailed logs saved to:\n  - {ITER_CSV_FILE}\n  - {TIME_CSV_FILE}")

    if not args.keep_artifacts:
        cleanup_paths(target_baseline, target_tuned)