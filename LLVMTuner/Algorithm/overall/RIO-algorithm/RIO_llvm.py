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

from llvm_pipeline_utils import (
    build_ast,
    assign_flags,
    prune_ast_in_place,
    get_pipeline_string_from_root,
)


# ==========================================
# LLVM RIO Main Logic
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


def generate_random_conf(x: int, total_length: int) -> List[int]:
    comb = bin(x).replace("0b", "")
    comb = "0" * (total_length - len(comb)) + comb
    return [int(s) for s in comb]


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


def append_csv_row(csv_file: str, row: List) -> None:
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def build_pipeline_from_seq(master_root, all_flags: List[str], seq: List[int]) -> str:
    selected_ids = {flag for flag, bit in zip(all_flags, seq) if bit == 1}

    current_root = copy.deepcopy(master_root)
    new_children = []
    for ch in current_root.children:
        if prune_ast_in_place(ch, selected_ids):
            new_children.append(ch)
    current_root.children = new_children

    if not current_root.children:
        return "no-op-module"

    return get_pipeline_string_from_root(current_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLVM RIO Tuning (unified protocol, CSV logging)"
    )
    parser.add_argument("--include_dir", type=str, default="",
                        help="Optional include directory for compiling C sources")
    parser.add_argument("--extra_c_file", type=str, default="",
                        help="Optional extra C file to compile together, e.g. polybench.c")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save iter.csv and time.csv")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to source program directory")
    parser.add_argument("--clang_path", type=str, required=True,
                        help="Path to clang")
    parser.add_argument("--opt_path", type=str, required=True,
                        help="Path to opt")
    parser.add_argument("--exec_param", type=str, default="",
                        help="Execution parameter for the output executable")
    parser.add_argument("--pipeline_file", type=str, default="o3_pipeline.txt",
                        help="Path to O3 pipeline file")
    parser.add_argument("--budget", type=int, default=5000,
                        help="Time budget in seconds")
    parser.add_argument("--bootstrap_timeout", type=float, default=600.0,
                        help="Timeout for the baseline warmup run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed command failure info")
    parser.add_argument("--keep_artifacts", action="store_true",
                        help="Keep intermediate artifacts after tuning")

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
        "Seq",
        "Pipeline"
    ])

    append_csv_row(TIME_CSV_FILE, [
        "Timestamp",
        "Best_Speedup_So_Far",
        "Best_Iter",
        "Best_Seq"
    ])

    # =========================

    # =========================
    print(f"Loading pipeline template from {args.pipeline_file}...")
    try:
        with open(args.pipeline_file, "r", encoding="utf-8") as f:
            pipeline_text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    master_root = build_ast(pipeline_text)
    all_flags = assign_flags(master_root)
    total_flags = len(all_flags)

    print(f"Total flags found in pipeline: {total_flags}")

    # =========================

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


    exec_args = shlex.split(args.exec_param) if args.exec_param.strip() else []

    # =========================

    # =========================
    cleanup_paths(base_bc, baseline_bc, tuned_bc, exe_baseline, exe_tuned)
    cleanup_glob_patterns("*.bc", "*.o")

    # =========================

    # =========================
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

    # =========================

    # baseline = O3 pipeline + clang -O0 link
    # =========================
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

    baseline_run_cmd = [os.path.abspath(exe_baseline)] + exec_args

    print("4. Warm-up baseline executable to establish baseline_time_ref ...")
    ok_baseline_warmup, baseline_time_ref = run_and_measure_once(
        baseline_run_cmd,
        timeout=args.bootstrap_timeout,
    )
    if not ok_baseline_warmup:
        print("Error: baseline executable failed during warmup.")
        sys.exit(1)

    dynamic_timeout = max(2.0 * baseline_time_ref + 10.0, 10.0)

    print(f"Baseline reference time established: {baseline_time_ref:.6f}s")
    print(f"Dynamic timeout set to: {dynamic_timeout:.2f}s")

    # =========================

    # =========================
    print(f"5. Starting Tuning Loop. Results will be saved in: {OUT_DIR}")
    print(f"Time budget: {args.budget}s")

    time_zero = time.perf_counter()
    last_log_time = 0.0
    iter_count = 0

    best_speedup_so_far = float("-inf")
    best_seq_so_far: List[int] = []
    best_tuned_time_so_far = -1.0
    best_baseline_time_so_far = -1.0
    best_iter_so_far = -1
    best_timestamp_so_far = -1.0

    while True:
        elapsed = time.perf_counter() - time_zero
        if elapsed >= args.budget:
            break

        iter_count += 1

        # -------------------------

        # -------------------------
        x = random.randint(0, 2 ** total_flags - 1)
        seq = generate_random_conf(x, total_flags)
        pipeline_str = build_pipeline_from_seq(master_root, all_flags, seq)

        # -------------------------

        # -------------------------
        cmd_opt_tuned = [
            args.opt_path,
            f"-passes={pipeline_str}",
            base_bc,
            "-o",
            tuned_bc,
        ]
        if not execute_terminal_command(cmd_opt_tuned, verbose=args.verbose):
            continue

        cmd_link_tuned = [
            args.clang_path,
            tuned_bc,
            "-O0",
            "-o",
            exe_tuned,
            "-lm",
        ]
        if not execute_terminal_command(cmd_link_tuned, verbose=args.verbose):
            continue

        if not os.path.exists(exe_tuned):
            print("Tuned executable not found after successful build.")
            continue

        # -------------------------

        # -------------------------
        tuned_run_cmd = [os.path.abspath(exe_tuned)] + exec_args
        ok_tuned, time_tuned = run_and_measure_once(
            tuned_run_cmd,
            timeout=dynamic_timeout,
        )
        if not ok_tuned:
            continue

        # -------------------------

        # -------------------------
        ok_baseline_iter, time_baseline_iter = run_and_measure_once(
            baseline_run_cmd,
            timeout=dynamic_timeout,
        )
        if not ok_baseline_iter:
            continue

        # -------------------------

        # -------------------------
        current_speedup = time_baseline_iter / time_tuned
        current_time = time.perf_counter() - time_zero

        is_new_best = current_speedup > best_speedup_so_far

        if is_new_best:
            best_speedup_so_far = current_speedup
            best_seq_so_far = seq
            best_tuned_time_so_far = time_tuned
            best_baseline_time_so_far = time_baseline_iter
            best_iter_so_far = iter_count
            best_timestamp_so_far = current_time
            print(f"[New Record] Speedup: {best_speedup_so_far:.6f} at Iter {iter_count}")

        best_display = best_speedup_so_far if best_speedup_so_far != float("-inf") else current_speedup

        print(
            f"Iter {iter_count:04d} | "
            f"Tuned: {time_tuned:.6f}s | "
            f"Baseline: {time_baseline_iter:.6f}s | "
            f"Speedup: {current_speedup:.6f} | "
            f"Best So Far: {best_display:.6f}"
        )

        # -------------------------

        # -------------------------
        seq_str = ",".join(map(str, seq))
        append_csv_row(ITER_CSV_FILE, [
            iter_count,
            f"{current_time:.2f}",
            f"{time_tuned:.6f}",
            f"{time_baseline_iter:.6f}",
            f"{current_speedup:.6f}",
            is_new_best,
            seq_str,
            pipeline_str,
        ])

        # -------------------------

        # -------------------------
        if current_time - last_log_time >= 20:
            best_seq_str = ",".join(map(str, best_seq_so_far)) if best_seq_so_far else ""
            append_csv_row(TIME_CSV_FILE, [
                f"{current_time:.2f}",
                f"{best_speedup_so_far:.6f}",
                best_iter_so_far,
                best_seq_str,
            ])
            last_log_time = current_time

    # =========================

    # =========================
    total_elapsed = time.perf_counter() - time_zero

    print("\n=== Tuning Finished ===")
    print(f"Total Iterations: {iter_count}")
    print(f"Total Time: {total_elapsed:.2f}s")
    print(f"Best Iter: {best_iter_so_far}")
    print(f"Best Speedup: {best_speedup_so_far:.6f}")
    print(f"Best Tuned Time: {best_tuned_time_so_far:.6f}s")
    print(f"Best Baseline Time: {best_baseline_time_so_far:.6f}s")
    print(f"Best Found At: {best_timestamp_so_far:.2f}s")
    print(f"Detailed logs saved to:\n  - {ITER_CSV_FILE}\n  - {TIME_CSV_FILE}")

    # =========================

    # =========================
    if not args.keep_artifacts:
        cleanup_paths(base_bc, baseline_bc, tuned_bc, exe_baseline, exe_tuned)
        cleanup_glob_patterns("*.bc", "*.o")