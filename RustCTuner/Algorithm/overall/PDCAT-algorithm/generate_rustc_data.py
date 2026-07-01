import argparse
import copy
import glob
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
            if verbose:
                print(f"Command failed with return code {result.returncode}")
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


def get_median_execution_time(
    command: List[str],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    runs: int = 5,
    timeout: Optional[float] = None,
) -> float:
    times = []
    for _ in range(runs):
        ok, elapsed = run_and_measure_once(
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
        if ok:
            times.append(elapsed)

    if not times:
        return -1.0

    times.sort()
    return max(times[len(times) // 2], 0.0001)


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


def build_binary_path(target_dir: str, bin_name: str) -> str:
    return os.path.join(target_dir, "release", bin_name)


def build_mir_flag_string(mir_passes: List[str], mir_bits: List[int]) -> str:
    items = []
    for pass_name, bit in zip(mir_passes, mir_bits):
        prefix = "+" if bit else "-"
        items.append(f"{prefix}{pass_name}")
    return ",".join(items)


def build_llvm_pipeline_from_seq(master_root, llvm_flags: List[str], llvm_seq: List[int]) -> str:
    selected_llvm_ids = {flag for flag, bit in zip(llvm_flags, llvm_seq) if bit == 1}

    current_root = copy.deepcopy(master_root)
    new_children = []
    for ch in current_root.children:
        if prune_ast_in_place(ch, selected_llvm_ids):
            new_children.append(ch)
    current_root.children = new_children

    return "no-op-module" if not current_root.children else get_pipeline_string_from_root(current_root)


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


def generate_random_conf(x: int, total_length: int) -> List[int]:
    comb = bin(x).replace("0b", "")
    comb = "0" * (total_length - len(comb)) + comb
    return [int(s) for s in comb]


def discover_programs(src_bin_dir: str) -> List[str]:
    rs_files = sorted(glob.glob(os.path.join(src_bin_dir, "*.rs")))
    programs = []
    for f in rs_files:
        name = os.path.splitext(os.path.basename(f))[0]
        programs.append(name)
    return programs


def count_existing_samples(output_file: str) -> int:
    if not os.path.exists(output_file):
        return 0
    cnt = 0
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt


def parse_program_list(s: str) -> List[str]:
    if not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# ==========================================
# Profile One Program
# ==========================================

def profile_program(
    prog_name: str,
    args,
    mir_passes: List[str],
    llvm_flags: List[str],
    master_root,
):
    project_dir = args.project_dir
    output_file = os.path.join(args.output_dir, f"data_{prog_name}.jsonl")

    print(f"\n[{prog_name}] Project Dir: {project_dir}")
    print(f"[{prog_name}] Output Data: {output_file}")


    target_baseline = os.path.join(project_dir, f"target_pdcat_baseline_{prog_name}")
    target_tuned = os.path.join(project_dir, f"target_pdcat_tuned_{prog_name}")

    exe_baseline = build_binary_path(target_baseline, prog_name)
    exec_args = shlex.split(args.exec_param) if args.exec_param.strip() else []

    cleanup_paths(target_baseline, target_tuned)


    if not os.path.exists(output_file):
        open(output_file, "w", encoding="utf-8").close()

    existing_count = count_existing_samples(output_file)
    print(f"[{prog_name}] Existing samples in file: {existing_count}")

    # --------------------------------------

    # --------------------------------------
    print(f"[{prog_name}] 1. Pre-compiling baseline executable ...")
    baseline_cmd = build_cargo_rustc_command(prog_name, extra_rustc_args=[])
    baseline_env = make_build_env(target_baseline, args.use_rustc_bootstrap)

    if not execute_terminal_command(
        baseline_cmd,
        cwd=project_dir,
        env=baseline_env,
        verbose=args.verbose,
    ):
        print(f"[{prog_name}] Error compiling baseline executable.")
        return

    if not os.path.exists(exe_baseline):
        print(f"[{prog_name}] Error: baseline executable not found: {exe_baseline}")
        return

    baseline_run_cmd = [exe_baseline] + exec_args

    # --------------------------------------
    # 2. baseline warmup + median reference
    # --------------------------------------
    print(f"[{prog_name}] 2. Profiling baseline ...")
    ok_warmup, warmup_time = run_and_measure_once(
        baseline_run_cmd,
        cwd=project_dir,
        timeout=args.bootstrap_timeout,
    )
    if not ok_warmup:
        print(f"[{prog_name}] Error: baseline executable failed during warmup.")
        return

    dynamic_timeout = max(2.0 * warmup_time + 10.0, 10.0)

    baseline_time = get_median_execution_time(
        baseline_run_cmd,
        cwd=project_dir,
        runs=5,
        timeout=dynamic_timeout,
    )
    if baseline_time < 0:
        print(f"[{prog_name}] Error: failed to profile baseline.")
        return

    print(f"[{prog_name}] --> Baseline Median Time: {baseline_time:.6f}s")

    # --------------------------------------

    # --------------------------------------
    print(f"[{prog_name}] 3. Starting Random Sampling ({args.samples} attempts)...")

    success_count = 0
    total_flags = len(mir_passes) + len(llvm_flags)

    with open(output_file, "a", encoding="utf-8") as f_out:
        for i in range(args.samples):
            x = random.randint(0, 2 ** total_flags - 1)
            seq = generate_random_conf(x, total_flags)

            mir_seq = seq[:len(mir_passes)]
            llvm_seq = seq[len(mir_passes):]

            mir_str = build_mir_flag_string(mir_passes, mir_seq)
            llvm_str = build_llvm_pipeline_from_seq(master_root, llvm_flags, llvm_seq)

            tuned_extra_args = build_rustc_extra_args(
                mir_flag_str=mir_str,
                llvm_pipeline_str=llvm_str,
                use_unsound_mir_opts=args.use_unsound_mir_opts,
            )
            tuned_cmd = build_cargo_rustc_command(prog_name, tuned_extra_args)
            tuned_env = make_build_env(target_tuned, args.use_rustc_bootstrap)

            if not execute_terminal_command(
                tuned_cmd,
                cwd=project_dir,
                env=tuned_env,
                verbose=args.verbose,
            ):
                continue

            exe_tuned = build_binary_path(target_tuned, prog_name)
            if not os.path.exists(exe_tuned):
                continue

            tuned_run_cmd = [exe_tuned] + exec_args
            ok_tuned, time_tuned = run_and_measure_once(
                tuned_run_cmd,
                cwd=project_dir,
                timeout=dynamic_timeout,
            )
            if not ok_tuned:
                continue

            time_tuned = max(time_tuned, 0.0001)
            speedup = baseline_time / time_tuned

            if speedup > 1.0:
                success_count += 1
                record = {
                    "MIR": mir_seq,
                    "LLVM": llvm_seq,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                print(
                    f"[{prog_name}]  🌟 [{i+1}/{args.samples}] "
                    f"Found valid sequence! Speedup: {speedup:.4f}"
                )

    final_count = count_existing_samples(output_file)
    added_count = final_count - existing_count

    cleanup_paths(target_baseline, target_tuned)
    print(
        f"[{prog_name}] Finished. Added {added_count} new sequences. "
        f"Total now: {final_count}. File: {output_file}\n"
    )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Pool Data Generator for Rustc PDCAT")
    parser.add_argument("--pipeline_file", type=str, default="pass.txt")
    parser.add_argument("--exec_param", type=str, default="")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of random attempts per program")
    parser.add_argument("--bootstrap_timeout", type=float, default=600.0,
                        help="Timeout for baseline warmup run")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_unsound_mir_opts", action="store_true",
                        help="Append -Z unsound-mir-opts when necessary")
    parser.add_argument("--use_rustc_bootstrap", action="store_true",
                        help="Set RUSTC_BOOTSTRAP=1 if not using nightly")

    parser.add_argument(
        "--programs",
        type=str,
        default="",
        help="Comma-separated program list to run, e.g. seidel_2d,symm,syr2k"
    )
    parser.add_argument(
        "--force_programs",
        type=str,
        default="",
        help="Comma-separated program list to force rerun even if data file already exists"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip programs whose data_xxx.jsonl already exists, unless in --force_programs"
    )

    args = parser.parse_args()


    args.project_dir = str(Path(__file__).resolve().parents[3] / "Benchmarks" / "polybench-rs")
    args.output_dir = os.path.join(os.getcwd(), "data")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    src_bin_dir = os.path.join(args.project_dir, "src", "bin")
    if not os.path.isdir(src_bin_dir):
        print(f"Error: src/bin directory not found: {src_bin_dir}")
        sys.exit(1)

    print(f"Loading MIR and LLVM pipelines from {args.pipeline_file}...")
    try:
        mir_passes, llvm_pipeline_text = parse_pass_txt(args.pipeline_file)
    except FileNotFoundError:
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    master_root = build_ast(llvm_pipeline_text)
    llvm_flags = assign_flags(master_root)

    print(f"Total MIR passes: {len(mir_passes)}")
    print(f"Total LLVM flags: {len(llvm_flags)}")
    print(f"Total Search Space Dimension: {len(mir_passes) + len(llvm_flags)}")

    discovered_programs = discover_programs(src_bin_dir)
    if not discovered_programs:
        print("Error: no Rust benchmark programs found under src/bin.")
        sys.exit(1)

    requested_programs = parse_program_list(args.programs)
    force_programs = set(parse_program_list(args.force_programs))

    if requested_programs:
        programs = []
        discovered_set = set(discovered_programs)
        for p in requested_programs:
            if p not in discovered_set:
                print(f"Warning: program '{p}' not found under src/bin, skipped.")
            else:
                programs.append(p)
    else:
        programs = list(discovered_programs)

    if args.skip_existing:
        filtered = []
        for p in programs:
            output_file = os.path.join(args.output_dir, f"data_{p}.jsonl")
            if os.path.exists(output_file) and p not in force_programs:
                print(f"[Skip Existing] {p} -> {output_file}")
                continue
            filtered.append(p)
        programs = filtered

    if not programs:
        print("No programs to run after filtering.")
        sys.exit(0)

    print("=================================================================")
    print("Starting Batch Data Generation for Rust PolyBench Programs")
    print(f"Project Directory: {args.project_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Programs to run ({len(programs)}): {programs}")
    print("=================================================================")

    total_progs = len(programs)
    for idx, prog_name in enumerate(programs, start=1):
        print(f"\n--- Processing {idx}/{total_progs}: {prog_name} ---")
        profile_program(
            prog_name=prog_name,
            args=args,
            mir_passes=mir_passes,
            llvm_flags=llvm_flags,
            master_root=master_root,
        )

    print("=================================================================")
    print("All selected programs have been processed.")
    print("=================================================================")
