#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ALGO_ROOT = PROJECT_ROOT / "Algorithm" / "overall"
BENCH_ROOT = PROJECT_ROOT / "Benchmarks" / "polyBench"
UTIL_DIR = BENCH_ROOT / "utilities"
POLYBENCH_C = UTIL_DIR / "polybench.c"


OUTPUT_ROOT = PROJECT_ROOT / "results" / "llvm_batch_results"


CLANG_PATH = "clang-20"
OPT_PATH = "opt-20"


DEFAULT_BUDGET = 5000


DRY_RUN = False


PDCAT_PERMAX = 2.0
PDCAT_PERMIN = 0.8

# ============================================================


# ============================================================

PROGRAMS = [
    "linear-algebra/kernels/2mm",
    "linear-algebra/kernels/3mm",
    "stencils/adi",
    "linear-algebra/kernels/atax",
    "linear-algebra/kernels/bicg",
    "linear-algebra/solvers/cholesky",
    "datamining/correlation",
    "datamining/covariance",
    "medley/deriche",
    "linear-algebra/kernels/doitgen",
    "linear-algebra/solvers/durbin",
    "stencils/fdtd-2d",
    "medley/floyd-warshall",
    "linear-algebra/blas/gemm",
    "linear-algebra/blas/gemver",
    "linear-algebra/blas/gesummv",
    "linear-algebra/solvers/gramschmidt",
    "stencils/heat-3d",
    "stencils/jacobi-1d",
    "stencils/jacobi-2d",
    "linear-algebra/solvers/lu",
    "linear-algebra/solvers/ludcmp",
    "linear-algebra/kernels/mvt",
    "medley/nussinov",
    "stencils/seidel-2d",
    "linear-algebra/blas/symm",
    "linear-algebra/blas/syr2k",
    "linear-algebra/blas/syrk",
    "linear-algebra/solvers/trisolv",
    "linear-algebra/blas/trmm",
]

# ============================================================

# ============================================================

METHODS = [
    "RIO",
    "SRTuner",
    "CompTuner",
    "CFSCA",
    "PDCAT",
    "GroupTuner",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def detect_next_repeat_index(output_root: Path) -> int:
    max_idx = 0
    pattern = re.compile(r"^repeat_(\d+)$")

    if not output_root.exists():
        return 1

    for item in output_root.iterdir():
        if not item.is_dir():
            continue
        m = pattern.match(item.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx

    return max_idx + 1


def run_cmd(cmd, cwd: Path, log_file: Path):
    ensure_dir(log_file.parent)

    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    append_text(log_file, f"[{now_str()}] CMD: {cmd_str}")
    append_text(log_file, f"[{now_str()}] CWD: {cwd}")

    if DRY_RUN:
        append_text(log_file, f"[{now_str()}] DRY_RUN=True, command not executed.")
        return True, 0

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )

    append_text(log_file, f"[{now_str()}] RETURN CODE: {proc.returncode}")
    if proc.stdout:
        append_text(log_file, "[STDOUT]")
        append_text(log_file, proc.stdout.rstrip())
    if proc.stderr:
        append_text(log_file, "[STDERR]")
        append_text(log_file, proc.stderr.rstrip())

    return proc.returncode == 0, proc.returncode


def get_program_name(program_rel: str) -> str:
    return Path(program_rel).name


def get_program_source_dir(program_rel: str) -> Path:
    return BENCH_ROOT / program_rel


def common_llvm_args(out_dir: Path, source_dir: Path):
    return [
        "--out_dir", str(out_dir),
        "--source_path", str(source_dir),
        "--clang_path", CLANG_PATH,
        "--opt_path", OPT_PATH,
        "--include_dir", str(UTIL_DIR),
        "--extra_c_file", str(POLYBENCH_C),
        "--budget", str(DEFAULT_BUDGET),
    ]


def build_method_command(method: str, program_rel: str, run_dir: Path):
    source_dir = get_program_source_dir(program_rel)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Program directory does not exist: {source_dir}")

    if method == "RIO":
        algo_dir = ALGO_ROOT / "RIO-algorithm"
        script = algo_dir / "RIO_llvm.py"
        cmd = [
            "python", str(script),
            *common_llvm_args(run_dir, source_dir),
            "--pipeline_file", str(algo_dir / "o3_pipeline.txt"),
        ]
        return algo_dir, cmd

    if method == "SRTuner":
        algo_dir = ALGO_ROOT / "SRTuner-algorithm"
        script = algo_dir / "SRTuner_llvm.py"
        cmd = [
            "python", str(script),
            *common_llvm_args(run_dir, source_dir),
            "--pipeline_file", str(algo_dir / "o3_pipeline.txt"),
        ]
        return algo_dir, cmd

    if method == "CompTuner":
        algo_dir = ALGO_ROOT / "CompTuner-algorithm"
        script = algo_dir / "CompTuner_llvm.py"
        cmd = [
            "python", str(script),
            *common_llvm_args(run_dir, source_dir),
            "--pipeline_file", str(algo_dir / "o3_pipeline.txt"),
        ]
        return algo_dir, cmd

    if method == "CFSCA":
        algo_dir = ALGO_ROOT / "CFSCA-algorithm"
        script = algo_dir / "CFSCA_llvm.py"
        cmd = [
            "python", str(script),
            *common_llvm_args(run_dir, source_dir),
            "--pipeline_file", str(algo_dir / "o3_pipeline.txt"),
        ]
        return algo_dir, cmd

    if method == "PDCAT":
        algo_dir = ALGO_ROOT / "PDCAT-algorithm"
        script = algo_dir / "PDCAT_llvm.py"
        merged_trainset_path = run_dir / "merged_trainset.txt"

        cmd = [
            "python", str(script),
            *common_llvm_args(run_dir, source_dir),
            "--pipeline_file", str(algo_dir / "o3_pipeline.txt"),
            "--data_dir", str(algo_dir / "data"),
            "--merged_trainset_path", str(merged_trainset_path),
            "--constraints_path", str(algo_dir / "constraints.txt"),
            "--permax", str(PDCAT_PERMAX),
            "--permin", str(PDCAT_PERMIN),
        ]
        return algo_dir, cmd

    if method == "GroupTuner":
        algo_dir = ALGO_ROOT / "GroupTuner"
        script = algo_dir / "GroupTuner_llvm.py"
        cmd = [
            "python", str(script),
            *common_llvm_args(run_dir, source_dir),
            "--pipeline_file", str(algo_dir / "o3_pipeline.txt"),
            "--group_file", str(algo_dir / "opts_group.txt"),
        ]
        return algo_dir, cmd

    raise ValueError(f"Unknown method: {method}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LLVM batch runner with resumable repeat indexing"
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        required=True,
        help="How many new repeats to run this time"
    )
    parser.add_argument(
        "--start_repeat",
        type=int,
        default=0,
        help="Optional manual starting repeat index; 0 means auto-detect"
    )
    args = parser.parse_args()

    if args.num_repeats <= 0:
        print("--num_repeats must be greater than 0")
        sys.exit(1)

    if not PROGRAMS:
        print("PROGRAMS is empty; configure the program list first.")
        sys.exit(1)

    ensure_dir(OUTPUT_ROOT)

    if args.start_repeat > 0:
        repeat_start = args.start_repeat
    else:
        repeat_start = detect_next_repeat_index(OUTPUT_ROOT)

    repeat_end = repeat_start + args.num_repeats - 1

    master_log = OUTPUT_ROOT / "master_run.log"
    failed_log = OUTPUT_ROOT / "failed_programs.log"

    append_text(
        master_log,
        f"\n========== LLVM Batch Run Start: {now_str()} | repeats {repeat_start}-{repeat_end} =========="
    )
    append_text(
        failed_log,
        f"\n========== LLVM Batch Failure Log Start: {now_str()} | repeats {repeat_start}-{repeat_end} =========="
    )

    total_jobs = 0
    success_jobs = 0
    failed_jobs = 0

    for repeat_idx in range(repeat_start, repeat_end + 1):
        repeat_dir = OUTPUT_ROOT / f"repeat_{repeat_idx}"
        ensure_dir(repeat_dir)

        append_text(master_log, f"[{now_str()}] ===== Start repeat_{repeat_idx} =====")

        for program_rel in PROGRAMS:
            program_name = get_program_name(program_rel)
            append_text(master_log, f"[{now_str()}] ---- Program: {program_name} ----")

            for method in METHODS:
                total_jobs += 1

                method_dir = repeat_dir / method
                ensure_dir(method_dir)

                run_dir = method_dir / program_name
                ensure_dir(run_dir)

                cmd_log = run_dir / "run.log"

                try:
                    cwd, cmd = build_method_command(method, program_rel, run_dir)
                except Exception as e:
                    failed_jobs += 1
                    msg = (
                        f"[{now_str()}] BUILD_CMD_FAILED | repeat={repeat_idx} | "
                        f"method={method} | program={program_name} | reason={e}"
                    )
                    append_text(master_log, msg)
                    append_text(failed_log, msg)
                    continue

                append_text(
                    master_log,
                    f"[{now_str()}] START | repeat={repeat_idx} | method={method} | program={program_name}"
                )

                ok, retcode = run_cmd(cmd, cwd, cmd_log)

                if ok:
                    success_jobs += 1
                    append_text(
                        master_log,
                        f"[{now_str()}] DONE  | repeat={repeat_idx} | method={method} | program={program_name}"
                    )
                else:
                    failed_jobs += 1
                    msg = (
                        f"[{now_str()}] FAILED | repeat={repeat_idx} | "
                        f"method={method} | program={program_name} | return_code={retcode}"
                    )
                    append_text(master_log, msg)
                    append_text(failed_log, msg)

        append_text(master_log, f"[{now_str()}] ===== End repeat_{repeat_idx} =====")

    append_text(
        master_log,
        f"[{now_str()}] TOTAL={total_jobs}, SUCCESS={success_jobs}, FAILED={failed_jobs}"
    )
    append_text(
        master_log,
        f"========== LLVM Batch Run End: {now_str()} | repeats {repeat_start}-{repeat_end} ==========\n"
    )

    print("Batch jobs completed.")
    print(f"Repeat range for this run: {repeat_start} - {repeat_end}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Master log: {master_log}")
    print(f"Failure log: {failed_log}")
    print(f"TOTAL={total_jobs}, SUCCESS={success_jobs}, FAILED={failed_jobs}")


if __name__ == "__main__":
    main()
