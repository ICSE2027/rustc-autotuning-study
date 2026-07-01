#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


ALGO_ROOT = PROJECT_ROOT / "Algorithm" / "ablation"

POLYBENCH_ROOT = PROJECT_ROOT / "Benchmarks" / "polybench-rs"
RUNTIMEBENCH_ROOT = (
    PROJECT_ROOT
    / "Benchmarks"
    / "rustc-perf"
    / "collector"
    / "runtime-benchmarks"
)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "rustc_ablation_results"


DEFAULT_BUDGET = 2500

DRY_RUN = False

PDCAT_PERMAX = 2.5
PDCAT_PERMIN = 0.8

USE_RUSTC_BOOTSTRAP = False
USE_UNSOUND_MIR_OPTS = False

TUNE_SCOPES = ["mir", "llvm"]

PROGRAMS = [
    ("polybench", "src/bin/2mm.rs"),
    ("polybench", "src/bin/3mm.rs"),
    ("polybench", "src/bin/adi.rs"),
    ("polybench", "src/bin/atax.rs"),
    ("polybench", "src/bin/bicg.rs"),
    ("polybench", "src/bin/cholesky.rs"),
    ("polybench", "src/bin/correlation.rs"),
    ("polybench", "src/bin/covariance.rs"),
    ("polybench", "src/bin/deriche.rs"),
    ("polybench", "src/bin/doitgen.rs"),
    ("polybench", "src/bin/durbin.rs"),
    ("polybench", "src/bin/fdtd_2d.rs"),
    ("polybench", "src/bin/floyd_warshall.rs"),
    ("polybench", "src/bin/gemm.rs"),
    ("polybench", "src/bin/gemver.rs"),
    ("polybench", "src/bin/gesummv.rs"),
    ("polybench", "src/bin/gramschmidt.rs"),
    ("polybench", "src/bin/heat_3d.rs"),
    ("polybench", "src/bin/jacobi_1d.rs"),
    ("polybench", "src/bin/jacobi_2d.rs"),
    ("polybench", "src/bin/lu.rs"),
    ("polybench", "src/bin/ludcmp.rs"),
    ("polybench", "src/bin/mvt.rs"),
    ("polybench", "src/bin/nussinov.rs"),
    ("polybench", "src/bin/seidel_2d.rs"),
    ("polybench", "src/bin/symm.rs"),
    ("polybench", "src/bin/syr2k.rs"),
    ("polybench", "src/bin/syrk.rs"),
    ("polybench", "src/bin/trisolv.rs"),
    ("polybench", "src/bin/trmm.rs"),
    ("runtimebench", "bufreader"),
    ("runtimebench", "compression"),
    ("runtimebench", "css"),
    ("runtimebench", "fmt"),
    ("runtimebench", "hashmap"),
    ("runtimebench", "nbody"),
    ("runtimebench", "nes"),
    ("runtimebench", "parsing"),
    ("runtimebench", "raytracer"),
    ("runtimebench", "svg"),
    ("runtimebench", "text-search"),
]

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
            max_idx = max(max_idx, idx)

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


def get_program_name(program_spec) -> str:
    bench_type, program_rel = program_spec

    if bench_type == "polybench":
        return Path(program_rel).stem

    if bench_type == "runtimebench":
        return program_rel

    raise ValueError(f"Unknown benchmark type: {bench_type}")


def get_program_source_file(program_spec) -> Path:
    bench_type, program_rel = program_spec

    if bench_type == "polybench":
        return POLYBENCH_ROOT / program_rel

    if bench_type == "runtimebench":
        return RUNTIMEBENCH_ROOT / program_rel / "src" / "main.rs"

    raise ValueError(f"Unknown benchmark type: {bench_type}")


def common_rustc_args(out_dir: Path, source_file: Path, tune_scope: str):
    return [
        "--out_dir", str(out_dir),
        "--source_path", str(source_file),
        "--budget", str(DEFAULT_BUDGET),
        "--tune_scope", tune_scope,
    ]


def maybe_add_common_rust_flags(cmd: list):
    if USE_UNSOUND_MIR_OPTS:
        cmd.append("--use_unsound_mir_opts")
    if USE_RUSTC_BOOTSTRAP:
        cmd.append("--use_rustc_bootstrap")
    return cmd


def build_method_command(method: str, program_rel: str, run_dir: Path, tune_scope: str):
    source_file = get_program_source_file(program_rel)

    if not source_file.is_file():
        raise FileNotFoundError(f"Program file does not exist: {source_file}")

    if method == "RIO":
        algo_dir = ALGO_ROOT / "RIO-algorithm"
        script = algo_dir / "RIO_rustc.py"
        cmd = [
            "python", str(script),
            *common_rustc_args(run_dir, source_file, tune_scope),
            "--pipeline_file", str(algo_dir / "pass.txt"),
        ]
        return algo_dir, maybe_add_common_rust_flags(cmd)

    if method == "SRTuner":
        algo_dir = ALGO_ROOT / "SRTuner-algorithm"
        script = algo_dir / "SRTuner_rustc.py"
        cmd = [
            "python", str(script),
            *common_rustc_args(run_dir, source_file, tune_scope),
            "--pipeline_file", str(algo_dir / "pass.txt"),
        ]
        return algo_dir, maybe_add_common_rust_flags(cmd)

    if method == "CompTuner":
        algo_dir = ALGO_ROOT / "CompTuner-algorithm"
        script = algo_dir / "CompTuner_rustc.py"
        cmd = [
            "python", str(script),
            *common_rustc_args(run_dir, source_file, tune_scope),
            "--pipeline_file", str(algo_dir / "pass.txt"),
        ]
        return algo_dir, maybe_add_common_rust_flags(cmd)

    if method == "CFSCA":
        algo_dir = ALGO_ROOT / "CFSCA-algorithm"
        script = algo_dir / "CFSCA_rustc.py"
        cmd = [
            "python", str(script),
            *common_rustc_args(run_dir, source_file, tune_scope),
            "--pipeline_file", str(algo_dir / "pass.txt"),
        ]
        return algo_dir, maybe_add_common_rust_flags(cmd)

    if method == "PDCAT":
        algo_dir = ALGO_ROOT / "PDCAT-algorithm"
        script = algo_dir / "PDCAT_rustc.py"
        merged_trainset_path = run_dir / "merged_trainset.jsonl"

        cmd = [
            "python", str(script),
            *common_rustc_args(run_dir, source_file, tune_scope),
            "--pipeline_file", str(algo_dir / "pass.txt"),
            "--data_dir", str(algo_dir / "data"),
            "--merged_trainset_path", str(merged_trainset_path),
            "--constraints_path", str(algo_dir / "constraints.txt"),
            "--permax", str(PDCAT_PERMAX),
            "--permin", str(PDCAT_PERMIN),
        ]
        return algo_dir, maybe_add_common_rust_flags(cmd)

    if method == "GroupTuner":
        algo_dir = ALGO_ROOT / "GroupTuner-algorithm"
        script = algo_dir / "GroupTuner_rustc.py"
        cmd = [
            "python", str(script),
            *common_rustc_args(run_dir, source_file, tune_scope),
            "--pipeline_file", str(algo_dir / "pass.txt"),
            "--group_file", str(algo_dir / "pass_group.txt"),
        ]
        return algo_dir, maybe_add_common_rust_flags(cmd)

    raise ValueError(f"Unknown method: {method}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rustc MIR/LLVM ablation batch runner"
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
        f"\n========== Rustc Ablation Batch Start: {now_str()} | repeats {repeat_start}-{repeat_end} | budget={DEFAULT_BUDGET}s per scope =========="
    )
    append_text(
        failed_log,
        f"\n========== Rustc Ablation Failure Log Start: {now_str()} | repeats {repeat_start}-{repeat_end} =========="
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
                for tune_scope in TUNE_SCOPES:
                    total_jobs += 1

                    run_dir = repeat_dir / method / program_name / tune_scope
                    ensure_dir(run_dir)

                    cmd_log = run_dir / "run.log"

                    try:
                        cwd, cmd = build_method_command(
                            method=method,
                            program_rel=program_rel,
                            run_dir=run_dir,
                            tune_scope=tune_scope,
                        )
                    except Exception as e:
                        failed_jobs += 1
                        msg = (
                            f"[{now_str()}] BUILD_CMD_FAILED | repeat={repeat_idx} | "
                            f"method={method} | program={program_name} | scope={tune_scope} | reason={e}"
                        )
                        append_text(master_log, msg)
                        append_text(failed_log, msg)
                        continue

                    append_text(
                        master_log,
                        f"[{now_str()}] START | repeat={repeat_idx} | method={method} | program={program_name} | scope={tune_scope}"
                    )

                    ok, retcode = run_cmd(cmd, cwd, cmd_log)

                    if ok:
                        success_jobs += 1
                        append_text(
                            master_log,
                            f"[{now_str()}] DONE  | repeat={repeat_idx} | method={method} | program={program_name} | scope={tune_scope}"
                        )
                    else:
                        failed_jobs += 1
                        msg = (
                            f"[{now_str()}] FAILED | repeat={repeat_idx} | "
                            f"method={method} | program={program_name} | scope={tune_scope} | return_code={retcode}"
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
        f"========== Rustc Ablation Batch End: {now_str()} | repeats {repeat_start}-{repeat_end} ==========\n"
    )

    print("Batch jobs completed.")
    print(f"Repeat range for this run: {repeat_start} - {repeat_end}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Master log: {master_log}")
    print(f"Failure log: {failed_log}")
    print(f"TOTAL={total_jobs}, SUCCESS={success_jobs}, FAILED={failed_jobs}")


if __name__ == "__main__":
    main()
