# Rust Compiler Autotuning Study Artifact

This repository contains the compact artifact for an empirical study of compiler autotuning in `rustc` and its relationship to LLVM-side tuning. It is designed for reviewers to inspect the reported data, connect each research question to its figures, and rerun selected experiments if needed.

The full evaluation is expensive: the paper reports 3,791 CPU-hours. For most checks, start from the summarized JSON files in `data/`.

## What Is Included

```text
.
|-- data/       summarized speedup data used by the paper
|-- figures/    rendered figures for the RQs
|-- LLVMTuner/  LLVM-side tuning methods and batch runner
`-- RustCTuner/ rustc-side tuning methods and batch runners
```

Benchmark source trees and raw output directories are intentionally omitted to keep the artifact compact and anonymous-review friendly.

## Research Questions, Data, and Figures

### RQ1: rustc tuning effectiveness

Focus: Effectiveness of six tuning methods on 41 Rust benchmarks.

Data: `data/rustc_benchmarks_speedups.json`

Figure: `figures/rustc_method_improvement_bins_line.png`

Runners: `RustCTuner/run_rustc_batch.py`, `RustCTuner/run_runtimebench_batch.py`

### RQ2: LLVM guidance for rustc method selection

Focus: Whether LLVM-side behavior predicts rustc-side behavior on 30 paired PolyBench programs.

Data: `data/llvm_polybench_speedups.json`, plus the 30 paired PolyBench-rs entries in `data/rustc_benchmarks_speedups.json`

Figures: `figures/llvm_rustc_avg_improvement_slope_dumbbell.png`, `figures/Rustc_minus_LLVM_program_method_direction_heatmap_landscape.png`

Runners: `LLVMTuner/run_llvm_batch.py`, `RustCTuner/run_rustc_batch.py`

### RQ3: direct LLVM-to-rustc configuration reuse

Focus: Whether LLVM-selected pass configurations can be reused directly in rustc.

Data: `data/llvm_to_rustc_polybench_transfer_speedups.json`

Figure: `figures/rerun_llvm_to_rustc_transfer_improvement_bins_bar.png`

Rerun path: summarized transfer results in `data/`

### RQ4: MIR and LLVM stage allocation

Focus: MIR-only, LLVM-only, joint MIR+LLVM tuning, and simple MIR+LLVM configuration combination.

Data: `data/rustc_mir_llvm_ablation_speedups.json` for MIR-only and LLVM-only results; `data/rustc_benchmarks_speedups.json` for joint MIR+LLVM checkpoints; `data/rustc_mir_llvm_combined_joint_speedups.json` for combined single-stage configurations versus jointly tuned configurations

Figure: `figures/rustc_ablation_best_count_grouped_bar.png`

Runner: `RustCTuner/run_rustc_ablation_batch.py`

## Data Format

All JSON files store speedups, not percentage improvements. For example, `1.20` means the tuned configuration is 20% faster than the default optimized baseline.

`data/llvm_polybench_speedups.json`

Median LLVM-side results for 30 PolyBench/C programs and six methods.

`data/rustc_benchmarks_speedups.json`

Median rustc-side results for 41 Rust benchmarks: 30 PolyBench-rs programs and 11 rustc-perf runtime benchmarks.

Both files use:

```json
{
  "program": {
    "method": {
      "1250s": 1.0,
      "2500s": 1.0,
      "5000s": 1.0
    }
  }
}
```

`data/llvm_to_rustc_polybench_transfer_speedups.json`

Direct reuse results for the top five LLVM-selected configurations on each paired PolyBench program:

```json
{
  "program": {
    "method": {
      "rank_1": 1.0,
      "rank_2": 1.0,
      "rank_3": 1.0,
      "rank_4": 1.0,
      "rank_5": 1.0
    }
  }
}
```

`data/rustc_mir_llvm_ablation_speedups.json`

Single-stage ablation results for 41 Rust benchmarks using `CompTuner`, `PDCAT`, and `GroupTuner`. The ablation runs use a 2500s budget, so this file records only `1250s` and `2500s` checkpoints. Stage keys are lowercase in the JSON:

```json
{
  "program": {
    "method": {
      "mir": {"1250s": 1.0, "2500s": 1.0},
      "llvm": {"1250s": 1.0, "2500s": 1.0}
    }
  }
}
```

For RQ4, joint MIR+LLVM checkpoints are the `CompTuner`, `PDCAT`, and `GroupTuner` entries in `data/rustc_benchmarks_speedups.json`.

`data/rustc_mir_llvm_combined_joint_speedups.json`

RQ4 comparison between simple MIR+LLVM configuration combination and jointly tuned MIR+LLVM configurations. `combined_mir_llvm` is the simple-combination speedup used in the paper's RQ4 combination comparison; `joint_mir_llvm` is the corresponding 5000s joint MIR+LLVM speedup from `data/rustc_benchmarks_speedups.json`:

```json
{
  "program": {
    "method": {
      "combined_mir_llvm": 1.0,
      "joint_mir_llvm": 1.0
    }
  }
}
```

## Figures

The `figures/` directory contains PNG versions of the paper figures referenced in the RQ descriptions above. The short notes below summarize what each figure is intended to show.

**RQ1: rustc improvement distribution**

![RQ1 rustc improvement distribution](figures/rustc_method_improvement_bins_line.png)

This figure shows the per-program final improvement distribution after the 5000s rustc tuning budget. Negative bins indicate degradation relative to the default Rust release configuration. It is used to check whether each method usually improves Rust programs or occasionally hurts performance.

**RQ2: average LLVM-to-rustc method trend**

![RQ2 average improvement slope](figures/llvm_rustc_avg_improvement_slope_dumbbell.png)

This figure compares the average improvement of the same six tuning methods on paired LLVM and rustc benchmarks. It shows that absolute improvement values decrease from LLVM to rustc, while the coarse strong/weak grouping of methods is partly preserved.

**RQ2: program-method transfer differences**

![RQ2 program-method heatmap](figures/Rustc_minus_LLVM_program_method_direction_heatmap_landscape.png)

This heatmap shows rustc-minus-LLVM improvement differences for each program and method. Green cells mean the Rust version benefits more from tuning, while red cells mean the LLVM/C version benefits more. It highlights that transfer behavior is strongly program-dependent.

**RQ3: direct configuration reuse**

![RQ3 transfer distribution](figures/rerun_llvm_to_rustc_transfer_improvement_bins_bar.png)

This figure reports the improvement distribution when LLVM-selected pass configurations are reused directly in rustc. Negative values indicate performance degradation on rustc. It supports the paper's conclusion that LLVM configurations may provide candidate signals but should not be used without rustc-side validation.

**RQ4: MIR, LLVM, and joint tuning**

![RQ4 ablation best counts](figures/rustc_ablation_best_count_grouped_bar.png)

This figure counts how often MIR-only, LLVM-only, and joint MIR+LLVM tuning achieve the best result under the 2500s budget. It shows that short budgets do not consistently favor joint tuning, and that both MIR and LLVM can be useful search targets depending on the method and program.

## Rerunning Experiments

The full experiment is expensive. For availability checks, reviewers can inspect the JSON files in `data/` directly. To rerun selected experiments, use the following steps.

### Reference Environment

The paper used the following environment:

```text
OS: Ubuntu 22.04
CPU: 32 vCPUs on an AMD EPYC 9654 96-Core Processor
Memory: 60 GB
rustc: 1.90.0-nightly
LLVM/Clang: 20.1.8
```

GPU is not required. The tuning scripts measure CPU runtime.

### Step 0: Clone the Artifact

```bash
git clone https://github.com/ICSE2027/rustc-autotuning-study.git
cd rustc-autotuning-study
```

Install the Python packages used by the tuning methods:

```bash
python -m pip install numpy scipy scikit-learn pandas
```

### Step 1: Install Compiler Toolchains

Install a nightly Rust toolchain and LLVM/Clang 20. The scripts assume these commands are available:

```bash
rustc --version
cargo --version
clang-20 --version
opt-20 --version
```

The paper uses `rustc 1.90.0-nightly` and LLVM/Clang `20.1.8`. Other versions may run, but exact performance numbers may differ.

### Step 2: Clone Benchmark Suites

Benchmark source trees are not included in this compact anonymous artifact. From the artifact repository root, clone the upstream suites into the paths expected by the batch runners.

LLVM experiments use [MatthiasJReisinger/PolyBenchC-4.2.1](https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1):

```bash
mkdir -p LLVMTuner/Benchmarks
cd LLVMTuner/Benchmarks
git clone https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git polyBench
cd ../..
```

Rust experiments use [JRF63/polybench-rs](https://github.com/JRF63/polybench-rs) and [rust-lang/rustc-perf](https://github.com/rust-lang/rustc-perf):

```bash
mkdir -p RustCTuner/Benchmarks
cd RustCTuner/Benchmarks
git clone https://github.com/JRF63/polybench-rs.git polybench-rs
git clone https://github.com/rust-lang/rustc-perf.git rustc-perf
cd ../..
```

The runtime benchmark runner uses `RustCTuner/Benchmarks/rustc-perf/collector/runtime-benchmarks/` inside the cloned rustc-perf tree.

The batch runners derive `PROJECT_ROOT` from their own file location. If your local layout differs, edit the configuration constants at the top of each runner:

```python
BENCH_ROOT = Path(...)
OUTPUT_ROOT = Path(...)
CLANG_PATH = "clang-20"
OPT_PATH = "opt-20"
```

### Step 3: Run Selected Experiments

Run a single repeat first:

```bash
cd LLVMTuner
python run_llvm_batch.py --num_repeats 1

cd ../RustCTuner
python run_rustc_batch.py --num_repeats 1
python run_runtimebench_batch.py --num_repeats 1
python run_rustc_ablation_batch.py --num_repeats 1
```

The paper reports five independent repeats for most experiments. To reproduce that protocol, use:

```bash
cd LLVMTuner
python run_llvm_batch.py --num_repeats 5

cd ../RustCTuner
python run_rustc_batch.py --num_repeats 5
python run_runtimebench_batch.py --num_repeats 5
python run_rustc_ablation_batch.py --num_repeats 5
```

All runners support `--start_repeat N` for manual resume control. Logs are written under `OUTPUT_ROOT`, including `master_run.log`, `failed_programs.log`, and per-program `run.log` files. The default output directories are under `LLVMTuner/results/` and `RustCTuner/results/`, which are ignored by Git.
