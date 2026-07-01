import argparse
import copy
import csv
import glob
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from llvm_pipeline_utils import (
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
    k_iter: int,
    all_flags: List[str],
    master_root,
    base_bc: str,
    opt_path: str,
    clang_path: str,
    baseline_run_cmd: List[str],
    baseline_time_ref: float,
    iter_csv_file: str,
    time_csv_file: str,
    state: dict,
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

    current_time = time.perf_counter() - state["time_zero"]
    state["iter_counter"] += 1
    iter_id = state["iter_counter"]

    is_new_best = speedup > state["best_speedup_so_far"]
    if is_new_best:
        state["best_speedup_so_far"] = speedup
        state["best_seq_so_far"] = list(independent)
        state["best_pipeline_so_far"] = pipeline_str
        state["best_tuned_time_so_far"] = time_tuned
        state["best_baseline_time_so_far"] = time_baseline
        state["best_iter_so_far"] = iter_id
        state["best_timestamp_so_far"] = current_time
        print(f"[New Record] Speedup: {speedup:.6f} at Iter {iter_id}")

    best_display = state["best_speedup_so_far"]
    print(
        f"Iter {iter_id:04d} | "
        f"Tuned: {time_tuned:.6f}s | "
        f"Baseline: {time_baseline:.6f}s | "
        f"Speedup: {speedup:.6f} | "
        f"Best So Far: {best_display:.6f}"
    )

    seq_str = ",".join(map(str, independent))
    append_csv_row(iter_csv_file, [
        iter_id,
        f"{current_time:.2f}",
        f"{time_tuned:.6f}",
        f"{time_baseline:.6f}",
        f"{speedup:.6f}",
        is_new_best,
        seq_str,
        pipeline_str,
    ])

    if current_time - state["last_log_time"] >= 20:
        best_seq_str = ",".join(map(str, state["best_seq_so_far"])) if state["best_seq_so_far"] else ""
        append_csv_row(time_csv_file, [
            f"{current_time:.2f}",
            f"{state['best_speedup_so_far']:.6f}",
            state["best_iter_so_far"],
            best_seq_str,
        ])
        state["last_log_time"] = current_time

    return speedup, time_tuned, time_baseline, pipeline_str


# ==========================================
# CompTuner
# ==========================================

class CompTuner:
    def __init__(
        self,
        dim: int,
        c1: float,
        c2: float,
        w: float,
        get_objective_score,
        random_seed: int,
        flags: List[str],
        state: dict,
        **kwargs,
    ):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim
        self.V = []
        self.pbest = []
        self.gbest = []
        self.p_fit = []
        self.fit = float("-inf")

        self.get_objective_score = get_objective_score
        self.random_seed = random_seed
        self.all_flags = flags
        self.kwargs = kwargs
        self.state = state

        self.kwargs["state"] = self.state

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def generate_random_conf(self, x: int) -> List[int]:
        comb = bin(x).replace("0b", "")
        comb = "0" * (self.dim - len(comb)) + comb
        return [1 if s == "1" else 0 for s in comb]

    def get_ei(self, preds, eta):
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f

    def runtime_predict(self, model, wait_for_train):
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        for tree in estimators:
            predictions = tree.predict(wait_for_train)
            sum_of_predictions += predictions
        average_prediction = sum_of_predictions / len(estimators)
        return [[wait_for_train[i], average_prediction[i]] for i in range(len(wait_for_train))]

    def getPrecision(self, model, seq):
        true_running, _, _, _ = self.get_objective_score(
            seq,
            k_iter=1000001,
            **self.kwargs,
        )

        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)

        if true_running <= 0:
            return 1.0, 0.0001
        return abs(true_running - acc_predict) / true_running, true_running

    def selectByDistribution(self, merged_predicted_objectives):
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        if diffs_sum == 0:
            return 0
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        idx = np.random.choice(index, p=probabilities)
        return idx

    def build_RF_by_CompTuner(self):
        initial_indep = []
        all_acc = []

        while len(initial_indep) < 2:
            x = random.randint(0, 2 ** self.dim - 1)
            seq = self.generate_random_conf(x)
            if seq not in initial_indep:
                initial_indep.append(seq)

        initial_dep = []
        for indep in initial_indep:
            speedup, _, _, _ = self.get_objective_score(
                indep,
                k_iter=0,
                **self.kwargs,
            )
            initial_dep.append(speedup)

        model = RandomForestRegressor(random_state=self.random_seed)
        model.fit(np.array(initial_indep), np.array(initial_dep))

        rec_size = 2
        while rec_size < 50:
            global_best = max(initial_dep)
            estimators = model.estimators_

            neighbors = []
            while len(neighbors) < 30000:
                x = random.randint(0, 2 ** self.dim - 1)
                seq = self.generate_random_conf(x)
                if seq not in neighbors:
                    neighbors.append(seq)

            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))

            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)

            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in initial_indep:
                    initial_indep.append(x[0])
                    acc, label = self.getPrecision(model, x[0])
                    initial_dep.append(label)
                    all_acc.append(acc)
                    flag = True

            rec_size += 1

            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[indx][0] in initial_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                initial_indep.append(merged_predicted_objectives[indx][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[indx][0])
                initial_dep.append(label)
                all_acc.append(acc)
                rec_size += 1

            model = RandomForestRegressor(random_state=self.random_seed)
            model.fit(np.array(initial_indep), np.array(initial_dep))

            if rec_size > 50 and len(all_acc) > 0 and np.mean(all_acc) < 0.04:
                break

        return model, initial_indep, initial_dep

    def getDistance(self, seq1, seq2):
        t1 = np.array(seq1)
        t2 = np.array(seq2)
        s1_norm = np.linalg.norm(t1)
        s2_norm = np.linalg.norm(t2)
        if s1_norm == 0.0 or s2_norm == 0.0:
            return 0.0
        return np.dot(t1, t2) / (s1_norm * s2_norm)

    def init_v(self, n, d, V_max, V_min):
        v = []
        for _ in range(n):
            vi = []
            for _ in range(d):
                a = random.random() * (V_max - V_min) + V_min
                vi.append(a)
            v.append(vi)
        return v

    def update_v(self, v, x, m, n, pbest, g, w, c1, c2, vmax, vmin):
        for i in range(m):
            a = random.random()
            b = random.random()
            for j in range(n):
                v[i][j] = (
                    w * v[i][j]
                    + c1 * a * (pbest[i][j] - x[i][j])
                    + c2 * b * (g[j] - x[i][j])
                )
                if v[i][j] < vmin:
                    v[i][j] = vmin
                if v[i][j] > vmax:
                    v[i][j] = vmax
        return v

    def run(self, budget=5000):
        model, initial_indep, initial_dep = self.build_RF_by_CompTuner()

        self.V = self.init_v(len(initial_indep), len(initial_indep[0]), 10, -10)
        self.fit = max(initial_dep)
        self.pbest = list(initial_indep)
        self.p_fit = list(initial_dep)
        self.gbest = initial_indep[initial_dep.index(max(initial_dep))]

        t = 0
        while True:
            elapsed = time.perf_counter() - self.state["time_zero"]
            if elapsed >= budget:
                break

            if t == 0:
                self.V = self.update_v(
                    self.V, initial_indep, len(initial_indep), len(initial_indep[0]),
                    self.pbest, self.gbest, self.w, self.c1, self.c2, 10, -10
                )
                for i in range(len(initial_indep)):
                    for j in range(len(initial_indep[0])):
                        a = random.random()
                        if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                            initial_indep[i][j] = 1
                        else:
                            initial_indep[i][j] = 0
                t += 1
            else:
                merged_predicted_objectives = self.runtime_predict(model, initial_indep)
                for i in range(len(merged_predicted_objectives)):
                    if merged_predicted_objectives[i][1] > self.p_fit[i]:
                        self.p_fit[i] = merged_predicted_objectives[i][1]
                        self.pbest[i] = merged_predicted_objectives[i][0]

                sort_merged_predicted_objectives = sorted(
                    merged_predicted_objectives,
                    key=lambda x: x[1],
                    reverse=True
                )
                current_best_seq = sort_merged_predicted_objectives[0][0]

                temp, _, _, _ = self.get_objective_score(
                    current_best_seq,
                    k_iter=1000002,
                    **self.kwargs,
                )

                if temp > self.fit:
                    self.gbest = current_best_seq
                    self.fit = temp
                    self.V = self.update_v(
                        self.V, initial_indep, len(initial_indep), len(initial_indep[0]),
                        self.pbest, self.gbest, self.w, self.c1, self.c2, 10, -10
                    )
                    for i in range(len(initial_indep)):
                        for j in range(len(initial_indep[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                                initial_indep[i][j] = 1
                            else:
                                initial_indep[i][j] = 0
                else:
                    avg_dis = 0.0
                    for i in range(1, len(merged_predicted_objectives)):
                        avg_dis += self.getDistance(merged_predicted_objectives[i][0], current_best_seq)
                    avg_dis = avg_dis / (len(initial_indep) - 1)

                    better_seed_indep = []
                    worse_seed_indep = []
                    better_seed_seq = []
                    worse_seed_seq = []
                    better_seed_pbest = []
                    worse_seed_pbest = []
                    better_seed_V = []
                    worse_seed_V = []

                    for i in range(len(merged_predicted_objectives)):
                        if self.getDistance(merged_predicted_objectives[i][0], current_best_seq) > avg_dis:
                            worse_seed_indep.append(i)
                            worse_seed_seq.append(merged_predicted_objectives[i][0])
                            worse_seed_pbest.append(self.pbest[i])
                            worse_seed_V.append(self.V[i])
                        else:
                            better_seed_indep.append(i)
                            better_seed_seq.append(merged_predicted_objectives[i][0])
                            better_seed_pbest.append(self.pbest[i])
                            better_seed_V.append(self.V[i])

                    if better_seed_seq:
                        V_for_better = self.update_v(
                            better_seed_V, better_seed_seq, len(better_seed_seq),
                            len(better_seed_seq[0]), better_seed_pbest, self.gbest,
                            self.w, 2 * self.c1, self.c2, 10, -10
                        )
                        for i in range(len(better_seed_seq)):
                            for j in range(len(better_seed_seq[0])):
                                a = random.random()
                                if 1.0 / (1 + math.exp(-V_for_better[i][j])) > a:
                                    better_seed_seq[i][j] = 1
                                else:
                                    better_seed_seq[i][j] = 0

                    if worse_seed_seq:
                        V_for_worse = self.update_v(
                            worse_seed_V, worse_seed_seq, len(worse_seed_seq),
                            len(worse_seed_seq[0]), worse_seed_pbest, self.gbest,
                            self.w, self.c1, 2 * self.c2, 10, -10
                        )
                        for i in range(len(worse_seed_seq)):
                            for j in range(len(worse_seed_seq[0])):
                                a = random.random()
                                if 1.0 / (1 + math.exp(-V_for_worse[i][j])) > a:
                                    worse_seed_seq[i][j] = 1
                                else:
                                    worse_seed_seq[i][j] = 0

                    for i in range(len(better_seed_seq)):
                        initial_indep[better_seed_indep[i]] = better_seed_seq[i]
                    for i in range(len(worse_seed_seq)):
                        initial_indep[worse_seed_indep[i]] = worse_seed_seq[i]

                t += 1

        total_elapsed = time.perf_counter() - self.state["time_zero"]

        print("\n=== Tuning Finished ===")
        print(f"Total Iterations: {self.state['iter_counter']}")
        print(f"Total Time: {total_elapsed:.2f}s")
        print(f"Best Iter: {self.state['best_iter_so_far']}")
        print(f"Best Speedup: {self.state['best_speedup_so_far']:.6f}")
        print(f"Best Tuned Time: {self.state['best_tuned_time_so_far']:.6f}s")
        print(f"Best Baseline Time: {self.state['best_baseline_time_so_far']:.6f}s")
        print(f"Best Seq: {self.state['best_seq_so_far']}")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CompTuner adapted for LLVM pipeline tuning (unified protocol, CSV logging)"
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
                        help="Execution parameter")
    parser.add_argument("--pipeline_file", type=str, default="o3_pipeline.txt",
                        help="Path to O3 pipeline file")
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
    cleanup_glob_patterns("*.bc", "*.o")

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

    state = {
        "time_zero": time.perf_counter(),
        "last_log_time": 0.0,
        "iter_counter": 0,
        "best_speedup_so_far": float("-inf"),
        "best_seq_so_far": [],
        "best_pipeline_so_far": "",
        "best_tuned_time_so_far": -1.0,
        "best_baseline_time_so_far": -1.0,
        "best_iter_so_far": -1,
        "best_timestamp_so_far": -1.0,
    }

    print(f"5. Starting CompTuner Loop. Results will be saved in: {OUT_DIR}")

    com_params = {
        "dim": len(all_flags),
        "get_objective_score": get_objective_score_llvm,
        "c1": 2,
        "c2": 2,
        "w": 0.6,
        "random_seed": 456,
        "flags": all_flags,
        "state": state,
        "all_flags": all_flags,
        "master_root": master_root,
        "base_bc": base_bc,
        "opt_path": args.opt_path,
        "clang_path": args.clang_path,
        "baseline_run_cmd": baseline_run_cmd,
        "baseline_time_ref": baseline_time_ref,
        "iter_csv_file": ITER_CSV_FILE,
        "time_csv_file": TIME_CSV_FILE,
        "exec_param": args.exec_param,
        "verbose": args.verbose,
    }

    com = CompTuner(**com_params)
    com.run(budget=args.budget)

    if not args.keep_artifacts:
        cleanup_paths(base_bc, baseline_bc, tuned_bc, exe_baseline, exe_tuned)
        cleanup_glob_patterns("*.bc", "*.o")