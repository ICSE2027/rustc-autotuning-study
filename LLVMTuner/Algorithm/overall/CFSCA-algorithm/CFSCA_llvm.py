import argparse
import copy
import csv
import glob
import itertools
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

from getRelated_llvm import (
    obtain_c_code,
    remove_commentsandinclude_from_c_code,
    get_related_flags,
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

    print(
        f"Iter {iter_id:04d} | "
        f"Tuned: {time_tuned:.6f}s | "
        f"Baseline: {time_baseline:.6f}s | "
        f"Speedup: {speedup:.6f} | "
        f"Best So Far: {state['best_speedup_so_far']:.6f}"
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
# CFSCA
# ==========================================

class CFSCA:
    def __init__(
        self,
        dim,
        get_objective_score,
        seed,
        related_flags,
        flags,
        state,
        **kwargs,
    ):
        self.dim = dim
        self.get_objective_score = get_objective_score
        self.seed = seed
        self.related = related_flags
        self.all_flags = flags

        self.critical = []
        self.global_best_per = float("-inf")
        self.global_best_seq = []
        self.global_best_pipeline = ""

        self.state = state
        self.kwargs = kwargs
        self.kwargs["state"] = self.state

        random.seed(self.seed)
        np.random.seed(self.seed)

    def generate_random_conf(self, x):
        comb = bin(x).replace("0b", "")
        comb = "0" * (self.dim - len(comb)) + comb
        conf = []
        for s in comb:
            conf.append(1 if s == "1" else 0)
        return conf

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
            all_flags=self.all_flags,
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
                all_flags=self.all_flags,
                **self.kwargs,
            )
            initial_dep.append(speedup)

        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(initial_indep), np.array(initial_dep))

        rec_size = 2
        all_acc = []

        while rec_size < 11:
            model = RandomForestRegressor(random_state=self.seed)
            model.fit(np.array(initial_indep), np.array(initial_dep))

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

        self.global_best_per = max(initial_dep)
        self.global_best_seq = initial_indep[initial_dep.index(max(initial_dep))]
        return model, initial_indep, initial_dep

    def get_critical_flags(self, model, initial_indep, initial_dep):
        if not self.related:
            print("No related flags identified. Falling back to all flags.")
            return [], model

        candidate_seq = []
        initial_indep_temp = copy.deepcopy(initial_indep)
        initial_dep_temp = copy.deepcopy(initial_dep)

        while len(candidate_seq) < 30000:
            x = random.randint(0, 2 ** self.dim - 1)
            seq = self.generate_random_conf(x)
            if seq not in candidate_seq:
                candidate_seq.append(seq)

        all_per = self.runtime_predict(model, candidate_seq)
        candidate_per = [item[1] for item in all_per]
        pos_seq = [0] * len(self.related)

        now_best = max(candidate_per)
        now_best_seq = candidate_seq[candidate_per.index(now_best)]

        now_best, _, _, _ = self.get_objective_score(
            now_best_seq,
            k_iter=1000002,
            all_flags=self.all_flags,
            **self.kwargs,
        )

        if now_best > self.global_best_per:
            self.global_best_per = now_best
            self.global_best_seq = now_best_seq

        initial_indep_temp.append(now_best_seq)
        initial_dep_temp.append(now_best)

        model_new = RandomForestRegressor(random_state=self.seed)
        model_new.fit(np.array(initial_indep_temp), np.array(initial_dep_temp))

        for idx in range(len(self.related)):
            new_candidate = []
            for j in range(len(candidate_seq)):
                seq = copy.deepcopy(candidate_seq[j])
                seq[self.related[idx]] = 1 - seq[self.related[idx]]
                new_candidate.append(seq)

            predicted_pairs = self.runtime_predict(model_new, new_candidate)
            new_per = [item[1] for item in predicted_pairs]
            new_seq = [item[0] for item in predicted_pairs]
            new_best_seq = new_seq[new_per.index(max(new_per))]

            new_best, _, _, _ = self.get_objective_score(
                new_best_seq,
                k_iter=1000002,
                all_flags=self.all_flags,
                **self.kwargs,
            )

            if new_best > self.global_best_per:
                self.global_best_per = new_best
                self.global_best_seq = new_best_seq

            for l in range(len(new_candidate)):
                if (candidate_per[l] > new_per[l] and new_candidate[l][self.related[idx]] == 1) or \
                   (candidate_per[l] < new_per[l] and new_candidate[l][self.related[idx]] == 0):
                    pos_seq[idx] -= 1
                else:
                    pos_seq[idx] += 1

            initial_indep_temp.append(new_best_seq)
            initial_dep_temp.append(new_best)

            model_new = RandomForestRegressor(random_state=self.seed)
            model_new.fit(np.array(initial_indep_temp), np.array(initial_dep_temp))

        sort_pos = sorted(enumerate(pos_seq), key=lambda x: x[1], reverse=True)
        critical_flag_idx = []
        limit = min(10, len(self.related))
        for i in range(limit):
            critical_flag_idx.append(self.related[sort_pos[i][0]])

        return critical_flag_idx, model_new

    def searchBycritical(self, critical_flag):
        if not critical_flag:
            seqs = []
            while len(seqs) < 1024 * 40:
                x = random.randint(0, 2 ** self.dim - 1)
                seq = self.generate_random_conf(x)
                if seq not in seqs:
                    seqs.append(seq)
            return seqs

        permutations = list(itertools.product([0, 1], repeat=len(critical_flag)))
        seqs = []
        while len(seqs) < 1024 * 40:
            x = random.randint(0, 2 ** self.dim - 1)
            seq = self.generate_random_conf(x)
            if seq not in seqs:
                seqs.append(seq)

        num_perms = len(permutations)
        repeats = len(seqs) // num_perms

        for i in range(num_perms):
            for rep in range(repeats):
                pos = i + rep * num_perms
                if pos < len(seqs):
                    for idx in range(len(critical_flag)):
                        seqs[pos][critical_flag[idx]] = permutations[i][idx]
        return seqs

    def run(self, budget=5000):
        print("--- Stage 1: Building Model ---")
        model, initial_indep, initial_dep = self.build_RF_by_CompTuner()

        print("--- Stage 2: Extracting Critical Flags ---")
        critical_flag, model_new = self.get_critical_flags(model, initial_indep, initial_dep)

        print("--- Stage 3: Fine-tuning around Critical Flags ---")
        while True:
            elapsed = time.perf_counter() - self.state["time_zero"]
            if elapsed >= budget:
                break

            seq = self.searchBycritical(critical_flag)
            result = self.runtime_predict(model_new, seq)
            sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

            true_result, _, _, _ = self.get_objective_score(
                sorted_result[0][0],
                k_iter=0,
                all_flags=self.all_flags,
                **self.kwargs,
            )

            if true_result > self.global_best_per:
                self.global_best_per = true_result
                self.global_best_seq = sorted_result[0][0]

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
    parser = argparse.ArgumentParser(description="CFSCA adapted for LLVM (unified protocol, CSV logging)")
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
    parser.add_argument("--exec_param", type=str, default="")
    parser.add_argument("--pipeline_file", type=str, default="o3_pipeline.txt")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--bootstrap_timeout", type=float, default=600.0,
                        help="Timeout for baseline warmup run")
    parser.add_argument("--keep_artifacts", action="store_true",
                        help="Keep temporary artifacts after tuning")
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

    cleanup_paths("base.bc", "baseline.bc", "tuned.bc", "baseline.out", "tuned.out")
    cleanup_glob_patterns("*.bc", "*.o", "*.I", "*.s")

    try:
        with open(args.pipeline_file, "r", encoding="utf-8") as f:
            pipeline_text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    master_root = build_ast(pipeline_text)
    all_flags = assign_flags(master_root)

    print(f"Analyzing source code at {args.source_path} for related flags...")
    raw_code = obtain_c_code(args.source_path)
    clean_code = remove_commentsandinclude_from_c_code(raw_code)
    related_flags_list = get_related_flags(clean_code, all_flags)

    if not related_flags_list:
        related_flags_list = list(range(len(all_flags)))
        print("No related flags automatically identified. Falling back to all tunable passes.")

    print(f"Extracted {len(all_flags)} tunable passes.")
    print(f"Automatically identified {len(related_flags_list)} related flags: {related_flags_list}")

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

    print(f"5. Starting CFSCA Loop. Results will be saved in: {OUT_DIR}")

    cfsca_params = {
        "dim": len(all_flags),
        "get_objective_score": get_objective_score_llvm,
        "seed": 456,
        "related_flags": related_flags_list,
        "flags": all_flags,
        "state": state,
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

    cfsca = CFSCA(**cfsca_params)
    cfsca.run(budget=args.budget)

    if not args.keep_artifacts:
        cleanup_paths(base_bc, baseline_bc, tuned_bc, exe_baseline, exe_tuned)
        cleanup_glob_patterns("*.bc", "*.o", "*.I", "*.s")