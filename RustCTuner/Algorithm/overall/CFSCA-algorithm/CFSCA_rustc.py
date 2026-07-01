import argparse
import copy
import csv
import itertools
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import Tuple, List, Optional, Dict

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from rust_pipeline_utils import (
    build_ast,
    assign_flags,
    prune_ast_in_place,
    get_pipeline_string_from_root,
)

from getRelated_rustc import (
    obtain_rust_code,
    get_related_flags,
    parse_pass_txt,
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


# ==========================================
# Objective Evaluation
# ==========================================

def get_objective_score_rustc(
    independent: List[int],
    k_iter: int,
    all_flags: List[str],
    mir_passes: List[str],
    llvm_flags: List[str],
    master_root,
    project_dir: str,
    bin_name: str,
    target_tuned: str,
    baseline_run_cmd: List[str],
    baseline_time_ref: float,
    iter_csv_file: str,
    time_csv_file: str,
    state: dict,
    use_unsound_mir_opts: bool = False,
    use_rustc_bootstrap: bool = False,
    exec_param: str = "",
    verbose: bool = False,
):
    total_mir = len(mir_passes)
    mir_seq = independent[:total_mir]
    llvm_seq = independent[total_mir:]

    mir_str = build_mir_flag_string(mir_passes, mir_seq)
    llvm_str = build_llvm_pipeline_from_seq(master_root, llvm_flags, llvm_seq)

    tuned_extra_args = build_rustc_extra_args(
        mir_flag_str=mir_str,
        llvm_pipeline_str=llvm_str,
        use_unsound_mir_opts=use_unsound_mir_opts,
    )
    tuned_cmd = build_cargo_rustc_command(bin_name, tuned_extra_args)
    tuned_env = make_build_env(target_tuned, use_rustc_bootstrap)

    if not execute_terminal_command(
        tuned_cmd,
        cwd=project_dir,
        env=tuned_env,
        verbose=verbose,
    ):
        return 0.0001, -1.0, -1.0, mir_str, llvm_str

    exe_tuned = build_binary_path(target_tuned, bin_name)
    if not os.path.exists(exe_tuned):
        print("Tuned executable not found after successful build.")
        return 0.0001, -1.0, -1.0, mir_str, llvm_str

    dynamic_timeout = max(2.0 * baseline_time_ref + 10.0, 10.0)

    tuned_run_cmd = [exe_tuned]
    if exec_param.strip():
        tuned_run_cmd.extend(shlex.split(exec_param))

    ok_tuned, time_tuned = run_and_measure_once(
        tuned_run_cmd,
        cwd=project_dir,
        timeout=dynamic_timeout,
    )
    if not ok_tuned:
        return 0.0001, -1.0, -1.0, mir_str, llvm_str

    ok_baseline, time_baseline = run_and_measure_once(
        baseline_run_cmd,
        cwd=project_dir,
        timeout=dynamic_timeout,
    )
    if not ok_baseline:
        return 0.0001, time_tuned, -1.0, mir_str, llvm_str

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
        state["best_mir_pipeline_so_far"] = mir_str
        state["best_llvm_pipeline_so_far"] = llvm_str
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

    mir_seq_str = ",".join(map(str, mir_seq))
    llvm_seq_str = ",".join(map(str, llvm_seq))
    append_csv_row(iter_csv_file, [
        iter_id,
        f"{current_time:.2f}",
        f"{time_tuned:.6f}",
        f"{time_baseline:.6f}",
        f"{speedup:.6f}",
        is_new_best,
        mir_seq_str,
        llvm_seq_str,
        mir_str,
        llvm_str,
    ])

    if current_time - state["last_log_time"] >= 20:
        best_mir_seq = ""
        best_llvm_seq = ""
        if state["best_seq_so_far"]:
            best_mir_seq = ",".join(map(str, state["best_seq_so_far"][:total_mir]))
            best_llvm_seq = ",".join(map(str, state["best_seq_so_far"][total_mir:]))

        append_csv_row(time_csv_file, [
            f"{current_time:.2f}",
            f"{state['best_speedup_so_far']:.6f}",
            state["best_iter_so_far"],
            best_mir_seq,
            best_llvm_seq,
        ])
        state["last_log_time"] = current_time

    return speedup, time_tuned, time_baseline, mir_str, llvm_str


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
        self.global_best_mir_pipeline = ""
        self.global_best_llvm_pipeline = ""

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
        true_running, _, _, _, _ = self.get_objective_score(
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
            speedup, _, _, _, _ = self.get_objective_score(
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

        now_best, _, _, _, _ = self.get_objective_score(
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

            new_best, _, _, _, _ = self.get_objective_score(
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

            true_result, _, _, _, _ = self.get_objective_score(
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
    parser = argparse.ArgumentParser(description="CFSCA adapted for Rustc (unified protocol, CSV logging)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save iter.csv and time.csv")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to Rust source file, e.g. src/main.rs or src/bin/foo.rs")
    parser.add_argument("--exec_param", type=str, default="")
    parser.add_argument("--pipeline_file", type=str, default="pass.txt")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--bootstrap_timeout", type=float, default=600.0,
                        help="Timeout for baseline warmup run")
    parser.add_argument("--use_unsound_mir_opts", action="store_true",
                        help="Append -Z unsound-mir-opts when necessary")
    parser.add_argument("--use_rustc_bootstrap", action="store_true",
                        help="Set RUSTC_BOOTSTRAP=1 if not using nightly")
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

    try:
        mir_passes, llvm_pipeline_text = parse_pass_txt(args.pipeline_file)
    except FileNotFoundError:
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    master_root = build_ast(llvm_pipeline_text)
    llvm_flags = assign_flags(master_root)

    total_flags = len(mir_passes) + len(llvm_flags)
    all_flags = mir_passes + llvm_flags

    print(f"Analyzing Rust source at {args.source_path} for related flags...")
    raw_code = obtain_rust_code(args.source_path)
    related_flags_list = get_related_flags(raw_code, mir_passes, llvm_flags)

    if not related_flags_list:
        related_flags_list = list(range(total_flags))
        print("No related flags automatically identified. Falling back to all tunable passes.")

    print(f"Total MIR passes: {len(mir_passes)}")
    print(f"Total LLVM flags: {len(llvm_flags)}")
    print(f"Total Search Space Dimension: {total_flags}")
    print(f"Automatically identified {len(related_flags_list)} related flags: {related_flags_list}")

    try:
        project_dir, bin_name = infer_project_info(args.source_path)
    except Exception as e:
        print(f"Error inferring Cargo project info: {e}")
        sys.exit(1)

    print(f"Project dir: {project_dir}")
    print(f"Binary name: {bin_name}")

    target_baseline = os.path.join(project_dir, "target_cfsca_baseline")
    target_tuned = os.path.join(project_dir, "target_cfsca_tuned")

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

    state = {
        "time_zero": time.perf_counter(),
        "last_log_time": 0.0,
        "iter_counter": 0,
        "best_speedup_so_far": float("-inf"),
        "best_seq_so_far": [],
        "best_mir_pipeline_so_far": "",
        "best_llvm_pipeline_so_far": "",
        "best_tuned_time_so_far": -1.0,
        "best_baseline_time_so_far": -1.0,
        "best_iter_so_far": -1,
        "best_timestamp_so_far": -1.0,
    }

    print(f"3. Starting CFSCA Loop. Results will be saved in: {OUT_DIR}")

    cfsca_params = {
        "dim": total_flags,
        "get_objective_score": get_objective_score_rustc,
        "seed": 456,
        "related_flags": related_flags_list,
        "flags": all_flags,
        "state": state,
        "mir_passes": mir_passes,
        "llvm_flags": llvm_flags,
        "master_root": master_root,
        "project_dir": project_dir,
        "bin_name": bin_name,
        "target_tuned": target_tuned,
        "baseline_run_cmd": baseline_run_cmd,
        "baseline_time_ref": baseline_time_ref,
        "iter_csv_file": ITER_CSV_FILE,
        "time_csv_file": TIME_CSV_FILE,
        "use_unsound_mir_opts": args.use_unsound_mir_opts,
        "use_rustc_bootstrap": args.use_rustc_bootstrap,
        "exec_param": args.exec_param,
        "verbose": args.verbose,
    }

    cfsca = CFSCA(**cfsca_params)
    cfsca.run(budget=args.budget)

    if not args.keep_artifacts:
        cleanup_paths(target_baseline, target_tuned)