import argparse
import copy
import csv
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from rust_pipeline_utils import (
    build_ast,
    assign_flags,
    prune_ast_in_place,
    get_pipeline_string_from_root,
)

FLOAT_MAX = float("inf")


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


# ==========================================
# Parse pass.txt / pass_group.txt
# ==========================================

def parse_pass_txt(filepath: str) -> Tuple[List[str], str]:


    mir_passes: List[str] = []
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


def parse_joint_group_file(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:


    mir_groups: List[List[str]] = []
    llvm_groups: List[List[str]] = []

    current_group: List[str] = []
    mode = None

    def flush_group():
        nonlocal current_group, mode
        if not current_group:
            return
        if mode == "MIR":
            mir_groups.append(current_group)
        elif mode == "LLVM":
            llvm_groups.append(current_group)
        current_group = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s == "MIR:":
                flush_group()
                mode = "MIR"
                continue

            if s == "LLVM:":
                flush_group()
                mode = "LLVM"
                continue

            if s.startswith("-") and set(s) == {"-"}:
                flush_group()
                continue

            current_group.append(s)

    flush_group()
    return mir_groups, llvm_groups


def validate_joint_groups(
    mir_passes: List[str],
    llvm_flags: List[str],
    mir_groups: List[List[str]],
    llvm_groups: List[List[str]],
) -> None:
    mir_set = set(mir_passes)
    llvm_set = set(llvm_flags)

    mir_group_items = [x for g in mir_groups for x in g]
    llvm_group_items = [x for g in llvm_groups for x in g]

    mir_group_set = set(mir_group_items)
    llvm_group_set = set(llvm_group_items)

    missing_mir = [x for x in mir_passes if x not in mir_group_set]
    extra_mir = [x for x in mir_group_items if x not in mir_set]

    missing_llvm = [x for x in llvm_flags if x not in llvm_group_set]
    extra_llvm = [x for x in llvm_group_items if x not in llvm_set]

    if missing_mir:
        print("Error: these MIR passes are missing in pass_group.txt:")
        for x in missing_mir:
            print("  ", x)
        sys.exit(1)

    if extra_mir:
        print("Error: these MIR entries in pass_group.txt are not found in pass.txt:")
        for x in extra_mir:
            print("  ", x)
        sys.exit(1)

    if missing_llvm:
        print("Error: these LLVM flags are missing in pass_group.txt:")
        for x in missing_llvm:
            print("  ", x)
        sys.exit(1)

    if extra_llvm:
        print("Error: these LLVM entries in pass_group.txt are not found in llvm flags:")
        for x in extra_llvm:
            print("  ", x)
        sys.exit(1)


# ==========================================
# Rust/Cargo Helpers
# ==========================================

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
        prefix = "+" if bit == 1 else "-"
        items.append(f"{prefix}{pass_name}")
    return ",".join(items)


def build_rustc_extra_args(
    mir_flag_str: str,
    llvm_pipeline_str: str,
    use_unsound_mir_opts: bool = False,
) -> List[str]:
    extra: List[str] = []

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
# LLVM / MIR state helpers
# ==========================================

def build_llvm_pipeline_from_state(
    master_root,
    llvm_flags: List[str],
    llvm_state: Dict[str, bool],
) -> Tuple[List[int], str]:
    llvm_seq = [1 if llvm_state[f] else 0 for f in llvm_flags]
    selected_llvm_ids = {flag for flag in llvm_flags if llvm_state[flag]}

    current_root = copy.deepcopy(master_root)
    new_children = []
    for ch in current_root.children:
        if prune_ast_in_place(ch, selected_llvm_ids):
            new_children.append(ch)
    current_root.children = new_children

    if not current_root.children:
        return llvm_seq, "no-op-module"

    return llvm_seq, get_pipeline_string_from_root(current_root)


def build_mir_seq_from_state(
    mir_passes: List[str],
    mir_state: Dict[str, bool],
) -> Tuple[List[int], str]:
    mir_seq = [1 if mir_state[p] else 0 for p in mir_passes]
    mir_str = build_mir_flag_string(mir_passes, mir_seq)
    return mir_seq, mir_str


def build_joint_signature(
    mir_passes: List[str],
    llvm_flags: List[str],
    master_root,
    state: Dict[str, Dict[str, bool]],
) -> Tuple[List[int], List[int], str, str, Tuple[str, str]]:
    mir_seq, mir_str = build_mir_seq_from_state(mir_passes, state["mir"])
    llvm_seq, llvm_pipeline = build_llvm_pipeline_from_state(master_root, llvm_flags, state["llvm"])
    signature = (mir_str, llvm_pipeline)
    return mir_seq, llvm_seq, mir_str, llvm_pipeline, signature


# ==========================================
# Evaluator (RIO-style)
# ==========================================

class RustcEvaluator:

    def __init__(
        self,
        mir_passes: List[str],
        llvm_flags: List[str],
        master_root,
        project_dir: str,
        bin_name: str,
        target_tuned: str,
        baseline_run_cmd: List[str],
        iter_csv_file: str,
        time_csv_file: str,
        baseline_time_ref: float,
        exec_param: str = "",
        verbose: bool = False,
        use_unsound_mir_opts: bool = False,
        use_rustc_bootstrap: bool = False,
    ):
        self.mir_passes = mir_passes
        self.llvm_flags = llvm_flags
        self.master_root = master_root
        self.project_dir = project_dir
        self.bin_name = bin_name
        self.target_tuned = target_tuned
        self.baseline_run_cmd = baseline_run_cmd
        self.iter_csv_file = iter_csv_file
        self.time_csv_file = time_csv_file
        self.baseline_time_ref = baseline_time_ref
        self.exec_param = exec_param
        self.verbose = verbose
        self.use_unsound_mir_opts = use_unsound_mir_opts
        self.use_rustc_bootstrap = use_rustc_bootstrap

        self.exe_tuned = build_binary_path(target_tuned, bin_name)
        self.exec_args = shlex.split(exec_param) if exec_param.strip() else []

        self.dynamic_timeout = max(2.0 * self.baseline_time_ref + 10.0, 10.0)

        self.best_speedup_so_far = float("-inf")
        self.best_mir_seq_so_far: List[int] = []
        self.best_llvm_seq_so_far: List[int] = []
        self.best_tuned_time_so_far = -1.0
        self.best_baseline_time_so_far = -1.0
        self.best_iter_so_far = -1
        self.best_timestamp_so_far = -1.0

        self.iter_count = 0
        self.time_zero = time.perf_counter()
        self.last_log_time = 0.0

    def evaluate(
        self,
        state: Dict[str, Dict[str, bool]],
    ) -> Tuple[float, float, str, str, Tuple[str, str], List[int], List[int]]:

        self.iter_count += 1

        mir_seq, llvm_seq, mir_str, llvm_pipeline, signature = build_joint_signature(
            self.mir_passes,
            self.llvm_flags,
            self.master_root,
            state,
        )

        tuned_extra_args = build_rustc_extra_args(
            mir_flag_str=mir_str,
            llvm_pipeline_str=llvm_pipeline,
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
            return -1.0, FLOAT_MAX, mir_str, llvm_pipeline, signature, mir_seq, llvm_seq

        if not os.path.exists(self.exe_tuned):
            print("Tuned executable not found after successful build.")
            return -1.0, FLOAT_MAX, mir_str, llvm_pipeline, signature, mir_seq, llvm_seq

        tuned_run_cmd = [self.exe_tuned] + self.exec_args
        ok_tuned, time_tuned = run_and_measure_once(
            tuned_run_cmd,
            cwd=self.project_dir,
            timeout=self.dynamic_timeout,
        )
        if not ok_tuned:
            return -1.0, FLOAT_MAX, mir_str, llvm_pipeline, signature, mir_seq, llvm_seq

        ok_baseline_iter, time_baseline_iter = run_and_measure_once(
            self.baseline_run_cmd,
            cwd=self.project_dir,
            timeout=self.dynamic_timeout,
        )
        if not ok_baseline_iter:
            return -1.0, FLOAT_MAX, mir_str, llvm_pipeline, signature, mir_seq, llvm_seq

        current_speedup = time_baseline_iter / time_tuned
        current_perf = time_tuned / time_baseline_iter
        current_time = time.perf_counter() - self.time_zero

        is_new_best = current_speedup > self.best_speedup_so_far
        if is_new_best:
            self.best_speedup_so_far = current_speedup
            self.best_mir_seq_so_far = mir_seq
            self.best_llvm_seq_so_far = llvm_seq
            self.best_tuned_time_so_far = time_tuned
            self.best_baseline_time_so_far = time_baseline_iter
            self.best_iter_so_far = self.iter_count
            self.best_timestamp_so_far = current_time
            print(f"[New Record] Speedup: {self.best_speedup_so_far:.6f} at Iter {self.iter_count}")

        best_display = self.best_speedup_so_far if self.best_speedup_so_far != float("-inf") else current_speedup
        print(
            f"Iter {self.iter_count:04d} | "
            f"Tuned: {time_tuned:.6f}s | "
            f"Baseline: {time_baseline_iter:.6f}s | "
            f"Speedup: {current_speedup:.6f} | "
            f"Best So Far: {best_display:.6f}"
        )

        mir_seq_str = ",".join(map(str, mir_seq))
        llvm_seq_str = ",".join(map(str, llvm_seq))

        append_csv_row(self.iter_csv_file, [
            self.iter_count,
            f"{current_time:.2f}",
            f"{time_tuned:.6f}",
            f"{time_baseline_iter:.6f}",
            f"{current_speedup:.6f}",
            is_new_best,
            mir_seq_str,
            llvm_seq_str,
            mir_str,
            llvm_pipeline,
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

        return current_speedup, current_perf, mir_str, llvm_pipeline, signature, mir_seq, llvm_seq


# ==========================================
# GroupTuner Core
# ==========================================

class GroupTunerRustc:
    def __init__(
        self,
        mir_passes: List[str],
        llvm_flags: List[str],
        mir_groups: List[List[str]],
        llvm_groups: List[List[str]],
        evaluator: RustcEvaluator,
    ):
        self.mir_passes = mir_passes
        self.llvm_flags = llvm_flags
        self.mir_groups = mir_groups
        self.llvm_groups = llvm_groups
        self.evaluator = evaluator


        self.joint_groups: List[Tuple[str, List[str]]] = []
        for g in self.mir_groups:
            self.joint_groups.append(("MIR", g))
        for g in self.llvm_groups:
            self.joint_groups.append(("LLVM", g))


        self.o3_state = {
            "mir": {p: True for p in self.mir_passes},
            "llvm": {f: True for f in self.llvm_flags},
        }



        self.current_state: List[List] = []
        self.worst_state = None
        self.worst_perf = FLOAT_MAX
        self.worst_signature = None


        self.seen_signatures: Set[Tuple[str, str]] = set()

    def bootstrap_default_o3(self) -> Tuple[Optional[Dict[str, Dict[str, bool]]], float]:
        _, o3_perf, _, _, o3_signature, _, _ = self.evaluator.evaluate(self.o3_state)
        if o3_perf == FLOAT_MAX:
            return None, FLOAT_MAX

        self.current_state = [[copy.deepcopy(self.o3_state), o3_perf, o3_signature]]
        self.worst_state = copy.deepcopy(self.o3_state)
        self.worst_perf = o3_perf
        self.worst_signature = o3_signature
        self.seen_signatures.add(o3_signature)
        return copy.deepcopy(self.o3_state), o3_perf

    def update_worst_state_perf(self, candidate_state, perf, signature):
        remove_idx = -1
        for i, item in enumerate(self.current_state):
            if item[1] == self.worst_perf and item[2] == self.worst_signature:
                remove_idx = i
                break

        if remove_idx != -1:
            self.current_state.pop(remove_idx)

        self.current_state.append([copy.deepcopy(candidate_state), perf, signature])

        self.worst_state = self.current_state[0][0]
        self.worst_perf = self.current_state[0][1]
        self.worst_signature = self.current_state[0][2]

        for i in range(1, len(self.current_state)):
            if self.current_state[i][1] > self.worst_perf:
                self.worst_perf = self.current_state[i][1]
                self.worst_state = self.current_state[i][0]
                self.worst_signature = self.current_state[i][2]

    def generate_candidate(
        self,
        base_state: Dict[str, Dict[str, bool]],
        base_signature: Tuple[str, str],
    ) -> Dict[str, Dict[str, bool]]:
        max_tries = 200

        for _ in range(max_tries):
            new_state = {
                "mir": dict(base_state["mir"]),
                "llvm": dict(base_state["llvm"]),
            }

            domain, selected_group = random.choice(self.joint_groups)

            flipped = False
            if domain == "MIR":
                for pass_name in selected_group:
                    if random.randint(0, 1) == 1:
                        new_state["mir"][pass_name] = not new_state["mir"][pass_name]
                        flipped = True
                if not flipped:
                    one = random.choice(selected_group)
                    new_state["mir"][one] = not new_state["mir"][one]
            else:
                for flag in selected_group:
                    if random.randint(0, 1) == 1:
                        new_state["llvm"][flag] = not new_state["llvm"][flag]
                        flipped = True
                if not flipped:
                    one = random.choice(selected_group)
                    new_state["llvm"][one] = not new_state["llvm"][one]

            _, _, _, _, signature = build_joint_signature(
                self.mir_passes,
                self.llvm_flags,
                self.evaluator.master_root,
                new_state,
            )

            if signature != base_signature and signature not in self.seen_signatures:
                return new_state

        while True:
            new_state = {
                "mir": dict(base_state["mir"]),
                "llvm": dict(base_state["llvm"]),
            }

            domain, selected_group = random.choice(self.joint_groups)
            one = random.choice(selected_group)

            if domain == "MIR":
                new_state["mir"][one] = not new_state["mir"][one]
            else:
                new_state["llvm"][one] = not new_state["llvm"][one]

            _, _, _, _, signature = build_joint_signature(
                self.mir_passes,
                self.llvm_flags,
                self.evaluator.master_root,
                new_state,
            )

            if signature != base_signature and signature not in self.seen_signatures:
                return new_state

    def init_state_until_pool_ready(self, time_budget_sec: float, pool_size: int = 10):
        best_state, best_perf = self.bootstrap_default_o3()
        if best_state is None:
            return None, FLOAT_MAX

        while len(self.current_state) < pool_size:
            elapsed = time.perf_counter() - self.evaluator.time_zero
            if elapsed >= time_budget_sec:
                break

            new_state = self.generate_candidate(self.o3_state, self.current_state[0][2])
            _, new_perf, _, _, new_signature, _, _ = self.evaluator.evaluate(new_state)

            if new_perf == FLOAT_MAX:
                continue

            self.seen_signatures.add(new_signature)

            if new_perf > self.worst_perf:
                self.worst_state = copy.deepcopy(new_state)
                self.worst_perf = new_perf
                self.worst_signature = new_signature

            if new_perf < best_perf:
                best_perf = new_perf
                best_state = copy.deepcopy(new_state)

            self.current_state.append([copy.deepcopy(new_state), new_perf, new_signature])

        return best_state, best_perf

    def tune(self, time_budget_sec: float):
        initial_temp = 1000.0
        min_temp = 1.0
        alpha = 5e-5

        best_state, best_perf = self.init_state_until_pool_ready(time_budget_sec, pool_size=10)
        if best_state is None:
            return None, FLOAT_MAX

        while True:
            elapsed = time.perf_counter() - self.evaluator.time_zero
            if elapsed >= time_budget_sec:
                break

            progress = min(max(elapsed / time_budget_sec, 0.0), 1.0)
            current_temp = initial_temp * ((min_temp / initial_temp) ** progress)

            selected_item = self.current_state[random.randint(0, len(self.current_state) - 1)]
            base_state = selected_item[0]
            base_signature = selected_item[2]

            candidate_state = self.generate_candidate(base_state, base_signature)
            _, candidate_perf, _, _, candidate_signature, _, _ = self.evaluator.evaluate(candidate_state)

            if candidate_perf == FLOAT_MAX:
                continue

            self.seen_signatures.add(candidate_signature)

            if candidate_perf < self.worst_perf:
                self.update_worst_state_perf(candidate_state, candidate_perf, candidate_signature)
            else:
                delta = (candidate_perf - self.worst_perf) / self.worst_perf
                accept_prob = math.exp(-(delta / (current_temp * alpha)))
                if random.random() < accept_prob:
                    self.update_worst_state_perf(candidate_state, candidate_perf, candidate_signature)

            if candidate_perf < best_perf:
                best_perf = candidate_perf
                best_state = copy.deepcopy(candidate_state)

        return best_state, best_perf


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rustc GroupTuner (MIR + LLVM joint tuning, RIO-style time-budget protocol)"
    )
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save iter.csv and time.csv")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to Rust source file, e.g. src/main.rs or src/bin/foo.rs")
    parser.add_argument("--exec_param", type=str, default="",
                        help="Execution parameter for the output executable")
    parser.add_argument("--pipeline_file", type=str, default="pass.txt",
                        help="Path to pass.txt")
    parser.add_argument("--group_file", type=str, required=True,
                        help="Path to pass_group.txt")
    parser.add_argument("--budget", type=int, default=5000,
                        help="Time budget in seconds")
    parser.add_argument("--bootstrap_timeout", type=float, default=600.0,
                        help="Timeout for the baseline warmup run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed command failure info")
    parser.add_argument("--use_unsound_mir_opts", action="store_true",
                        help="Append -Z unsound-mir-opts when necessary")
    parser.add_argument("--use_rustc_bootstrap", action="store_true",
                        help="Set RUSTC_BOOTSTRAP=1 if not using nightly")
    parser.add_argument("--keep_artifacts", action="store_true",
                        help="Keep target directories after tuning")

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

    # =========================

    # =========================
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

    # =========================

    # =========================
    print(f"Loading joint groups from {args.group_file}...")
    try:
        mir_groups, llvm_groups = parse_joint_group_file(args.group_file)
    except FileNotFoundError:
        print(f"Error: {args.group_file} not found.")
        sys.exit(1)

    validate_joint_groups(mir_passes, llvm_flags, mir_groups, llvm_groups)

    print(f"Total MIR groups: {len(mir_groups)}")
    print(f"Total LLVM groups: {len(llvm_groups)}")
    print(f"Total joint groups: {len(mir_groups) + len(llvm_groups)}")

    # =========================

    # =========================
    try:
        project_dir, bin_name = infer_project_info(args.source_path)
    except Exception as e:
        print(f"Error inferring Cargo project info: {e}")
        sys.exit(1)

    print(f"Project dir: {project_dir}")
    print(f"Binary name: {bin_name}")

    target_baseline = os.path.join(project_dir, "target_group_baseline")
    target_tuned = os.path.join(project_dir, "target_group_tuned")

    exe_baseline = build_binary_path(target_baseline, bin_name)

    exec_args = shlex.split(args.exec_param) if args.exec_param.strip() else []

    # =========================

    # =========================
    cleanup_paths(target_baseline, target_tuned)

    # =========================

    # =========================
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

    # =========================
    # 2. baseline warm-up
    # =========================
    print("2. Warm-up baseline executable to establish baseline_time_ref ...")

    ok_baseline_warmup, baseline_time_ref = run_and_measure_once(
        baseline_run_cmd,
        cwd=project_dir,
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
    print(f"3. Starting GroupTuner Loop. Results will be saved in: {OUT_DIR}")
    print(f"Time budget: {args.budget}s")

    evaluator = RustcEvaluator(
        mir_passes=mir_passes,
        llvm_flags=llvm_flags,
        master_root=master_root,
        project_dir=project_dir,
        bin_name=bin_name,
        target_tuned=target_tuned,
        baseline_run_cmd=baseline_run_cmd,
        iter_csv_file=ITER_CSV_FILE,
        time_csv_file=TIME_CSV_FILE,
        baseline_time_ref=baseline_time_ref,
        exec_param=args.exec_param,
        verbose=args.verbose,
        use_unsound_mir_opts=args.use_unsound_mir_opts,
        use_rustc_bootstrap=args.use_rustc_bootstrap,
    )

    grouptuner = GroupTunerRustc(
        mir_passes=mir_passes,
        llvm_flags=llvm_flags,
        mir_groups=mir_groups,
        llvm_groups=llvm_groups,
        evaluator=evaluator,
    )

    best_state, best_perf = grouptuner.tune(args.budget)

    total_elapsed = time.perf_counter() - evaluator.time_zero

    print("\n=== Tuning Finished ===")
    print(f"Total Iterations: {evaluator.iter_count}")
    print(f"Total Time: {total_elapsed:.2f}s")
    print(f"Best Iter: {evaluator.best_iter_so_far}")
    print(f"Best Speedup: {evaluator.best_speedup_so_far:.6f}")
    print(f"Best Tuned Time: {evaluator.best_tuned_time_so_far:.6f}s")
    print(f"Best Baseline Time: {evaluator.best_baseline_time_so_far:.6f}s")
    print(f"Best Found At: {evaluator.best_timestamp_so_far:.2f}s")

    best_mir_seq_str_final = ",".join(map(str, evaluator.best_mir_seq_so_far))
    best_llvm_seq_str_final = ",".join(map(str, evaluator.best_llvm_seq_so_far))
    print("Best Seq:")
    print(f"MIR: {best_mir_seq_str_final}")
    print(f"LLVM: {best_llvm_seq_str_final}")
    print(f"Detailed logs saved to:\n  - {ITER_CSV_FILE}\n  - {TIME_CSV_FILE}")

    if not args.keep_artifacts:
        cleanup_paths(target_baseline, target_tuned)