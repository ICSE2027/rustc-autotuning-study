import argparse
import subprocess
import time
import os
import copy
import sys
import glob
import ast


from llvm_pipeline_utils import (
    build_ast, 
    assign_flags, 
    prune_ast_in_place, 
    get_pipeline_string_from_root
)

def execute_terminal_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Error] Command failed: {command}")
            if result.stderr:
                print(result.stderr.strip())
            return False
        return True
    except Exception as e:
        print("Exec Error:", str(e))
        return False

def get_median_execution_time(cmd, runs=5):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        proc = subprocess.run(cmd, shell=True, capture_output=True)
        end = time.perf_counter()
        if proc.returncode == 0:
            times.append(end - start)
    
    if not times:
        return -1.0
        
    times.sort()
    return times[len(times) // 2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify specific binary sequence for LLVM Pipeline")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source program")
    parser.add_argument("--clang_path", type=str, required=True, help="Path to clang")
    parser.add_argument("--opt_path", type=str, required=True, help="Path to opt")
    parser.add_argument("--exec_param", type=str, default="", help="Execution parameter")
    parser.add_argument("--pipeline_file", type=str, default="o3_pipeline.txt", help="Base O3 pipeline file")
    parser.add_argument("--seq", type=str, required=True, help="The 0/1 binary sequence as a string list, e.g., '[0, 1, 0]'")

    args = parser.parse_args()


    try:
        seq = ast.literal_eval(args.seq)
        if not isinstance(seq, list) or not all(isinstance(x, int) for x in seq):
            raise ValueError
    except Exception:
        print("Error: --seq must be a valid list of integers, e.g., '[0, 1, 0, ...]'")
        sys.exit(1)


    if not os.path.exists(args.pipeline_file):
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    with open(args.pipeline_file, 'r') as f:
        pipeline_text = f.read().strip()

    master_root = build_ast(pipeline_text)
    all_flags = assign_flags(master_root)

    if len(seq) != len(all_flags):
        print(f"Error: Sequence length ({len(seq)}) does not match the number of tunable passes ({len(all_flags)}).")
        sys.exit(1)


    selected_ids = {flag for flag, bit in zip(all_flags, seq) if bit == 1}
    current_root = copy.deepcopy(master_root)
    new_children = []
    for ch in current_root.children:
        if prune_ast_in_place(ch, selected_ids):
            new_children.append(ch)
    current_root.children = new_children
    
    tuned_pipeline_str = "no-op-module" if not current_root.children else get_pipeline_string_from_root(current_root)

    print("=" * 60)
    print("🎯 Target Tuned Pipeline:")
    print(tuned_pipeline_str)
    print("=" * 60)


    default_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    project_root = os.getenv('PROJECT_ROOT', default_project_root)
    include_cmd = f'-I {project_root}/Benchmarks/polyBench/utilities {project_root}/Benchmarks/polyBench/utilities/polybench.c'
    
    base_bc = "base.bc"
    o3_bc = "o3.bc"
    tuned_bc = "tuned.bc"
    exe_o3 = "./test_o3"
    exe_tuned = "./test_tuned"

    version_suffix = args.clang_path.split("clang-")[-1] if "clang-" in args.clang_path else "20"
    llvm_link_bin = f"llvm-link-{version_suffix}" if "clang-" in args.clang_path else "llvm-link"

    print("\n[1/4] Compiling Source -> Base IR (base.bc)...")
    cmd_prep = f"{args.clang_path} -O0 -emit-llvm -c {include_cmd} {args.source_path}/*.c"
    
    if execute_terminal_command(cmd_prep):
        input_files = [f for f in glob.glob("*.bc") + glob.glob("*.o") if f not in [base_bc, o3_bc, tuned_bc]]
        if not input_files:
            print("Error: No bitcode files generated.")
            sys.exit(1)
        input_files_str = " ".join(input_files)
        cmd_link = f"{llvm_link_bin} {input_files_str} -o {base_bc}"
        execute_terminal_command(cmd_link)
        for f in input_files:
            try: os.remove(f)
            except: pass
    else:
        sys.exit(1)


    print("[2/4] Compiling and profiling O3 Baseline...")
    execute_terminal_command(f"{args.opt_path} -passes='{pipeline_text}' {base_bc} -o {o3_bc}")
    execute_terminal_command(f"{args.clang_path} {o3_bc} -O0 -o {exe_o3} -lm")
    
    time_o3 = get_median_execution_time(f"{exe_o3} {args.exec_param}", runs=5)
    if time_o3 < 0:
        print("Error: O3 execution failed.")
        sys.exit(1)


    print("[3/4] Compiling and profiling Tuned Pipeline...")
    if not execute_terminal_command(f"{args.opt_path} -passes=\"{tuned_pipeline_str}\" {base_bc} -o {tuned_bc}"):
        print("Error: Tuned pipeline opt failed.")
        sys.exit(1)
    
    if not execute_terminal_command(f"{args.clang_path} {tuned_bc} -O0 -o {exe_tuned} -lm"):
        print("Error: Tuned pipeline clang link failed.")
        sys.exit(1)

    time_tuned = get_median_execution_time(f"{exe_tuned} {args.exec_param}", runs=5)
    if time_tuned < 0:
        print("Error: Tuned execution failed.")
        sys.exit(1)


    speedup = time_o3 / time_tuned
    print("\n" + "=" * 60)
    print("📊 Final Verification Results:")
    print(f"O3 Median Time (5 runs):    {time_o3:.6f} s")
    print(f"Tuned Median Time (5 runs): {time_tuned:.6f} s")
    print(f"Strict Speedup:             {speedup:.4f}x")
    print("=" * 60)


    subprocess.run(f"rm -f *.bc *.o {exe_tuned} {exe_o3}", shell=True)
