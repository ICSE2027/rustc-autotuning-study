import argparse
import subprocess
import random
import time
import os
import copy
import sys
import glob
from pathlib import Path


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
            return False
        return True
    except Exception as e:
        print("Exec Error:", str(e))
        return False

def generate_random_conf(x, all_flags):
    comb = bin(x).replace('0b', '')
    comb = '0' * (len(all_flags) - len(comb)) + comb
    return [int(s) for s in comb]

def get_median_execution_time(cmd, runs=3):
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


POLYBENCH_PROGRAMS = {
    "nussinov": "medley/nussinov",
    "adi": "stencils/adi",
    "fdtd-2d": "stencils/fdtd-2d",
    "heat-3d": "stencils/heat-3d",
    "jacobi-1d": "stencils/jacobi-1d",
    "jacobi-2d": "stencils/jacobi-2d",
    "seidel-2d": "stencils/seidel-2d"
}

def profile_program(prog_name, rel_path, args, all_flags, master_root, pipeline_text, llvm_link_bin, include_cmd):
    

    subprocess.run("rm -f *.bc *.o test_tuned_data test_o3_data", shell=True, capture_output=True)
    
    full_source_path = os.path.join(args.polybench_dir, rel_path)
    output_file = os.path.join(args.output_dir, f"data_{prog_name}.txt")
    

    with open(output_file, 'w') as f:
        pass
        
    print(f"\n[{prog_name}] Source Path: {full_source_path}")
    print(f"[{prog_name}] Output Data: {output_file}")
    
    base_bc = "base.bc"
    tuned_bc = "tuned_data.bc"
    o3_bc = "o3.bc"
    
    exe_tuned = "./test_tuned_data"
    exe_o3 = "./test_o3_data"


    print(f"[{prog_name}] 1. Compiling Source -> Base IR...")
    

    cmd_prep = f"{args.clang_path} -O0 -emit-llvm -c {include_cmd} {full_source_path}/{prog_name}.c"
    
    if execute_terminal_command(cmd_prep):
        generated_files = glob.glob("*.bc") + glob.glob("*.o")
        input_files = [f for f in generated_files if f not in [base_bc, tuned_bc, o3_bc]]
        
        if not input_files:
            print(f"[{prog_name}] Error: No bitcode files found after compilation.")
            return
            
        input_files_str = " ".join(input_files)
        cmd_link = f"{llvm_link_bin} {input_files_str} -o {base_bc}"
        if not execute_terminal_command(cmd_link):
            print(f"[{prog_name}] Error linking Base IR.")
            return
            
        for f in input_files:
            try: os.remove(f)
            except: pass
    else:
        print(f"[{prog_name}] Error compiling source.")
        return


    print(f"[{prog_name}] 2. Pre-compiling O3 Executable & Profiling Baseline...")
    cmd_opt_o3 = f"{args.opt_path} -passes='{pipeline_text}' {base_bc} -o {o3_bc}"
    execute_terminal_command(cmd_opt_o3)
    
    cmd_link_o3 = f"{args.clang_path} {o3_bc} -O0 -o {exe_o3} -lm"
    execute_terminal_command(cmd_link_o3)

    time_o3 = get_median_execution_time(f"{exe_o3} {args.exec_param}", runs=5)
    if time_o3 < 0:
        print(f"[{prog_name}] Error: O3 baseline failed to execute.")
        return
        
    time_o3 = max(time_o3, 0.0001)
    print(f"[{prog_name}] --> O3 Baseline Median Time: {time_o3:.4f}s")


    print(f"[{prog_name}] 3. Starting Random Sampling ({args.samples} attempts)...")
    success_count = 0

    with open(output_file, 'a') as f_out:
        for i in range(args.samples):
            x = random.randint(0, 2 ** len(all_flags) - 1)
            seq = generate_random_conf(x, all_flags)
            
            selected_ids = {flag for flag, bit in zip(all_flags, seq) if bit == 1}
            current_root = copy.deepcopy(master_root)
            new_children = []
            for ch in current_root.children:
                if prune_ast_in_place(ch, selected_ids):
                    new_children.append(ch)
            current_root.children = new_children
            
            pipeline_str = "no-op-module" if not current_root.children else get_pipeline_string_from_root(current_root)

            def cleanup_temp_files():
                subprocess.run(f"rm -f {tuned_bc} {exe_tuned} *.o *.s", shell=True, capture_output=True)

            cmd_opt = f"{args.opt_path} -passes=\"{pipeline_str}\" {base_bc} -o {tuned_bc}"
            if not execute_terminal_command(cmd_opt): 
                cleanup_temp_files()
                continue

            cmd_compile = f"{args.clang_path} {tuned_bc} -O0 -o {exe_tuned} -lm"
            if not execute_terminal_command(cmd_compile): 
                cleanup_temp_files()
                continue

            start_tuned = time.perf_counter()
            proc_tuned = subprocess.run(f"./{exe_tuned} {args.exec_param}", shell=True, capture_output=True)
            time_tuned = time.perf_counter() - start_tuned

            if proc_tuned.returncode != 0: 
                cleanup_temp_files()
                continue
            
            time_tuned = max(time_tuned, 0.0001)
            cleanup_temp_files()

            speedup = time_o3 / time_tuned
            
            if speedup > 1.0:
                success_count += 1
                seq_str = ", ".join(map(str, seq))
                f_out.write(seq_str + "\n")
                f_out.flush()
                print(f"[{prog_name}]  🌟 [{i+1}/{args.samples}] Found valid sequence! Speedup: {speedup:.4f}")


    subprocess.run(f"rm -f *.bc *.o {exe_o3} {exe_tuned}", shell=True, capture_output=True)
    print(f"[{prog_name}] Finished. Found {success_count} good sequences saved to {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Pool Data Generator for LLVM PDCAT")
    parser.add_argument("--clang_path", type=str, default="clang-20")
    parser.add_argument("--opt_path", type=str, default="opt-20")
    parser.add_argument("--pipeline_file", type=str, default="o3_pipeline.txt")
    parser.add_argument("--exec_param", type=str, default="")
    parser.add_argument("--samples", type=int, default=500, help="Number of random attempts per program")
    args = parser.parse_args()


    args.polybench_dir = str(Path(__file__).resolve().parents[3] / "Benchmarks" / "polyBench")
    args.output_dir = os.path.join(os.getcwd(), 'data')
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading pipeline template from {args.pipeline_file}...")
    try:
        with open(args.pipeline_file, 'r') as f:
            pipeline_text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: {args.pipeline_file} not found.")
        sys.exit(1)

    master_root = build_ast(pipeline_text)
    all_flags = assign_flags(master_root)
    print(f"Total flags found in pipeline: {len(all_flags)}")

    include_cmd = f'-I {args.polybench_dir}/utilities {args.polybench_dir}/utilities/polybench.c'

    if "clang-" in args.clang_path:
        version_suffix = args.clang_path.split("clang-")[-1]
        llvm_link_bin = f"llvm-link-{version_suffix}"
    else:
        llvm_link_bin = "llvm-link-20"

    print("=================================================================")
    print("Starting Batch Data Generation for 30 PolyBench Programs")
    print(f"Output Directory: {args.output_dir}")
    print("=================================================================")

    total_progs = len(POLYBENCH_PROGRAMS)
    current = 0

    for prog_name, rel_path in POLYBENCH_PROGRAMS.items():
        current += 1
        print(f"\n--- Processing {current}/{total_progs}: {prog_name} ---")
        profile_program(prog_name, rel_path, args, all_flags, master_root, pipeline_text, llvm_link_bin, include_cmd)
        
    print("=================================================================")
    print("All 30 programs have been processed.")
    print("=================================================================")
