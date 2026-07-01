import re
import os
import argparse

from rust_pipeline_utils import build_ast, assign_flags
from typing import Tuple, List, Optional, Dict


# ==========================================


# ==========================================


loop_passes = {
    "MIR": [

    ],
    "LLVM": [
        "licm",
        "loop-rotate",
        "loop-unroll",
        "loop-unroll-full",
        "loop-vectorize",
        "loop-deletion",
        "indvars",
        "loop-idiom",
        "loop-simplifycfg",
        "simple-loop-unswitch",
        "loop-distribute",
        "loop-load-elim",
        "loop-sink",
        "loop-instsimplify",
        "extra-simple-loop-unswitch-passes",
    ]
}


branch_passes = {
    "MIR": [
        "UnreachableEnumBranching",
        "UnreachablePropagation",
        "SimplifyCfg-after-unreachable-enum-branching",
        "MultipleReturnTerminators",
        "MatchBranchSimplification",
        "SimplifyConstCondition-after-const-prop",
        "JumpThreading",
        "EarlyOtherwiseBranch",
        "SimplifyConstCondition-final",
        "SimplifyCfg-final",
    ],
    "LLVM": [
        "simplifycfg",
        "jump-threading",
        "correlated-propagation",
        "speculative-execution",
        "chr",
        "lower-expect",
    ]
}


function_passes = {
    "MIR": [
        "LowerSliceLenCalls",
        "Inline",
    ],
    "LLVM": [
        "inline",
        "always-inline",
        "argpromotion",
        "function-attrs",
        "rpo-function-attrs",
        "called-value-propagation",
        "deadargelim",
        "tailcallelim",
        "openmp-opt",
        "callsite-splitting",
        "openmp-opt-cgscc",
    ]
}


static_variable_passes = {
    "MIR": [
        "ReorderLocals",
        "EnumSizeOpt",
    ],
    "LLVM": [
        "globalopt",
        "globaldce",
        "constmerge",
        "elim-avail-extern",
        "recompute-globalsaa",
    ]
}


pointer_passes = {
    "MIR": [
        "ReferencePropagation",
        "ScalarReplacementOfAggregates",
        "DeadStoreElimination-initial",
        "DestinationPropagation",
        "DeadStoreElimination-final",
    ],
    "LLVM": [
        "sroa",
        "mem2reg",
        "memcpyopt",
        "dse",
        "mldst-motion",
        "alignment-from-assumptions",
        "infer-alignment",
    ]
}


string_passes = {
    "MIR": [
        "LowerSliceLenCalls",
    ],
    "LLVM": [
        "libcalls-shrinkwrap",
    ]
}


float_passes = {
    "MIR": [

    ],
    "LLVM": [
        "float2int",
    ]
}


# ==========================================

# ==========================================

def obtain_rust_code(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def remove_commentsanduse_from_rust_code(code: str) -> str:

    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    code = re.sub(r'//.*', '', code)

    code = re.sub(r'^\s*use\s+.*?;\s*$', '', code, flags=re.MULTILINE)
    # extern crate
    code = re.sub(r'^\s*extern\s+crate\s+.*?;\s*$', '', code, flags=re.MULTILINE)

    return "\n".join([line for line in code.split("\n") if line.strip() != ""])


def contain_loop(code: str) -> bool:
    patterns = [
        r'\bfor\b\s+.*?\s+\bin\b',
        r'\bwhile\b\s+.*?{',
        r'\bloop\b\s*{',
    ]
    for p in patterns:
        if re.search(p, code):
            return True
    return False


def contain_branch(code: str) -> bool:
    patterns = [
        r'\bif\b\s+.*?{',
        r'\belse\s+if\b\s+.*?{',
        r'\belse\b\s*{',
        r'\bmatch\b\s+.*?{',
    ]
    for p in patterns:
        if re.search(p, code):
            return True
    return False


def contain_function(code: str) -> bool:

    if re.search(r'\bfn\b\s+\w+\s*\(', code):
        return True


    calls = re.findall(r'\b([A-Za-z_]\w*)\s*\(', code)
    excluded = {
        "if", "for", "while", "loop", "match", "return",
        "Some", "Ok", "Err", "println", "print", "assert",
    }
    calls = [c for c in calls if c not in excluded]
    return len(calls) > 0


def contain_static_variable(code: str) -> bool:
    """
    Rust static / const / static mut
    """
    patterns = [
        r'\bstatic\b\s+(mut\s+)?\w+',
        r'\bconst\b\s+\w+',
    ]
    for p in patterns:
        if re.search(p, code):
            return True
    return False


def contain_pointer(code: str) -> bool:
    patterns = [
        r'&mut\s+\w+',
        r'&\s*\w+',
        r'\*\s*(const|mut)\b',
        r'\bBox\s*<',
        r'\bVec\s*<',
        r'&\s*\[',
        r'&mut\s*\[',
    ]
    for p in patterns:
        if re.search(p, code):
            return True
    return False


def contain_string(code: str) -> bool:
    patterns = [
        r'\bString\b',
        r'&str\b',
        r'\bformat!\s*\(',
        r'\bpush_str\s*\(',
        r'\blen\s*\(',
    ]
    for p in patterns:
        if re.search(p, code):
            return True
    return False


def contain_float_calculation(code: str) -> bool:
    patterns = [
        r'\bf32\b',
        r'\bf64\b',
        r'[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?',
    ]
    for p in patterns:
        if re.search(p, code):
            return True
    return False


# ==========================================

# ==========================================

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


# ==========================================

# ==========================================

def get_related_flags(code: str, mir_passes: List[str], llvm_flags: List[str]) -> List[int]:
    related_mir_names = set()
    related_llvm_names = set()

    code_without_comment = remove_commentsanduse_from_rust_code(code)

    if contain_loop(code_without_comment):
        related_mir_names.update(loop_passes["MIR"])
        related_llvm_names.update(loop_passes["LLVM"])

    if contain_branch(code_without_comment):
        related_mir_names.update(branch_passes["MIR"])
        related_llvm_names.update(branch_passes["LLVM"])

    if contain_function(code_without_comment):
        related_mir_names.update(function_passes["MIR"])
        related_llvm_names.update(function_passes["LLVM"])

    if contain_static_variable(code_without_comment):
        related_mir_names.update(static_variable_passes["MIR"])
        related_llvm_names.update(static_variable_passes["LLVM"])

    if contain_pointer(code_without_comment):
        related_mir_names.update(pointer_passes["MIR"])
        related_llvm_names.update(pointer_passes["LLVM"])

    if contain_string(code_without_comment):
        related_mir_names.update(string_passes["MIR"])
        related_llvm_names.update(string_passes["LLVM"])

    if contain_float_calculation(code_without_comment):
        related_mir_names.update(float_passes["MIR"])
        related_llvm_names.update(float_passes["LLVM"])

    related_idx = []


    for i, p in enumerate(mir_passes):
        if p in related_mir_names:
            related_idx.append(i)


    offset = len(mir_passes)
    for j, f in enumerate(llvm_flags):
        base_name = f.split("__")[0]
        if base_name in related_llvm_names:
            related_idx.append(offset + j)

    return sorted(list(set(related_idx)))


# ==========================================
# 5. Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtain program related flags for Rustc (MIR + LLVM)")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to Rust source file, e.g. src/main.rs or src/bin/foo.rs")
    parser.add_argument("--pipeline_file", type=str, required=True,
                        help="Path to pass.txt")
    args = parser.parse_args()

    mir_passes, llvm_pipeline_text = parse_pass_txt(args.pipeline_file)

    master_root = build_ast(llvm_pipeline_text)
    llvm_flags = assign_flags(master_root)

    code = obtain_rust_code(args.source_path)
    related_indices = get_related_flags(code, mir_passes, llvm_flags)

    print(",".join(map(str, related_indices)))