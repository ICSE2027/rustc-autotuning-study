import re, os, glob, argparse
from llvm_pipeline_utils import build_ast, assign_flags



loop_passes = [
    'licm', 'loop-rotate', 'loop-unroll', 'loop-unroll-full', 
    'loop-vectorize', 'loop-deletion', 'indvars', 'loop-idiom', 
    'loop-simplifycfg', 'simple-loop-unswitch', 'loop-distribute', 
    'loop-load-elim', 'loop-sink', 'loop-instsimplify', 
    'extra-simple-loop-unswitch-passes'
]


branch_passes = [
    'simplifycfg', 'jump-threading', 'correlated-propagation', 
    'speculative-execution', 'chr', 'lower-expect'
]


function_passes = [
    'inline', 'always-inline', 'argpromotion', 'function-attrs', 
    'rpo-function-attrs', 'called-value-propagation', 'deadargelim', 
    'tailcallelim', 'openmp-opt', 'callsite-splitting', 'openmp-opt-cgscc'
]


static_variable_passes = [
    'globalopt', 'globaldce', 'constmerge', 'elim-avail-extern', 
    'recompute-globalsaa'
]


pointer_passes = [
    'sroa', 'mem2reg', 'memcpyopt', 'dse', 'mldst-motion', 
    'alignment-from-assumptions', 'infer-alignment'
]


string_passes = [
    'libcalls-shrinkwrap'
]


float_passes = [
    'float2int'
]


def obtain_c_code(file_path):
    c_code = ""
    pattern = os.path.join(file_path, '*.c')
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)
        if filename != 'loop-wrap.c':
            with open(file_path, 'r') as file:
                c_code += file.read() + "\n"  
    return c_code

def contain_loop(code):
    for_loop_pattern = r"for\s*\(([^)]+)\)\s*{?"
    while_loop_pattern = r"while\s*\(([^)]+)\)\s*{?"
    do_while_loop_pattern = r"do\s*{?[^}]*}\s*while\s*\(([^)]+)\)"
    matches = re.findall(for_loop_pattern, code) + re.findall(while_loop_pattern, code) + re.findall(do_while_loop_pattern, code)
    return len(list(set(matches))) > 0

def contain_branch(code):
    pattern = r'if\s*\(.*?\)|else\s*if\s*\(.*?\)|else|switch\s*\(.*?\)'
    return len(list(set(re.findall(pattern, code)))) > 0

def contain_function(code):
    function_call_pattern = r'\b\w+\s*\([^)]*\)'
    function_declaration_pattern = r'\b\w+\s+\w+\s*\([^)]*\)'
    function_calls = re.findall(function_call_pattern, code)
    function_declarations = re.findall(function_declaration_pattern, code)
    function_calls_names = set([re.match(r'\b\w+', call).group() for call in function_calls])
    function_declarations_names = set([re.match(r'\b\w+\s+(\w+)', decl).group(1) for decl in function_declarations])
    matched_functions = function_calls_names.intersection(function_declarations_names)
    for kw in ['main', 'int', 'float', 'double', 'string', 'long']:
        if kw in matched_functions: matched_functions.remove(kw)
    return list(matched_functions)

def contain_static_variable(code):
    pattern = r'\bstatic\s+\w+\s+\w+\s*=?\s*[^;]*'
    return len(re.findall(pattern, code)) > 0

def contain_pointer(code):
    pattern = r'\b([_a-zA-Z][_a-zA-Z0-9]*\s+\*+\s*[_a-zA-Z][_a-zA-Z0-9]*\s*);'
    return len(list(set(re.findall(pattern, code)))) > 0

def contain_string(code):
    pattern = r'\b(str(?:len|cpy|ncpy|cat|ncat|cmp|ncmp|chr|rchr|str|tok|dup|ncpy))\b'
    return len(re.findall(pattern, code)) > 0

def contain_float_calculation(code):
    float_pattern = r'[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?'
    return len(re.findall(float_pattern, code)) > 0

def remove_commentsandinclude_from_c_code(c_code):
    c_code = re.sub(r'/\*.*?\*/', '', c_code, flags=re.DOTALL)
    c_code = re.sub(r'//.*', '', c_code)
    c_code = re.sub(r'".*?"', lambda x: x.group(0) if '/*' not in x.group(0) else '', c_code)
    c_code = re.sub(r'#include\s*<.*?>', '', c_code)
    c_code = re.sub(r'#include\s*".*?"', '', c_code)
    return "\n".join([line for line in c_code.split('\n') if line.strip() != ''])

def get_related_flags(code, all_flags):
    related_idx = []
    related_base_names = []
    code_without_comment = remove_commentsandinclude_from_c_code(code)
    
    if contain_loop(code_without_comment): related_base_names += loop_passes
    if contain_branch(code_without_comment): related_base_names += branch_passes
    if contain_function(code_without_comment): related_base_names += function_passes
    if contain_static_variable(code_without_comment): related_base_names += static_variable_passes
    if contain_pointer(code_without_comment): related_base_names += pointer_passes
    if contain_string(code_without_comment): related_base_names += string_passes
    if contain_float_calculation(code_without_comment): related_base_names += float_passes
    

    for position, flag in enumerate(all_flags):
        base_name = flag.split('__')[0]
        if base_name in related_base_names:
            related_idx.append(position)
            
    return sorted(list(set(related_idx)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtain program related flags for LLVM")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source program for tuning")
    parser.add_argument("--pipeline_file", type=str, required=True, help="LLVM O3 Pipeline file")
    args = parser.parse_args()


    with open(args.pipeline_file, 'r') as f:
        pipeline_text = f.read().strip()
    master_root = build_ast(pipeline_text)
    all_flags = assign_flags(master_root)

    code = obtain_c_code(args.source_path)
    new_code = remove_commentsandinclude_from_c_code(code)
    

    related_indices = get_related_flags(new_code, all_flags)
    print(",".join(map(str, related_indices)))