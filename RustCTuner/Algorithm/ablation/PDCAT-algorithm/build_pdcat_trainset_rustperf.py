import os
import re
import sys
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional


# ============================================================
# rustc-perf runtime-benchmarks:


# ============================================================


CASE_TO_CRATE = {
    "bufreader_snappy": "bufreader",

    "brotli-compress": "compression",
    "brotli-decompress": "compression",

    "css-parse-fb": "css",

    "fmt-debug-derive": "fmt",
    "fmt-write-str": "fmt",

    "hashmap_find_1m": "hashmap",
    "hashmap_find_misses_1m": "hashmap",
    "hashmap_insert_1m": "hashmap",
    "hashmap_iterate_1m": "hashmap",
    "hashmap_remove_1m": "hashmap",

    "nbody_5k": "nbody",

    "pinky-nes15": "nes",

    "nom-json": "parsing",

    "raytracer": "raytracer",

    "svg-parse-1": "svg",
    "svg-render-1": "svg",

    "regex-capture-1": "text-search",
    "regex-search-1": "text-search",
}


VALID_CRATES = {
    "bufreader",
    "compression",
    "css",
    "fmt",
    "hashmap",
    "nbody",
    "nes",
    "parsing",
    "raytracer",
    "svg",
    "text-search",
}

ALL_19_CASES = [
    "bufreader_snappy",
    "brotli-compress",
    "brotli-decompress",
    "css-parse-fb",
    "fmt-debug-derive",
    "fmt-write-str",
    "hashmap_find_1m",
    "hashmap_find_misses_1m",
    "hashmap_insert_1m",
    "hashmap_iterate_1m",
    "hashmap_remove_1m",
    "nbody_5k",
    "pinky-nes15",
    "nom-json",
    "raytracer",
    "svg-parse-1",
    "svg-render-1",
    "regex-capture-1",
    "regex-search-1",
]


def extract_prog_name_from_datafile(filename: str) -> str:
    base = os.path.basename(filename)
    m = re.match(r"data_(.+)\.jsonl$", base)
    if not m:
        return ""
    return m.group(1)


def canonicalize_target_name(name: str) -> str:
    name = name.strip()

    if name in VALID_CRATES:
        return name

    if name in CASE_TO_CRATE:
        return CASE_TO_CRATE[name]

    raise ValueError(
        f"Unknown target name: {name}. "
        f"Must be one of crate names {sorted(VALID_CRATES)} "
        f"or runtime case names {sorted(CASE_TO_CRATE.keys())}."
    )


def extract_prog_name_from_source_path(source_path: str) -> str:

    abs_path = os.path.abspath(source_path)
    parts = abs_path.split(os.sep)


    if len(parts) >= 3 and parts[-1] == "main.rs" and parts[-2] == "src":
        crate_name = parts[-3]
        return canonicalize_target_name(crate_name)


    base = os.path.basename(abs_path)
    stem = os.path.splitext(base)[0]

    if stem in VALID_CRATES or stem in CASE_TO_CRATE:
        return canonicalize_target_name(stem)

    return stem


def normalize_record(record: Dict) -> Dict:
    mir = [int(x) for x in record.get("MIR", [])]
    llvm = [int(x) for x in record.get("LLVM", [])]
    return {
        "MIR": mir,
        "LLVM": llvm,
    }


def record_to_key(record: Dict) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return (tuple(record["MIR"]), tuple(record["LLVM"]))


def load_jsonl_records(
    file_path: str,
    expected_mir_dim: Optional[int] = None,
    expected_llvm_dim: Optional[int] = None,
) -> List[Dict]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            obj = normalize_record(obj)

            if expected_mir_dim is not None and len(obj["MIR"]) != expected_mir_dim:
                continue
            if expected_llvm_dim is not None and len(obj["LLVM"]) != expected_llvm_dim:
                continue

            records.append(obj)

    return records


def datafile_belongs_to_target(datafile_prog_name: str, target_crate: str) -> bool:
    if datafile_prog_name == target_crate:
        return True

    if datafile_prog_name in CASE_TO_CRATE and CASE_TO_CRATE[datafile_prog_name] == target_crate:
        return True

    return False


def build_trainset(
    data_dir: str,
    target_prog: str,
    output_file: str,
    expected_mir_dim: Optional[int] = None,
    expected_llvm_dim: Optional[int] = None,
    dedup: bool = True,
):
    target_crate = canonicalize_target_name(target_prog)

    pattern = os.path.join(data_dir, "data_*.jsonl")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        raise FileNotFoundError(f"No data_*.jsonl files found in {data_dir}")

    selected_files = []
    skipped_files = []

    for fp in all_files:
        prog = extract_prog_name_from_datafile(fp)
        if datafile_belongs_to_target(prog, target_crate):
            skipped_files.append(fp)
        else:
            selected_files.append(fp)

    if not selected_files:
        raise ValueError(f"After excluding target crate '{target_crate}', no files remain.")

    merged_records = []
    seen = set()

    for fp in selected_files:
        records = load_jsonl_records(
            fp,
            expected_mir_dim=expected_mir_dim,
            expected_llvm_dim=expected_llvm_dim,
        )

        for record in records:
            key = record_to_key(record)

            if dedup:
                if key in seen:
                    continue
                seen.add(key)

            merged_records.append(record)

    if not merged_records:
        raise ValueError("Merged result is empty.")

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    info = {
        "target_prog": target_prog,
        "target_crate": target_crate,
        "skipped_files": [os.path.basename(x) for x in skipped_files],
        "used_file_count": len(selected_files),
        "total_merged_lines": len(merged_records),
        "output_file": output_file,
    }
    return info


def split_list_round_robin(items: List[str], num_shards: int) -> List[List[str]]:
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(items):
        shards[i % num_shards].append(item)
    return shards


def get_shard_cases(num_shards: int, shard_id: int) -> List[str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not (0 <= shard_id < num_shards):
        raise ValueError("shard_id must be in [0, num_shards)")
    shards = split_list_round_robin(ALL_19_CASES, num_shards)
    return shards[shard_id]


def main():
    parser = argparse.ArgumentParser(
        description="Build merged PDCAT trainset for rustc-perf runtime-benchmarks."
    )

    parser.add_argument("--data_dir", type=str, required=False,
                        help="Directory containing data_*.jsonl")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--target_prog", type=str,
                       help="Target crate name or benchmark case name")
    group.add_argument("--source_path", type=str,
                       help="Full source path of the target program, e.g. .../bufreader/src/main.rs")

    parser.add_argument("--output_file", type=str, default="merged_trainset.jsonl",
                        help="Output merged jsonl file")
    parser.add_argument("--expected_mir_dim", type=int, default=None,
                        help="Expected MIR dimension")
    parser.add_argument("--expected_llvm_dim", type=int, default=None,
                        help="Expected LLVM dimension")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable deduplication")


    parser.add_argument("--print_shard_cases", action="store_true",
                        help="Print case names for one shard and exit")
    parser.add_argument("--num_shards", type=int, default=3,
                        help="Number of shards, default 3")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard id, 0-based")

    args = parser.parse_args()

    if args.print_shard_cases:
        cases = get_shard_cases(args.num_shards, args.shard_id)
        print("====================================================")
        print(f"num_shards = {args.num_shards}")
        print(f"shard_id   = {args.shard_id}")
        print("assigned benchmark cases:")
        for x in cases:
            print(x)
        print("====================================================")
        sys.exit(0)

    if not args.data_dir:
        raise ValueError("--data_dir is required unless --print_shard_cases is used")

    if not args.target_prog and not args.source_path:
        raise ValueError("Must provide --target_prog or --source_path")

    if args.target_prog:
        target_prog = args.target_prog
    else:
        target_prog = extract_prog_name_from_source_path(args.source_path)

    info = build_trainset(
        data_dir=args.data_dir,
        target_prog=target_prog,
        output_file=args.output_file,
        expected_mir_dim=args.expected_mir_dim,
        expected_llvm_dim=args.expected_llvm_dim,
        dedup=(not args.no_dedup),
    )

    print("====================================================")
    print(f"Target program      : {info['target_prog']}")
    print(f"Target crate        : {info['target_crate']}")
    print(f"Skipped file(s)     : {info['skipped_files']}")
    print(f"Used file count     : {info['used_file_count']}")
    print(f"Total merged lines  : {info['total_merged_lines']}")
    print(f"Output file         : {info['output_file']}")
    print("====================================================")


if __name__ == "__main__":
    main()