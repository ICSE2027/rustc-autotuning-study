import os
import re
import json
import glob
import argparse
from typing import Dict, List, Tuple


def extract_prog_name_from_datafile(filename: str) -> str:
    base = os.path.basename(filename)
    m = re.match(r"data_(.+)\.jsonl$", base)
    if not m:
        return ""
    return m.group(1)


def extract_prog_name_from_source_path(source_path: str) -> str:
    base = os.path.basename(source_path)
    return os.path.splitext(base)[0]


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
    expected_mir_dim: int = None,
    expected_llvm_dim: int = None,
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


def build_trainset(
    data_dir: str,
    target_prog: str,
    output_file: str,
    expected_mir_dim: int = None,
    expected_llvm_dim: int = None,
    dedup: bool = True,
):
    pattern = os.path.join(data_dir, "data_*.jsonl")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        raise FileNotFoundError(f"No data_*.jsonl files found in {data_dir}")

    selected_files = []
    skipped_files = []

    for fp in all_files:
        prog = extract_prog_name_from_datafile(fp)
        if prog == target_prog:
            skipped_files.append(fp)
        else:
            selected_files.append(fp)

    if not selected_files:
        raise ValueError(f"After excluding target '{target_prog}', no files remain.")

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
        "skipped_files": [os.path.basename(x) for x in skipped_files],
        "used_file_count": len(selected_files),
        "total_merged_lines": len(merged_records),
        "output_file": output_file,
    }
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a merged Rustc PDCAT training set by excluding the target program."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing data_*.jsonl")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target_prog", type=str,
                       help="Target program name, e.g. gemm")
    group.add_argument("--source_path", type=str,
                       help="Full source path of the target program, e.g. .../src/bin/gemm.rs")

    parser.add_argument("--output_file", type=str, default="merged_trainset.jsonl",
                        help="Output merged jsonl file")
    parser.add_argument("--expected_mir_dim", type=int, default=None,
                        help="Expected MIR dimension")
    parser.add_argument("--expected_llvm_dim", type=int, default=None,
                        help="Expected LLVM dimension")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable deduplication")

    args = parser.parse_args()

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
    print(f"Skipped file(s)     : {info['skipped_files']}")
    print(f"Used file count     : {info['used_file_count']}")
    print(f"Total merged lines  : {info['total_merged_lines']}")
    print(f"Output file         : {info['output_file']}")
    print("====================================================")