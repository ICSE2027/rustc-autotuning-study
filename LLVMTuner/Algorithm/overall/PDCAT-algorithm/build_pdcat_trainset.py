import os
import re
import glob
import argparse


def normalize_seq_line(line: str) -> str:
    nums = re.findall(r'[01]', line)
    return ",".join(nums)


def extract_prog_name_from_datafile(filename: str) -> str:
    base = os.path.basename(filename)
    m = re.match(r"data_(.+)\.txt$", base)
    if not m:
        return ""
    return m.group(1)


def extract_prog_name_from_source_path(source_path: str) -> str:
    return os.path.basename(os.path.normpath(source_path))


def build_trainset(data_dir: str, target_prog: str, output_file: str, dedup: bool = True):
    pattern = os.path.join(data_dir, "data_*.txt")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        raise FileNotFoundError(f"No data_*.txt files found in {data_dir}")

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

    merged_lines = []
    seen = set()

    for fp in selected_files:
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                norm_line = normalize_seq_line(line)
                if not norm_line:
                    continue

                if dedup:
                    if norm_line in seen:
                        continue
                    seen.add(norm_line)

                merged_lines.append(norm_line)

    if not merged_lines:
        raise ValueError("Merged result is empty.")

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(output_file, "w") as f:
        for line in merged_lines:
            f.write(line + "\n")

    info = {
        "target_prog": target_prog,
        "skipped_files": [os.path.basename(x) for x in skipped_files],
        "used_file_count": len(selected_files),
        "total_merged_lines": len(merged_lines),
        "output_file": output_file
    }
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a merged PDCAT training set by excluding the target program."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing data_*.txt")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target_prog", type=str,
                       help="Target program name, e.g. gemm")
    group.add_argument("--source_path", type=str,
                       help="Full source path of the target program, e.g. .../blas/gemm")

    parser.add_argument("--output_file", type=str, default="merged_trainset.txt",
                        help="Output merged txt file")
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
        dedup=(not args.no_dedup)
    )

    print("====================================================")
    print(f"Target program      : {info['target_prog']}")
    print(f"Skipped file(s)     : {info['skipped_files']}")
    print(f"Used file count     : {info['used_file_count']}")
    print(f"Total merged lines  : {info['total_merged_lines']}")
    print(f"Output file         : {info['output_file']}")
    print("====================================================")