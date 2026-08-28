#!/usr/bin/env python3
"""Batch-convert selected TACO sequences to V15 multi-hand windows."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--taco-code-root", required=True)
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(paths):
    rows = []
    seen = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["local_split"], row["triplet"], row["sequence"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["local_split"], row["triplet"], row["sequence"]
        ),
    )


def output_is_current(stream_out, split):
    manifest = stream_out / f"{split}_windows.jsonl"
    summary = stream_out / "summary.json"
    if not manifest.is_file() or not summary.is_file():
        return False
    try:
        metadata = json.loads(summary.read_text(encoding="utf-8"))
        first = next(
            line for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        row = json.loads(first)
        label = Path(row["label_paths"][0])
        return metadata.get("mano_backend") == "manopth" and label.is_file()
    except (OSError, ValueError, KeyError, StopIteration):
        return False


def convert(row, args, converter, out_root, index, total):
    split = row["local_split"]
    stream_id = f"taco__{row['sequence']}"
    stream_out = out_root / "streams" / split / stream_id
    manifest = stream_out / f"{split}_windows.jsonl"
    if not args.overwrite and output_is_current(stream_out, split):
        return {
            "index": index,
            "stream_id": stream_id,
            "split": split,
            "status": "cached",
            "manifest": str(manifest),
        }

    stream_out.mkdir(parents=True, exist_ok=True)
    log_path = stream_out / "prepare.log"
    command = [
        sys.executable, "-u", str(converter),
        "--taco-root", str(args.taco_root),
        "--taco-code-root", str(args.taco_code_root),
        "--triplet", row["triplet"],
        "--sequence", row["sequence"],
        "--mano-model-folder", str(args.mano_model_folder),
        "--out-dir", str(stream_out),
        "--split", split,
        "--window-size", str(args.window_size),
        "--window-stride", str(args.window_stride),
        "--min-valid-frames", str(args.min_valid_frames),
        "--overlay-count", "0",
        "--jpeg-quality", str(args.jpeg_quality),
        "--overwrite",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    status = "done" if result.returncode == 0 else "failed"
    return {
        "index": index,
        "total": total,
        "stream_id": stream_id,
        "split": split,
        "status": status,
        "returncode": result.returncode,
        "manifest": str(manifest),
        "log": str(log_path),
    }


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window size and stride must be positive")

    out_root = Path(args.out_root).expanduser().resolve()
    converter = Path(__file__).with_name("prepare_taco_v15.py")
    rows = load_rows(args.selection)
    if not rows:
        raise RuntimeError("No selected TACO sequences")
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                convert, row, args, converter, out_root, index, len(rows)
            ): row
            for index, row in enumerate(rows, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{result['index']}/{len(rows)}] {result['status']} "
                f"{result['split']} {result['stream_id']}",
                flush=True,
            )

    merged = {}
    failures = []
    for result in results:
        if result["status"] == "failed":
            failures.append(result)
            continue
        manifest = Path(result["manifest"])
        if not manifest.is_file():
            failures.append({**result, "status": "missing_manifest"})
            continue
        split_rows = merged.setdefault(result["split"], [])
        split_rows.extend(
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    manifest_root = out_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split, split_rows in merged.items():
        split_rows.sort(key=lambda row: (row["stream_id"], row["start"]))
        output = manifest_root / f"{split}_windows.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    status = {
        "requested_sequences": len(rows),
        "completed_sequences": sum(
            result["status"] in {"done", "cached"} for result in results
        ),
        "failed_sequences": len(failures),
        "workers": args.workers,
        "windows": {
            split: len(split_rows) for split, split_rows in sorted(merged.items())
        },
        "failures": failures,
    }
    (out_root / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
