#!/usr/bin/env python3
"""Batch-convert H2O sequences and merge their V15 manifest."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audit-json")
    source.add_argument("--subject-root", action="append")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_sequences(subject_roots):
    """Discover sequence directories without parsing every per-frame pose file."""
    items = []
    skipped = []
    seen = set()
    for value in subject_roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"H2O subject root does not exist: {root}")
        for intrinsics in sorted(root.rglob("cam_intrinsics.txt")):
            sequence_dir = intrinsics.parent
            key = str(sequence_dir)
            if key in seen:
                continue
            seen.add(key)
            missing = [
                name for name in ("rgb", "hand_pose")
                if not (sequence_dir / name).is_dir()
            ]
            if missing:
                skipped.append({
                    "sequence_dir": key,
                    "reason": "missing " + ", ".join(missing),
                })
                continue
            items.append({
                "subject": root.name,
                "sequence": sequence_dir.relative_to(root).as_posix(),
                "sequence_dir": key,
            })
    items.sort(key=lambda item: (item["subject"], item["sequence"]))
    return items, skipped


def output_is_current(manifest):
    if not manifest.is_file():
        return False
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or not rows[0].get("label_paths"):
        return False
    try:
        with np.load(rows[0]["label_paths"][0], allow_pickle=False) as data:
            return {"seg", "joint_in_frame", "observation_valid"}.issubset(data.files)
    except (OSError, KeyError, ValueError):
        return False


def convert(item, args, converter, out_root):
    sequence_dir = Path(item["sequence_dir"])
    relative = item["sequence"].replace("/", "__")
    stream_id = f"h2o__{item['subject']}__{relative}"
    stream_out = out_root / "streams" / "train" / stream_id
    manifest = stream_out / "train_windows.jsonl"
    if not args.overwrite and output_is_current(manifest):
        return {"stream_id": stream_id, "manifest": str(manifest), "cached": True}
    command = [
        sys.executable, "-u", str(converter),
        "--sequence-dir", str(sequence_dir),
        "--out-dir", str(stream_out),
        "--split", "train",
        "--stream-id", stream_id,
        "--window-size", str(args.window_size),
        "--window-stride", str(args.window_stride),
        "--overwrite",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not output_is_current(manifest):
        return {
            "stream_id": stream_id,
            "error": f"returncode={result.returncode}",
            "output": result.stdout[-4000:],
        }
    return {"stream_id": stream_id, "manifest": str(manifest), "cached": False}


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    audit_path = None
    skipped = []
    if args.audit_json:
        audit_path = Path(args.audit_json).expanduser().resolve()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        items = audit["sequences"]
    else:
        items, skipped = discover_sequences(args.subject_root)
        if not items:
            raise RuntimeError("No H2O sequences found under --subject-root")
        print(
            f"Discovered {len(items)} sequences directly; skipped {len(skipped)}",
            flush=True,
        )
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    converter = Path(__file__).with_name("build_h2o_sequence_windows.py")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert, item, args, converter, out_root): item
            for item in items
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            state = "FAILED" if "error" in result else "cached" if result["cached"] else "done"
            print(f"[{index}/{len(items)}] {state}: {result['stream_id']}", flush=True)
            if "output" in result:
                print(result["output"], flush=True)

    rows = []
    failures = [result for result in results if "error" in result]
    for result in results:
        if "manifest" not in result:
            continue
        rows.extend(
            json.loads(line) for line in Path(result["manifest"]).read_text().splitlines()
            if line.strip()
        )
    rows.sort(key=lambda row: (row["stream_id"], row["start"]))
    manifest_root = out_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_root / "train_windows.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            row["dataset"] = "h2o"
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    status = {
        "audit_json": str(audit_path) if audit_path else None,
        "subject_roots": args.subject_root,
        "discovery_skipped": skipped,
        "sequences": len(items),
        "completed": len(results) - len(failures),
        "cached": sum(bool(result.get("cached")) for result in results),
        "windows": len(rows),
        "manifest": str(manifest),
        "failures": failures,
    }
    (out_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
