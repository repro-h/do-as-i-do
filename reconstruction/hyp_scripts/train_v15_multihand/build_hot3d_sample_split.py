#!/usr/bin/env python3
"""Build deterministic participant-diverse HOT3D train/validation manifests."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--sequence-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows, split):
    with path.open("w", encoding="utf-8") as handle:
        for source in rows:
            row = dict(source)
            row["split"] = split
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def participant(sequence):
    return sequence.split("_", 1)[0]


def main():
    args = parse_args()
    processed_root = Path(args.processed_root).expanduser().resolve()
    sequence_list = Path(args.sequence_list).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    outputs = [
        out_dir / "train_windows.jsonl",
        out_dir / "val_windows.jsonl",
        out_dir / "all_windows.jsonl",
        out_dir / "split.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Outputs already exist; pass --overwrite: {existing[0]}"
        )

    sequences = [
        line.strip()
        for line in sequence_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"Duplicate sequence in {sequence_list}")
    if not 0 < args.val_count < len(sequences):
        raise ValueError("--val-count must be between 1 and sequence count - 1")

    rows_by_sequence = {}
    summaries = {}
    failures = []
    for sequence in sequences:
        sequence_root = processed_root / sequence
        summary_path = sequence_root / "summary.json"
        windows_path = sequence_root / "train_windows.jsonl"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = read_jsonl(windows_path)
            if not rows:
                raise RuntimeError("manifest has no windows")
            stream_ids = {str(row["stream_id"]) for row in rows}
            if len(stream_ids) != 1:
                raise RuntimeError(f"expected one stream, found {len(stream_ids)}")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            failures.append({"sequence": sequence, "error": repr(error)})
            continue
        rows_by_sequence[sequence] = rows
        summaries[sequence] = summary
    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        raise RuntimeError(
            f"{len(failures)} HOT3D sequences are not ready; finish processing first"
        )

    grouped = defaultdict(list)
    for sequence in sequences:
        grouped[participant(sequence)].append(sequence)
    rng = random.Random(args.seed)
    participant_order = sorted(grouped)
    rng.shuffle(participant_order)
    for values in grouped.values():
        values.sort()
        rng.shuffle(values)

    validation = []
    for name in participant_order:
        if len(validation) >= args.val_count:
            break
        validation.append(grouped[name][0])
    if len(validation) < args.val_count:
        remaining = [item for item in sequences if item not in validation]
        rng.shuffle(remaining)
        validation.extend(remaining[:args.val_count - len(validation)])

    validation_set = set(validation)
    training = [item for item in sequences if item not in validation_set]
    train_rows = [row for item in training for row in rows_by_sequence[item]]
    val_rows = [row for item in validation for row in rows_by_sequence[item]]
    all_rows = train_rows + val_rows

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs[0], train_rows, "train")
    write_jsonl(outputs[1], val_rows, "val")
    with outputs[2].open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    report = {
        "schema_version": "hot3d_sample_split_v1",
        "seed": args.seed,
        "sequences": len(sequences),
        "train_sequences": training,
        "val_sequences": validation,
        "val_participants": [participant(item) for item in validation],
        "train_windows": len(train_rows),
        "val_windows": len(val_rows),
        "sequence_quality": {
            item: {
                "windows": len(rows_by_sequence[item]),
                "exported_frames": summaries[item].get("exported_frames"),
                "frames_with_hands": summaries[item].get("frames_with_hands"),
                "joint_in_frame_fraction": summaries[item].get(
                    "joint_in_frame_fraction"
                ),
            }
            for item in sequences
        },
    }
    outputs[3].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
