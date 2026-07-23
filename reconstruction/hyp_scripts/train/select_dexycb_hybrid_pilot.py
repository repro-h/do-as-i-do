#!/usr/bin/env python3
"""Select a deterministic, hand-side-balanced pilot from a hybrid manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--num-streams", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_path = Path(args.out_jsonl).expanduser().resolve()
    records = load_jsonl(manifest_path)
    if args.num_streams <= 0 or args.num_streams > len(records):
        raise ValueError(f"Invalid --num-streams={args.num_streams}")

    random.Random(args.seed).shuffle(records)
    target_by_side = {
        "left": args.num_streams // 2,
        "right": args.num_streams - args.num_streams // 2,
    }
    selected = []
    selected_ids = set()
    used_objects = set()
    side_counts = Counter()

    # First maximize object diversity while satisfying the requested side balance.
    for prefer_new_object in (True, False):
        for record in records:
            if len(selected) >= args.num_streams:
                break
            stream_id = record["stream_id"]
            side = record["hand_side"]
            object_name = record["object_name"]
            if stream_id in selected_ids or side_counts[side] >= target_by_side[side]:
                continue
            if prefer_new_object and object_name in used_objects:
                continue
            selected.append(record)
            selected_ids.add(stream_id)
            used_objects.add(object_name)
            side_counts[side] += 1

    if len(selected) != args.num_streams:
        raise RuntimeError(
            f"Could only select {len(selected)}/{args.num_streams} streams; "
            f"side counts={dict(side_counts)}"
        )

    selected.sort(key=lambda row: (row["hand_side"], row["object_name"], row["stream_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record) + "\n")

    summary = {
        "source_manifest": str(manifest_path),
        "out_jsonl": str(out_path),
        "num_streams": len(selected),
        "seed": args.seed,
        "hand_side_counts": dict(Counter(row["hand_side"] for row in selected)),
        "object_counts": dict(Counter(row["object_name"] for row in selected)),
        "streams": [
            {
                "stream_id": row["stream_id"],
                "hand_side": row["hand_side"],
                "object_name": row["object_name"],
            }
            for row in selected
        ],
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
