#!/usr/bin/env python3
"""Deterministically sample object/hand-balanced SE(3) training windows."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-windows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.num_windows <= 0:
        raise ValueError("num-windows must be positive")
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = load_jsonl(input_path)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["object_name"]), str(row["hand_side"]))].append(row)

    rng = random.Random(args.seed)
    queues = {}
    for key, values in sorted(groups.items()):
        rng.shuffle(values)
        queues[key] = deque(values)

    selected = []
    active = list(sorted(queues))
    while active and len(selected) < min(args.num_windows, len(rows)):
        next_active = []
        for key in active:
            if queues[key] and len(selected) < args.num_windows:
                selected.append(queues[key].popleft())
            if queues[key]:
                next_active.append(key)
        active = next_active

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")

    counts = defaultdict(int)
    for row in selected:
        counts[(str(row["object_name"]), str(row["hand_side"]))] += 1
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "seed": args.seed,
        "num_available": len(rows),
        "num_selected": len(selected),
        "strata": {
            f"{name}/{side}": count
            for (name, side), count in sorted(counts.items())
        },
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
