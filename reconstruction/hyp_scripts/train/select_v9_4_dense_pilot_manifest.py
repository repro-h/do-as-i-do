#!/usr/bin/env python3
"""Select a side-balanced stream manifest restricted to available windows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--streams-per-side", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    eligible = {
        str(row["stream_id"]) for row in load_jsonl(windows_path)
    }
    rows = [
        row for row in load_jsonl(manifest_path)
        if str(row["stream_id"]) in eligible
    ]
    by_side: dict[str, list[dict]] = {"left": [], "right": []}
    for row in rows:
        side = str(row.get("hand_side", "")).lower()
        if side in by_side:
            by_side[side].append(row)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for side in ("left", "right"):
        values = sorted(by_side[side], key=lambda row: row["stream_id"])
        rng.shuffle(values)
        chosen = values[: args.streams_per_side]
        selected.extend(chosen)
        counts[side] = len(chosen)
    selected.sort(key=lambda row: row["stream_id"])

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "windows": str(windows_path),
        "output": str(out_path),
        "eligible_streams": len(eligible),
        "available": {key: len(value) for key, value in by_side.items()},
        "selected": counts,
        "total": len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
