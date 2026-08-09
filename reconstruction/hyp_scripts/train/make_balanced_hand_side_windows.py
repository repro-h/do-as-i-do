#!/usr/bin/env python3
"""Select all windows from a balanced set of left/right streams."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--streams-per-side", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    return item.decode() if isinstance(item, bytes) else str(item)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_stream: dict[str, list[dict]] = {}
    for row in rows:
        by_stream.setdefault(str(row["stream_id"]), []).append(row)

    by_side: dict[str, list[str]] = {"left": [], "right": []}
    failures: list[dict] = []
    for stream_id, stream_rows in by_stream.items():
        supervision = Path(stream_rows[0]["supervision_npz"]).expanduser()
        try:
            with np.load(supervision, allow_pickle=False) as se3:
                global_path = Path(
                    scalar_text(se3["source_global_supervision"])
                ).expanduser()
            with np.load(global_path, allow_pickle=False) as glob:
                side = scalar_text(glob["hand_side"])
        except (OSError, KeyError, ValueError) as error:
            failures.append({"stream_id": stream_id, "error": str(error)})
            continue
        if side in by_side:
            by_side[side].append(stream_id)

    randomizer = random.Random(args.seed)
    selected: set[str] = set()
    selected_counts: dict[str, int] = {}
    for side, streams in by_side.items():
        streams.sort()
        randomizer.shuffle(streams)
        chosen = streams[: args.streams_per_side]
        selected.update(chosen)
        selected_counts[side] = len(chosen)

    output_rows = [row for row in rows if str(row["stream_id"]) in selected]
    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "available_streams": {
                    side: len(streams) for side, streams in by_side.items()
                },
                "selected_streams": selected_counts,
                "num_windows": len(output_rows),
                "num_failures": len(failures),
                "failures": failures[:20],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
