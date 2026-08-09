#!/usr/bin/env python3
"""Match ordinary supervision rows to exported dense Pi3X windows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from train_v9_camera_hand_residual import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_windows).expanduser().resolve()
    dense_root = Path(args.dense_root).expanduser().resolve()
    rows_by_stream: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(source_path):
        rows_by_stream[str(row["stream_id"])].append(row)

    output_rows: list[dict] = []
    missing_source: list[str] = []
    for summary_path in sorted(dense_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stream_id = summary_path.parent.name
        source_rows = rows_by_stream.get(stream_id, [])
        if not source_rows:
            missing_source.append(stream_id)
            continue
        template = source_rows[0]
        for dense in summary.get("windows", []):
            start, end = int(dense["start"]), int(dense["end"])
            covering = next(
                (
                    row for row in source_rows
                    if int(row["start"]) <= start
                    and int(row["end"]) >= end
                ),
                None,
            )
            row = dict(covering or template)
            row.update({
                "stream_id": stream_id,
                "start": start,
                "end": end,
                "dense_pi3x_npz": str(
                    summary_path.parent / "windows"
                    / f"window_{start:06d}_{end:06d}.npz"
                ),
            })
            output_rows.append(row)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({
        "source_windows": str(source_path),
        "dense_root": str(dense_root),
        "output": str(out_path),
        "streams": len({row["stream_id"] for row in output_rows}),
        "windows": len(output_rows),
        "missing_source_streams": missing_source,
    }, indent=2))


if __name__ == "__main__":
    main()
