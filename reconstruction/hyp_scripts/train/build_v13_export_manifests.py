#!/usr/bin/env python3
"""Build disjoint Pi3X export manifests from the active V13 window split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument(
        "--source-manifest",
        action="append",
        required=True,
        help="May be passed more than once; rows are joined by stream_id.",
    )
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stream_ids(path: Path) -> set[str]:
    return {str(row["stream_id"]) for row in load_jsonl(path)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def summarize(rows: list[dict]) -> dict:
    sides = Counter(str(row.get("hand_side", "unknown")) for row in rows)
    return {
        "streams": len(rows),
        "sides": dict(sorted(sides.items())),
    }


def main() -> None:
    args = parse_args()
    train_ids = stream_ids(Path(args.train_windows).expanduser().resolve())
    val_ids = stream_ids(Path(args.val_windows).expanduser().resolve())
    overlap = train_ids & val_ids
    if overlap:
        examples = sorted(overlap)[:10]
        raise RuntimeError(
            f"Train/val overlap contains {len(overlap)} streams: {examples}"
        )

    records: dict[str, dict] = {}
    for manifest in args.source_manifest:
        path = Path(manifest).expanduser().resolve()
        for row in load_jsonl(path):
            stream_id = str(row["stream_id"])
            previous = records.get(stream_id)
            if (
                previous is not None
                and Path(previous["stream_dir"]).expanduser().resolve()
                != Path(row["stream_dir"]).expanduser().resolve()
            ):
                raise RuntimeError(
                    f"Conflicting manifest rows for {stream_id}"
                )
            records.setdefault(stream_id, row)

    requested = train_ids | val_ids
    missing = sorted(requested - records.keys())
    if missing:
        raise KeyError(
            f"Missing {len(missing)} streams from source manifests: "
            f"{missing[:10]}"
        )

    train_rows = [records[key] for key in sorted(train_ids)]
    val_rows = [records[key] for key in sorted(val_ids)]
    train_out = Path(args.train_out).expanduser().resolve()
    val_out = Path(args.val_out).expanduser().resolve()
    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    print(json.dumps({
        "train": {**summarize(train_rows), "output": str(train_out)},
        "val": {**summarize(val_rows), "output": str(val_out)},
        "overlap": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
