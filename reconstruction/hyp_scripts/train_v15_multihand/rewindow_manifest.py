#!/usr/bin/env python3
"""Rebuild fixed-length windows from one or more existing JSONL manifests."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def row_dataset(row):
    if row.get("dataset"):
        return str(row["dataset"])
    return str(row.get("schema_version", "unknown")).split("_", 1)[0]


def consecutive_runs(positions):
    positions = sorted(positions)
    if not positions:
        return
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            yield list(range(start, previous + 1))
            start = position
        previous = position
    yield list(range(start, previous + 1))


def window_positions(run, size, stride):
    """Cover a contiguous run with full windows and an anchored tail window."""
    if len(run) < size:
        return []
    starts = list(range(0, len(run) - size + 1, stride))
    final = len(run) - size
    if not starts or starts[-1] != final:
        starts.append(final)
    return [run[start:start + size] for start in starts]


def main():
    args = parse_args()
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("Window size and stride must be positive")

    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    for path in (output, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    source_rows = 0
    for value in args.input:
        source = Path(value).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                grouped[(row_dataset(row), str(row["stream_id"]))].append(row)
                source_rows += 1

    output_rows = []
    totals = defaultdict(lambda: {
        "streams": 0,
        "source_windows": 0,
        "recoverable_frames": 0,
        "windows": 0,
        "covered_frames": set(),
        "short_run_frames": 0,
        "gapped_or_conflicting_frames": 0,
    })

    for (dataset, stream), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["start"]), int(row["end"])))
        stats = totals[dataset]
        stats["streams"] += 1
        stats["source_windows"] += len(rows)

        frame_records = {}
        template_by_position = {}
        conflicts = set()
        for row in rows:
            frame_count = len(row["frame_indices"])
            start = int(row["start"])
            frame_keys = {
                key for key, value in row.items()
                if isinstance(value, list) and len(value) == frame_count
            }
            for offset in range(frame_count):
                position = start + offset
                record = {key: row[key][offset] for key in frame_keys}
                if position in frame_records and frame_records[position] != record:
                    conflicts.add(position)
                    continue
                frame_records[position] = record
                template_by_position[position] = row

        for position in conflicts:
            frame_records.pop(position, None)
            template_by_position.pop(position, None)
        stats["gapped_or_conflicting_frames"] += len(conflicts)
        stats["recoverable_frames"] += len(frame_records)

        for run in consecutive_runs(frame_records):
            windows = window_positions(run, args.window_size, args.window_stride)
            if not windows:
                stats["short_run_frames"] += len(run)
                continue
            for positions in windows:
                template = template_by_position[positions[0]]
                row = {
                    key: value for key, value in template.items()
                    if not isinstance(value, list) and key not in ("start", "end")
                }
                common_keys = set.intersection(*(
                    set(frame_records[position]) for position in positions
                ))
                for key in sorted(common_keys):
                    row[key] = [frame_records[position][key] for position in positions]
                row["dataset"] = dataset
                row["stream_id"] = stream
                row["start"] = positions[0]
                row["end"] = positions[-1] + 1
                if len(row.get("frame_indices", [])) != args.window_size:
                    raise RuntimeError(f"Failed to construct {dataset}:{stream}:{positions[0]}")
                output_rows.append(row)
                stats["windows"] += 1
                stats["covered_frames"].update(
                    (stream, position) for position in positions
                )

    output_rows.sort(key=lambda row: (
        str(row["dataset"]), str(row["stream_id"]), int(row["start"])
    ))
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    serializable = {}
    for dataset, stats in sorted(totals.items()):
        item = dict(stats)
        item["covered_frames"] = len(stats["covered_frames"])
        item["uncovered_recoverable_frames"] = (
            stats["recoverable_frames"] - item["covered_frames"]
        )
        serializable[dataset] = item
    report = {
        "inputs": [str(Path(value).expanduser().resolve()) for value in args.input],
        "output": str(output),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "source_rows": source_rows,
        "output_rows": len(output_rows),
        "datasets": serializable,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
