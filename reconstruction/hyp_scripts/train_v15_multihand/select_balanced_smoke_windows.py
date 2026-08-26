#!/usr/bin/env python3
"""Build a balanced smoke manifest from complete original-camera caches."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--pi3x-root", action="append", required=True)
    parser.add_argument("--visibility-root", required=True)
    parser.add_argument("--left-streams", type=int, default=2)
    parser.add_argument("--right-streams", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cache_path(root, row):
    return (
        root / str(row["stream_id"]) / "windows"
        / f"window_{int(row['start']):06d}_{int(row['end']):06d}.npz"
    )


def usable_cache(path):
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if bool(np.asarray(data.get("horizontal_mirror", False)).item()):
                return False
            required = (
                "geometry_patch_features",
                "geometry_feature_grid_hw",
                "resized_wh",
                "intrinsics_resized",
                "metric_window_features",
            )
            return all(key in data.files for key in required)
    except (OSError, ValueError, KeyError):
        return False


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    roots = [Path(value).expanduser().resolve() for value in args.pi3x_root]
    visibility_root = Path(args.visibility_root).expanduser().resolve()
    streams = defaultdict(list)
    with Path(args.input).expanduser().resolve().open(
        "r", encoding="utf-8"
    ) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                streams[str(row["stream_id"])].append(row)

    candidates = {"left": [], "right": []}
    rejected = defaultdict(int)
    for stream_id, rows in sorted(streams.items()):
        side = str(rows[0].get("hand_side_metadata_only", "unknown")).lower()
        if side not in candidates:
            continue
        visibility = visibility_root / stream_id / "visibility_cache.npz"
        if not visibility.is_file():
            rejected[f"{side}_missing_visibility"] += 1
            continue
        selected_rows = []
        for row in sorted(rows, key=lambda value: (value["start"], value["end"])):
            selected_cache = None
            for root in roots:
                candidate = cache_path(root, row)
                if usable_cache(candidate):
                    selected_cache = candidate
                    break
            if selected_cache is None:
                selected_rows = []
                break
            selected = dict(row)
            selected["dense_pi3x_npz"] = str(selected_cache)
            selected_rows.append(selected)
        if selected_rows:
            candidates[side].append((stream_id, selected_rows))
        else:
            rejected[f"{side}_missing_original_camera_pi3x"] += 1

    requested = {"left": args.left_streams, "right": args.right_streams}
    chosen = {
        side: candidates[side][:requested[side]] for side in candidates
    }
    missing = {
        side: max(0, requested[side] - len(chosen[side])) for side in candidates
    }
    rows = [
        row
        for side in ("left", "right")
        for _, stream_rows in chosen[side]
        for row in stream_rows
    ]
    if not rows:
        raise RuntimeError("No complete original-camera streams found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "output": str(output),
        "windows": len(rows),
        "selected": {
            side: [stream for stream, _ in chosen[side]] for side in chosen
        },
        "missing_requested_streams": missing,
        "rejected": dict(rejected),
        "pi3x_roots_in_priority_order": [str(root) for root in roots],
    }
    print(json.dumps(report, indent=2))
    if any(missing.values()):
        raise RuntimeError(
            "Balanced selection is incomplete; export original-camera Pi3X "
            f"for the missing streams: {missing}"
        )


if __name__ == "__main__":
    main()
