#!/usr/bin/env python3
"""Audit H2O sequence lengths and expected V15 window counts."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    return parser.parse_args()


def frame_number(path):
    try:
        return int(path.stem)
    except ValueError:
        return None


def frame_map(paths):
    result = {}
    for path in paths:
        number = frame_number(path)
        if number is not None:
            result[number] = path
    return result


def window_count(length, size, stride):
    if length < size:
        return 0
    starts = list(range(0, length - size + 1, stride))
    return len(starts) + int(starts[-1] != length - size)


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def audit_sequence(sequence_dir, subject_root, size, stride):
    images = frame_map(list((sequence_dir / "rgb").glob("*.png")) +
                       list((sequence_dir / "rgb").glob("*.jpg")))
    poses = frame_map((sequence_dir / "hand_pose").glob("*.txt"))
    frames = sorted(set(images) & set(poses))
    side_valid = np.zeros(2, dtype=np.int64)
    invalid_pose_files = 0
    for frame in frames:
        try:
            values = np.loadtxt(poses[frame], dtype=np.float32).reshape(-1)
            if len(values) != 128:
                invalid_pose_files += 1
                continue
            side_valid += (np.asarray([values[0], values[64]]) > 0.5)
        except (OSError, ValueError):
            invalid_pose_files += 1
    return {
        "subject": subject_root.name,
        "sequence": str(sequence_dir.relative_to(subject_root)),
        "sequence_dir": str(sequence_dir),
        "frames": len(frames),
        "windows": window_count(len(frames), size, stride),
        "left_valid_frames": int(side_valid[0]),
        "right_valid_frames": int(side_valid[1]),
        "both_valid_frames_lower_bound": int(max(0, side_valid.sum() - len(frames))),
        "invalid_pose_files": invalid_pose_files,
    }


def main():
    args = parse_args()
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("Window size and stride must be positive")
    rows = []
    roots = [Path(value).expanduser().resolve() for value in args.subject_root]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for intrinsics in sorted(root.rglob("cam_intrinsics.txt")):
            sequence_dir = intrinsics.parent
            if (sequence_dir / "rgb").is_dir() and (sequence_dir / "hand_pose").is_dir():
                rows.append(audit_sequence(
                    sequence_dir, root, args.window_size, args.window_stride
                ))
    rows.sort(key=lambda row: (row["subject"], row["sequence"]))
    by_subject = {}
    for root in roots:
        selected = [row for row in rows if row["subject"] == root.name]
        by_subject[root.name] = {
            "sequences": len(selected),
            "frames": int(sum(row["frames"] for row in selected)),
            "windows": int(sum(row["windows"] for row in selected)),
            "left_valid_frames": int(sum(row["left_valid_frames"] for row in selected)),
            "right_valid_frames": int(sum(row["right_valid_frames"] for row in selected)),
        }
    report = {
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "totals": {
            "sequences": len(rows),
            "frames": int(sum(row["frames"] for row in rows)),
            "windows": int(sum(row["windows"] for row in rows)),
        },
        "frame_distribution": distribution([row["frames"] for row in rows]),
        "window_distribution": distribution([row["windows"] for row in rows]),
        "by_subject": by_subject,
        "sequences": rows,
    }
    output = Path(args.out_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "window_size", "window_stride", "totals", "frame_distribution",
        "window_distribution", "by_subject",
    )}, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
