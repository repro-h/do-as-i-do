#!/usr/bin/env python3
"""Aggregate Pi3X ray-depth metrics by hand side."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--worst-k", type=int, default=20)
    return parser.parse_args()


def stats(values: list[np.ndarray]) -> dict:
    if not values:
        return {"count": 0}
    array = np.concatenate(values).astype(np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median_mm": float(np.median(array)),
        "p90_mm": float(np.quantile(array, 0.9)),
        "max_mm": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    supervision_root = Path(args.supervision_root).expanduser().resolve()
    grouped = defaultdict(
        lambda: {
            "before": [],
            "after": [],
            "ray_before": [],
            "ray_after": [],
            "improved": 0,
            "degraded": 0,
            "wrong_sign": 0,
            "valid": 0,
            "streams": 0,
        }
    )
    stream_rows = []

    for supervision_path in sorted(supervision_root.glob("*.npz")):
        stream_id = supervision_path.stem
        prediction_path = (
            prediction_root
            / stream_id
            / "handflow_camera_result_pi3x_depth_refined.npz"
        )
        if not prediction_path.is_file():
            continue
        with np.load(supervision_path, allow_pickle=False) as archive:
            pred = np.asarray(archive["pred_joints_3d"], dtype=np.float32)
            gt = np.asarray(archive["gt_joints_3d"], dtype=np.float32)
            valid = np.asarray(archive["supervision_valid"], dtype=bool)
            side = str(np.asarray(archive["hand_side"]).item())
            normalized_left = bool(
                np.asarray(archive["normalized_left"]).item()
            )
        with np.load(prediction_path, allow_pickle=False) as archive:
            depth = np.asarray(
                archive["pi3x_depth_correction"], dtype=np.float32
            )
            predicted = np.asarray(
                archive["pi3x_depth_predicted"], dtype=bool
            )

        count = min(len(pred), len(gt), len(valid), len(depth), len(predicted))
        pred, gt = pred[:count], gt[:count]
        depth = depth[:count]
        valid = valid[:count] & predicted[:count]
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        ray = pred[:, 0] / np.maximum(
            np.linalg.norm(pred[:, 0], axis=-1, keepdims=True), 1e-8
        )
        target = np.sum((gt[:, 0] - pred[:, 0]) * ray, axis=-1)
        translation = depth[:, None] * ray
        before = np.linalg.norm(pred[:, 0] - gt[:, 0], axis=-1) * 1000.0
        after = (
            np.linalg.norm(pred[:, 0] + translation - gt[:, 0], axis=-1)
            * 1000.0
        )
        ray_before = np.abs(target) * 1000.0
        ray_after = np.abs(target - depth) * 1000.0
        wrong_sign = (np.abs(target) > 1e-6) & (target * depth < 0.0)

        group = grouped[side]
        group["before"].append(before[indices])
        group["after"].append(after[indices])
        group["ray_before"].append(ray_before[indices])
        group["ray_after"].append(ray_after[indices])
        group["improved"] += int(np.sum(after[indices] < before[indices]))
        group["degraded"] += int(np.sum(after[indices] > before[indices]))
        group["wrong_sign"] += int(np.sum(wrong_sign[indices]))
        group["valid"] += len(indices)
        group["streams"] += 1
        stream_rows.append(
            {
                "stream_id": stream_id,
                "hand_side": side,
                "normalized_left": normalized_left,
                "valid": int(len(indices)),
                "before_median_mm": float(np.median(before[indices])),
                "after_median_mm": float(np.median(after[indices])),
                "median_delta_mm": float(
                    np.median(after[indices]) - np.median(before[indices])
                ),
                "degraded_fraction": float(
                    np.mean(after[indices] > before[indices])
                ),
                "wrong_sign_fraction": float(np.mean(wrong_sign[indices])),
            }
        )

    sides = {}
    for side, group in grouped.items():
        valid = max(group["valid"], 1)
        sides[side] = {
            "num_streams": group["streams"],
            "num_valid_frames": group["valid"],
            "wrist_before": stats(group["before"]),
            "wrist_after": stats(group["after"]),
            "ray_before": stats(group["ray_before"]),
            "ray_after": stats(group["ray_after"]),
            "improved_fraction": group["improved"] / valid,
            "degraded_fraction": group["degraded"] / valid,
            "wrong_sign_fraction": group["wrong_sign"] / valid,
        }
    worst = sorted(
        stream_rows,
        key=lambda row: (row["median_delta_mm"], row["degraded_fraction"]),
        reverse=True,
    )[: args.worst_k]
    report = {"sides": sides, "worst_streams": worst, "streams": stream_rows}

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")
    for side in sorted(sides):
        print(f"\n===== {side} =====")
        print(json.dumps(sides[side], indent=2))
    print("\n===== worst streams =====")
    for row in worst:
        print(
            row["hand_side"], row["stream_id"],
            f"median {row['before_median_mm']:.3f}"
            f"->{row['after_median_mm']:.3f} mm",
            f"degraded={row['degraded_fraction']:.3f}",
            f"wrong_sign={row['wrong_sign_fraction']:.3f}",
        )


if __name__ == "__main__":
    main()
