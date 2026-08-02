#!/usr/bin/env python3
"""Audit one Pi3X ray-depth stream against DexYCB supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PALM_JOINT_IDS = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--worst-k", type=int, default=20)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.quantile(values, 0.9)),
        "max_mm": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()

    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(supervision_path, allow_pickle=False) as archive:
        supervision = {key: np.asarray(archive[key]) for key in archive.files}

    required_prediction = {
        "pi3x_depth_correction",
        "pi3x_translation_normalized",
        "pi3x_depth_predicted",
    }
    required_supervision = {
        "pred_joints_3d",
        "gt_joints_3d",
        "supervision_valid",
    }
    missing = (required_prediction - prediction.keys()) | (
        required_supervision - supervision.keys()
    )
    if missing:
        raise KeyError(f"Missing required arrays: {sorted(missing)}")

    pred = np.asarray(supervision["pred_joints_3d"], dtype=np.float32)
    gt = np.asarray(supervision["gt_joints_3d"], dtype=np.float32)
    correction = np.asarray(
        prediction["pi3x_translation_normalized"], dtype=np.float32
    )
    depth = np.asarray(
        prediction["pi3x_depth_correction"], dtype=np.float32
    )
    predicted = np.asarray(
        prediction["pi3x_depth_predicted"], dtype=bool
    )
    valid = np.asarray(supervision["supervision_valid"], dtype=bool)

    count = min(len(pred), len(gt), len(correction), len(depth), len(valid))
    pred = pred[:count]
    gt = gt[:count]
    correction = correction[:count]
    depth = depth[:count]
    predicted = predicted[:count]
    valid = valid[:count] & predicted

    camera_ray = pred[:, 0] / np.maximum(
        np.linalg.norm(pred[:, 0], axis=-1, keepdims=True), 1e-8
    )
    target_depth = np.sum((gt[:, 0] - pred[:, 0]) * camera_ray, axis=-1)
    corrected = pred + correction[:, None, :]

    wrist_before = np.linalg.norm(pred[:, 0] - gt[:, 0], axis=-1) * 1000.0
    wrist_after = (
        np.linalg.norm(corrected[:, 0] - gt[:, 0], axis=-1) * 1000.0
    )
    palm_before = np.median(
        np.linalg.norm(
            pred[:, PALM_JOINT_IDS] - gt[:, PALM_JOINT_IDS], axis=-1
        ),
        axis=-1,
    ) * 1000.0
    palm_after = np.median(
        np.linalg.norm(
            corrected[:, PALM_JOINT_IDS] - gt[:, PALM_JOINT_IDS], axis=-1
        ),
        axis=-1,
    ) * 1000.0
    target_mm = target_depth * 1000.0
    predicted_mm = depth * 1000.0
    remaining_mm = (target_depth - depth) * 1000.0

    rows = []
    for frame in range(count):
        sign_ok = bool(
            abs(target_mm[frame]) < 1e-3
            or target_mm[frame] * predicted_mm[frame] >= 0
        )
        rows.append(
            {
                "frame": frame,
                "valid": bool(valid[frame]),
                "target_depth_mm": float(target_mm[frame]),
                "predicted_depth_mm": float(predicted_mm[frame]),
                "remaining_depth_mm": float(remaining_mm[frame]),
                "sign_ok": sign_ok,
                "wrist_before_mm": float(wrist_before[frame]),
                "wrist_after_mm": float(wrist_after[frame]),
                "wrist_delta_mm": float(wrist_after[frame] - wrist_before[frame]),
                "palm_before_mm": float(palm_before[frame]),
                "palm_after_mm": float(palm_after[frame]),
                "palm_delta_mm": float(palm_after[frame] - palm_before[frame]),
            }
        )

    valid_rows = [row for row in rows if row["valid"]]
    worst = sorted(
        valid_rows, key=lambda row: row["wrist_delta_mm"], reverse=True
    )[: args.worst_k]
    best = sorted(valid_rows, key=lambda row: row["wrist_delta_mm"])[
        : args.worst_k
    ]
    valid_indices = np.flatnonzero(valid)
    report = {
        "prediction_npz": str(prediction_path),
        "supervision_npz": str(supervision_path),
        "objective": str(
            prediction.get("pi3x_depth_objective", np.asarray("not_recorded")).item()
        ),
        "checkpoint": str(
            prediction.get("pi3x_depth_checkpoint", np.asarray("not_recorded")).item()
        ),
        "num_frames": count,
        "num_valid": int(valid.sum()),
        "num_improved": int(
            np.sum(wrist_after[valid_indices] < wrist_before[valid_indices])
        ),
        "num_degraded": int(
            np.sum(wrist_after[valid_indices] > wrist_before[valid_indices])
        ),
        "num_wrong_sign": int(
            sum(not row["sign_ok"] for row in valid_rows)
        ),
        "metrics": {
            "wrist_before": quantiles(wrist_before[valid_indices]),
            "wrist_after": quantiles(wrist_after[valid_indices]),
            "palm_before": quantiles(palm_before[valid_indices]),
            "palm_after": quantiles(palm_after[valid_indices]),
            "ray_before": quantiles(np.abs(target_mm[valid_indices])),
            "ray_after": quantiles(np.abs(remaining_mm[valid_indices])),
        },
        "worst_frames": worst,
        "best_frames": best,
        "frames": rows,
    }

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    print(f"Objective: {report['objective']}")
    print(f"Checkpoint: {report['checkpoint']}")
    print(
        "Frames:", report["num_valid"],
        "improved:", report["num_improved"],
        "degraded:", report["num_degraded"],
        "wrong sign:", report["num_wrong_sign"],
    )
    for name, values in report["metrics"].items():
        print(name, values)
    print("\nWorst frames:")
    for row in worst:
        print(
            f"{row['frame']:06d}",
            f"wrist {row['wrist_before_mm']:.3f}->{row['wrist_after_mm']:.3f}",
            f"palm {row['palm_before_mm']:.3f}->{row['palm_after_mm']:.3f}",
            f"ray target/pred/remain "
            f"{row['target_depth_mm']:+.3f}/"
            f"{row['predicted_depth_mm']:+.3f}/"
            f"{row['remaining_depth_mm']:+.3f}",
            f"sign_ok={row['sign_ok']}",
        )


if __name__ == "__main__":
    main()
