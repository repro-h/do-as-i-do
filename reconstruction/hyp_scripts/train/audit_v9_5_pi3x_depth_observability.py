#!/usr/bin/env python3
"""Audit whether dense Pi3X points observe HandFlow ray-depth error."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_v9_2_pi3x_feature_trajectory_depth import finite_float
from train_v9_4_dense_joint_pi3x_noop_probe import (
    JOINT_IDS,
    bilinear_sample,
    image_uv,
    load_dense_npz,
    patch_center_points,
    patch_uv,
)
from train_v9_camera_hand_residual import load_jsonl, load_npz, scalar_text


FEATURE_NAMES = (
    "wrist_point_ray_minus_handflow",
    "joint_point_ray_minus_handflow_median",
    "joint_point_ray_minus_handflow_mean",
    "joint_point_ray_minus_handflow_std",
    "wrist_point_z_minus_handflow",
    "joint_point_z_minus_handflow_median",
    "wrist_point_ray_minus_object",
    "joint_point_ray_minus_object_median",
    "wrist_confidence",
    "joint_confidence_median",
    "wrist_hand_coverage",
    "joint_hand_coverage_median",
    "wrist_object_coverage",
    "joint_object_coverage_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--dense-train-root", required=True)
    parser.add_argument("--dense-val-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--min-object-coverage", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-2)
    return parser.parse_args()


def safe_median(value: np.ndarray, valid: np.ndarray) -> float:
    selected = value[valid & np.isfinite(value)]
    return float(np.median(selected)) if len(selected) else 0.0


def safe_mean(value: np.ndarray, valid: np.ndarray) -> float:
    selected = value[valid & np.isfinite(value)]
    return float(np.mean(selected)) if len(selected) else 0.0


def safe_std(value: np.ndarray, valid: np.ndarray) -> float:
    selected = value[valid & np.isfinite(value)]
    return float(np.std(selected)) if len(selected) else 0.0


def distribution(value: np.ndarray) -> dict:
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    return {
        "count": int(len(value)),
        "median_mm": float(np.median(value) * 1000.0) if len(value) else None,
        "p90_mm": float(np.percentile(value, 90) * 1000.0)
        if len(value) else None,
    }


def rank(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    result = np.empty(len(value), dtype=np.float64)
    result[order] = np.arange(len(value), dtype=np.float64)
    return result


def correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return 0.0, 0.0
    a, b = a[valid], b[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0, 0.0
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(np.corrcoef(rank(a), rank(b))[0, 1])
    return pearson, spearman


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    before = np.abs(target)
    after = np.abs(target - prediction)
    increase = after - before
    nontrivial = np.abs(target) >= 0.005
    direction = (
        float(np.mean(np.sign(target[nontrivial]) == np.sign(prediction[nontrivial])))
        if nontrivial.any() else None
    )
    return {
        "target": distribution(before),
        "error_after": distribution(after),
        "degraded_fraction": float(np.mean(increase > 1e-6)),
        "worse_2mm_fraction": float(np.mean(increase > 0.002)),
        "worse_5mm_fraction": float(np.mean(increase > 0.005)),
        "direction_accuracy_ge5mm": direction,
    }


def dense_path(row: dict, root: Path, stream_id: str) -> Path:
    start, end = int(row["start"]), int(row["end"])
    return Path(row.get(
        "dense_pi3x_npz",
        root / stream_id / "windows" / f"window_{start:06d}_{end:06d}.npz",
    )).expanduser().resolve()


def collect(
    windows: Path,
    global_root: Path,
    dense_root: Path,
    min_confidence: float,
    min_object_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = load_jsonl(windows)
    records: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"features": [], "target": [], "side": []}
    )
    global_cache: dict[str, dict[str, np.ndarray]] = {}

    for row_index, row in enumerate(rows, 1):
        stream_id = str(row["stream_id"])
        start, end = int(row["start"]), int(row["end"])
        se3 = load_npz(str(Path(row["supervision_npz"]).resolve()))
        global_path = Path(
            scalar_text(se3["source_global_supervision"])
        ).expanduser().resolve()
        if not global_path.is_file():
            global_path = global_root / f"{stream_id}.npz"
        cache_key = str(global_path)
        if cache_key not in global_cache:
            global_cache[cache_key] = load_npz(cache_key)
        glob = global_cache[cache_key]

        pred = np.asarray(glob["pred_joints_3d"], dtype=np.float32)[start:end].copy()
        target = np.asarray(glob["gt_joints_3d"], dtype=np.float32)[start:end].copy()
        if bool(np.asarray(glob.get("normalized_left", False)).item()):
            pred[..., 0] *= -1.0
            target[..., 0] *= -1.0
        side = 0 if scalar_text(glob["hand_side"]) == "left" else 1
        valid_frame = (
            np.asarray(glob["hand_valid"], dtype=bool)[start:end]
            & np.asarray(glob["gt_valid"], dtype=bool)[start:end]
            & np.asarray(glob["supervision_valid"], dtype=bool)[start:end]
            & np.isfinite(pred[:, 0]).all(axis=-1)
            & np.isfinite(target[:, 0]).all(axis=-1)
        )
        pred = finite_float(pred[:, JOINT_IDS])

        dense = load_dense_npz(dense_path(row, dense_root, stream_id))
        frames = np.asarray(dense["frame_indices"], dtype=np.int64)
        expected = np.arange(start, end, dtype=np.int64)
        if not np.array_equal(frames, expected):
            raise ValueError(f"Dense frames do not match {stream_id} [{start}, {end})")
        intrinsics = np.asarray(dense["intrinsics_resized"], dtype=np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.broadcast_to(intrinsics[None], (end - start, 3, 3))
        image_wh = np.asarray(dense["resized_wh"], dtype=np.float32).reshape(2)
        z = pred[..., 2]
        safe_z = np.maximum(z, 1e-6)
        pixels = np.stack((
            intrinsics[:, None, 0, 0] * pred[..., 0] / safe_z
            + intrinsics[:, None, 0, 2],
            intrinsics[:, None, 1, 1] * pred[..., 1] / safe_z
            + intrinsics[:, None, 1, 2],
        ), axis=-1)
        uv = image_uv(pixels, image_wh)
        projected = (
            np.isfinite(uv).all(axis=-1) & (z > 1e-5)
            & (uv[..., 0] >= 0) & (uv[..., 0] <= 1)
            & (uv[..., 1] >= 0) & (uv[..., 1] <= 1)
        )
        uv = finite_float(uv)
        patch_hw = tuple(int(x) for x in np.asarray(
            dense["geometry_feature_grid_hw"]
        ).reshape(2))
        feature_uv = patch_uv(pixels, image_wh, patch_hw)
        points = np.asarray(bilinear_sample(dense["local_points"], uv), dtype=np.float32)
        confidence = np.asarray(bilinear_sample(dense["confidence"], uv), dtype=np.float32)
        hand_coverage = np.asarray(
            bilinear_sample(dense["hand_patch_coverage"], feature_uv), dtype=np.float32
        )
        object_coverage = np.asarray(
            bilinear_sample(dense["object_patch_coverage"], feature_uv), dtype=np.float32
        )
        joint_valid = projected & np.isfinite(points).all(axis=-1) & (
            confidence >= min_confidence
        )

        object_points = patch_center_points(
            np.asarray(dense["local_points"], dtype=np.float32), patch_hw, image_wh
        )
        object_mask = np.asarray(
            dense["object_patch_coverage"], dtype=np.float32
        ) >= min_object_coverage
        object_center = np.zeros((end - start, 3), dtype=np.float32)
        object_valid = np.zeros(end - start, dtype=bool)
        for frame in range(end - start):
            values = object_points[frame][object_mask[frame]]
            values = values[np.isfinite(values).all(axis=-1)]
            if len(values) >= 3:
                object_center[frame] = np.median(values, axis=0)
                object_valid[frame] = True

        joint_ray = pred / np.maximum(
            np.linalg.norm(pred, axis=-1, keepdims=True), 1e-6
        )
        point_on_ray = np.sum(points * joint_ray, axis=-1)
        handflow_on_ray = np.linalg.norm(pred, axis=-1)
        discrepancy = point_on_ray - handflow_on_ray
        z_discrepancy = points[..., 2] - pred[..., 2]
        object_on_ray = np.sum(
            object_center[:, None] * joint_ray, axis=-1
        )
        point_minus_object = point_on_ray - object_on_ray

        wrist_initial = pred[:, 0]
        wrist_target = target[:, 0]
        wrist_ray = wrist_initial / np.maximum(
            np.linalg.norm(wrist_initial, axis=-1, keepdims=True), 1e-6
        )
        target_ray = np.sum(
            (wrist_target - wrist_initial) * wrist_ray, axis=-1
        )
        for local_frame, frame_index in enumerate(expected):
            valid = joint_valid[local_frame]
            if not valid_frame[local_frame] or not valid[0] or valid.sum() < 3:
                continue
            feature = np.asarray((
                discrepancy[local_frame, 0],
                safe_median(discrepancy[local_frame], valid),
                safe_mean(discrepancy[local_frame], valid),
                safe_std(discrepancy[local_frame], valid),
                z_discrepancy[local_frame, 0],
                safe_median(z_discrepancy[local_frame], valid),
                point_minus_object[local_frame, 0]
                if object_valid[local_frame] else 0.0,
                safe_median(point_minus_object[local_frame], valid)
                if object_valid[local_frame] else 0.0,
                confidence[local_frame, 0],
                safe_median(confidence[local_frame], valid),
                hand_coverage[local_frame, 0],
                safe_median(hand_coverage[local_frame], valid),
                object_coverage[local_frame, 0],
                safe_median(object_coverage[local_frame], valid),
            ), dtype=np.float64)
            record = records[(stream_id, int(frame_index))]
            record["features"].append(feature)
            record["target"].append(float(target_ray[local_frame]))
            record["side"].append(side)
        if row_index % 500 == 0 or row_index == len(rows):
            print(f"[{row_index}/{len(rows)}] unique_frames={len(records)}")

    features, targets, sides = [], [], []
    for record in records.values():
        features.append(np.median(np.stack(record["features"]), axis=0))
        targets.append(float(np.median(record["target"])))
        sides.append(int(record["side"][0]))
    return np.stack(features), np.asarray(targets), np.asarray(sides)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (x - mean) / scale
    design = np.concatenate((np.ones((len(x), 1)), normalized), axis=1)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weight = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "weight": weight}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    normalized = (x - model["mean"]) / model["scale"]
    design = np.concatenate((np.ones((len(x), 1)), normalized), axis=1)
    return design @ model["weight"]


def main() -> None:
    args = parse_args()
    train_x, train_y, train_side = collect(
        Path(args.train_windows), Path(args.global_train_root),
        Path(args.dense_train_root), args.min_confidence,
        args.min_object_coverage,
    )
    val_x, val_y, val_side = collect(
        Path(args.val_windows), Path(args.global_val_root),
        Path(args.dense_val_root), args.min_confidence,
        args.min_object_coverage,
    )
    model = fit_ridge(train_x, train_y, args.ridge)
    train_prediction = predict(model, train_x)
    val_prediction = predict(model, val_x)

    feature_audit = {}
    for index, name in enumerate(FEATURE_NAMES):
        train_corr = correlation(train_x[:, index], train_y)
        val_corr = correlation(val_x[:, index], val_y)
        feature_audit[name] = {
            "train_pearson": train_corr[0],
            "train_spearman": train_corr[1],
            "val_pearson": val_corr[0],
            "val_spearman": val_corr[1],
        }
    output = {
        "train_unique_frames": int(len(train_y)),
        "val_unique_frames": int(len(val_y)),
        "features": feature_audit,
        "ridge": {
            "train": metrics(train_y, train_prediction),
            "val": metrics(val_y, val_prediction),
            "val_by_side": {
                name: metrics(
                    val_y[val_side == value],
                    val_prediction[val_side == value],
                )
                for name, value in (("left", 0), ("right", 1))
            },
            "weights": {
                name: float(weight)
                for name, weight in zip(FEATURE_NAMES, model["weight"][1:])
            },
            "bias": float(model["weight"][0]),
        },
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
