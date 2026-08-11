#!/usr/bin/env python3
"""Refine HandFlow wrist depth with a ray-constrained temporal objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_VERSION = "handflow_wrist_depth_temporal_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-supervision-npz", required=True)
    parser.add_argument("--handflow-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--lambda-acceleration", type=float, default=0.2)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--max-correction-mm", type=float, default=0.0)
    parser.add_argument(
        "--confidence-key",
        help="Optional per-frame confidence key in the HandFlow archive.",
    )
    parser.add_argument("--confidence-power", type=float, default=8.0)
    parser.add_argument("--confidence-clamp-low", type=float, default=0.5)
    parser.add_argument("--confidence-clamp-high", type=float, default=1.5)
    parser.add_argument("--min-segment-frames", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def scalar_text(value: np.ndarray | str) -> str:
    return str(np.asarray(value).item())


def contiguous_segments(valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    return [part for part in np.split(indices, boundaries) if len(part)]


def confidence_weights(
    confidence: np.ndarray,
    valid: np.ndarray,
    power: float,
    clamp_low: float,
    clamp_high: float,
) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.ndim > 1:
        confidence = np.nanmedian(confidence.reshape(len(confidence), -1), axis=1)
    confidence = confidence.reshape(-1)
    if len(confidence) != len(valid):
        raise ValueError(
            f"Confidence length {len(confidence)} != frame count {len(valid)}"
        )
    finite = valid & np.isfinite(confidence)
    median = float(np.median(confidence[finite])) if finite.any() else 1.0
    median = max(median, 1e-8)
    normalized = np.nan_to_num(confidence / median, nan=1.0)
    return np.clip(normalized, clamp_low, clamp_high) ** power


def optimize_segment_depth(
    initial_wrist: np.ndarray,
    anchor_weights: np.ndarray,
    lambda_acceleration: float,
    max_correction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve weighted depth anchors plus zero world-space acceleration."""
    count = len(initial_wrist)
    depth = np.linalg.norm(initial_wrist, axis=-1)
    ray = initial_wrist / np.maximum(depth[:, None], 1e-8)
    if count < 3 or lambda_acceleration <= 0.0:
        return depth.copy(), ray

    anchor = np.maximum(np.asarray(anchor_weights, np.float64), 1e-8)
    rows = [np.diag(np.sqrt(anchor))]
    targets = [np.sqrt(anchor) * depth]
    scale = np.sqrt(lambda_acceleration)
    acceleration = np.zeros((3 * (count - 2), count), dtype=np.float64)
    for local in range(1, count - 1):
        row = 3 * (local - 1)
        acceleration[row : row + 3, local - 1] = ray[local - 1]
        acceleration[row : row + 3, local] = -2.0 * ray[local]
        acceleration[row : row + 3, local + 1] = ray[local + 1]
    rows.append(scale * acceleration)
    targets.append(np.zeros(len(acceleration), dtype=np.float64))
    matrix = np.concatenate(rows, axis=0)
    target = np.concatenate(targets, axis=0)
    optimized = np.linalg.lstsq(matrix, target, rcond=None)[0]
    if max_correction > 0.0:
        optimized = depth + np.clip(
            optimized - depth, -max_correction, max_correction
        )
    return optimized, ray


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "median_mm": float(np.median(values) * 1000.0) if values.size else None,
        "p90_mm": (
            float(np.percentile(values, 90) * 1000.0) if values.size else None
        ),
        "max_mm": float(np.max(values) * 1000.0) if values.size else None,
    }


def acceleration_errors(
    points: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    values = []
    for segment in contiguous_segments(valid):
        if len(segment) < 3:
            continue
        current = points[segment]
        values.append(np.linalg.norm(np.diff(current, n=2, axis=0), axis=-1))
    return np.concatenate(values) if values else np.empty(0)


def main() -> None:
    args = parse_args()
    if args.lambda_acceleration < 0.0 or args.anchor_weight <= 0.0:
        raise ValueError("Weights must be non-negative and anchor-weight positive")
    if args.min_segment_frames < 1:
        raise ValueError("min-segment-frames must be positive")

    supervision_path = Path(args.global_supervision_npz).expanduser().resolve()
    handflow_path = Path(args.handflow_npz).expanduser().resolve()
    output_path = Path(args.out_npz).expanduser().resolve()
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_path.with_suffix(".json")
    )
    for path in (output_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)

    supervision = load_npz(supervision_path)
    handflow = load_npz(handflow_path)
    required = ("pred_joints_3d", "hand_valid", "frame_ids")
    missing = [key for key in required if key not in supervision]
    if missing:
        raise KeyError(f"Global supervision lacks {missing}")
    if "verts_cam" not in handflow or "pred_valid" not in handflow:
        raise KeyError("HandFlow archive needs verts_cam and pred_valid")

    count = min(
        len(supervision["frame_ids"]),
        len(supervision["pred_joints_3d"]),
        len(handflow["verts_cam"]),
        len(handflow["pred_valid"]),
    )
    initial = np.asarray(supervision["pred_joints_3d"][:count, 0], np.float64)
    vertices = np.asarray(handflow["verts_cam"][:count], np.float32)
    valid = (
        np.asarray(supervision["hand_valid"][:count], bool)
        & np.asarray(handflow["pred_valid"][:count], bool)
        & np.isfinite(initial).all(axis=-1)
        & (np.linalg.norm(initial, axis=-1) > 1e-6)
        & np.isfinite(vertices).all(axis=(1, 2))
    )

    weights = np.full(count, args.anchor_weight, dtype=np.float64)
    if args.confidence_key:
        if args.confidence_key not in handflow:
            raise KeyError(
                f"HandFlow archive lacks confidence key {args.confidence_key!r}"
            )
        weights *= confidence_weights(
            handflow[args.confidence_key][:count],
            valid,
            args.confidence_power,
            args.confidence_clamp_low,
            args.confidence_clamp_high,
        )

    corrected = initial.copy()
    applied = np.zeros(count, dtype=bool)
    for segment in contiguous_segments(valid):
        if len(segment) < args.min_segment_frames:
            continue
        optimized_depth, ray = optimize_segment_depth(
            initial[segment],
            weights[segment],
            args.lambda_acceleration,
            args.max_correction_mm / 1000.0,
        )
        corrected[segment] = ray * optimized_depth[:, None]
        applied[segment] = True

    correction_normalized = corrected - initial
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    correction_camera = correction_normalized.copy()
    if normalized_left:
        correction_camera[:, 0] *= -1.0
    corrected_vertices = vertices.copy()
    corrected_vertices[applied] += correction_camera[applied, None].astype(
        np.float32
    )

    initial_acceleration = acceleration_errors(initial, applied)
    corrected_acceleration = acceleration_errors(corrected, applied)
    correction_size = np.linalg.norm(correction_normalized[applied], axis=-1)
    summary = {
        "method_version": METHOD_VERSION,
        "stream_id": scalar_text(
            supervision.get("stream_id", np.asarray(output_path.parent.name))
        ),
        "hand_side": scalar_text(
            supervision.get("hand_side", np.asarray("unknown"))
        ),
        "normalized_left": normalized_left,
        "frames": count,
        "valid_frames": int(valid.sum()),
        "optimized_frames": int(applied.sum()),
        "segments": len(
            [
                segment
                for segment in contiguous_segments(valid)
                if len(segment) >= args.min_segment_frames
            ]
        ),
        "lambda_acceleration": args.lambda_acceleration,
        "anchor_weight": args.anchor_weight,
        "confidence_key": args.confidence_key,
        "max_correction_mm": args.max_correction_mm,
        "correction": distribution(correction_size),
        "wrist_acceleration_before": distribution(initial_acceleration),
        "wrist_acceleration_after": distribution(corrected_acceleration),
    }

    if "gt_joints_3d" in supervision and "gt_valid" in supervision:
        target = np.asarray(supervision["gt_joints_3d"][:count, 0], np.float64)
        evaluation = (
            applied
            & np.asarray(supervision["gt_valid"][:count], bool)
            & np.isfinite(target).all(axis=-1)
        )
        initial_error = np.linalg.norm(target[evaluation] - initial[evaluation], axis=-1)
        corrected_error = np.linalg.norm(
            target[evaluation] - corrected[evaluation], axis=-1
        )
        ray = initial / np.maximum(
            np.linalg.norm(initial, axis=-1, keepdims=True), 1e-8
        )
        target_ray = np.abs(np.sum((target - initial) * ray, axis=-1))
        corrected_ray = np.abs(
            np.sum((target - corrected) * ray, axis=-1)
        )
        summary["evaluation"] = {
            "initial_translation": distribution(initial_error),
            "corrected_translation": distribution(corrected_error),
            "initial_ray_depth": distribution(target_ray[evaluation]),
            "corrected_ray_depth": distribution(corrected_ray[evaluation]),
            "degraded_fraction": (
                float(np.mean(corrected_error > initial_error))
                if evaluation.any()
                else None
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frame_ids=np.asarray(supervision["frame_ids"][:count]),
        verts_cam=corrected_vertices,
        faces=np.asarray(handflow.get("faces", np.empty((0, 3), np.int64))),
        pred_valid=valid,
        optimization_valid=applied,
        initial_wrist_normalized=initial.astype(np.float32),
        corrected_wrist_normalized=corrected.astype(np.float32),
        depth_correction_normalized=correction_normalized.astype(np.float32),
        depth_correction_camera=correction_camera.astype(np.float32),
        anchor_weight=weights.astype(np.float32),
        normalized_left=np.asarray(normalized_left),
        hand_side=np.asarray(summary["hand_side"]),
        stream_id=np.asarray(summary["stream_id"]),
        method_version=np.asarray(METHOD_VERSION),
        source_supervision=np.asarray(str(supervision_path)),
        source_handflow=np.asarray(str(handflow_path)),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
