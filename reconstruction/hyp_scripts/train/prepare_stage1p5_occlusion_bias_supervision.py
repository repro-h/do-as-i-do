#!/usr/bin/env python3
"""Prepare one stream of Stage1.5 occlusion-bias supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-hand-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--pi3x-cache", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--carry-enter", type=float, default=0.8)
    parser.add_argument("--carry-exit", type=float, default=0.5)
    parser.add_argument("--occlusion-enter", type=float, default=0.25)
    parser.add_argument("--occlusion-exit", type=float, default=0.15)
    parser.add_argument("--mask-dilation-px", type=int, default=3)
    parser.add_argument("--min-core-frames", type=int, default=2)
    parser.add_argument("--gate-threshold-mm", type=float, default=5.0)
    parser.add_argument("--max-bias-mm", type=float, default=25.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def window_starts(count: int, size: int, stride: int) -> list[int]:
    if count <= size:
        return [0]
    starts = list(range(0, count - size + 1, stride))
    tail = count - size
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def carry_segments(
    probability: np.ndarray,
    valid: np.ndarray,
    enter: float,
    exit_threshold: float,
    min_length: int,
) -> list[tuple[int, int]]:
    segments = []
    start = None
    for index, value in enumerate(probability):
        if start is None:
            if valid[index] and value >= enter:
                start = index
            continue
        if not valid[index] or value < exit_threshold:
            if index - start >= min_length:
                segments.append((start, index - 1))
            start = None
    if start is not None and len(probability) - start >= min_length:
        segments.append((start, len(probability) - 1))
    return segments


def masked_token_mean(
    value: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    weight = valid.astype(np.float32)
    return (value * weight).sum(axis=1) / np.maximum(weight.sum(axis=1), 1.0)


def load_segmentation(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as raw:
        for key in ("seg", "segmentation", "mask", "label"):
            if key in raw.files:
                value = np.asarray(raw[key])
                if value.ndim == 2:
                    return value
    raise KeyError(f"No segmentation array in {path}")


def render_hand_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    z = vertices[:, 2]
    projected = np.empty((len(vertices), 2), dtype=np.float64)
    safe_z = np.maximum(z, 1e-6)
    projected[:, 0] = (
        intrinsics[0, 0] * vertices[:, 0] / safe_z
        + intrinsics[0, 2]
    )
    projected[:, 1] = (
        intrinsics[1, 1] * vertices[:, 1] / safe_z
        + intrinsics[1, 2]
    )
    for face in faces:
        if np.any(z[face] <= 1e-6):
            continue
        polygon = np.rint(projected[face]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 1].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool)


def segments_from_mask(
    mask: np.ndarray,
    min_length: int,
) -> list[tuple[int, int]]:
    segments = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if not value and start is not None:
            if index - start >= min_length:
                segments.append((start, index - 1))
            start = None
    if start is not None and len(mask) - start >= min_length:
        segments.append((start, len(mask) - 1))
    return segments


def main() -> None:
    args = parse_args()
    hand_path = Path(args.v7_hand_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()
    pi3x_path = Path(args.pi3x_cache).expanduser().resolve()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "stage1p5_occlusion_bias_supervision.npz"
    audit_path = out_dir / "audit.json"
    if output_path.is_file() and not args.overwrite:
        print(f"Cached: {output_path}")
        return

    with np.load(hand_path, allow_pickle=False) as raw:
        hand = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(supervision_path, allow_pickle=False) as raw:
        supervision = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(pi3x_path, allow_pickle=False) as raw:
        pi3x = {key: np.asarray(raw[key]) for key in raw.files}
    frame_map = json.loads(frame_map_path.read_text(encoding="utf-8"))
    frame_rows = frame_map["frames"]

    pred_wrist = np.asarray(
        supervision["pred_joints_3d"][:, 0], dtype=np.float32
    )
    gt_wrist = np.asarray(
        supervision["gt_joints_3d"][:, 0], dtype=np.float32
    )
    v7_depth = np.asarray(
        hand["pi3x_depth_correction"], dtype=np.float32
    )
    geometry = np.asarray(
        hand["pi3x_geometry_depth_correction"], dtype=np.float32
    )
    motion = np.asarray(
        hand["pi3x_motion_depth_residual"], dtype=np.float32
    )
    carry = np.asarray(
        hand["pi3x_motion_anomaly_probability"], dtype=np.float32
    )
    count = min(
        len(pred_wrist),
        len(gt_wrist),
        len(v7_depth),
        len(pi3x["hand_valid"]),
        len(frame_rows),
    )
    pred_wrist = pred_wrist[:count]
    gt_wrist = gt_wrist[:count]
    v7_depth = v7_depth[:count]
    geometry = geometry[:count]
    motion = motion[:count]
    carry = carry[:count]
    valid = (
        np.asarray(supervision["supervision_valid"][:count]).astype(bool)
        & np.asarray(hand["pi3x_depth_predicted"][:count]).astype(bool)
    )
    ray = pred_wrist / np.maximum(
        np.linalg.norm(pred_wrist, axis=-1, keepdims=True), 1e-8
    )
    target_depth = np.sum((gt_wrist - pred_wrist) * ray, axis=-1)
    remaining_depth = target_depth - v7_depth

    hand_valid = np.asarray(pi3x["hand_valid"][:count]).astype(bool)
    object_valid = np.asarray(pi3x["object_valid"][:count]).astype(bool)
    hand_count = hand_valid.sum(axis=1).astype(np.float32)
    object_count = object_valid.sum(axis=1).astype(np.float32)
    hand_coverage = masked_token_mean(
        np.asarray(pi3x["hand_coverage"][:count], dtype=np.float32),
        hand_valid,
    )
    object_coverage = masked_token_mean(
        np.asarray(pi3x["object_coverage"][:count], dtype=np.float32),
        object_valid,
    )
    hand_confidence = masked_token_mean(
        np.asarray(pi3x["hand_confidence"][:count], dtype=np.float32),
        hand_valid,
    )
    object_confidence = masked_token_mean(
        np.asarray(pi3x["object_confidence"][:count], dtype=np.float32),
        object_valid,
    )
    hand_vertices = np.asarray(hand["verts_cam"][:count], dtype=np.float32)
    hand_faces = np.asarray(hand["faces"], dtype=np.int32)
    intrinsics = np.asarray(hand["intrinsics"], dtype=np.float64)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]
    hand_label = int(np.asarray(pi3x["hand_label"]).item())
    object_label = int(np.asarray(pi3x["object_label"]).item())
    dilation = max(0, args.mask_dilation_px)
    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation + 1, 2 * dilation + 1),
        )
        if dilation > 0
        else None
    )
    rendered_hand_pixels = np.zeros(count, dtype=np.float32)
    observed_hand_pixels = np.zeros(count, dtype=np.float32)
    visible_hand_fraction = np.zeros(count, dtype=np.float32)
    object_occlusion_fraction = np.zeros(count, dtype=np.float32)
    for frame in range(count):
        segmentation = load_segmentation(
            Path(frame_rows[frame]["label_path"]).expanduser().resolve()
        )
        rendered = render_hand_silhouette(
            hand_vertices[frame],
            hand_faces,
            intrinsics,
            *segmentation.shape,
        )
        observed_hand = segmentation == hand_label
        observed_object = segmentation == object_label
        if kernel is not None:
            observed_hand = cv2.dilate(
                observed_hand.astype(np.uint8), kernel
            ).astype(bool)
            observed_object = cv2.dilate(
                observed_object.astype(np.uint8), kernel
            ).astype(bool)
        rendered_count = int(rendered.sum())
        rendered_hand_pixels[frame] = rendered_count
        observed_hand_pixels[frame] = int(
            (rendered & observed_hand).sum()
        )
        if rendered_count:
            visible_hand_fraction[frame] = (
                observed_hand_pixels[frame] / rendered_count
            )
            object_occlusion_fraction[frame] = float(
                (rendered & observed_object).sum() / rendered_count
            )
    frame_features = np.stack(
        [
            v7_depth,
            geometry,
            motion,
            carry,
            hand_count / max(hand_valid.shape[1], 1),
            object_count / max(object_valid.shape[1], 1),
            hand_coverage,
            object_coverage,
            hand_confidence,
            object_confidence,
            visible_hand_fraction,
            object_occlusion_fraction,
            rendered_hand_pixels / 10000.0,
            observed_hand_pixels / 10000.0,
        ],
        axis=-1,
    ).astype(np.float32)
    feature_names = np.asarray(
        [
            "v7_depth_m",
            "v7_geometry_m",
            "v7_motion_m",
            "v7_carry_probability",
            "hand_token_fraction",
            "object_token_fraction",
            "hand_coverage_mean",
            "object_coverage_mean",
            "hand_confidence_mean",
            "object_confidence_mean",
            "visible_hand_fraction",
            "object_occlusion_fraction",
            "rendered_hand_pixels_10k",
            "observed_hand_pixels_10k",
        ]
    )

    carry_only_segments = carry_segments(
        carry,
        valid,
        args.carry_enter,
        args.carry_exit,
        args.min_core_frames,
    )
    occlusion_only_segments = carry_segments(
        object_occlusion_fraction,
        valid,
        args.occlusion_enter,
        args.occlusion_exit,
        args.min_core_frames,
    )
    core_mask = np.zeros(count, dtype=bool)
    carry_mask = np.zeros(count, dtype=bool)
    occlusion_mask = np.zeros(count, dtype=bool)
    for start, end in carry_only_segments:
        carry_mask[start : end + 1] = True
    for start, end in occlusion_only_segments:
        occlusion_mask[start : end + 1] = True
    core_mask = (carry_mask | occlusion_mask) & valid
    segments = segments_from_mask(core_mask, args.min_core_frames)
    core_mask[:] = False
    for start, end in segments:
        core_mask[start : end + 1] = True

    size = min(max(1, args.window_size), count)
    stride = max(1, args.window_stride)
    rows = []
    masks = []
    for start in window_starts(count, size, stride):
        end = min(count, start + size)
        local_core = core_mask[start:end] & valid[start:end]
        local_remaining = remaining_depth[start:end]
        if int(local_core.sum()) >= args.min_core_frames:
            raw_bias = float(np.median(local_remaining[local_core]))
            bias = float(
                np.clip(
                    raw_bias,
                    -args.max_bias_mm / 1000.0,
                    args.max_bias_mm / 1000.0,
                )
            )
            gate = float(abs(raw_bias) >= args.gate_threshold_mm / 1000.0)
        else:
            raw_bias = 0.0
            bias = 0.0
            gate = 0.0
        local_mask = np.zeros(size, dtype=bool)
        local_mask[: end - start] = local_core
        masks.append(local_mask)
        rows.append(
            {
                "start": start,
                "end": end,
                "core_frames": (np.flatnonzero(local_core) + start).tolist(),
                "gate_target": gate,
                "shared_bias_target_m": bias,
                "raw_shared_bias_target_m": raw_bias,
            }
        )

    stream_id = str(np.asarray(supervision["stream_id"]).item())
    starts = np.asarray([row["start"] for row in rows], dtype=np.int32)
    ends = np.asarray([row["end"] for row in rows], dtype=np.int32)
    gate_target = np.asarray(
        [row["gate_target"] for row in rows], dtype=np.float32
    )
    bias_target = np.asarray(
        [row["shared_bias_target_m"] for row in rows], dtype=np.float32
    )
    raw_bias_target = np.asarray(
        [row["raw_shared_bias_target_m"] for row in rows], dtype=np.float32
    )
    np.savez_compressed(
        output_path,
        stream_id=np.asarray(stream_id),
        frame_features=frame_features,
        frame_feature_names=feature_names,
        frame_valid=valid,
        remaining_depth_target=remaining_depth.astype(np.float32),
        carry_core_mask=core_mask,
        combined_core_mask=core_mask,
        segment_start=np.asarray([row[0] for row in segments], dtype=np.int32),
        segment_end=np.asarray([row[1] for row in segments], dtype=np.int32),
        window_start=starts,
        window_end=ends,
        window_core_mask=np.stack(masks),
        gate_target=gate_target,
        shared_bias_target=bias_target,
        raw_shared_bias_target=raw_bias_target,
        v7_hand_npz=np.asarray(str(hand_path)),
        supervision_npz=np.asarray(str(supervision_path)),
        pi3x_cache=np.asarray(str(pi3x_path)),
        frame_map_json=np.asarray(str(frame_map_path)),
        carry_only_mask=carry_mask,
        occlusion_only_mask=occlusion_mask,
        visible_hand_fraction=visible_hand_fraction,
        object_occlusion_fraction=object_occlusion_fraction,
    )
    audit = {
        "stream_id": stream_id,
        "v7_hand_npz": str(hand_path),
        "supervision_npz": str(supervision_path),
        "pi3x_cache": str(pi3x_path),
        "frame_map_json": str(frame_map_path),
        "settings": vars(args),
        "num_frames": count,
        "segments": [
            {
                "frames": [start, end],
                "carry": {
                    "min": float(carry[start : end + 1].min()),
                    "median": float(np.median(carry[start : end + 1])),
                    "max": float(carry[start : end + 1].max()),
                },
                "visibility": {
                    "visible_hand_fraction_median": float(
                        np.median(visible_hand_fraction[start : end + 1])
                    ),
                    "object_occlusion_fraction_median": float(
                        np.median(
                            object_occlusion_fraction[start : end + 1]
                        )
                    ),
                },
                "remaining_depth_mm": {
                    "median": float(
                        np.median(remaining_depth[start : end + 1]) * 1000.0
                    ),
                    "min": float(
                        remaining_depth[start : end + 1].min() * 1000.0
                    ),
                    "max": float(
                        remaining_depth[start : end + 1].max() * 1000.0
                    ),
                },
            }
            for start, end in segments
        ],
        "carry_only_segments": [
            [start, end] for start, end in carry_only_segments
        ],
        "occlusion_only_segments": [
            [start, end] for start, end in occlusion_only_segments
        ],
        "frames": [
            {
                "frame": frame,
                "valid": bool(valid[frame]),
                "core": bool(core_mask[frame]),
                "carry_core": bool(carry_mask[frame]),
                "occlusion_core": bool(occlusion_mask[frame]),
                "carry_probability": float(carry[frame]),
                "visible_hand_fraction": float(
                    visible_hand_fraction[frame]
                ),
                "object_occlusion_fraction": float(
                    object_occlusion_fraction[frame]
                ),
                "rendered_hand_pixels": int(
                    rendered_hand_pixels[frame]
                ),
                "observed_hand_pixels": int(
                    observed_hand_pixels[frame]
                ),
                "remaining_depth_mm": float(
                    remaining_depth[frame] * 1000.0
                ),
            }
            for frame in range(count)
        ],
        "num_windows": len(rows),
        "num_positive_windows": int(gate_target.sum()),
        "positive_windows": [
            {
                **row,
                "shared_bias_target_mm":
                    row["shared_bias_target_m"] * 1000.0,
                "raw_shared_bias_target_mm":
                    row["raw_shared_bias_target_m"] * 1000.0,
            }
            for row in rows
            if row["gate_target"] > 0
        ],
    }
    audit_path.write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
