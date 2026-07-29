#!/usr/bin/env python3
"""Export one compact, de-overlapped Pi3X geometry feature cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from export_pi3x_geometry_features import (
    load_rows,
    load_segmentation,
    patch_coverage,
    verify_pi3x_checkpoint,
    window_starts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--object-label", type=int, required=True)
    parser.add_argument("--hand-label", type=int, default=255)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--hand-topk", type=int, default=24)
    parser.add_argument("--object-topk", type=int, default=40)
    parser.add_argument("--context-topk", type=int, default=16)
    parser.add_argument("--hand-min-coverage", type=float, default=0.01)
    parser.add_argument("--object-min-coverage", type=float, default=0.01)
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def patch_average(values: np.ndarray, patch_hw: tuple[int, int]) -> np.ndarray:
    patch_h, patch_w = patch_hw
    if values.ndim == 2:
        return cv2.resize(
            values.astype(np.float32),
            (patch_w, patch_h),
            interpolation=cv2.INTER_AREA,
        )
    channels = [
        cv2.resize(
            values[..., channel].astype(np.float32),
            (patch_w, patch_h),
            interpolation=cv2.INTER_AREA,
        )
        for channel in range(values.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def select_tokens(
    features: np.ndarray,
    coverage: np.ndarray,
    confidence: np.ndarray,
    points: np.ndarray,
    count: int,
    min_coverage: float,
) -> dict[str, np.ndarray]:
    height, width, dim = features.shape
    score = coverage * (0.25 + 0.75 * confidence)
    valid_candidates = np.flatnonzero(
        (coverage.reshape(-1) >= min_coverage)
        & np.isfinite(score.reshape(-1))
    )
    order = valid_candidates[
        np.argsort(score.reshape(-1)[valid_candidates])[::-1]
    ]
    selected = order[:count]
    valid_count = len(selected)
    if valid_count < count:
        selected = np.pad(
            selected,
            (0, count - valid_count),
            constant_values=0,
        )
    ys, xs = np.unravel_index(selected, (height, width))
    valid = np.arange(count) < valid_count
    return {
        "features": features[ys, xs].reshape(count, dim),
        "indices": np.stack([ys, xs], axis=-1).astype(np.int16),
        "coverage": coverage[ys, xs].astype(np.float32),
        "confidence": confidence[ys, xs].astype(np.float32),
        "points": points[ys, xs].astype(np.float32),
        "valid": valid.astype(np.uint8),
    }


def select_context_tokens(
    features: np.ndarray,
    hand_coverage: np.ndarray,
    object_coverage: np.ndarray,
    confidence: np.ndarray,
    points: np.ndarray,
    count: int,
) -> dict[str, np.ndarray]:
    available = (hand_coverage < 0.01) & (object_coverage < 0.01)
    score = np.where(available, confidence, -np.inf)
    order = np.argsort(score.reshape(-1))[::-1]
    finite = np.isfinite(score.reshape(-1)[order])
    selected = order[finite][:count]
    valid_count = len(selected)
    if valid_count < count:
        selected = np.pad(
            selected,
            (0, count - valid_count),
            constant_values=0,
        )
    ys, xs = np.unravel_index(selected, score.shape)
    valid = np.arange(count) < valid_count
    return {
        "features": features[ys, xs],
        "indices": np.stack([ys, xs], axis=-1).astype(np.int16),
        "confidence": confidence[ys, xs].astype(np.float32),
        "points": points[ys, xs].astype(np.float32),
        "valid": valid.astype(np.uint8),
    }


def allocate(
    frame_count: int,
    count: int,
    feature_dim: int,
) -> dict[str, np.ndarray]:
    return {
        "features": np.zeros(
            (frame_count, count, feature_dim), dtype=np.float32
        ),
        "indices": np.zeros((frame_count, count, 2), dtype=np.int16),
        "coverage": np.zeros((frame_count, count), dtype=np.float32),
        "confidence": np.zeros((frame_count, count), dtype=np.float32),
        "points": np.zeros(
            (frame_count, count, 3), dtype=np.float32
        ),
        "valid": np.zeros((frame_count, count), dtype=np.uint8),
    }


def assign(target: dict[str, np.ndarray], index: int, source: dict) -> None:
    for key, value in source.items():
        target[key][index] = value


def main() -> None:
    args = parse_args()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    hand_uni_root = Path(args.hand_uni_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "pi3x_geometry_features_compact.npz"
    summary_path = out_dir / "summary.json"
    if output_path.is_file() and not args.overwrite:
        print(f"Cached: {output_path}")
        return

    if str(hand_uni_root) not in sys.path:
        sys.path.insert(0, str(hand_uni_root))
    from pi3_wilor_hand.factory import load_pi3x
    from pi3_wilor_hand.geometry import resize_intrinsics
    from pi3_wilor_hand.handsh_pipeline import Pi3XReconstructionBranch
    from pi3_wilor_hand.pi3_runner import load_images_for_pi3

    rows = load_rows(frame_map_path)
    frame_start = max(0, args.frame_start)
    frame_end = (
        len(rows) if args.frame_end < 0 else min(len(rows), args.frame_end)
    )
    if not frame_start < frame_end:
        raise ValueError(f"Invalid frame range [{frame_start}, {frame_end})")
    selected_rows = rows[frame_start:frame_end]
    frame_count = len(selected_rows)

    with np.load(hand_path, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]
    intrinsics = intrinsics.reshape(3, 3)

    verify_pi3x_checkpoint(checkpoint)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    model = load_pi3x(
        pi3_root=args.pi3_root,
        ckpt=str(checkpoint),
        device=str(device),
    )
    reconstruction = Pi3XReconstructionBranch(
        model,
        freeze_pi3=True,
        use_intrinsics=True,
    ).to(device).eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_point_decoder(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["ret_point"] = output.detach()

    hook = model.point_decoder.register_forward_hook(capture_point_decoder)
    size = min(max(1, args.window_size), frame_count)
    stride = max(1, min(args.window_stride, size))
    starts = window_starts(frame_count, size, stride)
    best_distance = np.full(frame_count, np.inf, dtype=np.float32)
    best_window_start = np.full(frame_count, -1, dtype=np.int32)
    hand_output = object_output = context_output = None
    metric_output = np.zeros(frame_count, dtype=np.float32)
    intrinsics_output = np.zeros((frame_count, 3, 3), dtype=np.float32)
    feature_grid_hw = None
    resized_wh_output = None
    original_wh_output = None

    try:
        for window_number, local_start in enumerate(starts):
            local_end = min(frame_count, local_start + size)
            window_rows = selected_rows[local_start:local_end]
            image_paths = [
                str(Path(row["image_path"]).expanduser().resolve())
                for row in window_rows
            ]
            images, resized_wh, original_wh = load_images_for_pi3(
                image_paths,
                pixel_limit=args.pixel_limit,
            )
            K_resized = resize_intrinsics(
                intrinsics,
                original_wh,
                resized_wh,
            )
            images_device = images[None].to(device)
            intrinsics_device = (
                torch.from_numpy(K_resized)
                .to(device=device, dtype=torch.float32)[None, None]
                .repeat(1, len(window_rows), 1, 1)
            )
            print(
                f"[{window_number + 1}/{len(starts)}] Pi3X compact "
                f"{frame_start + local_start}:{frame_start + local_end}",
                flush=True,
            )
            captured.clear()
            with torch.no_grad():
                outputs = reconstruction(
                    images_device,
                    intrinsics=intrinsics_device,
                )
            ret_point = captured["ret_point"]
            point_tokens = ret_point[:, int(model.patch_start_idx):]
            resized_w, resized_h = resized_wh
            patch_h = int(resized_h) // int(model.patch_size)
            patch_w = int(resized_w) // int(model.patch_size)
            geometry_features = (
                point_tokens.reshape(
                    len(window_rows),
                    patch_h,
                    patch_w,
                    point_tokens.shape[-1],
                )
                .float()
                .cpu()
                .numpy()
            )
            confidence_full = (
                torch.sigmoid(outputs["conf"][0, ..., 0])
                .float()
                .cpu()
                .numpy()
            )
            local_points_full = (
                outputs["local_points"][0].float().cpu().numpy()
            )
            metric = outputs.get("metric")
            metric_value = (
                1.0
                if metric is None
                else float(metric.float().cpu().numpy().reshape(-1)[0])
            )

            if hand_output is None:
                feature_dim = geometry_features.shape[-1]
                hand_output = allocate(
                    frame_count, args.hand_topk, feature_dim
                )
                object_output = allocate(
                    frame_count, args.object_topk, feature_dim
                )
                context_output = allocate(
                    frame_count, args.context_topk, feature_dim
                )
                feature_grid_hw = np.asarray(
                    [patch_h, patch_w], dtype=np.int32
                )
                resized_wh_output = np.asarray(
                    resized_wh, dtype=np.int32
                )
                original_wh_output = np.asarray(
                    original_wh, dtype=np.int32
                )

            center = (local_start + local_end - 1) / 2.0
            for offset, row in enumerate(window_rows):
                sequence_index = local_start + offset
                distance = abs(sequence_index - center)
                if distance >= best_distance[sequence_index]:
                    continue
                segmentation = load_segmentation(
                    Path(row["label_path"]).expanduser().resolve()
                )
                hand_coverage = patch_coverage(
                    segmentation,
                    args.hand_label,
                    resized_wh,
                    (patch_h, patch_w),
                ).astype(np.float32)
                object_coverage = patch_coverage(
                    segmentation,
                    args.object_label,
                    resized_wh,
                    (patch_h, patch_w),
                ).astype(np.float32)
                confidence_patch = patch_average(
                    confidence_full[offset],
                    (patch_h, patch_w),
                )
                points_patch = patch_average(
                    local_points_full[offset],
                    (patch_h, patch_w),
                )
                assign(
                    hand_output,
                    sequence_index,
                    select_tokens(
                        geometry_features[offset],
                        hand_coverage,
                        confidence_patch,
                        points_patch,
                        args.hand_topk,
                        args.hand_min_coverage,
                    ),
                )
                assign(
                    object_output,
                    sequence_index,
                    select_tokens(
                        geometry_features[offset],
                        object_coverage,
                        confidence_patch,
                        points_patch,
                        args.object_topk,
                        args.object_min_coverage,
                    ),
                )
                context = select_context_tokens(
                    geometry_features[offset],
                    hand_coverage,
                    object_coverage,
                    confidence_patch,
                    points_patch,
                    args.context_topk,
                )
                context["coverage"] = np.zeros(
                    args.context_topk, dtype=np.float32
                )
                assign(context_output, sequence_index, context)
                best_distance[sequence_index] = distance
                best_window_start[sequence_index] = (
                    frame_start + local_start
                )
                metric_output[sequence_index] = metric_value
                intrinsics_output[sequence_index] = K_resized
    finally:
        hook.remove()

    if np.any(best_window_start < 0):
        missing = np.flatnonzero(best_window_start < 0).tolist()
        raise RuntimeError(f"Frames not exported: {missing}")

    feature_dtype = (
        np.float16 if args.feature_dtype == "float16" else np.float32
    )
    payload: dict[str, np.ndarray] = {
        "frame_indices": np.arange(
            frame_start, frame_end, dtype=np.int32
        ),
        "selected_window_start": best_window_start,
        "selected_window_center_distance": best_distance,
        "intrinsics_resized": intrinsics_output,
        "metric": metric_output,
        "geometry_feature_layer": np.asarray(
            "point_decoder.final_output"
        ),
        "geometry_feature_grid_hw": feature_grid_hw,
        "geometry_feature_dim": np.int32(
            hand_output["features"].shape[-1]
        ),
        "geometry_feature_dtype": np.asarray(args.feature_dtype),
        "patch_size": np.int32(model.patch_size),
        "resized_wh": resized_wh_output,
        "original_wh": original_wh_output,
        "object_label": np.int32(args.object_label),
        "hand_label": np.int32(args.hand_label),
        "gt_intrinsics_conditioned": np.asarray(True),
    }
    for prefix, values in (
        ("hand", hand_output),
        ("object", object_output),
        ("context", context_output),
    ):
        for key, value in values.items():
            if key == "features":
                value = value.astype(feature_dtype)
            elif key in {"coverage", "confidence", "points"}:
                value = value.astype(np.float16)
            payload[f"{prefix}_{key}"] = value
    np.savez_compressed(output_path, **payload)

    summary = {
        "model": "Pi3X",
        "checkpoint": str(checkpoint),
        "frame_map_json": str(frame_map_path),
        "hand_npz": str(hand_path),
        "output": str(output_path),
        "compact": True,
        "deoverlap": "nearest_window_center",
        "num_frames": frame_count,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "window_size": size,
        "window_stride": stride,
        "hand_topk": args.hand_topk,
        "object_topk": args.object_topk,
        "context_topk": args.context_topk,
        "feature_dim": int(hand_output["features"].shape[-1]),
        "feature_dtype": args.feature_dtype,
        "size_bytes": output_path.stat().st_size,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
