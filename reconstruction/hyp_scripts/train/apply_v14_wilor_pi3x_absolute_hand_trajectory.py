#!/usr/bin/env python3
"""Apply a V14 checkpoint and stitch one stream's overlapping windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_v10_pi3x_hand_neighborhood_depth import disable_mha_fastpath
from train_v14_wilor_pi3x_absolute_hand_trajectory import (
    MODEL_VERSION,
    DenseTrajectoryDataset,
    Pi3XAbsoluteTrajectoryModel,
    compose_translation,
    model_args_from_checkpoint,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    disable_mha_fastpath()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Expected {MODEL_VERSION}, got {checkpoint.get('model_version')}"
        )

    data = DenseTrajectoryDataset(
        Path(args.windows),
        Path(args.global_root),
        Path(args.dense_root),
        Path(args.query_root),
        query_dropout=0.0,
    )
    data.rows = [
        row for row in data.rows
        if str(row["stream_id"]) == args.stream_id
    ]
    if not data.rows:
        raise RuntimeError(f"No windows for {args.stream_id}")
    data.stream_indices = {args.stream_id: 0}
    sample = data[0]

    cli = argparse.Namespace(feature_mode="normal")
    model_args = model_args_from_checkpoint(checkpoint, cli)
    model = Pi3XAbsoluteTrajectoryModel(
        int(checkpoint["point_feature_dim"]),
        int(checkpoint["metric_feature_dim"]),
        int(checkpoint["num_joints"]),
        model_args,
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()

    query_path = (
        Path(args.query_root) / args.stream_id / "wilor_query_cache.npz"
    ).resolve()
    with np.load(query_path, allow_pickle=False) as query:
        frame_ids = np.asarray(query["frame_ids"])
        cache_mirrored = bool(np.asarray(
            query["canonical_right_horizontal_mirror"]
        ).item())
        query_model_valid = np.asarray(query["model_valid"], dtype=bool)
    frame_count = len(frame_ids)
    translation_sum = np.zeros((frame_count, 3), dtype=np.float64)
    pixel_sum = np.zeros((frame_count, 2), dtype=np.float64)
    prediction_count = np.zeros(frame_count, dtype=np.int32)

    with torch.inference_mode():
        for index in range(len(data)):
            item = data[index]
            batch = {
                key: value[None].to(device)
                for key, value in item.items()
            }
            depth, image_offset = model(batch)
            translation, pixels = compose_translation(
                depth, image_offset, batch
            )
            start = int(item["start"])
            end = int(item["end"])
            translation_sum[start:end] += translation[0].cpu().numpy()
            pixel_sum[start:end] += pixels[0].cpu().numpy()
            prediction_count[start:end] += 1

    valid = prediction_count > 0
    canonical = np.full((frame_count, 3), np.nan, dtype=np.float32)
    root_pixels = np.full((frame_count, 2), np.nan, dtype=np.float32)
    canonical[valid] = (
        translation_sum[valid] / prediction_count[valid, None]
    ).astype(np.float32)
    root_pixels[valid] = (
        pixel_sum[valid] / prediction_count[valid, None]
    ).astype(np.float32)
    original = canonical.copy()
    if cache_mirrored:
        original[:, 0] *= -1.0

    output = Path(args.out_npz).expanduser().resolve()
    write_npz(output, {
        "frame_ids": frame_ids,
        "predicted_wrist_camera": original,
        "predicted_wrist_camera_canonical_right": canonical,
        "predicted_root_pixels_canonical_right": root_pixels,
        "prediction_count": prediction_count,
        "prediction_valid": valid & query_model_valid,
        "stream_id": np.asarray(args.stream_id),
        "cache_mirrored": np.asarray(cache_mirrored),
        "checkpoint": np.asarray(str(checkpoint_path)),
        "checkpoint_epoch": np.asarray(int(checkpoint["epoch"])),
        "model_version": np.asarray(MODEL_VERSION),
        "query_cache": np.asarray(str(query_path)),
    })
    print({
        "output": str(output),
        "stream_id": args.stream_id,
        "windows": len(data),
        "frames": frame_count,
        "valid": int((valid & query_model_valid).sum()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "cache_mirrored": cache_mirrored,
    })


if __name__ == "__main__":
    main()
