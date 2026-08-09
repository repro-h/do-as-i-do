#!/usr/bin/env python3
"""Apply V9.2 and audit blended unique validation frames."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apply_v9_1_camera_ray_depth_residual import (
    IndexedDataset,
    add_metrics,
    blend_weights,
    finalize_group,
    group_metrics,
    load_npz,
)
from train_v9_2_pi3x_feature_trajectory_depth import (
    FeatureTrajectoryDataset,
    FeatureTrajectoryDepthModel,
)
from train_v9_camera_hand_residual import scalar_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--pi3x-root", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows_path = Path(args.windows).expanduser().resolve()
    global_root = Path(args.global_root).expanduser().resolve()
    pi3x_root = Path(args.pi3x_root).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = argparse.Namespace(**checkpoint["args"])
    base_dataset = FeatureTrajectoryDataset(
        windows_path, global_root, pi3x_root
    )
    dataset = IndexedDataset(base_dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = FeatureTrajectoryDepthModel(
        int(checkpoint["local_hand_dim"]),
        int(checkpoint["pi3x_feature_dim"]),
        int(checkpoint["pi3x_metadata_dim"]),
        config,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    stream_info: dict[str, dict] = {}
    for row in base_dataset.rows:
        stream_id = str(row["stream_id"])
        if stream_id in stream_info:
            continue
        se3 = load_npz(Path(row["supervision_npz"]).expanduser().resolve())
        global_path = Path(
            scalar_text(se3["source_global_supervision"])
        ).expanduser().resolve()
        if not global_path.is_file():
            global_path = global_root / f"{stream_id}.npz"
        glob = load_npz(global_path)
        length = len(glob["frame_ids"])
        stream_info[stream_id] = {
            "global_path": global_path,
            "sum": np.zeros(length, dtype=np.float64),
            "weight": np.zeros(length, dtype=np.float64),
        }

    with torch.no_grad():
        for batch in tqdm(loader, desc="apply V9.2"):
            indices = batch.pop("window_index").numpy()
            device_batch = {
                key: value.to(args.device) for key, value in batch.items()
            }
            prediction = model(device_batch).cpu().numpy()
            for batch_index, window_index in enumerate(indices):
                row = base_dataset.rows[int(window_index)]
                stream_id = str(row["stream_id"])
                start, end = int(row["start"]), int(row["end"])
                weight = blend_weights(end - start)
                info = stream_info[stream_id]
                info["sum"][start:end] += prediction[batch_index] * weight
                info["weight"][start:end] += weight

    groups = defaultdict(group_metrics)
    stream_rows = []
    bin_ranges = {
        "0_5mm": (0.0, 0.005),
        "5_15mm": (0.005, 0.015),
        "15_30mm": (0.015, 0.030),
        "30_infmm": (0.030, np.inf),
    }
    for stream_id, info in tqdm(
        sorted(stream_info.items()), desc="write streams"
    ):
        output_path = out_root / stream_id / "v9_2_ray_depth_result.npz"
        if output_path.is_file() and not args.overwrite:
            raise FileExistsError(output_path)
        glob = load_npz(info["global_path"])
        initial = np.asarray(glob["pred_joints_3d"][:, 0], np.float32).copy()
        target = np.asarray(glob["gt_joints_3d"][:, 0], np.float32).copy()
        normalized_left = bool(
            np.asarray(glob.get("normalized_left", False)).item()
        )
        if normalized_left:
            initial[:, 0] *= -1.0
            target[:, 0] *= -1.0
        valid = (
            np.asarray(glob["hand_valid"], bool)
            & np.asarray(glob["gt_valid"], bool)
            & np.asarray(glob["supervision_valid"], bool)
            & np.isfinite(initial).all(axis=-1)
            & np.isfinite(target).all(axis=-1)
            & (info["weight"] > 0)
        )
        predicted_ray = (
            info["sum"] / np.maximum(info["weight"], 1e-8)
        ).astype(np.float32)
        ray = initial / np.maximum(
            np.linalg.norm(initial, axis=-1, keepdims=True), 1e-8
        )
        translation_camera = predicted_ray[:, None] * ray
        corrected = initial + translation_camera
        target_delta = target - initial
        target_ray = np.sum(target_delta * ray, axis=-1)
        lateral_vector = target_delta - target_ray[:, None] * ray
        initial_full = np.linalg.norm(target_delta, axis=-1)
        corrected_full = np.linalg.norm(target - corrected, axis=-1)
        initial_ray = np.abs(target_ray)
        corrected_ray = np.abs(target_ray - predicted_ray)
        lateral = np.linalg.norm(lateral_vector, axis=-1)
        side = scalar_text(glob.get("hand_side", np.asarray("unknown")))

        add_metrics(
            groups["all"], valid, initial_full, corrected_full,
            initial_ray, corrected_ray, lateral,
        )
        add_metrics(
            groups[f"side:{side}"], valid, initial_full, corrected_full,
            initial_ray, corrected_ray, lateral,
        )
        for name, (lower, upper) in bin_ranges.items():
            mask = valid & (initial_ray >= lower) & (initial_ray < upper)
            add_metrics(
                groups[f"ray:{name}"], mask,
                initial_full, corrected_full, initial_ray,
                corrected_ray, lateral,
            )

        handflow_path = handflow_root / stream_id / "handflow_camera_result.npz"
        handflow = load_npz(handflow_path)
        vertices = np.asarray(handflow["verts_cam"], dtype=np.float32)
        count = min(len(vertices), len(translation_camera))
        corrected_vertices = vertices[:count].copy()
        mesh_valid = valid[:count] & np.isfinite(vertices[:count]).all(
            axis=(1, 2)
        )
        corrected_vertices[mesh_valid] += translation_camera[
            :count
        ][mesh_valid, None]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            frame_ids=np.asarray(glob["frame_ids"][:count]),
            verts_cam=corrected_vertices,
            faces=np.asarray(handflow["faces"]),
            pred_valid=mesh_valid,
            initial_wrist_camera=initial[:count],
            target_wrist_camera=target[:count],
            corrected_wrist_camera=corrected[:count],
            predicted_ray_depth=predicted_ray[:count],
            target_ray_depth=target_ray[:count],
            translation_camera=translation_camera[:count],
            normalized_left=np.asarray(normalized_left),
            hand_side=np.asarray(side),
            stream_id=np.asarray(stream_id),
            checkpoint=np.asarray(str(checkpoint_path)),
            model_version=np.asarray(checkpoint["model_version"]),
            source_handflow=np.asarray(str(handflow_path)),
        )
        stream_rows.append({
            "stream_id": stream_id,
            "result": str(output_path),
            "side": side,
            "evaluated": int(valid.sum()),
        })

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_version": checkpoint["model_version"],
        "windows": str(windows_path),
        "num_windows": len(base_dataset),
        "num_streams": len(stream_rows),
        "groups": {
            name: finalize_group(group)
            for name, group in sorted(groups.items())
        },
        "streams": stream_rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        key: value for key, value in summary.items() if key != "streams"
    }, indent=2))


if __name__ == "__main__":
    main()
