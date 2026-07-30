#!/usr/bin/env python3
"""Apply and overlap-blend a Pi3X relative-depth refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_pi3x_relative_depth_refiner import (
    Pi3XRelativeDepthRefiner,
    Pi3XWindowDataset,
    quantiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--pi3x-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def blend_weights(length: int) -> np.ndarray:
    position = np.arange(length, dtype=np.float32)
    edge = np.minimum(position + 1.0, length - position)
    return edge / edge.max()


def main() -> None:
    args = parse_args()
    windows_path = Path(args.windows).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["args"]
    dataset = Pi3XWindowDataset(
        windows_path,
        Path(args.pi3x_root).expanduser().resolve(),
        float(config["max_target_mm"]) / 1000.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = Pi3XRelativeDepthRefiner(
        int(checkpoint["scalar_dim"]),
        int(checkpoint["feature_dim"]),
        int(checkpoint["metadata_dim"]),
        int(config["hidden_dim"]),
        int(config["spatial_layers"]),
        int(config["temporal_layers"]),
        int(config["heads"]),
        float(config["dropout"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lengths, supervision_paths = {}, {}
    for row in dataset.rows:
        stream_id = row["stream_id"]
        supervision_paths[stream_id] = row["supervision_npz"]
        supervision = np.load(row["supervision_npz"], allow_pickle=False)
        lengths[stream_id] = len(supervision["frame_ids"])
        supervision.close()
    sums = {
        stream_id: {
            "depth": np.zeros(length, dtype=np.float64),
            "gate": np.zeros(length, dtype=np.float64),
            "sign_probability": np.zeros(length, dtype=np.float64),
            "magnitude": np.zeros(length, dtype=np.float64),
            "weight": np.zeros(length, dtype=np.float64),
        }
        for stream_id, length in lengths.items()
    }

    with torch.no_grad():
        for batch in loader:
            model_output = model(
                batch["scalar"].to(args.device),
                batch["token_features"].to(args.device),
                batch["token_metadata"].to(args.device),
                batch["token_valid"].to(args.device),
                batch["token_types"].to(args.device),
                float(config["max_correction_mm"]) / 1000.0,
                return_aux=True,
            )
            depth = model_output["prediction"].cpu().numpy()
            gate = model_output["gate"].cpu().numpy()
            sign_probability = torch.sigmoid(
                model_output["sign_logits"]
            ).cpu().numpy()
            magnitude = model_output["magnitude"].cpu().numpy()
            for index, stream_id in enumerate(batch["stream_id"]):
                start = int(batch["start"][index])
                end = int(batch["end"][index])
                weight = blend_weights(end - start)
                sums[stream_id]["depth"][start:end] += depth[index] * weight
                sums[stream_id]["gate"][start:end] += gate[index] * weight
                sums[stream_id]["sign_probability"][start:end] += (
                    sign_probability[index] * weight
                )
                sums[stream_id]["magnitude"][start:end] += (
                    magnitude[index] * weight
                )
                sums[stream_id]["weight"][start:end] += weight

    handflow_root = Path(args.handflow_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stream_rows = []
    aggregate_before, aggregate_after = [], []
    aggregate_depth_before, aggregate_depth_after = [], []
    for stream_id, values in sorted(sums.items()):
        stream_out = out_root / stream_id
        result_path = (
            stream_out / "handflow_camera_result_pi3x_depth_refined.npz"
        )
        if result_path.is_file() and not args.overwrite:
            continue
        weight = values["weight"]
        predicted = weight > 0.0
        depth = (
            values["depth"] / np.maximum(weight, 1e-8)
        ).astype(np.float32)
        gate = (
            values["gate"] / np.maximum(weight, 1e-8)
        ).astype(np.float32)
        sign_probability = (
            values["sign_probability"] / np.maximum(weight, 1e-8)
        ).astype(np.float32)
        magnitude = (
            values["magnitude"] / np.maximum(weight, 1e-8)
        ).astype(np.float32)
        with np.load(
            supervision_paths[stream_id], allow_pickle=False
        ) as raw:
            supervision = {
                key: np.asarray(raw[key]) for key in raw.files
            }
        pred_wrist = np.asarray(
            supervision["pred_joints_3d"][:, 0], dtype=np.float32
        )
        gt_wrist = np.asarray(
            supervision["gt_joints_3d"][:, 0], dtype=np.float32
        )
        ray_normalized = pred_wrist / np.maximum(
            np.linalg.norm(pred_wrist, axis=-1, keepdims=True), 1e-8
        )
        translation_normalized = depth[:, None] * ray_normalized
        translation_camera = translation_normalized.copy()
        if bool(np.asarray(supervision["normalized_left"]).item()):
            translation_camera[:, 0] *= -1.0

        handflow_path = (
            handflow_root / stream_id / "handflow_camera_result.npz"
        )
        with np.load(handflow_path, allow_pickle=False) as raw:
            handflow = {key: np.asarray(raw[key]) for key in raw.files}
        vertices = np.asarray(handflow["verts_cam"], dtype=np.float32)
        count = min(len(vertices), len(depth))
        corrected_vertices = (
            vertices[:count] + translation_camera[:count, None]
        )
        valid = (
            np.asarray(supervision["supervision_valid"][:count]).astype(bool)
            & predicted[:count]
        )
        target_depth = np.sum(
            (gt_wrist[:count] - pred_wrist[:count])
            * ray_normalized[:count],
            axis=-1,
        )
        before = np.linalg.norm(
            pred_wrist[:count][valid] - gt_wrist[:count][valid], axis=-1
        )
        corrected_wrist = (
            pred_wrist[:count] + translation_normalized[:count]
        )
        after = np.linalg.norm(
            corrected_wrist[valid] - gt_wrist[:count][valid], axis=-1
        )
        depth_before = np.abs(target_depth[valid])
        depth_after = np.abs(target_depth[valid] - depth[:count][valid])
        aggregate_before.append(before)
        aggregate_after.append(after)
        aggregate_depth_before.append(depth_before)
        aggregate_depth_after.append(depth_after)
        metrics = {
            "initial_wrist": quantiles([before]),
            "corrected_wrist": quantiles([after]),
            "initial_ray_depth": quantiles([depth_before]),
            "corrected_ray_depth": quantiles([depth_after]),
            "num_frames": count,
            "num_predicted": int(predicted[:count].sum()),
        }
        stream_out.mkdir(parents=True, exist_ok=True)
        output = dict(handflow)
        output["verts_cam"] = corrected_vertices
        output["pi3x_depth_correction"] = depth[:count]
        output["pi3x_depth_gate"] = gate[:count]
        output["pi3x_depth_sign_probability"] = sign_probability[:count]
        output["pi3x_depth_magnitude"] = magnitude[:count]
        output["pi3x_translation_normalized"] = (
            translation_normalized[:count]
        )
        output["pi3x_translation_camera"] = translation_camera[:count]
        output["pi3x_depth_predicted"] = predicted[:count]
        output["pi3x_depth_checkpoint"] = np.asarray(
            str(checkpoint_path)
        )
        output["pi3x_depth_source_handflow"] = np.asarray(
            str(handflow_path)
        )
        np.savez_compressed(result_path, **output)
        (stream_out / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        stream_rows.append(
            {
                "stream_id": stream_id,
                "result": str(result_path),
                "metrics": metrics,
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_total": float(checkpoint["val_total"]),
        "windows": str(windows_path),
        "num_windows": len(dataset),
        "num_streams": len(stream_rows),
        "aggregate_metrics": {
            "initial_wrist": quantiles(aggregate_before),
            "corrected_wrist": quantiles(aggregate_after),
            "initial_ray_depth": quantiles(aggregate_depth_before),
            "corrected_ray_depth": quantiles(aggregate_depth_after),
        },
        "streams": stream_rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "streams"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
