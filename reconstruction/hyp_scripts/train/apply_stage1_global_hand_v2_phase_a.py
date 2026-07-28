#!/usr/bin/env python3
"""Apply and overlap-blend Stage1 Global Hand v2 Phase A predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_stage1_global_hand_v2_phase_a import (
    TranslationRefiner,
    WindowDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def blend_weights(length: int) -> np.ndarray:
    position = np.arange(length, dtype=np.float32)
    edge = np.minimum(position + 1.0, length - position)
    return edge / edge.max()


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64) * 1000.0
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.quantile(values, 0.9)),
        "max_mm": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    windows_path = Path(args.windows).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["args"]
    dataset = WindowDataset(
        windows_path,
        float(config["max_target_mm"]) / 1000.0,
        bool(config.get("include_camera_ray", False)),
        bool(config.get("include_surface_geometry", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = TranslationRefiner(
        int(checkpoint["input_dim"]),
        int(config["hidden_dim"]),
        int(config["layers"]),
        int(config["heads"]),
        float(config["dropout"]),
        bool(config.get("correction_gate", False)),
        str(config.get("prediction_mode", "translation3d")),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lengths, supervision_paths = {}, {}
    for row in dataset.rows:
        stream_id = row["stream_id"]
        supervision_paths[stream_id] = row["supervision_npz"]
        with np.load(row["supervision_npz"], allow_pickle=False) as raw:
            lengths[stream_id] = len(raw["frame_ids"])
    sums = {
        stream_id: {
            "translation": np.zeros((length, 3), dtype=np.float64),
            "gate": np.zeros(length, dtype=np.float64),
            "weight": np.zeros(length, dtype=np.float64),
        }
        for stream_id, length in lengths.items()
    }

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(args.device)
            prediction = model(
                features, float(config["max_translation_mm"]) / 1000.0
            )
            if bool(config.get("correction_gate", False)):
                raw_prediction, gate = prediction
                gate = gate.cpu().numpy()
            else:
                raw_prediction = prediction
                gate = np.ones(
                    raw_prediction.shape[:2], dtype=np.float32
                )
            raw_prediction = raw_prediction.cpu().numpy()
            wrist = batch["pred_joints"][:, :, 0].numpy()
            ray = wrist / np.maximum(
                np.linalg.norm(wrist, axis=-1, keepdims=True), 1e-8
            )
            if config.get("prediction_mode", "translation3d") == "ray_depth":
                ray_depth = raw_prediction[..., 0]
                translation = ray_depth[..., None] * ray
            else:
                translation = raw_prediction
                ray_depth = np.sum(translation * ray, axis=-1)
            for index, stream_id in enumerate(batch["stream_id"]):
                start = int(batch["start"][index])
                end = int(batch["end"][index])
                weights = blend_weights(end - start)
                sums[stream_id]["translation"][start:end] += (
                    translation[index] * weights[:, None]
                )
                sums[stream_id]["gate"][start:end] += (
                    gate[index] * weights
                )
                sums[stream_id]["weight"][start:end] += weights

    handflow_root = Path(args.handflow_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stream_rows = []
    aggregate_before, aggregate_after = [], []
    aggregate_ray_before, aggregate_ray_after = [], []
    for stream_id, values in sorted(sums.items()):
        stream_out = out_root / stream_id
        result_path = stream_out / "handflow_camera_result_global_v2_phase_a.npz"
        if result_path.is_file() and not args.overwrite:
            continue
        weight = values["weight"]
        predicted = weight > 0.0
        translation_normalized = (
            values["translation"] / np.maximum(weight, 1e-8)[:, None]
        ).astype(np.float32)
        correction_gate = (
            values["gate"] / np.maximum(weight, 1e-8)
        ).astype(np.float32)
        with np.load(supervision_paths[stream_id], allow_pickle=False) as raw:
            supervision = {key: np.asarray(raw[key]) for key in raw.files}
        translation_camera = translation_normalized.copy()
        if bool(np.asarray(supervision["normalized_left"]).item()):
            translation_camera[:, 0] *= -1.0

        handflow_path = (
            handflow_root / stream_id / "handflow_camera_result.npz"
        )
        with np.load(handflow_path, allow_pickle=False) as raw:
            handflow = {key: np.asarray(raw[key]) for key in raw.files}
        vertices = np.asarray(handflow["verts_cam"], dtype=np.float32)
        count = min(len(vertices), len(translation_camera))
        corrected = vertices[:count] + translation_camera[:count, None]

        pred_wrist = np.asarray(
            supervision["pred_joints_3d"][:count, 0], dtype=np.float32
        )
        gt_wrist = np.asarray(
            supervision["gt_joints_3d"][:count, 0], dtype=np.float32
        )
        valid = (
            np.asarray(supervision["supervision_valid"][:count]).astype(bool)
            & predicted[:count]
        )
        corrected_wrist = (
            pred_wrist + translation_normalized[:count]
        )
        before = np.linalg.norm(pred_wrist[valid] - gt_wrist[valid], axis=-1)
        after = np.linalg.norm(
            corrected_wrist[valid] - gt_wrist[valid], axis=-1
        )
        camera_ray_normalized = pred_wrist / np.maximum(
            np.linalg.norm(pred_wrist, axis=-1, keepdims=True), 1e-8
        )
        target_ray_depth = np.sum(
            (gt_wrist - pred_wrist) * camera_ray_normalized, axis=-1
        )
        predicted_ray_depth = np.sum(
            translation_normalized[:count] * camera_ray_normalized, axis=-1
        )
        ray_before = np.abs(target_ray_depth[valid])
        ray_after = np.abs(
            target_ray_depth[valid] - predicted_ray_depth[valid]
        )
        aggregate_before.append(before)
        aggregate_after.append(after)
        aggregate_ray_before.append(ray_before)
        aggregate_ray_after.append(ray_after)
        uncovered = (
            np.asarray(supervision["hand_valid"][:count]).astype(bool)
            & ~predicted[:count]
        )
        metrics = {
            "initial_wrist": quantiles(before),
            "corrected_wrist": quantiles(after),
            "initial_ray_depth": quantiles(ray_before),
            "corrected_ray_depth": quantiles(ray_after),
            "num_frames": count,
            "num_predicted": int(predicted[:count].sum()),
            "uncovered_hand_frames": np.flatnonzero(uncovered).tolist(),
        }
        stream_out.mkdir(parents=True, exist_ok=True)
        output = dict(handflow)
        output["verts_cam"] = corrected
        output["stage1_translation_normalized"] = translation_normalized[:count]
        output["stage1_translation_camera"] = translation_camera[:count]
        camera_ray = pred_wrist / np.maximum(
            np.linalg.norm(pred_wrist, axis=-1, keepdims=True), 1e-8
        )
        output["stage1_ray_depth"] = np.sum(
            translation_normalized[:count] * camera_ray, axis=-1
        ).astype(np.float32)
        output["stage1_prediction_mode"] = np.asarray(
            config.get("prediction_mode", "translation3d")
        )
        output["stage1_correction_gate"] = correction_gate[:count]
        output["stage1_predicted"] = predicted[:count]
        output["stage1_checkpoint"] = np.asarray(str(checkpoint_path))
        output["stage1_source_handflow"] = np.asarray(str(handflow_path))
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

    before_all = (
        np.concatenate(aggregate_before) if aggregate_before else np.empty(0)
    )
    after_all = (
        np.concatenate(aggregate_after) if aggregate_after else np.empty(0)
    )
    ray_before_all = (
        np.concatenate(aggregate_ray_before)
        if aggregate_ray_before else np.empty(0)
    )
    ray_after_all = (
        np.concatenate(aggregate_ray_after)
        if aggregate_ray_after else np.empty(0)
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_total": float(checkpoint["val_total"]),
        "windows": str(windows_path),
        "num_windows": len(dataset),
        "num_streams": len(stream_rows),
        "aggregate_metrics": {
            "initial_wrist": quantiles(before_all),
            "corrected_wrist": quantiles(after_all),
            "initial_ray_depth": quantiles(ray_before_all),
            "corrected_ray_depth": quantiles(ray_after_all),
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
