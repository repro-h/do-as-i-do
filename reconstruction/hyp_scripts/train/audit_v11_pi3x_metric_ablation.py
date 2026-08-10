#!/usr/bin/env python3
"""Audit V11 absolute-depth dependence on Pi3X feature branches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_v10_pi3x_hand_neighborhood_depth import disable_mha_fastpath
from train_v11_pi3x_metric_absolute_depth import (
    Pi3XMetricAbsoluteDepthModel,
    make_dataset,
)


MODES = {
    "normal", "point_zero", "metric_zero", "scalar_zero", "all_zero",
    "spatial_shuffle", "time_reverse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--modes",
        default=(
            "normal,point_zero,metric_zero,scalar_zero,all_zero,"
            "spatial_shuffle,time_reverse"
        ),
    )
    return parser.parse_args()


def distribution(value: np.ndarray) -> dict:
    value = np.asarray(value, dtype=np.float64)
    return {
        "count": int(len(value)),
        "median_mm": float(np.median(value) * 1000.0) if len(value) else None,
        "p90_mm": float(np.percentile(value, 90) * 1000.0)
        if len(value) else None,
        "max_mm": float(np.max(value) * 1000.0) if len(value) else None,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        empty = distribution(np.empty(0))
        return {
            "initial_ray_depth": empty,
            "predicted_ray_depth": empty,
            "initial_translation": empty,
            "predicted_translation": empty,
            "degraded_fraction": None,
        }
    initial = np.stack([row["initial"] for row in rows])
    target = np.stack([row["target"] for row in rows])
    predicted_depth = np.asarray([row["depth"] for row in rows])
    ray = initial / np.maximum(
        np.linalg.norm(initial, axis=-1, keepdims=True), 1e-6
    )
    initial_depth = np.linalg.norm(initial, axis=-1)
    target_depth = np.sum(target * ray, axis=-1)
    predicted = predicted_depth[:, None] * ray
    initial_full = np.linalg.norm(initial - target, axis=-1)
    predicted_full = np.linalg.norm(predicted - target, axis=-1)
    return {
        "initial_ray_depth": distribution(np.abs(initial_depth - target_depth)),
        "predicted_ray_depth": distribution(np.abs(predicted_depth - target_depth)),
        "initial_translation": distribution(initial_full),
        "predicted_translation": distribution(predicted_full),
        "degraded_fraction": float(
            np.mean(predicted_full - initial_full > 1e-6)
        ),
        "worse_2mm_fraction": float(
            np.mean(predicted_full - initial_full > 0.002)
        ),
        "worse_5mm_fraction": float(
            np.mean(predicted_full - initial_full > 0.005)
        ),
    }


def evaluate(
    model: Pi3XMetricAbsoluteDepthModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    frames: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"depths": []}
    )
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            depth = model(batch)
            initial = batch["initial_t"]
            ray = initial / torch.linalg.norm(
                initial, dim=-1, keepdim=True
            ).clamp_min(1e-6)
            target_depth = (batch["target_t"] * ray).sum(dim=-1)
            valid = (
                batch["valid"] & batch["observed"]
                & (target_depth > 1e-5)
            )
            for batch_index, time_index in zip(*torch.where(valid)):
                key = (
                    int(batch["stream_index"][batch_index, time_index]),
                    int(batch["frame_index"][batch_index, time_index]),
                )
                row = frames[key]
                row["depths"].append(float(depth[batch_index, time_index]))
                row["initial"] = initial[
                    batch_index, time_index
                ].cpu().numpy()
                row["target"] = batch["target_t"][
                    batch_index, time_index
                ].cpu().numpy()
                row["side"] = int(
                    batch["side"][batch_index, time_index]
                )
    rows = [{
        **row,
        "depth": float(np.mean(row["depths"])),
    } for row in frames.values()]
    output = summarize(rows)
    output["unique_frames"] = len(rows)
    output["by_side"] = {
        name: summarize([row for row in rows if row["side"] == value])
        for name, value in (("left", 0), ("right", 1))
    }
    return output


def main() -> None:
    args = parse_args()
    disable_mha_fastpath()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model_args = SimpleNamespace(**checkpoint["args"])
    dataset = make_dataset(
        args.windows, args.global_root, args.dense_root, model_args
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = Pi3XMetricAbsoluteDepthModel(
        int(checkpoint["point_feature_dim"]),
        int(checkpoint["metric_feature_dim"]),
        int(checkpoint["metadata_dim"]),
        int(checkpoint["num_joints"]),
        model_args,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    results = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        print(f"\n===== {mode} =====", flush=True)
        model.feature_mode = mode
        results[mode] = evaluate(model, loader, device)
        print(json.dumps(results[mode]), flush=True)
    output = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_version": checkpoint["model_version"],
        "results": results,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
