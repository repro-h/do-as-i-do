#!/usr/bin/env python3
"""Audit a frozen V9.4 no-op probe and offline residual selectors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_v9_3_joint_conditioned_noop_probe import (
    JointConditionedNoopModel,
    binary_metrics,
)
from train_v9_4_dense_joint_pi3x_noop_probe import DenseJointDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--pi3x-root", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--thresholds", default="0.3,0.5,0.7,0.8,0.9"
    )
    return parser.parse_args()


def distribution(value: np.ndarray) -> dict:
    value = np.asarray(value, dtype=np.float64)
    return {
        "count": int(len(value)),
        "median": float(np.median(value)) if len(value) else None,
        "p10": float(np.percentile(value, 10)) if len(value) else None,
        "p90": float(np.percentile(value, 90)) if len(value) else None,
    }


def correction_metrics(
    initial: np.ndarray,
    target: np.ndarray,
    ray: np.ndarray,
    correction: np.ndarray,
) -> dict:
    if not len(initial):
        empty = distribution(np.empty(0))
        return {
            "ray_error_mm": empty,
            "translation_error_mm": empty,
            "degraded_fraction": None,
            "worse_2mm_fraction": None,
            "worse_5mm_fraction": None,
            "worse_10mm_fraction": None,
        }
    corrected = initial + correction[:, None] * ray
    initial_error = np.linalg.norm(initial - target, axis=-1) * 1000.0
    corrected_error = np.linalg.norm(corrected - target, axis=-1) * 1000.0
    target_ray = np.sum((target - initial) * ray, axis=-1)
    ray_error = np.abs(correction - target_ray) * 1000.0
    increase = corrected_error - initial_error
    # Reconstructing an unchanged point can introduce sub-micrometer floating
    # noise. Do not count that as a behavioral degradation.
    degradation_tolerance_mm = 1e-3
    return {
        "ray_error_mm": distribution(ray_error),
        "translation_error_mm": distribution(corrected_error),
        "degraded_fraction": float(
            np.mean(increase > degradation_tolerance_mm)
        ),
        "worse_2mm_fraction": float(np.mean(increase > 2.0)),
        "worse_5mm_fraction": float(np.mean(increase > 5.0)),
        "worse_10mm_fraction": float(np.mean(increase > 10.0)),
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model_args = SimpleNamespace(**checkpoint["args"])
    dataset = DenseJointDataset(
        Path(args.windows), Path(args.global_root), Path(args.pi3x_root),
        Path(args.dense_root), model_args.min_confidence,
        model_args.min_object_coverage,
    )
    sample = dataset[0]
    model = JointConditionedNoopModel(
        int(checkpoint.get(
            "local_hand_dim", sample["local_hand_features"].shape[-1]
        )),
        int(checkpoint.get(
            "pi3x_feature_dim", sample["hand_token_features"].shape[-1]
        )),
        int(checkpoint.get(
            "pi3x_metadata_dim", sample["hand_token_metadata"].shape[-1]
        )),
        int(checkpoint.get(
            "joint_metadata_dim", sample["joint_token_metadata"].shape[-1]
        )),
        model_args,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    # A frame can occur in overlapping dense windows. Average frozen-model
    # outputs first, then report unique-frame metrics.
    frames: dict[tuple[int, int], dict] = defaultdict(lambda: {
        "correction": [], "score": [],
    })
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            correction, logit = model(batch)
            score = torch.sigmoid(logit)
            valid = batch["valid"] & batch["hand_observed"]
            for b, t in zip(*torch.where(valid)):
                key = (
                    int(batch["stream_index"][b, t]),
                    int(batch["frame_index"][b, t]),
                )
                row = frames[key]
                row["correction"].append(float(correction[b, t]))
                row["score"].append(float(score[b, t]))
                row["initial"] = batch["initial_t"][b, t].cpu().numpy()
                row["target"] = batch["target_t"][b, t].cpu().numpy()
                row["side"] = int(batch["side"][b, t])

    rows = list(frames.values())
    initial = np.stack([row["initial"] for row in rows])
    target = np.stack([row["target"] for row in rows])
    correction = np.asarray([
        np.mean(row["correction"]) for row in rows
    ])
    score = np.asarray([np.mean(row["score"]) for row in rows])
    side = np.asarray([row["side"] for row in rows])
    ray = initial / np.maximum(
        np.linalg.norm(initial, axis=-1, keepdims=True), 1e-6
    )
    target_ray = np.sum((target - initial) * ray, axis=-1)
    target_ray_mm = np.abs(target_ray) * 1000.0
    noop_target = target_ray_mm < model_args.noop_threshold_mm

    bins = {}
    for name, low, high in (
        ("0_5mm", 0.0, 5.0),
        ("5_15mm", 5.0, 15.0),
        ("15_30mm", 15.0, 30.0),
        ("30_infmm", 30.0, float("inf")),
    ):
        mask = (target_ray_mm >= low) & (target_ray_mm < high)
        bins[name] = {
            "noop_score": distribution(score[mask]),
            "candidate": correction_metrics(
                initial[mask], target[mask], ray[mask], correction[mask]
            ),
        }

    thresholds = [float(value) for value in args.thresholds.split(",")]
    selectors = {}
    for threshold in thresholds:
        selected = np.where(score >= threshold, 0.0, correction)
        selectors[f"hard_{threshold:g}"] = correction_metrics(
            initial, target, ray, selected
        )
    selectors["soft"] = correction_metrics(
        initial, target, ray, (1.0 - score) * correction
    )

    output = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_version": checkpoint["model_version"],
        "unique_frames": len(rows),
        "noop_probe": binary_metrics(noop_target, score),
        "noop_probe_by_side": {
            name: binary_metrics(noop_target[side == value], score[side == value])
            for name, value in (("left", 0), ("right", 1))
        },
        "initial": correction_metrics(
            initial, target, ray, np.zeros_like(correction)
        ),
        "candidate": correction_metrics(
            initial, target, ray, correction
        ),
        "score_bins": bins,
        "offline_selectors": selectors,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
