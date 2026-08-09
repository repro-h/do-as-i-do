#!/usr/bin/env python3
"""Ablate dense joint samples in a trained V9.4 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from train_v9_3_joint_conditioned_noop_probe import (
    JointConditionedNoopModel,
    run_epoch,
)
from train_v9_4_dense_joint_pi3x_noop_probe import DenseJointDataset


MODES = (
    "normal",
    "dense_feature_zero",
    "dense_geometry_zero",
    "dense_all_zero",
    "dense_time_mean",
    "dense_time_reverse",
    "compact_feature_zero",
    "compact_all_zero",
    "compact_time_reverse",
    "all_pi3x_zero",
)


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
        "--modes",
        default=",".join(MODES),
        help="Comma-separated ablation modes.",
    )
    return parser.parse_args()


class AblationDataset(Dataset):
    def __init__(self, source: DenseJointDataset, mode: str):
        self.source = source
        self.mode = mode

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.source[index]
        features = sample["joint_token_features"]
        metadata = sample["joint_token_metadata"]
        if self.mode in ("dense_feature_zero", "dense_all_zero"):
            sample["joint_token_features"] = torch.zeros_like(features)
        if self.mode == "dense_geometry_zero":
            metadata = metadata.clone()
            # point_relative, confidence, hand coverage, object coverage
            metadata[..., :6] = 0
            sample["joint_token_metadata"] = metadata
        if self.mode == "dense_all_zero":
            sample["joint_token_metadata"] = torch.zeros_like(metadata)
        if self.mode == "dense_time_mean":
            sample["joint_token_features"] = features.mean(
                dim=0, keepdim=True
            ).expand_as(features)
            sample["joint_token_metadata"] = metadata.mean(
                dim=0, keepdim=True
            ).expand_as(metadata)
        if self.mode == "dense_time_reverse":
            sample["joint_token_features"] = features.flip(0)
            sample["joint_token_metadata"] = metadata.flip(0)
            sample["joint_token_valid"] = sample[
                "joint_token_valid"
            ].flip(0)
        if self.mode in (
            "compact_feature_zero", "compact_all_zero", "all_pi3x_zero"
        ):
            for key in (
                "hand_token_features", "object_token_features"
            ):
                sample[key] = torch.zeros_like(sample[key])
        if self.mode in ("compact_all_zero", "all_pi3x_zero"):
            for key in (
                "hand_token_metadata", "object_token_metadata"
            ):
                sample[key] = torch.zeros_like(sample[key])
        if self.mode == "compact_time_reverse":
            for key in (
                "hand_token_features",
                "hand_token_metadata",
                "hand_token_valid",
                "object_token_features",
                "object_token_metadata",
                "object_token_valid",
            ):
                sample[key] = sample[key].flip(0)
        if self.mode == "all_pi3x_zero":
            sample["joint_token_features"] = torch.zeros_like(features)
            sample["joint_token_metadata"] = torch.zeros_like(metadata)
        return sample


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model_args = SimpleNamespace(**checkpoint["args"])
    source = DenseJointDataset(
        Path(args.windows),
        Path(args.global_root),
        Path(args.pi3x_root),
        Path(args.dense_root),
        model_args.min_confidence,
        model_args.min_object_coverage,
    )
    sample = source[0]
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
    model.to(device)

    results: dict[str, dict] = {}
    modes = tuple(
        mode.strip() for mode in args.modes.split(",") if mode.strip()
    )
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")
    for mode in modes:
        print(f"\n===== {mode} =====", flush=True)
        loader = DataLoader(
            AblationDataset(source, mode),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        metrics = run_epoch(model, loader, device, model_args)
        results[mode] = metrics
        print(json.dumps({
            "ray_after_mm": metrics[
                "corrected_ray_depth"
            ]["median_mm"],
            "degraded_fraction": metrics["degraded_fraction"],
            "noop_auc": metrics["noop_probe"].get("auc"),
            "noop_balanced_accuracy": metrics[
                "noop_probe"
            ].get("balanced_accuracy"),
        }), flush=True)

    output = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_version": checkpoint["model_version"],
        "results": results,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
