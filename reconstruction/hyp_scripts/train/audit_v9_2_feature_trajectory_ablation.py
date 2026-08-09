#!/usr/bin/env python3
"""Evaluate V9.2 with individual Pi3X feature groups ablated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from train_v9_2_pi3x_feature_trajectory_depth import (
    FeatureTrajectoryDataset,
    FeatureTrajectoryDepthModel,
    run_epoch,
)


MODES = (
    "normal",
    "hand_feature_zero",
    "object_feature_zero",
    "all_feature_zero",
    "feature_time_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--pi3x-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class AblationDataset(Dataset):
    def __init__(self, source: FeatureTrajectoryDataset, mode: str):
        self.source = source
        self.mode = mode

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.source[index]
        if self.mode in ("hand_feature_zero", "all_feature_zero"):
            sample["hand_token_features"] = torch.zeros_like(
                sample["hand_token_features"]
            )
        if self.mode in ("object_feature_zero", "all_feature_zero"):
            sample["object_token_features"] = torch.zeros_like(
                sample["object_token_features"]
            )
        if self.mode == "feature_time_mean":
            for key in ("hand_token_features", "object_token_features"):
                value = sample[key]
                sample[key] = value.mean(dim=0, keepdim=True).expand_as(value)
        return sample


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model_args = SimpleNamespace(**checkpoint["args"])
    source = FeatureTrajectoryDataset(
        Path(args.windows), Path(args.global_root), Path(args.pi3x_root)
    )
    sample = source[0]
    model = FeatureTrajectoryDepthModel(
        int(checkpoint.get(
            "local_hand_dim", sample["local_hand_features"].shape[-1]
        )),
        int(checkpoint.get(
            "pi3x_feature_dim", sample["hand_token_features"].shape[-1]
        )),
        int(checkpoint.get(
            "pi3x_metadata_dim", sample["hand_token_metadata"].shape[-1]
        )),
        model_args,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device)

    results: dict[str, dict] = {}
    for mode in MODES:
        print(f"\n===== {mode} =====", flush=True)
        loader = DataLoader(
            AblationDataset(source, mode),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        results[mode] = run_epoch(model, loader, device, model_args)
        summary = results[mode]
        print(
            json.dumps(
                {
                    "ray_after_mm": summary[
                        "corrected_ray_depth"
                    ]["median_mm"],
                    "left_after_mm": summary["by_side"]["left"][
                        "corrected_ray_depth"
                    ]["median_mm"],
                    "right_after_mm": summary["by_side"]["right"][
                        "corrected_ray_depth"
                    ]["median_mm"],
                    "degraded_fraction": summary["degraded_fraction"],
                }
            ),
            flush=True,
        )

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
