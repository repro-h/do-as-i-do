#!/usr/bin/env python3
"""Audit V11.2 dependence on HandFlow latent and Pi3X branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from audit_v11_1_pi3x_metric_point_ablation import evaluate
from train_v10_pi3x_hand_neighborhood_depth import disable_mha_fastpath
from train_v11_2_handflow_latent_pi3x_ray_residual import (
    FEATURE_MODES,
    HandFlowLatentPi3XRayResidual,
    make_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default=",".join(FEATURE_MODES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disable_mha_fastpath()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model_args = SimpleNamespace(**checkpoint["args"])
    dataset = make_dataset(
        args.windows, args.global_root, args.dense_root,
        args.handflow_root, model_args,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = HandFlowLatentPi3XRayResidual(
        int(checkpoint["decoder_feature_dim"]),
        int(checkpoint["metric_feature_dim"]),
        int(checkpoint["metadata_dim"]),
        int(checkpoint["latent_dim"]),
        int(checkpoint["num_joints"]),
        model_args,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    results = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        if mode not in FEATURE_MODES:
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
