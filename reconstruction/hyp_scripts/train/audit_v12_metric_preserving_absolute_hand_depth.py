#!/usr/bin/env python3
"""Evaluate V12 feature dependence on unique validation frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from train_v10_pi3x_hand_neighborhood_depth import disable_mha_fastpath
from train_v12_metric_preserving_absolute_hand_depth import (
    FEATURE_MODES,
    MetricPreservingAbsoluteHandDepth,
    make_dataset,
    run_epoch,
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
    parser.add_argument(
        "--modes",
        default="normal,decoder_zero,geometry_zero,metric_zero,pi3x_zero,"
        "handflow_zero,joint_shuffle,time_reverse",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disable_mha_fastpath()
    checkpoint = torch.load(
        Path(args.checkpoint).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    train_args = SimpleNamespace(**checkpoint["args"])
    data = make_dataset(
        args.windows,
        args.global_root,
        args.dense_root,
        args.handflow_root,
        train_args,
    )
    model = MetricPreservingAbsoluteHandDepth(
        int(checkpoint["decoder_feature_dim"]),
        int(checkpoint["metric_feature_dim"]),
        int(checkpoint["metadata_dim"]),
        int(checkpoint["handflow_dim"]),
        int(checkpoint["num_joints"]),
        train_args,
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    invalid = [value for value in modes if value not in FEATURE_MODES]
    if invalid:
        raise ValueError(f"Unknown feature modes: {invalid}")
    results = {}
    for mode in modes:
        model.feature_mode = mode
        print(f"\n===== {mode} =====", flush=True)
        metrics = run_epoch(model, loader, device, train_args)
        results[mode] = metrics
        print(json.dumps(metrics), flush=True)
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_version": checkpoint["model_version"],
        "unique_frame_metrics": True,
        "results": results,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
