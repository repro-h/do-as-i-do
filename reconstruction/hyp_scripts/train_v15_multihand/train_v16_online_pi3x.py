#!/usr/bin/env python3
"""Train V15 trajectory head with one frozen Pi3X pass per clip, cached in RAM."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from model import MultiHandPi3XTrajectoryModel  # noqa: E402
from online_pi3x import (  # noqa: E402
    Pi3XWindowMaterializer,
    RamFeatureProvider,
    row_key,
)
from train import run_epoch  # noqa: E402


MODEL_VERSION = "v16_online_pi3x_multihand_trajectory_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--visibility-train-root", required=True)
    parser.add_argument("--visibility-val-root", required=True)
    parser.add_argument("--track-train-root", required=True)
    parser.add_argument("--track-val-root", required=True)
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--pi3x-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--max-window-size", type=int, default=128)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--global-noise-px", type=float, default=4.0)
    parser.add_argument("--temporal-noise-px", type=float, default=0.5)
    parser.add_argument("--joint-noise-px", type=float, default=2.0)
    parser.add_argument("--outlier-probability", type=float, default=0.03)
    parser.add_argument("--query-dropout", type=float, default=0.1)
    parser.add_argument("--near-anchor-frames", type=int, default=4)
    parser.add_argument("--max-anchor-frames", type=int, default=8)
    parser.add_argument("--near-missing-weight", type=float, default=0.5)
    parser.add_argument("--far-missing-weight", type=float, default=0.2)
    parser.add_argument("--w-depth", type=float, default=0.5)
    parser.add_argument("--w-relative", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-reprojection", type=float, default=0.1)
    parser.add_argument("--reprojection-beta-px", type=float, default=2.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--translation-parameterization", choices=("ray_depth_uv", "direct_xyz"), default="ray_depth_uv")
    parser.add_argument("--max-image-offset-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def load_rows(path):
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_dataset(args, split, training, provider):
    noise = QueryNoise(
        global_sigma_px=args.global_noise_px,
        temporal_sigma_px=args.temporal_noise_px,
        joint_sigma_px=args.joint_noise_px,
        outlier_probability=args.outlier_probability,
        dropout_probability=args.query_dropout,
    )
    return DexYCBMultiHandWindowDataset(
        getattr(args, f"{split}_windows"),
        None,
        max_hands=args.max_hands,
        training=training,
        noise=noise,
        visibility_source="detector",
        visibility_root=getattr(args, f"visibility_{split}_root"),
        track_root=getattr(args, f"track_{split}_root"),
        near_anchor_frames=args.near_anchor_frames,
        max_anchor_frames=args.max_anchor_frames,
        near_missing_weight=args.near_missing_weight,
        far_missing_weight=args.far_missing_weight,
        dense_provider=provider,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.train_windows) + load_rows(args.val_windows)
    unique = {}
    for row in rows:
        length = len(row["frame_indices"])
        if length > args.max_window_size:
            raise ValueError(
                f"Window {row_key(row)} has {length} frames, above "
                f"--max-window-size {args.max_window_size}"
            )
        unique.setdefault(row_key(row), row)
    print(
        f"Materializing {len(unique)} unique Pi3X clips exactly once in RAM",
        flush=True,
    )
    extractor = Pi3XWindowMaterializer(
        args.hand_uni_root,
        args.pi3_root,
        args.pi3x_checkpoint,
        device=args.device,
        pixel_limit=args.pixel_limit,
        feature_dtype=args.feature_dtype,
    )
    payloads = {}
    try:
        for key, row in tqdm(unique.items(), desc="Pi3X/RAM"):
            payloads[key] = extractor(row)
    finally:
        extractor.close()
    provider = RamFeatureProvider(payloads)
    print(f"Pi3X RAM cache: {provider.nbytes / 2**30:.3f} GiB", flush=True)

    train_data = make_dataset(args, "train", True, provider)
    val_data = make_dataset(args, "val", False, provider)
    sample = train_data[0]
    audit = {
        "model_version": MODEL_VERSION,
        "pi3x_execution": "once_per_unique_manifest_clip_then_host_ram",
        "disk_feature_export": False,
        "unique_pi3x_clips": len(unique),
        "pi3x_ram_gib": provider.nbytes / 2**30,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "window_frames": int(sample["point_features"].shape[0]),
        "point_feature_shape": list(sample["point_features"].shape),
        "joint_query_shape": list(sample["joint_uv"].shape),
        "max_window_size": args.max_window_size,
    }
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        return

    device = torch.device(args.device)
    model = MultiHandPi3XTrajectoryModel(
        point_dim=sample["point_features"].shape[-1],
        metric_dim=sample["metric_window_features"].shape[-1],
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        temporal_layers=args.temporal_layers,
        max_window_size=args.max_window_size,
        dropout=args.dropout,
        translation_parameterization=args.translation_parameterization,
        max_image_offset_fraction=args.max_image_offset_fraction,
    ).to(device)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====", flush=True)
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "epoch": epoch,
            "model_version": MODEL_VERSION,
            "model": model.state_dict(),
            "args": vars(args),
            "val": val_metrics,
        }
        torch.save(payload, out_dir / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(payload, out_dir / "best.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
