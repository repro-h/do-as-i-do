#!/usr/bin/env python3
"""Run frozen Pi3X once, retain compact candidates, then train trajectory."""

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

from compact_dataset import CompactWindowDataset  # noqa: E402
from compact_model import CompactMultiHandPi3XTrajectoryModel  # noqa: E402
from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from online_pi3x import (  # noqa: E402
    CompactFeatureProvider,
    DiskCompactFeatureProvider,
    DummyDenseProvider,
    Pi3XWindowMaterializer,
    row_key,
)
from train import DatasetStreamBalancedSampler, row_dataset, run_epoch  # noqa: E402
from train_v16_online_pi3x import load_rows, window_layout  # noqa: E402


MODEL_VERSION = "v16_1_compact_pi3x_multihand_trajectory_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--visibility-train-root")
    parser.add_argument("--visibility-val-root")
    parser.add_argument("--track-train-root")
    parser.add_argument("--track-val-root")
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--pi3x-checkpoint", required=True)
    parser.add_argument(
        "--compact-cache-root",
        help="Read precomputed compact window caches instead of running Pi3X",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--max-window-size", type=int, default=128)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument(
        "--feature-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--joint-patch-radius", type=int, default=1)
    parser.add_argument("--global-grid-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
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
    parser.add_argument(
        "--translation-parameterization",
        choices=("ray_depth_uv", "direct_xyz"), default="ray_depth_uv",
    )
    parser.add_argument("--max-image-offset-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-windows-per-dataset", type=int, default=0,
        help="Equal per-dataset epoch budget; zero uses ordinary window shuffle",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def noise_from_args(args):
    return QueryNoise(
        global_sigma_px=args.global_noise_px,
        temporal_sigma_px=args.temporal_noise_px,
        joint_sigma_px=args.joint_noise_px,
        outlier_probability=args.outlier_probability,
        dropout_probability=args.query_dropout,
    )


def metadata_dataset(args, split, training):
    return DexYCBMultiHandWindowDataset(
        getattr(args, f"{split}_windows"),
        None,
        max_hands=args.max_hands,
        training=training,
        noise=noise_from_args(args),
        visibility_source="detector",
        visibility_root=getattr(args, f"visibility_{split}_root"),
        track_root=getattr(args, f"track_{split}_root"),
        near_anchor_frames=args.near_anchor_frames,
        max_anchor_frames=args.max_anchor_frames,
        near_missing_weight=args.near_missing_weight,
        far_missing_weight=args.far_missing_weight,
        dense_provider=DummyDenseProvider(),
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_rows(args.train_windows)
    val_rows = load_rows(args.val_windows)
    if not train_rows or not val_rows:
        raise RuntimeError("Training and validation manifests must both be non-empty")
    for row in train_rows + val_rows:
        if len(row["frame_indices"]) > args.max_window_size:
            raise ValueError(
                f"Window {row_key(row)} exceeds --max-window-size "
                f"{args.max_window_size}"
            )

    clean_sets = {
        "train": metadata_dataset(args, "train", False),
        "val": metadata_dataset(args, "val", False),
    }
    payloads = None
    if args.compact_cache_root:
        provider = DiskCompactFeatureProvider(
            args.compact_cache_root,
            patch_radius=args.joint_patch_radius,
            global_grid_size=args.global_grid_size,
        )
        print(f"Reading compact Pi3X caches from {provider.root}", flush=True)
    else:
        print(
            "Materializing compact Pi3X candidates once per unique clip",
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
            total = sum(len(dataset) for dataset in clean_sets.values())
            progress = tqdm(total=total, desc="Pi3X/compact-RAM")
            for dataset in clean_sets.values():
                for index, row in enumerate(dataset.rows):
                    key = row_key(row)
                    if key not in payloads:
                        metadata = dataset[index]
                        payloads[key] = extractor.compact(
                            row,
                            metadata["joint_uv"].numpy(),
                            patch_radius=args.joint_patch_radius,
                            global_grid_size=args.global_grid_size,
                        )
                    progress.update(1)
            progress.close()
        finally:
            extractor.close()
        provider = CompactFeatureProvider(payloads)
    train_metadata = metadata_dataset(args, "train", True)
    val_metadata = metadata_dataset(args, "val", False)
    train_data = CompactWindowDataset(train_metadata, provider)
    val_data = CompactWindowDataset(val_metadata, provider)
    sample = train_data[0]
    compact_thj = tuple(sample["joint_patch_features"].shape[:3])
    query_thj = tuple(sample["joint_uv"].shape[:3])
    if compact_thj != query_thj:
        raise ValueError(
            "Compact Pi3X/query shape mismatch: "
            f"cache [T,H,J]={compact_thj}, metadata [T,H,J]={query_thj}. "
            "Set --max-hands to the value used during compact cache export."
        )
    dense_equivalent = None
    if payloads is not None:
        dense_equivalent = 0
        for payload in payloads.values():
            feature = payload["joint_patch_features"]
            time, _, _, _, channels = feature.shape
            grid_h, grid_w = np.asarray(payload["source_grid_hw"]).reshape(2)
            dense_equivalent += (
                time * int(grid_h) * int(grid_w)
                * channels * feature.dtype.itemsize
            )
    audit = {
        "model_version": MODEL_VERSION,
        "pi3x_execution": (
            "precomputed_compact_disk_cache"
            if args.compact_cache_root
            else "once_per_unique_clip_then_compact_host_ram"
        ),
        "disk_feature_export": bool(args.compact_cache_root),
        "unique_pi3x_clips": None if payloads is None else len(payloads),
        "compact_ram_gib": (
            None if payloads is None else provider.nbytes / 2**30
        ),
        "compact_cache_root": (
            str(provider.root) if args.compact_cache_root else None
        ),
        "dense_feature_equivalent_gib": (
            None if dense_equivalent is None else dense_equivalent / 2**30
        ),
        "compression_ratio": (
            dense_equivalent / provider.nbytes
            if payloads is not None and provider.nbytes else None
        ),
        "joint_patch_radius": args.joint_patch_radius,
        "joint_candidates": int(sample["joint_patch_features"].shape[-2]),
        "global_candidates": int(sample["global_features"].shape[-2]),
        "train_window_layout": window_layout(train_rows),
        "val_window_layout": window_layout(val_rows),
        "joint_patch_shape": list(sample["joint_patch_features"].shape),
        "global_feature_shape": list(sample["global_features"].shape),
        "visibility_source": "separate_detector_cache",
        "train_datasets": sorted({row_dataset(row) for row in train_rows}),
        "val_datasets": sorted({row_dataset(row) for row in val_rows}),
        "train_windows_per_dataset": args.train_windows_per_dataset,
        "data_parallel_requested": args.data_parallel,
        "visible_cuda_devices": torch.cuda.device_count(),
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    device = torch.device(args.device)
    model = CompactMultiHandPi3XTrajectoryModel(
        point_dim=sample["joint_patch_features"].shape[-1],
        metric_dim=sample["metric_window_features"].shape[-1],
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        temporal_layers=args.temporal_layers,
        dropout=args.dropout,
        max_window_size=args.max_window_size,
        translation_parameterization=args.translation_parameterization,
        max_image_offset_fraction=args.max_image_offset_fraction,
    ).to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires --device cuda")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                "--data-parallel requires at least two visible CUDA devices"
            )
        model = torch.nn.DataParallel(model)
    sampler = (
        DatasetStreamBalancedSampler(
            train_data.rows, args.train_windows_per_dataset, args.seed
        )
        if args.train_windows_per_dataset > 0 else None
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
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
        train_metrics = run_epoch(
            model, train_loader, device, args, optimizer,
            dataset_names=train_metadata.dataset_names,
        )
        val_metrics = run_epoch(
            model, val_loader, device, args,
            dataset_names=val_metadata.dataset_names,
        )
        epoch_row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        checkpoint = {
            "epoch": epoch,
            "model_version": MODEL_VERSION,
            "model": model.state_dict(),
            "args": vars(args),
            "audit": audit,
            "val": val_metrics,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
