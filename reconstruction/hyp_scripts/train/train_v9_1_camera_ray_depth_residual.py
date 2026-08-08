#!/usr/bin/env python3
"""Train an observation-only camera-ray hand depth residual refiner."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import RelativeCrossAttention
from train_v9_camera_hand_residual import (
    CameraWindowDataset,
    masked_mean,
    smooth_l1,
    temporal_loss,
)


MODEL_VERSION = "v9_1_camera_ray_depth_residual_observation_only_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pi3x-relation-dim", type=int, default=128)
    parser.add_argument("--pi3x-heads", type=int, default=8)
    parser.add_argument("--max-ray-correction-mm", type=float, default=250.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--small-anchor-mm", type=float, default=10.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-residual", type=float, default=0.001)
    parser.add_argument("--w-small-anchor", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


class RayDepthResidualModel(nn.Module):
    def __init__(
        self,
        local_dim: int,
        feature_dim: int,
        metadata_dim: int,
        args: argparse.Namespace,
    ):
        super().__init__()
        if args.hidden_dim % 2:
            raise ValueError("hidden-dim must be even")
        self.max_correction = args.max_ray_correction_mm / 1000.0
        self.relation = RelativeCrossAttention(
            feature_dim,
            metadata_dim,
            args.pi3x_relation_dim,
            args.pi3x_heads,
            args.dropout,
        )
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.frame_encoder = nn.Sequential(
            nn.Linear(
                args.hidden_dim + args.pi3x_relation_dim,
                args.hidden_dim,
            ),
            nn.LayerNorm(args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.temporal = nn.GRU(
            args.hidden_dim,
            args.hidden_dim // 2,
            num_layers=args.layers,
            batch_first=True,
            bidirectional=True,
            dropout=args.dropout if args.layers > 1 else 0.0,
        )
        self.depth_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.GELU(),
            nn.Linear(args.hidden_dim, 1),
        )
        nn.init.zeros_(self.depth_head[-1].weight)
        nn.init.zeros_(self.depth_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        relation = self.relation(
            batch["hand_token_features"],
            batch["hand_token_metadata"],
            batch["hand_token_valid"],
            batch["key_token_features"],
            batch["key_token_metadata"],
            batch["key_token_valid"],
            batch["key_token_types"],
        )
        local = self.local_encoder(batch["local_hand_features"])
        frame = self.frame_encoder(torch.cat((local, relation), dim=-1))
        temporal, _ = self.temporal(frame)
        return (
            torch.tanh(self.depth_head(temporal).squeeze(-1))
            * self.max_correction
        )


def distribution(chunks: list[np.ndarray]) -> dict:
    values = np.concatenate(chunks) if chunks else np.empty(0)
    return {
        "count": int(values.size),
        "median_mm": float(np.median(values) * 1000.0) if values.size else None,
        "p90_mm": float(np.percentile(values, 90) * 1000.0) if values.size else None,
        "max_mm": float(np.max(values) * 1000.0) if values.size else None,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    sums = {
        key: 0.0
        for key in (
            "total",
            "depth",
            "velocity",
            "acceleration",
            "residual",
            "small_anchor",
        )
    }
    metrics = {
        key: []
        for key in (
            "initial_full",
            "corrected_full",
            "initial_ray",
            "corrected_ray",
            "lateral",
        )
    }
    improved = degraded = evaluated = batches = 0
    iterator = tqdm(loader, desc="train" if training else "val")
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        bad = [
            key
            for key, value in batch.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if bad:
            raise RuntimeError(f"non-finite batch inputs: {bad}")

        initial_t = batch["initial_t"]
        target_t = batch["target_t"]
        valid = batch["valid"]
        ray = initial_t / torch.linalg.norm(
            initial_t, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        target_delta = target_t - initial_t
        target_ray = (target_delta * ray).sum(dim=-1)
        target_lateral = target_delta - target_ray[..., None] * ray

        with torch.set_grad_enabled(training):
            predicted_ray = model(batch)
            corrected_t = initial_t + predicted_ray[..., None] * ray
            depth = masked_mean(
                smooth_l1(
                    predicted_ray - target_ray,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                valid,
            )
            velocity = temporal_loss(
                predicted_ray[..., None],
                target_ray[..., None],
                valid,
                1,
                args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration = temporal_loss(
                predicted_ray[..., None],
                target_ray[..., None],
                valid,
                2,
                args.smooth_l1_beta_mm / 1000.0,
            )
            residual = masked_mean(
                smooth_l1(predicted_ray, 0.02), valid
            )
            small = valid & (
                target_ray.abs() <= args.small_anchor_mm / 1000.0
            )
            small_anchor = masked_mean(
                smooth_l1(predicted_ray, 0.005), small
            )
            total = (
                args.w_depth * depth
                + args.w_velocity * velocity
                + args.w_acceleration * acceleration
                + args.w_residual * residual
                + args.w_small_anchor * small_anchor
            )
            if not torch.isfinite(total):
                raise RuntimeError(
                    "non-finite loss: "
                    f"depth={depth.item()} velocity={velocity.item()} "
                    f"acceleration={acceleration.item()} "
                    f"residual={residual.item()} "
                    f"small_anchor={small_anchor.item()}"
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for key, value in (
            ("total", total),
            ("depth", depth),
            ("velocity", velocity),
            ("acceleration", acceleration),
            ("residual", residual),
            ("small_anchor", small_anchor),
        ):
            sums[key] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        initial_full = torch.linalg.norm(initial_t - target_t, dim=-1)
        corrected_full = torch.linalg.norm(corrected_t - target_t, dim=-1)
        initial_ray = target_ray.abs()
        corrected_ray = (predicted_ray - target_ray).abs()
        lateral = torch.linalg.norm(target_lateral, dim=-1)
        valid_np = valid.detach().cpu().numpy().astype(bool)
        for key, value in (
            ("initial_full", initial_full),
            ("corrected_full", corrected_full),
            ("initial_ray", initial_ray),
            ("corrected_ray", corrected_ray),
            ("lateral", lateral),
        ):
            metrics[key].append(value.detach().cpu().numpy()[valid_np])
        initial_np = initial_full.detach().cpu().numpy()[valid_np]
        corrected_np = corrected_full.detach().cpu().numpy()[valid_np]
        improved += int((corrected_np < initial_np).sum())
        degraded += int((corrected_np > initial_np).sum())
        evaluated += int(len(initial_np))

    return {
        **{key: value / max(batches, 1) for key, value in sums.items()},
        "initial_translation": distribution(metrics["initial_full"]),
        "corrected_translation": distribution(metrics["corrected_full"]),
        "initial_ray_depth": distribution(metrics["initial_ray"]),
        "corrected_ray_depth": distribution(metrics["corrected_ray"]),
        "irreducible_lateral": distribution(metrics["lateral"]),
        "evaluated": evaluated,
        "improved": improved,
        "degraded": degraded,
        "degraded_fraction": degraded / max(evaluated, 1),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data = CameraWindowDataset(
        Path(args.train_windows),
        Path(args.global_train_root),
        Path(args.pi3x_train_root),
    )
    val_data = CameraWindowDataset(
        Path(args.val_windows),
        Path(args.global_val_root),
        Path(args.pi3x_val_root),
    )
    sample = train_data[0]
    model = RayDepthResidualModel(
        int(sample["local_hand_features"].shape[-1]),
        int(sample["hand_token_features"].shape[-1]),
        int(sample["hand_token_metadata"].shape[-1]),
        args,
    )
    device = torch.device(args.device)
    model.to(device)
    if args.data_parallel:
        model = nn.DataParallel(model)

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_total = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====", flush=True)
        train_metrics = run_epoch(
            model, train_loader, device, args, optimizer
        )
        val_metrics = run_epoch(model, val_loader, device, args)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "epoch": epoch,
            "model_version": MODEL_VERSION,
            "model_state": (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "args": vars(args),
            "local_hand_dim": int(sample["local_hand_features"].shape[-1]),
            "pi3x_feature_dim": int(
                sample["hand_token_features"].shape[-1]
            ),
            "pi3x_metadata_dim": int(
                sample["hand_token_metadata"].shape[-1]
            ),
            "val_total": val_metrics["total"],
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["total"] < best_total:
            best_total = val_metrics["total"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
