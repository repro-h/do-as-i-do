#!/usr/bin/env python3
"""Predict absolute camera-ray wrist depth from Pi3X features only."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_v10_pi3x_hand_neighborhood_depth import (
    HandNeighborhoodDataset,
    disable_mha_fastpath,
)
from train_v9_2_pi3x_feature_trajectory_depth import distribution
from train_v9_camera_hand_residual import (
    masked_mean,
    smooth_l1,
    temporal_loss,
)


MODEL_VERSION = "v11_pi3x_metric_absolute_ray_depth_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--dense-train-root", required=True)
    parser.add_argument("--dense-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=2.5)
    parser.add_argument("--initial-depth-m", type=float, default=0.8)
    parser.add_argument(
        "--use-metric-scalar",
        action="store_true",
        help="Also condition on Pi3X's final metric scalar.",
    )
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--feature-mode",
        choices=(
            "normal",
            "point_zero",
            "metric_zero",
            "scalar_zero",
            "all_zero",
            "spatial_shuffle",
            "time_reverse",
        ),
        default="normal",
    )
    return parser.parse_args()


class Pi3XMetricAbsoluteDepthModel(nn.Module):
    def __init__(
        self,
        point_dim: int,
        metric_dim: int,
        metadata_dim: int,
        num_joints: int,
        args: argparse.Namespace,
    ):
        super().__init__()
        if args.hidden_dim % 2:
            raise ValueError("hidden-dim must be even")
        if not 0.0 < args.initial_depth_m < args.max_depth_m:
            raise ValueError("initial-depth-m must be in (0, max-depth-m)")
        self.max_depth = float(args.max_depth_m)
        self.feature_mode = args.feature_mode
        self.use_metric_scalar = bool(args.use_metric_scalar)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_dim),
            nn.Linear(point_dim, args.token_dim),
        )
        self.metric_encoder = nn.Sequential(
            nn.LayerNorm(metric_dim),
            nn.Linear(metric_dim, args.token_dim),
        )
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, args.token_dim),
            nn.GELU(),
            nn.Linear(args.token_dim, args.token_dim),
        )
        if self.use_metric_scalar:
            self.metric_scalar_encoder = nn.Sequential(
                nn.LayerNorm(1),
                nn.Linear(1, args.token_dim),
                nn.GELU(),
                nn.Linear(args.token_dim, args.token_dim),
            )
        else:
            self.metric_scalar_encoder = None
        self.fusion = nn.Sequential(
            nn.LayerNorm(
                args.token_dim * (3 if self.use_metric_scalar else 2)
            ),
            nn.Linear(
                args.token_dim * (3 if self.use_metric_scalar else 2),
                args.token_dim,
            ),
            nn.GELU(),
        )
        self.joint_embedding = nn.Embedding(num_joints, args.token_dim)
        self.local_score = nn.Sequential(
            nn.LayerNorm(args.token_dim),
            nn.Linear(args.token_dim, args.token_dim // 2),
            nn.GELU(),
            nn.Linear(args.token_dim // 2, 1),
        )
        encoder = nn.TransformerEncoderLayer(
            d_model=args.token_dim,
            nhead=args.heads,
            dim_feedforward=args.token_dim * 2,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_encoder = nn.TransformerEncoder(encoder, num_layers=1)
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(args.token_dim * 2),
            nn.Linear(args.token_dim * 2, args.hidden_dim),
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
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.depth_head[-1].weight)
        ratio = args.initial_depth_m / args.max_depth_m
        nn.init.constant_(
            self.depth_head[-1].bias,
            math.log(ratio / (1.0 - ratio)),
        )

    @staticmethod
    def scalar_feature(value: torch.Tensor) -> torch.Tensor:
        return torch.sign(value) * torch.log1p(value.abs())

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        point = batch["neighborhood_features"]
        metric = batch["metric_neighborhood_features"]
        scalar = batch.get("metric_scalar")
        mode = self.feature_mode
        if mode in ("point_zero", "all_zero"):
            point = torch.zeros_like(point)
        if mode in ("metric_zero", "all_zero"):
            metric = torch.zeros_like(metric)
        if scalar is not None and mode in ("scalar_zero", "all_zero"):
            scalar = torch.zeros_like(scalar)
        if mode == "spatial_shuffle":
            point = torch.roll(point, shifts=7, dims=3)
            metric = torch.roll(metric, shifts=7, dims=3)
        if mode == "time_reverse":
            point = torch.flip(point, dims=(1,))
            metric = torch.flip(metric, dims=(1,))
            if scalar is not None:
                scalar = torch.flip(scalar, dims=(1,))

        inputs = [
            self.point_encoder(point),
            self.metric_encoder(metric),
        ]
        if self.use_metric_scalar:
            if scalar is None or self.metric_scalar_encoder is None:
                raise KeyError("metric_scalar is required by this checkpoint")
            scalar_token = self.metric_scalar_encoder(
                self.scalar_feature(scalar)
            )
            inputs.append(
                scalar_token[:, :, None, None].expand(
                    *point.shape[:-1], scalar_token.shape[-1]
                )
            )
        token = self.fusion(torch.cat(inputs, dim=-1))
        token = token + self.metadata_encoder(
            batch["neighborhood_metadata"]
        )
        joint_ids = torch.arange(
            token.shape[2], device=token.device
        ).view(1, 1, -1, 1)
        token = token + self.joint_embedding(joint_ids)

        valid = batch["neighborhood_valid"]
        score = self.local_score(token).squeeze(-1)
        score = score.masked_fill(~valid, -1e4)
        weight = torch.softmax(score, dim=3) * valid.to(score.dtype)
        weight = weight / weight.sum(dim=3, keepdim=True).clamp_min(1e-6)
        joint = (token * weight[..., None]).sum(dim=3)

        batch_size, time, joints, dim = joint.shape
        joint_valid = valid.any(dim=3).reshape(batch_size * time, joints)
        safe_valid = joint_valid.clone()
        safe_valid[~safe_valid.any(dim=1), 0] = True
        encoded = self.joint_encoder(
            joint.reshape(batch_size * time, joints, dim),
            src_key_padding_mask=~safe_valid,
        ).reshape(batch_size, time, joints, dim)
        joint_weight = joint_valid.reshape(
            batch_size, time, joints
        ).to(encoded.dtype)
        pooled = (encoded * joint_weight[..., None]).sum(dim=2)
        pooled = pooled / joint_weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        wrist = encoded[:, :, 0]
        temporal, _ = self.temporal(
            self.frame_encoder(torch.cat((wrist, pooled), dim=-1))
        )
        return torch.sigmoid(
            self.depth_head(temporal).squeeze(-1)
        ) * self.max_depth


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    names = ("total", "depth", "velocity", "acceleration")
    sums = {name: 0.0 for name in names}
    metric_names = (
        "initial_full", "predicted_full",
        "initial_ray", "predicted_ray",
        "absolute_target", "absolute_prediction",
    )
    metrics = {name: [] for name in metric_names}
    side_metrics = {
        side: {name: [] for name in metric_names}
        for side in ("left", "right")
    }
    improved = degraded = evaluated = batches = 0
    iterator = tqdm(loader, desc="train" if training else "val")
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        required = ("metric_neighborhood_features",)
        if args.use_metric_scalar:
            required += ("metric_scalar", "metric_scalar_valid")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                f"Dense cache lacks Pi3X metric features: {missing}. "
                "Re-export with --export-metric-features."
            )
        bad = [
            key for key, value in batch.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if bad:
            raise RuntimeError(f"non-finite batch inputs: {bad}")

        initial_t, target_t = batch["initial_t"], batch["target_t"]
        valid = batch["valid"] & batch["observed"]
        ray = initial_t / torch.linalg.norm(
            initial_t, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        initial_depth = torch.linalg.norm(initial_t, dim=-1)
        target_depth = (target_t * ray).sum(dim=-1)
        valid &= target_depth > 1e-5

        with torch.set_grad_enabled(training):
            predicted_depth = model(batch)
            predicted_t = predicted_depth[..., None] * ray
            depth = masked_mean(
                smooth_l1(
                    predicted_depth - target_depth,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                valid,
            )
            velocity = temporal_loss(
                predicted_depth[..., None], target_depth[..., None],
                valid, 1, args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration = temporal_loss(
                predicted_depth[..., None], target_depth[..., None],
                valid, 2, args.smooth_l1_beta_mm / 1000.0,
            )
            total = (
                args.w_depth * depth
                + args.w_velocity * velocity
                + args.w_acceleration * acceleration
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for name, value in zip(names, (total, depth, velocity, acceleration)):
            sums[name] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        values = {
            "initial_full": torch.linalg.norm(initial_t - target_t, dim=-1),
            "predicted_full": torch.linalg.norm(predicted_t - target_t, dim=-1),
            "initial_ray": (initial_depth - target_depth).abs(),
            "predicted_ray": (predicted_depth - target_depth).abs(),
            "absolute_target": target_depth,
            "absolute_prediction": predicted_depth,
        }
        valid_np = valid.detach().cpu().numpy().astype(bool)
        side_np = batch["side"].detach().cpu().numpy()
        for name, value in values.items():
            array = value.detach().cpu().numpy()
            metrics[name].append(array[valid_np])
            for side_name, side_value in (("left", 0), ("right", 1)):
                mask = valid_np & (side_np == side_value)
                side_metrics[side_name][name].append(array[mask])
        before = values["initial_full"].detach().cpu().numpy()[valid_np]
        after = values["predicted_full"].detach().cpu().numpy()[valid_np]
        improved += int((after < before).sum())
        degraded += int((after > before + 1e-6).sum())
        evaluated += len(before)

    def summarize(source: dict[str, list[np.ndarray]]) -> dict:
        return {
            "initial_translation": distribution(source["initial_full"]),
            "predicted_translation": distribution(source["predicted_full"]),
            "initial_ray_depth": distribution(source["initial_ray"]),
            "predicted_ray_depth": distribution(source["predicted_ray"]),
            "absolute_target_depth": distribution(source["absolute_target"]),
            "absolute_predicted_depth": distribution(
                source["absolute_prediction"]
            ),
        }

    return {
        **{name: value / max(batches, 1) for name, value in sums.items()},
        **summarize(metrics),
        "by_side": {
            side: summarize(source) for side, source in side_metrics.items()
        },
        "evaluated": evaluated,
        "improved": improved,
        "degraded": degraded,
        "degraded_fraction": degraded / max(evaluated, 1),
    }


def make_dataset(
    windows: str,
    global_root: str,
    dense_root: str,
    args: argparse.Namespace,
) -> HandNeighborhoodDataset:
    return HandNeighborhoodDataset(
        Path(windows), Path(global_root), Path(dense_root),
        args.neighborhood_size, args.min_confidence,
    )


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    args: argparse.Namespace,
    sample: dict[str, torch.Tensor],
    val: dict,
) -> dict:
    return {
        "epoch": epoch,
        "model_version": MODEL_VERSION,
        "model_state": (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        ),
        "args": vars(args),
        "point_feature_dim": int(sample["neighborhood_features"].shape[-1]),
        "metric_feature_dim": int(
            sample["metric_neighborhood_features"].shape[-1]
        ),
        "metadata_dim": int(sample["neighborhood_metadata"].shape[-1]),
        "num_joints": int(sample["neighborhood_features"].shape[1]),
        "initial_pose_usage": "2d_patch_localization_and_output_ray_only",
        "uses_metric_scalar": bool(args.use_metric_scalar),
        "val_total": val["total"],
        "val_ray_median_mm": val["predicted_ray_depth"]["median_mm"],
        "val_degraded_fraction": val["degraded_fraction"],
    }


def main() -> None:
    args = parse_args()
    disable_mha_fastpath()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data = make_dataset(
        args.train_windows, args.global_train_root,
        args.dense_train_root, args,
    )
    val_data = make_dataset(
        args.val_windows, args.global_val_root,
        args.dense_val_root, args,
    )
    sample = train_data[0]
    required = ("metric_neighborhood_features",)
    if args.use_metric_scalar:
        required += ("metric_scalar", "metric_scalar_valid")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(
            f"Dense cache lacks {missing}; re-export with "
            "--export-metric-features"
        )
    audit = {
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "point_feature_shape": list(sample["neighborhood_features"].shape),
        "metric_feature_shape": list(
            sample["metric_neighborhood_features"].shape
        ),
        "metric_scalar_shape": list(sample["metric_scalar"].shape),
        "uses_metric_scalar": bool(args.use_metric_scalar),
        "valid_tokens": int(sample["neighborhood_valid"].sum()),
        "valid_frames": int(sample["valid"].sum()),
        "initial_pose_usage": "2d_patch_localization_and_output_ray_only",
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    model = Pi3XMetricAbsoluteDepthModel(
        int(sample["neighborhood_features"].shape[-1]),
        int(sample["metric_neighborhood_features"].shape[-1]),
        int(sample["neighborhood_metadata"].shape[-1]),
        int(sample["neighborhood_features"].shape[1]),
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
    history = []
    best_total = best_ray = best_degraded = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====", flush=True)
        train = run_epoch(model, train_loader, device, args, optimizer)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val = run_epoch(model, val_loader, device, args)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train,
            "val": val,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = checkpoint_payload(model, epoch, args, sample, val)
        torch.save(checkpoint, out_dir / "last.pt")
        if val["total"] < best_total:
            best_total = val["total"]
            torch.save(checkpoint, out_dir / "best.pt")
        ray = val["predicted_ray_depth"]["median_mm"]
        if ray < best_ray:
            best_ray = ray
            torch.save(checkpoint, out_dir / "best_ray.pt")
        if val["degraded_fraction"] < best_degraded:
            best_degraded = val["degraded_fraction"]
            torch.save(checkpoint, out_dir / "best_degraded.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
