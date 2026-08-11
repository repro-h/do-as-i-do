#!/usr/bin/env python3
"""Predict absolute hand depth while preserving Pi3X metric scale."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from functools import lru_cache
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


MODEL_VERSION = "v12_metric_preserving_absolute_hand_depth_v1"
FEATURE_MODES = (
    "normal",
    "decoder_zero",
    "geometry_zero",
    "metric_zero",
    "pi3x_zero",
    "handflow_zero",
    "joint_shuffle",
    "time_reverse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--dense-train-root", required=True)
    parser.add_argument("--dense-val-root", required=True)
    parser.add_argument("--handflow-train-root", required=True)
    parser.add_argument("--handflow-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
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
    parser.add_argument("--initial-depth-m", type=float, default=0.85)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--w-wrist", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--feature-mode", choices=FEATURE_MODES, default="normal"
    )
    return parser.parse_args()


@lru_cache(maxsize=128)
def load_handflow_image_tokens(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "handflow_image_tokens" not in data.files:
            raise KeyError(f"Missing handflow_image_tokens: {path}")
        tokens = np.asarray(data["handflow_image_tokens"], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 512:
        raise ValueError(f"Unexpected HandFlow token shape {tokens.shape}: {path}")
    if not np.isfinite(tokens).all():
        raise ValueError(f"Non-finite HandFlow image tokens: {path}")
    return tokens


class MetricAbsoluteDataset(HandNeighborhoodDataset):
    def __init__(self, *args, handflow_root: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.handflow_root = handflow_root

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        output = super().__getitem__(index)
        row = self.rows[index]
        stream_id = str(row["stream_id"])
        start, end = int(row["start"]), int(row["end"])
        path = self.handflow_root / stream_id / "handflow_camera_result.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        tokens = load_handflow_image_tokens(str(path.resolve()))
        if end > len(tokens):
            raise ValueError(
                f"Window {start}:{end} exceeds token length {len(tokens)}: {path}"
            )
        output["handflow_image_tokens"] = torch.from_numpy(
            tokens[start:end].copy()
        )
        return output


def make_dataset(
    windows: str,
    global_root: str,
    dense_root: str,
    handflow_root: str,
    args: argparse.Namespace,
) -> MetricAbsoluteDataset:
    return MetricAbsoluteDataset(
        Path(windows),
        Path(global_root),
        Path(dense_root),
        args.neighborhood_size,
        args.min_confidence,
        handflow_root=Path(handflow_root),
    )


class MetricPreservingAbsoluteHandDepth(nn.Module):
    """Use HandFlow queries to read metric-preserving local Pi3X tokens."""

    def __init__(
        self,
        decoder_dim: int,
        metric_dim: int,
        metadata_dim: int,
        handflow_dim: int,
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
        dim = args.token_dim
        self.decoder_encoder = nn.Sequential(
            nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, dim)
        )
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        # These scalar paths intentionally have no LayerNorm: their magnitude
        # carries the Pi3X metric scale needed for absolute depth.
        self.radial_depth_encoder = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.metric_scalar_encoder = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.metric_feature_encoder = nn.Sequential(
            nn.LayerNorm(metric_dim), nn.Linear(metric_dim, dim), nn.GELU()
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.handflow_encoder = nn.Sequential(
            nn.LayerNorm(handflow_dim), nn.Linear(handflow_dim, dim), nn.GELU()
        )
        self.joint_embedding = nn.Embedding(num_joints, dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, args.heads, dropout=args.dropout, batch_first=True
        )
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, args.hidden_dim),
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
        nn.init.normal_(self.depth_head[-1].weight, mean=0.0, std=1e-3)
        ratio = args.initial_depth_m / args.max_depth_m
        nn.init.constant_(
            self.depth_head[-1].bias,
            math.log(ratio / (1.0 - ratio)),
        )

    @staticmethod
    def log_positive(value: torch.Tensor) -> torch.Tensor:
        return torch.log(value.clamp_min(1e-4))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        decoder = batch["neighborhood_features"]
        points = batch["neighborhood_points"]
        metadata = batch["neighborhood_metadata"]
        valid = batch["neighborhood_valid"]
        metric_features = batch["metric_window_features"]
        metric_scalar = batch["metric_scalar"]
        handflow = batch["handflow_image_tokens"]
        mode = self.feature_mode

        if mode == "joint_shuffle":
            decoder = torch.roll(decoder, shifts=1, dims=2)
            points = torch.roll(points, shifts=1, dims=2)
            metadata = torch.roll(metadata, shifts=1, dims=2)
            valid = torch.roll(valid, shifts=1, dims=2)
        elif mode == "time_reverse":
            decoder = torch.flip(decoder, dims=(1,))
            points = torch.flip(points, dims=(1,))
            metadata = torch.flip(metadata, dims=(1,))
            valid = torch.flip(valid, dims=(1,))
            metric_features = torch.flip(metric_features, dims=(1,))
            metric_scalar = torch.flip(metric_scalar, dims=(1,))

        radial = torch.linalg.norm(points, dim=-1, keepdim=True)
        direction = points / radial.clamp_min(1e-6)
        decoder_token = self.decoder_encoder(decoder)
        geometry_token = (
            self.direction_encoder(direction)
            + self.radial_depth_encoder(self.log_positive(radial))
        )
        metadata_token = self.metadata_encoder(metadata)
        metric_token = (
            self.metric_feature_encoder(metric_features)
            + self.metric_scalar_encoder(self.log_positive(metric_scalar))
        )
        handflow_token = self.handflow_encoder(handflow)

        if mode in ("decoder_zero", "pi3x_zero"):
            decoder_token = torch.zeros_like(decoder_token)
        if mode in ("geometry_zero", "pi3x_zero"):
            geometry_token = torch.zeros_like(geometry_token)
        if mode in ("metric_zero", "pi3x_zero"):
            metric_token = torch.zeros_like(metric_token)
        if mode == "pi3x_zero":
            metadata_token = torch.zeros_like(metadata_token)
        if mode == "handflow_zero":
            handflow_token = torch.zeros_like(handflow_token)

        neighbor_token = decoder_token + geometry_token + metadata_token
        batch_size, time, joints, neighbors, dim = neighbor_token.shape
        joint_ids = torch.arange(joints, device=neighbor_token.device)
        joint_token = self.joint_embedding(joint_ids).view(1, 1, joints, dim)
        query = handflow_token[:, :, None] + joint_token

        flat_neighbor = neighbor_token.reshape(-1, neighbors, dim)
        flat_valid = valid.reshape(-1, neighbors)
        safe_valid = flat_valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        if empty.any():
            flat_neighbor = flat_neighbor.clone()
            flat_neighbor[empty, 0] = 0.0
        attended, _ = self.cross_attention(
            query.reshape(-1, 1, dim),
            flat_neighbor,
            flat_neighbor,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        attended = attended.reshape(batch_size, time, joints, dim)
        attended = attended * flat_valid.any(dim=1).reshape(
            batch_size, time, joints, 1
        ).to(attended.dtype)

        metric_joint = metric_token[:, :, None].expand(-1, -1, joints, -1)
        frame = self.frame_encoder(torch.cat((attended, query, metric_joint), -1))
        temporal_input = frame.permute(0, 2, 1, 3).reshape(
            batch_size * joints, time, -1
        )
        temporal, _ = self.temporal(temporal_input)
        temporal = temporal.reshape(batch_size, joints, time, -1).permute(
            0, 2, 1, 3
        )
        return torch.sigmoid(self.depth_head(temporal).squeeze(-1)) * self.max_depth


def unique_metrics(records: dict[tuple[int, int], list[dict]]) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for rows in records.values():
        row = rows[0].copy()
        row["predicted_depth"] = float(np.mean([
            value["predicted_depth"] for value in rows
        ]))
        target = row["target_depth"]
        initial = row["initial_depth"]
        predicted = row["predicted_depth"]
        initial_ray = abs(initial - target)
        predicted_ray = abs(predicted - target)
        lateral = row["lateral"]
        values = {
            "initial_ray": initial_ray,
            "predicted_ray": predicted_ray,
            "initial_full": math.hypot(initial_ray, lateral),
            "predicted_full": math.hypot(predicted_ray, lateral),
        }
        groups = ("all", "left" if row["side"] == 0 else "right")
        groups += ("observed" if row["observed"] else "unobserved",)
        for group in groups:
            for name, value in values.items():
                grouped[group][name].append(np.asarray([value]))

    def summarize(values: dict[str, list[np.ndarray]]) -> dict:
        before = np.concatenate(values["initial_full"])
        after = np.concatenate(values["predicted_full"])
        return {
            "initial_translation": distribution(values["initial_full"]),
            "predicted_translation": distribution(values["predicted_full"]),
            "initial_ray_depth": distribution(values["initial_ray"]),
            "predicted_ray_depth": distribution(values["predicted_ray"]),
            "evaluated": int(len(before)),
            "degraded_fraction": float(np.mean(after > before + 1e-6)),
        }

    empty = {
        "initial_full": [], "predicted_full": [],
        "initial_ray": [], "predicted_ray": [],
    }
    return {
        group: summarize(grouped[group]) if grouped[group] else summarize_empty()
        for group in ("all", "left", "right", "observed", "unobserved")
    }


def summarize_empty() -> dict:
    empty = {"count": 0, "median_mm": None, "p90_mm": None, "max_mm": None}
    return {
        "initial_translation": empty,
        "predicted_translation": empty,
        "initial_ray_depth": empty,
        "predicted_ray_depth": empty,
        "evaluated": 0,
        "degraded_fraction": None,
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
    sums = {name: 0.0 for name in (
        "total", "wrist", "joint", "velocity", "acceleration"
    )}
    records: dict[tuple[int, int], list[dict]] = defaultdict(list)
    batches = 0
    iterator = tqdm(loader, desc="train" if training else "val")
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        bad = [
            key for key, value in batch.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if bad:
            raise RuntimeError(f"non-finite batch inputs: {bad}")
        required = (
            "metric_window_features", "metric_scalar", "metric_scalar_valid",
            "handflow_image_tokens",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Training input lacks {missing}")

        pred_joints = batch["pred_joints"]
        target_joints = batch["target_joints"]
        joint_ray = pred_joints / torch.linalg.norm(
            pred_joints, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        initial_joint_depth = torch.linalg.norm(pred_joints, dim=-1)
        target_joint_depth = (target_joints * joint_ray).sum(dim=-1)
        joint_mask = (
            batch["valid"][:, :, None]
            & batch["joint_valid"]
            & (target_joint_depth > 1e-5)
            & batch["metric_scalar_valid"][:, :, None]
        )
        wrist_mask = joint_mask[:, :, 0] & batch["metric_scalar_valid"]

        with torch.set_grad_enabled(training):
            predicted_joint_depth = model(batch)
            wrist_loss = masked_mean(
                smooth_l1(
                    predicted_joint_depth[:, :, 0] - target_joint_depth[:, :, 0],
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                wrist_mask,
            )
            joint_loss = masked_mean(
                smooth_l1(
                    predicted_joint_depth - target_joint_depth,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                joint_mask,
            )
            velocity = temporal_loss(
                predicted_joint_depth[:, :, :1],
                target_joint_depth[:, :, :1],
                wrist_mask,
                1,
                args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration = temporal_loss(
                predicted_joint_depth[:, :, :1],
                target_joint_depth[:, :, :1],
                wrist_mask,
                2,
                args.smooth_l1_beta_mm / 1000.0,
            )
            total = (
                args.w_wrist * wrist_loss
                + args.w_joint * joint_loss
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

        for name, value in zip(
            sums, (total, wrist_loss, joint_loss, velocity, acceleration)
        ):
            sums[name] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        wrist_ray = joint_ray[:, :, 0]
        target_wrist = target_joints[:, :, 0]
        target_depth = target_joint_depth[:, :, 0]
        target_on_ray = target_depth[..., None] * wrist_ray
        lateral = torch.linalg.norm(target_wrist - target_on_ray, dim=-1)
        valid_np = wrist_mask.detach().cpu().numpy()
        arrays = {
            "stream": batch["stream_index"].detach().cpu().numpy(),
            "frame": batch["frame_index"].detach().cpu().numpy(),
            "side": batch["side"].detach().cpu().numpy(),
            "observed": batch["observed"].detach().cpu().numpy(),
            "initial": initial_joint_depth[:, :, 0].detach().cpu().numpy(),
            "target": target_depth.detach().cpu().numpy(),
            "predicted": predicted_joint_depth[:, :, 0].detach().cpu().numpy(),
            "lateral": lateral.detach().cpu().numpy(),
        }
        for b, t in zip(*np.nonzero(valid_np)):
            key = (int(arrays["stream"][b, t]), int(arrays["frame"][b, t]))
            records[key].append({
                "side": int(arrays["side"][b, t]),
                "observed": bool(arrays["observed"][b, t]),
                "initial_depth": float(arrays["initial"][b, t]),
                "target_depth": float(arrays["target"][b, t]),
                "predicted_depth": float(arrays["predicted"][b, t]),
                "lateral": float(arrays["lateral"][b, t]),
            })

    metrics = unique_metrics(records)
    return {
        **{name: value / max(batches, 1) for name, value in sums.items()},
        **metrics["all"],
        "by_side": {"left": metrics["left"], "right": metrics["right"]},
        "by_visibility": {
            "observed": metrics["observed"],
            "unobserved": metrics["unobserved"],
        },
    }


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
        "decoder_feature_dim": int(sample["neighborhood_features"].shape[-1]),
        "metric_feature_dim": int(sample["metric_window_features"].shape[-1]),
        "metadata_dim": int(sample["neighborhood_metadata"].shape[-1]),
        "handflow_dim": int(sample["handflow_image_tokens"].shape[-1]),
        "num_joints": int(sample["neighborhood_features"].shape[1]),
        "output": "absolute_joint_and_wrist_ray_depth",
        "initial_pose_usage": "2d_sampling_ray_and_output_composition_only",
        "explicit_hand_depth_input": False,
        "pi3x_metric_scale_preserved": True,
        "val": val,
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
        args.train_windows, args.global_train_root, args.dense_train_root,
        args.handflow_train_root, args,
    )
    val_data = make_dataset(
        args.val_windows, args.global_val_root, args.dense_val_root,
        args.handflow_val_root, args,
    )
    sample = train_data[0]
    required = (
        "neighborhood_features", "neighborhood_points",
        "metric_window_features", "metric_scalar", "metric_scalar_valid",
        "neighborhood_metadata", "handflow_image_tokens",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"Training input lacks {missing}")
    audit = {
        "model": MODEL_VERSION,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "decoder_feature_shape": list(sample["neighborhood_features"].shape),
        "point_shape": list(sample["neighborhood_points"].shape),
        "metric_feature_shape": list(sample["metric_window_features"].shape),
        "metric_scalar_shape": list(sample["metric_scalar"].shape),
        "handflow_image_shape": list(sample["handflow_image_tokens"].shape),
        "explicit_hand_depth_input": False,
        "pi3x_metric_scale_preserved": True,
    }
    print(json.dumps(audit, indent=2), flush=True)
    model = MetricPreservingAbsoluteHandDepth(
        int(sample["neighborhood_features"].shape[-1]),
        int(sample["metric_window_features"].shape[-1]),
        int(sample["neighborhood_metadata"].shape[-1]),
        int(sample["handflow_image_tokens"].shape[-1]),
        int(sample["neighborhood_features"].shape[1]),
        args,
    )
    if args.audit_only:
        model.eval()
        with torch.no_grad():
            output = model({key: value.unsqueeze(0) for key, value in sample.items()})
        print(json.dumps({
            "forward_shape": list(output.shape),
            "forward_finite": bool(torch.isfinite(output).all()),
            "forward_min_m": float(output.min()),
            "forward_max_m": float(output.max()),
        }, indent=2), flush=True)
        return

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
        degraded = val["degraded_fraction"]
        if degraded < best_degraded:
            best_degraded = degraded
            torch.save(checkpoint, out_dir / "best_degraded.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
