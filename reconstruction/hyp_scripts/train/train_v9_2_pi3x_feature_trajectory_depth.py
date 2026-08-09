#!/usr/bin/env python3
"""Train a Pi3X feature-trajectory camera-ray depth refiner."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import RelativeCrossAttention
from train_v9_camera_hand_residual import (
    KEY_JOINTS,
    load_jsonl,
    load_npz,
    masked_mean,
    scalar_text,
    smooth_l1,
    temporal_loss,
)


MODEL_VERSION = "v9_2_pi3x_feature_trajectory_ray_depth_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pi3x-relation-dim", type=int, default=128)
    parser.add_argument("--pi3x-heads", type=int, default=8)
    parser.add_argument("--max-ray-correction-mm", type=float, default=120.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--small-anchor-mm", type=float, default=5.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-trajectory", type=float, default=0.2)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-residual", type=float, default=0.001)
    parser.add_argument("--w-small-anchor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


def finite_float(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        value, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


class FeatureTrajectoryDataset(Dataset):
    """Load HandFlow and Pi3X tokens in the original camera coordinate frame."""

    def __init__(self, windows: Path, global_root: Path, pi3x_root: Path):
        self.rows = load_jsonl(windows)
        self.global_root = global_root
        self.pi3x_root = pi3x_root
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        stream_id = str(row["stream_id"])
        start, end = int(row["start"]), int(row["end"])
        se3 = load_npz(str(Path(row["supervision_npz"]).resolve()))
        global_path = Path(
            scalar_text(se3["source_global_supervision"])
        ).expanduser().resolve()
        if not global_path.is_file():
            global_path = self.global_root / f"{stream_id}.npz"
        glob = load_npz(str(global_path))

        pred_joints = np.asarray(
            glob["pred_joints_3d"], dtype=np.float32
        )[start:end].copy()
        gt_joints = np.asarray(
            glob["gt_joints_3d"], dtype=np.float32
        )[start:end].copy()
        normalized_left = bool(
            np.asarray(glob.get("normalized_left", False)).item()
        )
        hand_side = scalar_text(glob["hand_side"])
        if normalized_left:
            # Pi3X cache is extracted from the original RGB. Undo the mirrored
            # left-hand supervision so every input shares that camera frame.
            pred_joints[..., 0] *= -1.0
            gt_joints[..., 0] *= -1.0

        initial_t = pred_joints[:, 0]
        target_t = gt_joints[:, 0]
        valid = (
            np.asarray(glob["hand_valid"], dtype=bool)[start:end]
            & np.asarray(glob["gt_valid"], dtype=bool)[start:end]
            & np.asarray(glob["supervision_valid"], dtype=bool)[start:end]
            & np.isfinite(initial_t).all(axis=-1)
            & np.isfinite(target_t).all(axis=-1)
        )
        local = pred_joints[:, KEY_JOINTS] - pred_joints[:, 0, None]
        local = finite_float((local / 0.1).reshape(len(local), -1))

        cache_path = (
            self.pi3x_root
            / stream_id
            / "pi3x_geometry_features_compact.npz"
        )
        cache = load_npz(str(cache_path.resolve()))
        frame_indices = np.asarray(cache["frame_indices"], dtype=np.int64)
        expected = np.arange(start, end, dtype=np.int64)
        positions = np.searchsorted(frame_indices, expected)
        covered = (
            np.all(positions < len(frame_indices))
            and np.array_equal(frame_indices[positions], expected)
        )
        if not covered:
            raise ValueError(f"{cache_path} does not cover [{start}, {end})")
        grid = np.asarray(cache["geometry_feature_grid_hw"], dtype=np.float32)

        def tokens(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            features = finite_float(cache[f"{prefix}_features"][positions])
            points = finite_float(cache[f"{prefix}_points"][positions])
            indices = np.asarray(
                cache[f"{prefix}_indices"][positions], dtype=np.float32
            ).copy()
            indices[..., 0] /= max(float(grid[0] - 1), 1.0)
            indices[..., 1] /= max(float(grid[1] - 1), 1.0)
            coverage = finite_float(cache[f"{prefix}_coverage"][positions])
            confidence = finite_float(
                cache[f"{prefix}_confidence"][positions]
            )
            metadata = np.concatenate(
                (points, indices, coverage[..., None], confidence[..., None]),
                axis=-1,
            )
            valid_tokens = np.asarray(
                cache[f"{prefix}_valid"][positions], dtype=bool
            )
            return features, finite_float(metadata), valid_tokens

        hand_f, hand_m, hand_v = tokens("hand")
        object_f, object_m, object_v = tokens("object")
        return {
            "local_hand_features": torch.from_numpy(local),
            "hand_token_features": torch.from_numpy(hand_f),
            "hand_token_metadata": torch.from_numpy(hand_m),
            "hand_token_valid": torch.from_numpy(hand_v),
            "object_token_features": torch.from_numpy(object_f),
            "object_token_metadata": torch.from_numpy(object_m),
            "object_token_valid": torch.from_numpy(object_v),
            "initial_t": torch.from_numpy(finite_float(initial_t)),
            "target_t": torch.from_numpy(finite_float(target_t)),
            "valid": torch.from_numpy(valid),
            "side": torch.full(
                (end - start,), 0 if hand_side == "left" else 1
            ),
        }


def temporal_differences(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = torch.zeros_like(value)
    velocity[:, 1:] = value[:, 1:] - value[:, :-1]
    acceleration = torch.zeros_like(value)
    acceleration[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]
    return velocity, acceleration


class FeatureTrajectoryDepthModel(nn.Module):
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
            nn.Linear(local_dim, args.hidden_dim // 2),
            nn.GELU(),
        )
        fusion_dim = args.pi3x_relation_dim * 3 + args.hidden_dim // 2 + 3
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.temporal = nn.GRU(
            args.hidden_dim,
            args.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.anchor_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        for head in (self.anchor_head, self.trajectory_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        key_types = torch.zeros_like(
            batch["object_token_valid"], dtype=torch.long
        )
        relation = self.relation(
            batch["hand_token_features"],
            batch["hand_token_metadata"],
            batch["hand_token_valid"],
            batch["object_token_features"],
            batch["object_token_metadata"],
            batch["object_token_valid"],
            key_types,
        )
        relation_velocity, relation_acceleration = temporal_differences(relation)

        depth = torch.linalg.norm(batch["initial_t"], dim=-1)
        masked_depth = depth.masked_fill(~batch["valid"], float("nan"))
        depth_median = torch.nanmedian(
            masked_depth, dim=1, keepdim=True
        ).values.nan_to_num()
        depth_relative = (depth - depth_median).masked_fill(
            ~batch["valid"], 0.0
        )
        depth_velocity, depth_acceleration = temporal_differences(
            depth_relative[..., None]
        )
        depth_trajectory = torch.cat(
            (depth_relative[..., None], depth_velocity, depth_acceleration),
            dim=-1,
        )
        frame = torch.cat(
            (
                relation,
                relation_velocity,
                relation_acceleration,
                self.local_encoder(batch["local_hand_features"]),
                depth_trajectory,
            ),
            dim=-1,
        )
        encoded = self.frame_encoder(frame)
        temporal, _ = self.temporal(encoded)
        anchor = self.anchor_head(temporal.mean(dim=1)).squeeze(-1)
        trajectory = self.trajectory_head(temporal).squeeze(-1)
        trajectory = trajectory - trajectory.mean(dim=1, keepdim=True)
        return torch.tanh(anchor[:, None] + trajectory) * self.max_correction


def distribution(values: list[np.ndarray]) -> dict:
    array = np.concatenate(values) if values else np.empty(0)
    return {
        "count": int(array.size),
        "median_mm": float(np.median(array) * 1000.0) if array.size else None,
        "p90_mm": float(np.percentile(array, 90) * 1000.0) if array.size else None,
        "max_mm": float(np.max(array) * 1000.0) if array.size else None,
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
    names = ("total", "depth", "trajectory", "acceleration", "residual", "small_anchor")
    sums = {name: 0.0 for name in names}
    metric_names = ("initial_full", "corrected_full", "initial_ray", "corrected_ray")
    metrics = {name: [] for name in metric_names}
    side_metrics = {
        side: {name: [] for name in metric_names} for side in ("left", "right")
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
        initial_t, target_t, valid = (
            batch["initial_t"],
            batch["target_t"],
            batch["valid"],
        )
        ray = initial_t / torch.linalg.norm(
            initial_t, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        target_ray = ((target_t - initial_t) * ray).sum(dim=-1)

        with torch.set_grad_enabled(training):
            predicted_ray = model(batch)
            corrected_t = initial_t + predicted_ray[..., None] * ray
            depth_loss = masked_mean(
                smooth_l1(
                    predicted_ray - target_ray,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                valid,
            )
            trajectory_loss = temporal_loss(
                predicted_ray[..., None],
                target_ray[..., None],
                valid,
                1,
                args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration_loss = temporal_loss(
                predicted_ray[..., None],
                target_ray[..., None],
                valid,
                2,
                args.smooth_l1_beta_mm / 1000.0,
            )
            residual_loss = masked_mean(
                smooth_l1(predicted_ray, 0.02), valid
            )
            small = valid & (
                target_ray.abs() <= args.small_anchor_mm / 1000.0
            )
            small_anchor = masked_mean(
                smooth_l1(predicted_ray, 0.005), small
            )
            total = (
                args.w_depth * depth_loss
                + args.w_trajectory * trajectory_loss
                + args.w_acceleration * acceleration_loss
                + args.w_residual * residual_loss
                + args.w_small_anchor * small_anchor
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        loss_values = (
            total,
            depth_loss,
            trajectory_loss,
            acceleration_loss,
            residual_loss,
            small_anchor,
        )
        for name, value in zip(names, loss_values):
            sums[name] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        initial_full = torch.linalg.norm(initial_t - target_t, dim=-1)
        corrected_full = torch.linalg.norm(corrected_t - target_t, dim=-1)
        values = {
            "initial_full": initial_full,
            "corrected_full": corrected_full,
            "initial_ray": target_ray.abs(),
            "corrected_ray": (predicted_ray - target_ray).abs(),
        }
        valid_np = valid.detach().cpu().numpy().astype(bool)
        side_np = batch["side"].detach().cpu().numpy()
        for name, value in values.items():
            array = value.detach().cpu().numpy()
            metrics[name].append(array[valid_np])
            for side_name, side_value in (("left", 0), ("right", 1)):
                mask = valid_np & (side_np == side_value)
                side_metrics[side_name][name].append(array[mask])
        before = initial_full.detach().cpu().numpy()[valid_np]
        after = corrected_full.detach().cpu().numpy()[valid_np]
        improved += int((after < before).sum())
        degraded += int((after > before).sum())
        evaluated += len(before)

    def summarize(source: dict[str, list[np.ndarray]]) -> dict:
        return {
            "initial_translation": distribution(source["initial_full"]),
            "corrected_translation": distribution(source["corrected_full"]),
            "initial_ray_depth": distribution(source["initial_ray"]),
            "corrected_ray_depth": distribution(source["corrected_ray"]),
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data = FeatureTrajectoryDataset(
        Path(args.train_windows),
        Path(args.global_train_root),
        Path(args.pi3x_train_root),
    )
    val_data = FeatureTrajectoryDataset(
        Path(args.val_windows),
        Path(args.global_val_root),
        Path(args.pi3x_val_root),
    )
    sample = train_data[0]
    model = FeatureTrajectoryDepthModel(
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
            "pi3x_feature_dim": int(sample["hand_token_features"].shape[-1]),
            "pi3x_metadata_dim": int(sample["hand_token_metadata"].shape[-1]),
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
