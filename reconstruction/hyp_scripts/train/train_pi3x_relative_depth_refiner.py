#!/usr/bin/env python3
"""Train a Pi3X-conditioned hand relative-depth refiner."""

from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PALM = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
TOKEN_GROUPS = ("hand", "object", "context")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-correction-mm", type=float, default=60.0)
    parser.add_argument("--max-target-mm", type=float, default=120.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-wrist", type=float, default=0.25)
    parser.add_argument("--w-projection", type=float, default=0.1)
    parser.add_argument("--w-velocity", type=float, default=0.5)
    parser.add_argument("--w-acceleration", type=float, default=1.0)
    parser.add_argument("--w-residual", type=float, default=0.05)
    parser.add_argument("--w-gate", type=float, default=0.5)
    parser.add_argument("--w-sign", type=float, default=0.5)
    parser.add_argument("--w-gate-temporal", type=float, default=0.1)
    parser.add_argument("--gate-zero-error-mm", type=float, default=15.0)
    parser.add_argument("--gate-full-error-mm", type=float, default=25.0)
    parser.add_argument("--sign-valid-threshold-mm", type=float, default=15.0)
    parser.add_argument("--accurate-anchor-mm", type=float, default=15.0)
    parser.add_argument("--w-accurate-anchor", type=float, default=1.0)
    parser.add_argument("--error-weight-reference-mm", type=float, default=20.0)
    parser.add_argument("--error-weight-min", type=float, default=0.5)
    parser.add_argument("--error-weight-max", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="do-as-i-do-pi3x-depth")
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=16)
def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, :, :2].transpose(0, 2, 1).reshape(len(matrix), 6)


class Pi3XWindowDataset(Dataset):
    def __init__(
        self,
        windows: Path,
        pi3x_root: Path,
        max_target_m: float,
    ):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.pi3x_root = pi3x_root
        self.max_target_m = max_target_m

    def __len__(self) -> int:
        return len(self.rows)

    def pi3x_path(self, stream_id: str) -> Path:
        return (
            self.pi3x_root
            / stream_id
            / "pi3x_geometry_features_compact.npz"
        )

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        supervision = load_npz(row["supervision_npz"])
        pi3x_path = self.pi3x_path(row["stream_id"])
        if not pi3x_path.is_file():
            raise FileNotFoundError(pi3x_path)
        pi3x = load_npz(str(pi3x_path))

        frame_indices = np.asarray(pi3x["frame_indices"], dtype=np.int64)
        expected = np.arange(start, end, dtype=np.int64)
        positions = np.searchsorted(frame_indices, expected)
        if (
            np.any(positions >= len(frame_indices))
            or not np.array_equal(frame_indices[positions], expected)
        ):
            raise ValueError(
                f"{pi3x_path} does not cover frames [{start}, {end})"
            )

        pred = np.asarray(
            supervision["pred_joints_3d"][start:end], dtype=np.float32
        )
        gt3d = np.asarray(
            supervision["gt_joints_3d"][start:end], dtype=np.float32
        )
        gt2d = np.asarray(
            supervision["gt_joints_2d"][start:end], dtype=np.float32
        )
        object_pose = np.asarray(
            supervision["object_pose"][start:end], dtype=np.float32
        )
        valid = np.asarray(
            supervision["supervision_valid"][start:end]
        ).astype(bool)
        wrist_error = np.linalg.norm(gt3d[:, 0] - pred[:, 0], axis=-1)
        valid &= np.isfinite(wrist_error) & (
            wrist_error <= self.max_target_m
        )

        pred = np.nan_to_num(pred)
        gt3d = np.nan_to_num(gt3d)
        gt2d = np.nan_to_num(gt2d)
        object_pose = np.nan_to_num(object_pose)
        wrist = pred[:, 0]
        object_center = object_pose[:, :3, 3]
        object_rotation = object_pose[:, :3, :3]
        camera_ray = wrist / np.maximum(
            np.linalg.norm(wrist, axis=-1, keepdims=True), 1e-8
        )
        hand_velocity = np.zeros_like(wrist)
        object_velocity = np.zeros_like(object_center)
        hand_velocity[1:] = wrist[1:] - wrist[:-1]
        object_velocity[1:] = object_center[1:] - object_center[:-1]
        relative = wrist - object_center
        relative_velocity = hand_velocity - object_velocity
        palm_local = pred[:, PALM] - wrist[:, None]
        object_extents = np.asarray(
            supervision["object_extents_metric"], dtype=np.float32
        ).reshape(3)
        if not np.isfinite(object_extents).all() or np.any(
            object_extents <= 1e-5
        ):
            raise ValueError(
                f"{row['supervision_npz']} has invalid object extents: "
                f"{object_extents.tolist()}"
            )
        safe_extents = np.maximum(object_extents, 1e-3)
        object_local_wrist = np.einsum(
            "ti,tij->tj", wrist - object_center, object_rotation
        )
        object_local_palm = np.einsum(
            "tki,tij->tkj",
            pred[:, PALM] - object_center[:, None],
            object_rotation,
        )
        scalar = np.concatenate(
            [
                wrist,
                palm_local.reshape(len(wrist), -1),
                hand_velocity,
                object_center,
                object_velocity,
                relative,
                relative_velocity,
                rotation_6d(object_rotation),
                camera_ray,
                np.broadcast_to(
                    object_extents / 0.2, (len(wrist), 3)
                ),
                object_local_wrist / safe_extents,
                (object_local_palm / safe_extents).reshape(len(wrist), -1),
                valid[:, None],
            ],
            axis=-1,
        ).astype(np.float32)

        token_features, token_metadata, token_valid, token_types = [], [], [], []
        mirrored = bool(np.asarray(supervision["normalized_left"]).item())
        for type_index, prefix in enumerate(TOKEN_GROUPS):
            feature = np.asarray(
                pi3x[f"{prefix}_features"][positions], dtype=np.float32
            )
            points = np.asarray(
                pi3x[f"{prefix}_points"][positions], dtype=np.float32
            )
            if mirrored:
                points = points.copy()
                points[..., 0] *= -1.0
            coverage = np.asarray(
                pi3x[f"{prefix}_coverage"][positions], dtype=np.float32
            )
            confidence = np.asarray(
                pi3x[f"{prefix}_confidence"][positions], dtype=np.float32
            )
            indices = np.asarray(
                pi3x[f"{prefix}_indices"][positions], dtype=np.float32
            )
            grid = np.asarray(
                pi3x["geometry_feature_grid_hw"], dtype=np.float32
            )
            indices[..., 0] /= max(float(grid[0] - 1), 1.0)
            indices[..., 1] /= max(float(grid[1] - 1), 1.0)
            metadata = np.concatenate(
                [
                    points,
                    coverage[..., None],
                    confidence[..., None],
                    indices,
                ],
                axis=-1,
            )
            group_valid = np.asarray(
                pi3x[f"{prefix}_valid"][positions]
            ).astype(bool)
            token_features.append(feature)
            token_metadata.append(metadata)
            token_valid.append(group_valid)
            token_types.append(
                np.full(group_valid.shape, type_index, dtype=np.int64)
            )

        return {
            "scalar": torch.from_numpy(scalar),
            "token_features": torch.from_numpy(
                np.concatenate(token_features, axis=1)
            ),
            "token_metadata": torch.from_numpy(
                np.concatenate(token_metadata, axis=1)
            ),
            "token_valid": torch.from_numpy(
                np.concatenate(token_valid, axis=1)
            ),
            "token_types": torch.from_numpy(
                np.concatenate(token_types, axis=1)
            ),
            "pred_joints": torch.from_numpy(pred),
            "gt_joints": torch.from_numpy(gt3d),
            "gt_joints_2d": torch.from_numpy(gt2d),
            "intrinsics": torch.from_numpy(
                np.asarray(supervision["intrinsics"], dtype=np.float32)
            ),
            "valid": torch.from_numpy(valid),
            "stream_id": row["stream_id"],
            "start": start,
            "end": end,
        }


class Pi3XRelativeDepthRefiner(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        feature_dim: int,
        metadata_dim: int,
        hidden_dim: int,
        spatial_layers: int,
        temporal_layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
        )
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_embedding = nn.Embedding(len(TOKEN_GROUPS), hidden_dim)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=spatial_layers
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.frame_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers
        )
        self.position = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        self.depth_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sign_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.depth_head[-1].weight)
        nn.init.zeros_(self.depth_head[-1].bias)
        nn.init.zeros_(self.sign_head[-1].weight)
        nn.init.zeros_(self.sign_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.0)

    @staticmethod
    def masked_pool(
        value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        weight = mask.to(value.dtype).unsqueeze(-1)
        return (value * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)

    def forward(
        self,
        scalar: torch.Tensor,
        token_features: torch.Tensor,
        token_metadata: torch.Tensor,
        token_valid: torch.Tensor,
        token_types: torch.Tensor,
        max_correction: float,
        return_aux: bool = False,
    ):
        batch, frames, tokens, _ = token_features.shape
        token = self.feature_projection(token_features)
        token = token + self.metadata_projection(token_metadata)
        token = token + self.type_embedding(token_types)
        flat = token.reshape(batch * frames, tokens, -1)
        padding = ~token_valid.reshape(batch * frames, tokens)
        encoded = self.spatial_encoder(flat, src_key_padding_mask=padding)
        encoded = encoded.reshape(batch, frames, tokens, -1)
        pooled = []
        for type_index in range(len(TOKEN_GROUPS)):
            mask = token_valid & (token_types == type_index)
            pooled.append(self.masked_pool(encoded, mask))
        frame_token = self.frame_fusion(
            torch.cat([self.scalar_projection(scalar), *pooled], dim=-1)
        )
        frame_token = frame_token + self.position[:, :frames]
        temporal = self.temporal_encoder(frame_token)
        prediction = (
            torch.tanh(self.depth_head(temporal).squeeze(-1))
            * max_correction
        )
        sign_logits = self.sign_head(temporal).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(temporal).squeeze(-1))
        if return_aux:
            return {
                "prediction": prediction,
                "gate": gate,
                "sign_logits": sign_logits,
                "magnitude": torch.abs(prediction),
            }
        return prediction


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.to(value.dtype).expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def smooth_l1(value, target, mask, beta):
    loss = F.smooth_l1_loss(value, target, reduction="none", beta=beta)
    return masked_mean(loss, mask)


def weighted_smooth_l1(value, target, mask, weight, beta):
    loss = F.smooth_l1_loss(value, target, reduction="none", beta=beta)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return masked_mean(loss * weight, mask)


def temporal(value, target, valid, order, beta):
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
        target = target[:, 1:] - target[:, :-1]
        valid = valid[:, 1:] & valid[:, :-1]
    return smooth_l1(value, target, valid, beta)


def project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp_min(1e-4)
    u = intrinsics[:, None, None, 0, 0] * points[..., 0] / z
    u += intrinsics[:, None, None, 0, 2]
    v = intrinsics[:, None, None, 1, 1] * points[..., 1] / z
    v += intrinsics[:, None, None, 1, 2]
    return torch.stack([u, v], dim=-1)


def compute(model, batch, args):
    batch = {
        key: value.to(args.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model_output = model(
        batch["scalar"],
        batch["token_features"],
        batch["token_metadata"],
        batch["token_valid"],
        batch["token_types"],
        args.max_correction_mm / 1000.0,
        return_aux=True,
    )
    ray_depth = model_output["prediction"]
    gate = model_output["gate"]
    sign_logits = model_output["sign_logits"]
    pred = batch["pred_joints"]
    gt = batch["gt_joints"]
    valid = batch["valid"]
    camera_ray = F.normalize(pred[:, :, 0], dim=-1, eps=1e-8)
    target_translation = gt[:, :, 0] - pred[:, :, 0]
    target_depth = torch.sum(target_translation * camera_ray, dim=-1)
    translation = ray_depth.unsqueeze(-1) * camera_ray
    corrected = pred + translation[:, :, None]
    beta = args.smooth_l1_beta_mm / 1000.0
    initial_depth_error = torch.abs(target_depth)
    supervision_weight = (
        initial_depth_error
        / max(args.error_weight_reference_mm / 1000.0, 1e-8)
    ).clamp(args.error_weight_min, args.error_weight_max)
    projection_mask = (
        valid[:, :, None]
        & (corrected[:, :, PALM, 2] > 1e-4)
        & (gt[:, :, PALM, 2] > 1e-4)
    )
    losses = {
        "depth": weighted_smooth_l1(
            ray_depth, target_depth, valid, supervision_weight, beta
        ),
        "wrist": weighted_smooth_l1(
            corrected[:, :, 0],
            gt[:, :, 0],
            valid,
            supervision_weight,
            beta,
        ),
        "projection": smooth_l1(
            project(corrected[:, :, PALM], batch["intrinsics"]) / 100.0,
            batch["gt_joints_2d"][:, :, PALM] / 100.0,
            projection_mask,
            beta,
        ),
        "velocity": temporal(
            corrected[:, :, 0], gt[:, :, 0], valid, 1, beta
        ),
        "acceleration": temporal(
            corrected[:, :, 0], gt[:, :, 0], valid, 2, beta
        ),
        "residual": smooth_l1(
            ray_depth, torch.zeros_like(ray_depth), valid, beta
        ),
    }
    gate_range = max(
        args.gate_full_error_mm - args.gate_zero_error_mm, 1e-6
    )
    gate_target = (
        (initial_depth_error * 1000.0 - args.gate_zero_error_mm)
        / gate_range
    ).clamp(0.0, 1.0)
    gate_loss = F.binary_cross_entropy(
        gate, gate_target, reduction="none"
    )
    losses["gate"] = masked_mean(gate_loss, valid)
    sign_valid = valid & (
        initial_depth_error >= args.sign_valid_threshold_mm / 1000.0
    )
    sign_target = (target_depth > 0.0).to(pred.dtype)
    sign_loss = F.binary_cross_entropy_with_logits(
        sign_logits, sign_target, reduction="none"
    )
    losses["sign"] = masked_mean(sign_loss, sign_valid)
    losses["gate_temporal"] = temporal(
        gate,
        gate_target,
        valid,
        1,
        beta=0.1,
    )
    accurate = valid & (
        initial_depth_error < args.accurate_anchor_mm / 1000.0
    )
    losses["accurate_anchor"] = smooth_l1(
        ray_depth, torch.zeros_like(ray_depth), accurate, beta
    )
    total = (
        args.w_depth * losses["depth"]
        + args.w_wrist * losses["wrist"]
        + args.w_projection * losses["projection"]
        + args.w_velocity * losses["velocity"]
        + args.w_acceleration * losses["acceleration"]
        + args.w_residual * losses["residual"]
        + args.w_gate * losses["gate"]
        + args.w_sign * losses["sign"]
        + args.w_gate_temporal * losses["gate_temporal"]
        + args.w_accurate_anchor * losses["accurate_anchor"]
    )
    before = torch.linalg.norm(pred[:, :, 0] - gt[:, :, 0], dim=-1)
    after = torch.linalg.norm(corrected[:, :, 0] - gt[:, :, 0], dim=-1)
    depth_after = torch.abs(target_depth - ray_depth)
    return (
        total,
        losses,
        before[valid],
        after[valid],
        initial_depth_error[valid],
        depth_after[valid],
    )


def quantiles(chunks: list[np.ndarray]) -> dict:
    values = np.concatenate(chunks) * 1000.0 if chunks else np.empty(0)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.quantile(values, 0.9)),
        "max_mm": float(values.max()),
    }


def run_epoch(model, loader, args, optimizer=None, split="train"):
    training = optimizer is not None
    model.train(training)
    sums, count = {}, 0
    before_chunks, after_chunks, depth_before, depth_after = [], [], [], []
    progress = tqdm(loader, desc=split, dynamic_ncols=True)
    for batch in progress:
        if training:
            optimizer.zero_grad(set_to_none=True)
        result = compute(model, batch, args)
        total, losses, before, after, before_depth, after_depth = result
        if training:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = int(batch["scalar"].shape[0])
        count += batch_size
        sums["total"] = sums.get("total", 0.0) + float(total) * batch_size
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
        before_chunks.append(before.detach().cpu().numpy())
        after_chunks.append(after.detach().cpu().numpy())
        depth_before.append(before_depth.detach().cpu().numpy())
        depth_after.append(after_depth.detach().cpu().numpy())
        progress.set_postfix(loss=f"{sums['total'] / count:.5f}")
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["initial_wrist"] = quantiles(before_chunks)
    metrics["corrected_wrist"] = quantiles(after_chunks)
    metrics["initial_ray_depth"] = quantiles(depth_before)
    metrics["corrected_ray_depth"] = quantiles(depth_after)
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset = Pi3XWindowDataset(
        Path(args.train_windows).expanduser().resolve(),
        Path(args.pi3x_train_root).expanduser().resolve(),
        args.max_target_mm / 1000.0,
    )
    val_dataset = Pi3XWindowDataset(
        Path(args.val_windows).expanduser().resolve(),
        Path(args.pi3x_val_root).expanduser().resolve(),
        args.max_target_mm / 1000.0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    sample = train_dataset[0]
    scalar_dim = int(sample["scalar"].shape[-1])
    feature_dim = int(sample["token_features"].shape[-1])
    metadata_dim = int(sample["token_metadata"].shape[-1])
    model = Pi3XRelativeDepthRefiner(
        scalar_dim,
        feature_dim,
        metadata_dim,
        args.hidden_dim,
        args.spatial_layers,
        args.temporal_layers,
        args.heads,
        args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            dir=str(out_dir),
        )
    history, best_depth, best_total = [], float("inf"), float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, args, optimizer, "train"
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, args, split="val")
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "lr": row["lr"],
                    "train/total": train_metrics["total"],
                    "val/total": val_metrics["total"],
                    "val/wrist_after_mm":
                        val_metrics["corrected_wrist"]["median_mm"],
                    "val/depth_after_mm":
                        val_metrics["corrected_ray_depth"]["median_mm"],
                },
                step=epoch,
            )
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "scalar_dim": scalar_dim,
            "feature_dim": feature_dim,
            "metadata_dim": metadata_dim,
            "epoch": epoch,
            "val_total": val_metrics["total"],
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        primary = val_metrics["corrected_ray_depth"]["median_mm"]
        if primary < best_depth or (
            primary == best_depth and val_metrics["total"] < best_total
        ):
            best_depth = primary
            best_total = val_metrics["total"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
