#!/usr/bin/env python3
"""Train Stage1 Global Hand v2 Phase A (translation only)."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-translation-mm", type=float, default=80.0)
    parser.add_argument(
        "--prediction-mode",
        choices=("translation3d", "ray_depth"),
        default="ray_depth",
        help="Predict an unconstrained 3D translation or one signed camera-ray depth.",
    )
    parser.add_argument(
        "--include-camera-ray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the normalized wrist camera ray to each frame feature.",
    )
    parser.add_argument(
        "--include-surface-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append object rotation/extent, object-local wrist/palm, and "
            "precomputed semantic hand-to-object surface statistics."
        ),
    )
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--max-target-mm", type=float, default=120.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--signed-magnitude-head", action="store_true")
    parser.add_argument(
        "--auxiliary-sign-head",
        action="store_true",
        help=(
            "Keep direct signed ray regression while using a separate sign "
            "classifier only as auxiliary supervision."
        ),
    )
    parser.add_argument("--sign-valid-threshold-mm", type=float, default=5.0)
    parser.add_argument("--w-sign", type=float, default=0.5)
    parser.add_argument("--w-wrist", type=float, default=1.0)
    parser.add_argument("--w-palm", type=float, default=0.0)
    parser.add_argument("--w-projection", type=float, default=1.0)
    parser.add_argument("--w-velocity", type=float, default=0.5)
    parser.add_argument("--w-acceleration", type=float, default=1.0)
    parser.add_argument("--w-residual", type=float, default=0.05)
    parser.add_argument("--z-axis-weight", type=float, default=2.0)
    parser.add_argument("--error-weight-reference-mm", type=float, default=20.0)
    parser.add_argument("--error-weight-min", type=float, default=0.5)
    parser.add_argument("--error-weight-max", type=float, default=2.0)
    parser.add_argument("--anchor-accurate-mm", type=float, default=15.0)
    parser.add_argument("--anchor-large-error-mm", type=float, default=30.0)
    parser.add_argument("--anchor-accurate-weight", type=float, default=4.0)
    parser.add_argument("--anchor-medium-weight", type=float, default=1.0)
    parser.add_argument("--anchor-large-error-weight", type=float, default=0.2)
    parser.add_argument("--correction-gate", action="store_true")
    parser.add_argument("--gate-zero-error-mm", type=float, default=10.0)
    parser.add_argument("--gate-full-error-mm", type=float, default=30.0)
    parser.add_argument("--w-gate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="do-as-i-do-global-hand-v2")
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=128)
def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def rotation_6d_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    valid = np.isfinite(rotvec).all(axis=-1)
    matrix = np.tile(np.eye(3, dtype=np.float32), (len(rotvec), 1, 1))
    if valid.any():
        matrix[valid] = Rotation.from_rotvec(rotvec[valid]).as_matrix()
    return matrix[:, :, :2].transpose(0, 2, 1).reshape(len(rotvec), 6)


def rotation_6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, :, :2].transpose(0, 2, 1).reshape(len(matrix), 6)


class WindowDataset(Dataset):
    def __init__(
        self,
        path: Path,
        max_target_m: float,
        include_camera_ray: bool = False,
        include_surface_geometry: bool = False,
    ):
        self.rows = load_jsonl(path)
        if not self.rows:
            raise RuntimeError(f"No windows in {path}")
        self.max_target_m = max_target_m
        self.include_camera_ray = include_camera_ray
        self.include_surface_geometry = include_surface_geometry

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        raw = load_npz(row["supervision_npz"])
        pred = np.asarray(raw["pred_joints_3d"][start:end], dtype=np.float32)
        gt3d = np.asarray(raw["gt_joints_3d"][start:end], dtype=np.float32)
        gt2d = np.asarray(raw["gt_joints_2d"][start:end], dtype=np.float32)
        object_pose = np.asarray(raw["object_pose"][start:end], dtype=np.float32)
        initial_root = np.asarray(
            raw["initial_root_rotvec"][start:end], dtype=np.float32
        )
        valid = np.asarray(raw["supervision_valid"][start:end]).astype(bool)
        wrist_error = np.linalg.norm(gt3d[:, 0] - pred[:, 0], axis=-1)
        valid &= np.isfinite(wrist_error) & (wrist_error <= self.max_target_m)

        pred = np.nan_to_num(pred)
        gt3d = np.nan_to_num(gt3d)
        gt2d = np.nan_to_num(gt2d)
        object_pose = np.nan_to_num(object_pose)
        initial_root = np.nan_to_num(initial_root)
        wrist = pred[:, 0]
        palm_local = pred[:, PALM] - wrist[:, None]
        object_center = object_pose[:, :3, 3]
        object_rotation = object_pose[:, :3, :3]
        hand_velocity = np.zeros_like(wrist)
        object_velocity = np.zeros_like(object_center)
        hand_velocity[1:] = wrist[1:] - wrist[:-1]
        object_velocity[1:] = object_center[1:] - object_center[:-1]
        relative = wrist - object_center
        relative_velocity = hand_velocity - object_velocity
        camera_ray = wrist / np.maximum(
            np.linalg.norm(wrist, axis=-1, keepdims=True), 1e-8
        )
        hand_acceleration = np.zeros_like(wrist)
        hand_acceleration[1:] = hand_velocity[1:] - hand_velocity[:-1]
        geometry_parts = []
        if self.include_surface_geometry:
            required = {
                "object_extents_metric",
                "surface_geometry_features",
                "surface_geometry_feature_names",
            }
            missing = required.difference(raw)
            if missing:
                raise KeyError(
                    f"{row['supervision_npz']} lacks geometry fields: "
                    f"{sorted(missing)}"
                )
            extents = np.asarray(
                raw["object_extents_metric"], dtype=np.float32
            ).reshape(3)
            safe_extents = np.maximum(extents, 1e-3)
            local_wrist = np.einsum(
                "ti,tij->tj", wrist - object_center, object_rotation
            )
            local_palm = np.einsum(
                "tki,tij->tkj",
                pred[:, PALM] - object_center[:, None],
                object_rotation,
            )
            surface = np.asarray(
                raw["surface_geometry_features"][start:end],
                dtype=np.float32,
            ).copy()
            names = [
                str(value)
                for value in raw["surface_geometry_feature_names"].tolist()
            ]
            for column, name in enumerate(names):
                if "distance_" in name or "ray_direction_" in name:
                    surface[:, column] /= 0.1
            geometry_parts = [
                rotation_6d_from_matrix(object_rotation),
                np.broadcast_to(extents / 0.2, (len(wrist), 3)),
                local_wrist / safe_extents,
                (local_palm / safe_extents).reshape(len(wrist), -1),
                surface,
            ]
        features = np.concatenate(
            [
                wrist,
                palm_local.reshape(len(wrist), -1),
                hand_velocity,
                hand_acceleration,
                object_center,
                relative,
                object_velocity,
                relative_velocity,
                rotation_6d_from_rotvec(initial_root),
                *([camera_ray] if self.include_camera_ray else []),
                *geometry_parts,
                valid[:, None],
            ],
            axis=-1,
        ).astype(np.float32)
        return {
            "features": torch.from_numpy(features),
            "pred_joints": torch.from_numpy(pred),
            "gt_joints": torch.from_numpy(gt3d),
            "gt_joints_2d": torch.from_numpy(gt2d),
            "intrinsics": torch.from_numpy(
                np.asarray(raw["intrinsics"], dtype=np.float32)
            ),
            "valid": torch.from_numpy(valid),
            "stream_id": row["stream_id"],
            "start": start,
            "end": end,
        }


class TranslationRefiner(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, layers: int,
        heads: int,
        dropout: float,
        correction_gate: bool = False,
        prediction_mode: str = "translation3d",
        signed_magnitude_head: bool = False,
        auxiliary_sign_head: bool = False,
    ):
        super().__init__()
        self.correction_gate = correction_gate
        self.prediction_mode = prediction_mode
        self.signed_magnitude_head = signed_magnitude_head
        self.auxiliary_sign_head = auxiliary_sign_head
        if signed_magnitude_head and auxiliary_sign_head:
            raise ValueError(
                "Choose either signed_magnitude_head or auxiliary_sign_head"
            )
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.position = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1 if prediction_mode == "ray_depth" else 3),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        if signed_magnitude_head or auxiliary_sign_head:
            if prediction_mode != "ray_depth":
                raise ValueError(
                    "sign heads require prediction_mode=ray_depth"
                )
            self.sign_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.sign_head[-1].weight)
            nn.init.zeros_(self.sign_head[-1].bias)
        if signed_magnitude_head:
            self.magnitude_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.magnitude_head[-1].weight)
            nn.init.constant_(self.magnitude_head[-1].bias, -2.2)
        if correction_gate:
            self.gate = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        max_translation: float,
        return_aux: bool = False,
    ):
        token = self.input(features) + self.position[:, : features.shape[1]]
        encoded = self.encoder(token)
        sign_logits = (
            self.sign_head(encoded).squeeze(-1)
            if self.signed_magnitude_head or self.auxiliary_sign_head
            else None
        )
        if self.signed_magnitude_head:
            magnitude = (
                torch.sigmoid(self.magnitude_head(encoded).squeeze(-1))
                * max_translation
            )
            raw_translation = (
                torch.tanh(sign_logits) * magnitude
            ).unsqueeze(-1)
        else:
            raw_translation = (
                torch.tanh(self.output(encoded)) * max_translation
            )
        gate = None
        if self.correction_gate:
            gate = torch.sigmoid(self.gate(encoded)).squeeze(-1)
            raw_translation = raw_translation * gate.unsqueeze(-1)
        if return_aux:
            return {
                "prediction": raw_translation,
                "gate": gate,
                "sign_logits": sign_logits,
            }
        if gate is not None:
            return raw_translation, gate
        return raw_translation


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(value.dtype).expand_as(value)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


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
        batch["features"],
        args.max_translation_mm / 1000.0,
        return_aux=True,
    )
    raw_prediction = model_output["prediction"]
    gate = model_output["gate"]
    sign_logits = model_output["sign_logits"]
    pred = batch["pred_joints"]
    gt = batch["gt_joints"]
    valid = batch["valid"]
    camera_ray = pred[:, :, 0]
    camera_ray = F.normalize(camera_ray, dim=-1, eps=1e-8)
    target_translation = gt[:, :, 0] - pred[:, :, 0]
    target_ray_depth = torch.sum(target_translation * camera_ray, dim=-1)
    if args.prediction_mode == "ray_depth":
        ray_depth = raw_prediction.squeeze(-1)
        translation = ray_depth.unsqueeze(-1) * camera_ray
    else:
        translation = raw_prediction
        ray_depth = torch.sum(translation * camera_ray, dim=-1)
    corrected = pred + translation[:, :, None]
    beta = args.smooth_l1_beta_mm / 1000.0
    palm_mask = valid[:, :, None]
    projection_mask = (
        palm_mask
        & (corrected[:, :, PALM, 2] > 1e-4)
        & (gt[:, :, PALM, 2] > 1e-4)
    )
    initial_error = torch.linalg.norm(target_translation, dim=-1)
    initial_ray_error = torch.abs(target_ray_depth)
    weighting_error = (
        initial_ray_error
        if args.prediction_mode == "ray_depth"
        else initial_error
    )
    supervision_weight = (
        weighting_error
        / max(args.error_weight_reference_mm / 1000.0, 1e-8)
    ).clamp(args.error_weight_min, args.error_weight_max)
    axis_weight = torch.tensor(
        [1.0, 1.0, args.z_axis_weight],
        dtype=pred.dtype,
        device=pred.device,
    )
    anchor_weight = torch.where(
        weighting_error < args.anchor_accurate_mm / 1000.0,
        torch.full_like(initial_error, args.anchor_accurate_weight),
        torch.where(
            weighting_error < args.anchor_large_error_mm / 1000.0,
            torch.full_like(initial_error, args.anchor_medium_weight),
            torch.full_like(initial_error, args.anchor_large_error_weight),
        ),
    )
    losses = {
        "depth": weighted_smooth_l1(
            ray_depth,
            target_ray_depth,
            valid,
            supervision_weight,
            beta,
        ),
        "wrist": weighted_smooth_l1(
            corrected[:, :, 0],
            gt[:, :, 0],
            valid,
            supervision_weight[:, :, None] * axis_weight,
            beta,
        ),
        "palm": weighted_smooth_l1(
            corrected[:, :, PALM],
            gt[:, :, PALM],
            palm_mask,
            supervision_weight[:, :, None, None] * axis_weight,
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
        "residual": weighted_smooth_l1(
            ray_depth,
            torch.zeros_like(ray_depth),
            valid,
            anchor_weight,
            beta,
        ),
    }
    if sign_logits is not None:
        sign_valid = (
            valid
            & (
                initial_ray_error
                >= args.sign_valid_threshold_mm / 1000.0
            )
        )
        sign_target = (target_ray_depth > 0.0).to(pred.dtype)
        sign_loss = F.binary_cross_entropy_with_logits(
            sign_logits, sign_target, reduction="none"
        )
        losses["sign"] = masked_mean(sign_loss, sign_valid)
    if gate is not None:
        gate_range = max(
            args.gate_full_error_mm - args.gate_zero_error_mm, 1e-6
        )
        gate_target = (
            (initial_ray_error * 1000.0 - args.gate_zero_error_mm) / gate_range
        ).clamp(0.0, 1.0)
        losses["gate"] = smooth_l1(
            gate, gate_target, valid, beta=0.1
        )
    total = (
        args.w_depth * losses["depth"]
        + (
            args.w_wrist * losses["wrist"]
            if args.prediction_mode == "translation3d"
            else 0.0
        )
        + args.w_palm * losses["palm"]
        + args.w_projection * losses["projection"]
        + args.w_velocity * losses["velocity"]
        + args.w_acceleration * losses["acceleration"]
        + args.w_residual * losses["residual"]
    )
    if gate is not None:
        total = total + args.w_gate * losses["gate"]
    if sign_logits is not None:
        total = total + args.w_sign * losses["sign"]
    before = torch.linalg.norm(pred[:, :, 0] - gt[:, :, 0], dim=-1)
    after = torch.linalg.norm(corrected[:, :, 0] - gt[:, :, 0], dim=-1)
    ray_after = torch.abs(target_ray_depth - ray_depth)
    return (
        total,
        losses,
        before[valid],
        after[valid],
        initial_ray_error[valid],
        ray_after[valid],
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
    before_chunks, after_chunks = [], []
    ray_before_chunks, ray_after_chunks = [], []
    progress = tqdm(loader, desc=split, dynamic_ncols=True)
    for batch in progress:
        if training:
            optimizer.zero_grad(set_to_none=True)
        total, losses, before, after, ray_before, ray_after = compute(
            model, batch, args
        )
        if training:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = int(batch["features"].shape[0])
        count += batch_size
        sums["total"] = sums.get("total", 0.0) + float(total) * batch_size
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
        before_chunks.append(before.detach().cpu().numpy())
        after_chunks.append(after.detach().cpu().numpy())
        ray_before_chunks.append(ray_before.detach().cpu().numpy())
        ray_after_chunks.append(ray_after.detach().cpu().numpy())
        progress.set_postfix(loss=f"{sums['total'] / count:.5f}")
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["initial_wrist"] = quantiles(before_chunks)
    metrics["corrected_wrist"] = quantiles(after_chunks)
    metrics["initial_ray_depth"] = quantiles(ray_before_chunks)
    metrics["corrected_ray_depth"] = quantiles(ray_after_chunks)
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset = WindowDataset(
        Path(args.train_windows).expanduser().resolve(),
        args.max_target_mm / 1000.0,
        args.include_camera_ray,
        args.include_surface_geometry,
    )
    val_dataset = WindowDataset(
        Path(args.val_windows).expanduser().resolve(),
        args.max_target_mm / 1000.0,
        args.include_camera_ray,
        args.include_surface_geometry,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    input_dim = int(train_dataset[0]["features"].shape[-1])
    model = TranslationRefiner(
        input_dim,
        args.hidden_dim,
        args.layers,
        args.heads,
        args.dropout,
        args.correction_gate,
        args.prediction_mode,
        args.signed_magnitude_head,
        args.auxiliary_sign_head,
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
    history, best_primary, best_total = [], float("inf"), float("inf")
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
                    "val/wrist_before_mm":
                        val_metrics["initial_wrist"]["median_mm"],
                    "val/wrist_after_mm":
                        val_metrics["corrected_wrist"]["median_mm"],
                    "val/ray_depth_before_mm":
                        val_metrics["initial_ray_depth"]["median_mm"],
                    "val/ray_depth_after_mm":
                        val_metrics["corrected_ray_depth"]["median_mm"],
                },
                step=epoch,
            )
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "input_dim": input_dim,
            "epoch": epoch,
            "val_total": val_metrics["total"],
        }
        torch.save(checkpoint, out_dir / "last.pt")
        primary_median = (
            val_metrics["corrected_ray_depth"]["median_mm"]
            if args.prediction_mode == "ray_depth"
            else val_metrics["corrected_wrist"]["median_mm"]
        )
        is_better = (
            primary_median < best_primary
            or (
                primary_median == best_primary
                and val_metrics["total"] < best_total
            )
        )
        if is_better:
            best_primary = primary_median
            best_total = val_metrics["total"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
