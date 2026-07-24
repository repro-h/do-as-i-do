#!/usr/bin/env python3
"""Train a temporal hand/object center residual refiner."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--w-hand", type=float, default=1.0)
    parser.add_argument("--w-object", type=float, default=1.0)
    parser.add_argument("--w-relative", type=float, default=2.0)
    parser.add_argument("--w-projection", type=float, default=0.25)
    parser.add_argument("--w-velocity", type=float, default=0.25)
    parser.add_argument("--w-acceleration", type=float, default=0.1)
    parser.add_argument("--w-residual", type=float, default=0.01)
    parser.add_argument("--max-residual-mm", type=float, default=100.0)
    parser.add_argument("--max-target-hand-mm", type=float, default=120.0)
    parser.add_argument("--max-target-object-mm", type=float, default=100.0)
    parser.add_argument("--max-target-relative-mm", type=float, default=150.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode",
        choices=("joint", "object_only", "hand_only"),
        default="joint",
        help="Select which rigid translation residuals the model may predict.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=128)
def load_supervision(path_text: str) -> dict[str, np.ndarray]:
    with np.load(path_text, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


class WindowDataset(Dataset):
    def __init__(
        self,
        path: Path,
        max_target_hand_m: float,
        max_target_object_m: float,
        max_target_relative_m: float,
    ):
        self.rows = load_jsonl(path)
        if not self.rows:
            raise RuntimeError(f"No windows in {path}")
        self.max_target_hand_m = max_target_hand_m
        self.max_target_object_m = max_target_object_m
        self.max_target_relative_m = max_target_relative_m

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        raw = load_supervision(row["supervision_npz"])
        hand = np.asarray(raw["pred_hand_center"][start:end], dtype=np.float32)
        obj = np.asarray(raw["pred_object_center"][start:end], dtype=np.float32)
        rot6d = np.asarray(raw["pred_object_rot6d"][start:end], dtype=np.float32)
        gt_hand = np.asarray(raw["gt_hand_center"][start:end], dtype=np.float32)
        gt_obj = np.asarray(raw["gt_object_center"][start:end], dtype=np.float32)
        hand_mask = np.asarray(
            raw["hand_supervision_valid"][start:end], dtype=bool
        )
        object_mask = np.asarray(
            raw["object_supervision_valid"][start:end], dtype=bool
        )
        relative_mask = np.asarray(
            raw["relative_supervision_valid"][start:end], dtype=bool
        )
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32)

        hand_safe = np.nan_to_num(hand)
        obj_safe = np.nan_to_num(obj)
        gt_hand_safe = np.nan_to_num(gt_hand)
        gt_obj_safe = np.nan_to_num(gt_obj)
        hand_delta = gt_hand_safe - hand_safe
        object_delta = gt_obj_safe - obj_safe
        relative_delta = hand_delta - object_delta
        hand_mask &= np.linalg.norm(hand_delta, axis=-1) <= self.max_target_hand_m
        object_mask &= (
            np.linalg.norm(object_delta, axis=-1) <= self.max_target_object_m
        )
        relative_mask &= hand_mask & object_mask
        relative_mask &= (
            np.linalg.norm(relative_delta, axis=-1) <= self.max_target_relative_m
        )
        rot_safe = np.nan_to_num(rot6d)
        hand_velocity = np.zeros_like(hand_safe)
        object_velocity = np.zeros_like(obj_safe)
        hand_velocity[1:] = hand_safe[1:] - hand_safe[:-1]
        object_velocity[1:] = obj_safe[1:] - obj_safe[:-1]
        features = np.concatenate(
            [
                hand_safe,
                obj_safe,
                hand_safe - obj_safe,
                hand_velocity,
                object_velocity,
                rot_safe,
                hand_mask[:, None],
                object_mask[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        return {
            "features": torch.from_numpy(features),
            "pred_hand": torch.from_numpy(hand_safe),
            "pred_object": torch.from_numpy(obj_safe),
            "gt_hand": torch.from_numpy(gt_hand_safe),
            "gt_object": torch.from_numpy(gt_obj_safe),
            "hand_mask": torch.from_numpy(hand_mask),
            "object_mask": torch.from_numpy(object_mask),
            "relative_mask": torch.from_numpy(relative_mask),
            "intrinsics": torch.from_numpy(intrinsics),
        }


class RigidTemporalRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        mode: str = "joint",
    ):
        super().__init__()
        if mode not in {"joint", "object_only", "hand_only"}:
            raise ValueError(f"Unsupported refinement mode: {mode}")
        self.mode = mode
        self.input = nn.Sequential(
            nn.Linear(23, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.position = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6 if mode == "joint" else 3),
        )

    def forward(self, features: torch.Tensor, max_residual: float):
        if features.shape[1] > self.position.shape[1]:
            raise ValueError(
                f"Window length {features.shape[1]} exceeds "
                f"{self.position.shape[1]}"
            )
        tokens = self.input(features) + self.position[:, : features.shape[1]]
        encoded = self.encoder(tokens)
        residual = torch.tanh(self.output(encoded)) * max_residual
        if self.mode == "joint":
            return residual[..., :3], residual[..., 3:]
        zeros = torch.zeros_like(residual)
        if self.mode == "object_only":
            return zeros, residual
        return residual, zeros


def masked_smooth_l1(value, target, mask, beta):
    error = F.smooth_l1_loss(
        value, target, reduction="none", beta=beta
    ).mean(dim=-1)
    weights = mask.float()
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def temporal_loss(value, target, mask, order: int, beta: float):
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
        target = target[:, 1:] - target[:, :-1]
        mask = mask[:, 1:] & mask[:, :-1]
    return masked_smooth_l1(value, target, mask, beta)


def project_points(points, intrinsics):
    z = points[..., 2].clamp_min(1e-4)
    x = points[..., 0] / z
    y = points[..., 1] / z
    fx = intrinsics[:, None, 0, 0]
    fy = intrinsics[:, None, 1, 1]
    cx = intrinsics[:, None, 0, 2]
    cy = intrinsics[:, None, 1, 2]
    return torch.stack((fx * x + cx, fy * y + cy), dim=-1)


def compute_loss(model, batch, args):
    batch = {key: value.to(args.device) for key, value in batch.items()}
    beta = args.smooth_l1_beta_mm / 1000.0
    hand_delta, object_delta = model(
        batch["features"], args.max_residual_mm / 1000.0
    )
    corrected_hand = batch["pred_hand"] + hand_delta
    corrected_object = batch["pred_object"] + object_delta
    corrected_relative = corrected_hand - corrected_object
    gt_relative = batch["gt_hand"] - batch["gt_object"]
    hand_projection_mask = (
        batch["hand_mask"]
        & (corrected_hand[..., 2] > 1e-4)
        & (batch["gt_hand"][..., 2] > 1e-4)
    )
    object_projection_mask = (
        batch["object_mask"]
        & (corrected_object[..., 2] > 1e-4)
        & (batch["gt_object"][..., 2] > 1e-4)
    )
    hand_projection = masked_smooth_l1(
        project_points(corrected_hand, batch["intrinsics"]) / 100.0,
        project_points(batch["gt_hand"], batch["intrinsics"]) / 100.0,
        hand_projection_mask,
        beta,
    )
    object_projection = masked_smooth_l1(
        project_points(corrected_object, batch["intrinsics"]) / 100.0,
        project_points(batch["gt_object"], batch["intrinsics"]) / 100.0,
        object_projection_mask,
        beta,
    )
    projection = 0.5 * (hand_projection + object_projection)

    losses = {
        "hand": masked_smooth_l1(
            corrected_hand, batch["gt_hand"], batch["hand_mask"], beta
        ),
        "object": masked_smooth_l1(
            corrected_object, batch["gt_object"], batch["object_mask"], beta
        ),
        "relative": masked_smooth_l1(
            corrected_relative, gt_relative, batch["relative_mask"], beta
        ),
        "projection": projection,
        "hand_projection": hand_projection,
        "object_projection": object_projection,
        "velocity": temporal_loss(
            corrected_relative, gt_relative, batch["relative_mask"], 1, beta
        ),
        "acceleration": temporal_loss(
            corrected_relative, gt_relative, batch["relative_mask"], 2, beta
        ),
        "object_velocity": temporal_loss(
            corrected_object,
            batch["gt_object"],
            batch["object_mask"],
            1,
            beta,
        ),
        "object_acceleration": temporal_loss(
            corrected_object,
            batch["gt_object"],
            batch["object_mask"],
            2,
            beta,
        ),
        "hand_velocity": temporal_loss(
            corrected_hand, batch["gt_hand"], batch["hand_mask"], 1, beta
        ),
        "hand_acceleration": temporal_loss(
            corrected_hand, batch["gt_hand"], batch["hand_mask"], 2, beta
        ),
        "residual": (hand_delta.square().mean() + object_delta.square().mean()),
    }
    if args.mode == "object_only":
        total = (
            args.w_object * losses["object"]
            + args.w_projection * losses["object_projection"]
            + args.w_velocity * losses["object_velocity"]
            + args.w_acceleration * losses["object_acceleration"]
            + args.w_residual * losses["residual"]
        )
    elif args.mode == "hand_only":
        total = (
            args.w_hand * losses["hand"]
            + args.w_relative * losses["relative"]
            + args.w_projection * losses["hand_projection"]
            + args.w_velocity * losses["hand_velocity"]
            + args.w_acceleration * losses["hand_acceleration"]
            + args.w_residual * losses["residual"]
        )
    else:
        total = (
            args.w_hand * losses["hand"]
            + args.w_object * losses["object"]
            + args.w_relative * losses["relative"]
            + args.w_projection * losses["projection"]
            + args.w_velocity * losses["velocity"]
            + args.w_acceleration * losses["acceleration"]
            + args.w_residual * losses["residual"]
        )
    return total, losses, {
        "corrected_hand": corrected_hand,
        "corrected_object": corrected_object,
        "corrected_relative": corrected_relative,
        "gt_relative": gt_relative,
        "batch": batch,
    }


def error_quantiles(values):
    if not values:
        return {"median_mm": None, "p90_mm": None}
    array = torch.cat(values)
    if not len(array):
        return {"median_mm": None, "p90_mm": None}
    return {
        "median_mm": float(torch.quantile(array, 0.5) * 1000.0),
        "p90_mm": float(torch.quantile(array, 0.9) * 1000.0),
    }


def run_epoch(model, loader, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {}
    count = 0
    errors = {
        "initial_hand": [],
        "corrected_hand": [],
        "initial_object": [],
        "corrected_object": [],
        "initial_relative": [],
        "corrected_relative": [],
    }
    for batch in loader:
        with torch.set_grad_enabled(training):
            total, losses, aux = compute_loss(model, batch, args)
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                values = aux["batch"]
                definitions = (
                    (
                        "hand",
                        values["pred_hand"],
                        aux["corrected_hand"],
                        values["gt_hand"],
                        values["hand_mask"],
                    ),
                    (
                        "object",
                        values["pred_object"],
                        aux["corrected_object"],
                        values["gt_object"],
                        values["object_mask"],
                    ),
                    (
                        "relative",
                        values["pred_hand"] - values["pred_object"],
                        aux["corrected_relative"],
                        aux["gt_relative"],
                        values["relative_mask"],
                    ),
                )
                for name, initial, corrected, target, mask in definitions:
                    errors[f"initial_{name}"].append(
                        torch.linalg.norm(initial[mask] - target[mask], dim=-1).cpu()
                    )
                    errors[f"corrected_{name}"].append(
                        torch.linalg.norm(corrected[mask] - target[mask], dim=-1).cpu()
                    )
        values = {"total": total, **losses}
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach())
        count += 1
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    if not training:
        metrics["errors"] = {
            key: error_quantiles(value) for key, value in errors.items()
        }
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_args = (
        args.max_target_hand_mm / 1000.0,
        args.max_target_object_mm / 1000.0,
        args.max_target_relative_mm / 1000.0,
    )
    train_data = WindowDataset(
        Path(args.train_windows).expanduser().resolve(), *dataset_args
    )
    val_data = WindowDataset(
        Path(args.val_windows).expanduser().resolve(), *dataset_args
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = RigidTemporalRefiner(
        args.hidden_dim, args.layers, args.heads, args.dropout, args.mode
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, args)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row))
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_total": val_metrics["total"],
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
