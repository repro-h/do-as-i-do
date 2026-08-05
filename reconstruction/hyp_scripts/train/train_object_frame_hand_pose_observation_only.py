#!/usr/bin/env python3
"""Train absolute object-frame hand pose from observation features only."""

from __future__ import annotations

import argparse
import json
import math
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import (
    ObjectFrameWindowDataset,
    RelativeCrossAttention,
    metric_distribution,
    rotation_6d_to_matrix,
    rotation_angle,
    seed_worker,
    smooth_l1_vector,
    temporal_loss,
)


MODEL_VERSION = "object_frame_observation_only_pi3x_bigru_absolute_pose_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pi3x-relation-dim", type=int, default=128)
    parser.add_argument("--pi3x-heads", type=int, default=8)
    parser.add_argument("--max-normalized-translation", type=float, default=3.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--w-translation", type=float, default=1.0)
    parser.add_argument("--w-rotation", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


class ObservationOnlyAbsolutePoseModel(nn.Module):
    def __init__(
        self,
        local_hand_dim: int,
        pi3x_feature_dim: int,
        pi3x_metadata_dim: int,
        relation_dim: int,
        heads: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        max_normalized_translation: float,
    ):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden-dim must be even")
        self.max_normalized_translation = max_normalized_translation
        self.relation = RelativeCrossAttention(
            pi3x_feature_dim,
            pi3x_metadata_dim,
            relation_dim,
            heads,
            dropout,
        )
        self.local_hand_encoder = nn.Sequential(
            nn.LayerNorm(local_hand_dim),
            nn.Linear(local_hand_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frame_encoder = nn.Sequential(
            nn.Linear(hidden_dim + relation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.translation_head[-1].weight)
        nn.init.zeros_(self.translation_head[-1].bias)
        nn.init.zeros_(self.rotation_head[-1].weight)
        self.rotation_head[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )

    def forward(
        self,
        local_hand_features: torch.Tensor,
        hand_token_features: torch.Tensor,
        hand_token_metadata: torch.Tensor,
        hand_token_valid: torch.Tensor,
        key_token_features: torch.Tensor,
        key_token_metadata: torch.Tensor,
        key_token_valid: torch.Tensor,
        key_token_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation = self.relation(
            hand_token_features,
            hand_token_metadata,
            hand_token_valid,
            key_token_features,
            key_token_metadata,
            key_token_valid,
            key_token_types,
        )
        local_hand = self.local_hand_encoder(local_hand_features)
        frame = self.frame_encoder(torch.cat((local_hand, relation), dim=-1))
        temporal, _ = self.temporal(frame)
        translation = (
            torch.tanh(self.translation_head(temporal))
            * self.max_normalized_translation
        )
        rotation = rotation_6d_to_matrix(self.rotation_head(temporal))
        return translation, rotation


def weighted_masked_mean(
    value: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {
        "total": 0.0,
        "translation": 0.0,
        "rotation": 0.0,
        "velocity": 0.0,
        "acceleration": 0.0,
    }
    batches = 0
    values = {name: [] for name in ("initial_t", "predicted_t", "initial_r", "predicted_r")}
    iterator = tqdm(loader, desc="train" if training else "val")
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            predicted_t_normalized, predicted_r = model(
                batch["local_hand_features"],
                batch["hand_token_features"],
                batch["hand_token_metadata"],
                batch["hand_token_valid"],
                batch["key_token_features"],
                batch["key_token_metadata"],
                batch["key_token_valid"],
                batch["key_token_types"],
            )
            scale = batch["object_scale"][..., None]
            predicted_t = predicted_t_normalized * scale
            target_t = batch["target_translation"] * scale
            initial_t = batch["initial_translation"] * scale
            beta = (
                args.smooth_l1_beta_mm
                / 1000.0
                / batch["object_scale"]
            )
            translation_frame = smooth_l1_vector(
                predicted_t_normalized, batch["target_translation"], beta
            )
            translation = weighted_masked_mean(
                translation_frame, batch["valid_translation"]
            )
            predicted_r_error = rotation_angle(
                predicted_r, batch["target_rotation"]
            )
            initial_r_error = rotation_angle(
                batch["initial_rotation"], batch["target_rotation"]
            )
            rotation_weight = (
                batch["valid_rotation"].to(predicted_r_error.dtype)
                * batch["rotation_weight"]
            )
            rotation = weighted_masked_mean(
                predicted_r_error, rotation_weight
            )
            velocity = temporal_loss(
                predicted_t,
                target_t,
                batch["valid_translation"],
                1,
                args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration = temporal_loss(
                predicted_t,
                target_t,
                batch["valid_translation"],
                2,
                args.smooth_l1_beta_mm / 1000.0,
            )
            total = (
                args.w_translation * translation
                + args.w_rotation * rotation
                + args.w_velocity * velocity
                + args.w_acceleration * acceleration
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for name, metric in (
            ("total", total),
            ("translation", translation),
            ("rotation", rotation),
            ("velocity", velocity),
            ("acceleration", acceleration),
        ):
            totals[name] += float(metric.detach())
        batches += 1
        iterator.set_postfix(loss=f"{totals['total'] / batches:.5f}")
        valid_t = batch["valid_translation"].detach().cpu().numpy().astype(bool)
        valid_r = batch["valid_rotation"].detach().cpu().numpy().astype(bool)
        initial_t_error = torch.linalg.norm(initial_t - target_t, dim=-1)
        predicted_t_error = torch.linalg.norm(predicted_t - target_t, dim=-1)
        for name, metric, valid in (
            ("initial_t", initial_t_error, valid_t),
            ("predicted_t", predicted_t_error, valid_t),
            ("initial_r", initial_r_error, valid_r),
            ("predicted_r", predicted_r_error, valid_r),
        ):
            array = metric.detach().cpu().numpy()
            values[name].append(array[valid])

    output = {name: value / max(batches, 1) for name, value in totals.items()}
    output.update(
        initial_translation=metric_distribution(values["initial_t"], 1000.0),
        predicted_translation=metric_distribution(values["predicted_t"], 1000.0),
        initial_rotation=metric_distribution(values["initial_r"], 180.0 / math.pi),
        predicted_rotation=metric_distribution(values["predicted_r"], 180.0 / math.pi),
    )
    return output


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_args = Namespace(
        translation_noise_mm=0.0,
        rotation_noise_deg=0.0,
        initial_pose_dropout=0.0,
    )
    train_windows = Path(args.train_windows).expanduser().resolve()
    val_windows = Path(args.val_windows).expanduser().resolve()
    train_rows = [json.loads(line) for line in train_windows.read_text().splitlines() if line.strip()]
    val_rows = [json.loads(line) for line in val_windows.read_text().splitlines() if line.strip()]
    object_names = sorted({str(row["object_name"]) for row in train_rows})
    unknown = sorted({str(row["object_name"]) for row in val_rows} - set(object_names))
    if unknown:
        raise KeyError(f"Validation contains unknown objects: {unknown}")
    object_to_index = {name: index for index, name in enumerate(object_names)}
    train_data = ObjectFrameWindowDataset(
        train_windows,
        dataset_args,
        augment=False,
        object_to_index=object_to_index,
        pi3x_root=Path(args.pi3x_train_root).expanduser().resolve(),
    )
    val_data = ObjectFrameWindowDataset(
        val_windows,
        dataset_args,
        augment=False,
        object_to_index=object_to_index,
        pi3x_root=Path(args.pi3x_val_root).expanduser().resolve(),
    )
    sample = train_data[0]
    model = ObservationOnlyAbsolutePoseModel(
        int(sample["local_hand_features"].shape[-1]),
        int(sample["hand_token_features"].shape[-1]),
        int(sample["hand_token_metadata"].shape[-1]),
        args.pi3x_relation_dim,
        args.pi3x_heads,
        args.hidden_dim,
        args.layers,
        args.dropout,
        args.max_normalized_translation,
    )
    device = torch.device(args.device)
    model.to(device)
    if args.data_parallel:
        model = nn.DataParallel(model)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_total = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====", flush=True)
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args, None)
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
            "object_names": object_names,
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
