#!/usr/bin/env python3
"""Train frozen-v3 selectors between initial and absolute hand-pose candidates."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import (
    AbsoluteObjectFramePoseModel,
    ObjectFrameWindowDataset,
    rotation_angle,
    seed_worker,
)


MODEL_VERSION = "object_frame_frozen_absolute_pose_selector_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--object-embedding-dim", type=int, default=16)
    parser.add_argument("--side-embedding-dim", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--translation-margin-mm", type=float, default=2.0)
    parser.add_argument("--rotation-margin-deg", type=float, default=2.0)
    parser.add_argument("--w-translation-selector", type=float, default=1.0)
    parser.add_argument("--w-rotation-selector", type=float, default=1.0)
    parser.add_argument("--w-temporal", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel-base", action="store_true")
    return parser.parse_args()


def rotation_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


class PoseCandidateSelector(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        num_objects: int,
        object_embedding_dim: int,
        side_embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.object_embedding = nn.Embedding(num_objects, object_embedding_dim)
        self.side_embedding = nn.Embedding(2, side_embedding_dim)
        pose_dim = 3 + 3 + 3 + 6 + 6
        input_dim = context_dim + pose_dim + object_embedding_dim + side_embedding_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.translation_head = nn.Linear(hidden_dim, 1)
        self.rotation_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        initial_t: torch.Tensor,
        candidate_t: torch.Tensor,
        initial_r: torch.Tensor,
        candidate_r: torch.Tensor,
        object_index: torch.Tensor,
        hand_side_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_feature = self.object_embedding(object_index)[:, None].expand(
            -1, context.shape[1], -1
        )
        side_feature = self.side_embedding(hand_side_index)[:, None].expand(
            -1, context.shape[1], -1
        )
        features = torch.cat(
            (
                context,
                initial_t,
                candidate_t,
                candidate_t - initial_t,
                rotation_to_6d(initial_r),
                rotation_to_6d(candidate_r),
                object_feature,
                side_feature,
            ),
            dim=-1,
        )
        encoded = self.encoder(features)
        return (
            self.translation_head(encoded).squeeze(-1),
            self.rotation_head(encoded).squeeze(-1),
        )


def build_base(checkpoint: dict) -> AbsoluteObjectFramePoseModel:
    config = Namespace(**checkpoint["args"])
    model = AbsoluteObjectFramePoseModel(
        checkpoint["input_dim"],
        config.hidden_dim,
        config.layers,
        config.dropout,
        config.max_normalized_translation,
        len(checkpoint["object_names"]),
        checkpoint.get("object_embedding_dim", 0),
        checkpoint.get("pi3x_feature_dim", 0),
        checkpoint.get("pi3x_metadata_dim", 0),
        checkpoint.get("pi3x_relation_dim", 128),
        checkpoint.get("pi3x_heads", 8),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.requires_grad_(False)
    return model


def balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected_logits = logits[mask]
    selected_target = target[mask].to(logits.dtype)
    if not selected_logits.numel():
        return logits.sum() * 0.0
    positive = selected_target.sum()
    negative = selected_target.numel() - positive
    if positive > 0 and negative > 0:
        positive_weight = selected_target.numel() / (2.0 * positive)
        negative_weight = selected_target.numel() / (2.0 * negative)
        weight = torch.where(
            selected_target > 0.5, positive_weight, negative_weight
        )
    else:
        weight = torch.ones_like(selected_target)
    return F.binary_cross_entropy_with_logits(
        selected_logits, selected_target, weight=weight
    )


def distribution(values: list[np.ndarray], scale: float) -> dict:
    if not values:
        return {"count": 0}
    array = np.concatenate(values).astype(np.float64) * scale
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def temporal_probability_loss(
    logits: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    pair_valid = valid[:, 1:] & valid[:, :-1]
    if not pair_valid.any():
        return logits.sum() * 0.0
    probability = torch.sigmoid(logits)
    return torch.abs(probability[:, 1:] - probability[:, :-1])[pair_valid].mean()


def run_epoch(
    base: nn.Module,
    selector: PoseCandidateSelector,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    base.eval()
    selector.train(training)
    totals = {
        "total": 0.0,
        "translation_selector": 0.0,
        "rotation_selector": 0.0,
        "temporal": 0.0,
    }
    batches = 0
    metric_values = {
        name: []
        for name in (
            "initial_t", "candidate_t", "selected_t", "oracle_t",
            "initial_r", "candidate_r", "selected_r", "oracle_r",
        )
    }
    counts = {
        "translation_decisive": 0,
        "translation_target_positive": 0,
        "translation_predicted_positive": 0,
        "translation_correct": 0,
        "rotation_decisive": 0,
        "rotation_target_positive": 0,
        "rotation_predicted_positive": 0,
        "rotation_correct": 0,
    }
    iterator = tqdm(loader, desc="train selector" if training else "val selector")
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            candidate_t, candidate_r, context = base(
                batch["features"],
                batch["object_index"],
                batch.get("hand_token_features"),
                batch.get("hand_token_metadata"),
                batch.get("hand_token_valid"),
                batch.get("key_token_features"),
                batch.get("key_token_metadata"),
                batch.get("key_token_valid"),
                batch.get("key_token_types"),
                return_context=True,
            )
        translation_logits, rotation_logits = selector(
            context,
            batch["initial_translation"],
            candidate_t,
            batch["initial_rotation"],
            candidate_r,
            batch["object_index"],
            batch["hand_side_index"],
        )
        scale = batch["object_scale"][..., None]
        initial_t = batch["initial_translation"] * scale
        target_t = batch["target_translation"] * scale
        candidate_t_metric = candidate_t * scale
        initial_t_error = torch.linalg.norm(initial_t - target_t, dim=-1)
        candidate_t_error = torch.linalg.norm(
            candidate_t_metric - target_t, dim=-1
        )
        initial_r_error = rotation_angle(
            batch["initial_rotation"], batch["target_rotation"]
        )
        candidate_r_error = rotation_angle(candidate_r, batch["target_rotation"])
        translation_improvement = initial_t_error - candidate_t_error
        rotation_improvement = initial_r_error - candidate_r_error
        translation_margin = args.translation_margin_mm / 1000.0
        rotation_margin = math.radians(args.rotation_margin_deg)
        translation_mask = batch["valid_translation"] & (
            torch.abs(translation_improvement) >= translation_margin
        )
        rotation_mask = batch["valid_rotation"] & (
            batch["rotation_weight"] > 0
        ) & (torch.abs(rotation_improvement) >= rotation_margin)
        translation_target = translation_improvement > 0
        rotation_target = rotation_improvement > 0
        translation_loss = balanced_bce(
            translation_logits, translation_target, translation_mask
        )
        rotation_loss = balanced_bce(
            rotation_logits, rotation_target, rotation_mask
        )
        temporal = 0.5 * (
            temporal_probability_loss(
                translation_logits, batch["valid_translation"]
            )
            + temporal_probability_loss(rotation_logits, batch["valid_rotation"])
        )
        total = (
            args.w_translation_selector * translation_loss
            + args.w_rotation_selector * rotation_loss
            + args.w_temporal * temporal
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()

        for name, value in (
            ("total", total),
            ("translation_selector", translation_loss),
            ("rotation_selector", rotation_loss),
            ("temporal", temporal),
        ):
            totals[name] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{totals['total'] / batches:.4f}")

        translation_choose = translation_logits >= 0
        rotation_choose = rotation_logits >= 0
        selected_t = torch.where(
            translation_choose[..., None], candidate_t_metric, initial_t
        )
        selected_r = torch.where(
            rotation_choose[..., None, None],
            candidate_r,
            batch["initial_rotation"],
        )
        selected_t_error = torch.linalg.norm(selected_t - target_t, dim=-1)
        selected_r_error = rotation_angle(selected_r, batch["target_rotation"])
        oracle_t_error = torch.minimum(initial_t_error, candidate_t_error)
        oracle_r_error = torch.minimum(initial_r_error, candidate_r_error)
        valid_t = batch["valid_translation"].detach().cpu().numpy().astype(bool)
        valid_r = batch["valid_rotation"].detach().cpu().numpy().astype(bool)
        for name, value, valid in (
            ("initial_t", initial_t_error, valid_t),
            ("candidate_t", candidate_t_error, valid_t),
            ("selected_t", selected_t_error, valid_t),
            ("oracle_t", oracle_t_error, valid_t),
            ("initial_r", initial_r_error, valid_r),
            ("candidate_r", candidate_r_error, valid_r),
            ("selected_r", selected_r_error, valid_r),
            ("oracle_r", oracle_r_error, valid_r),
        ):
            array = value.detach().cpu().numpy()
            metric_values[name].append(array[valid])
        for prefix, choose, target, mask in (
            ("translation", translation_choose, translation_target, translation_mask),
            ("rotation", rotation_choose, rotation_target, rotation_mask),
        ):
            mask_cpu = mask.detach().cpu().numpy().astype(bool)
            choose_cpu = choose.detach().cpu().numpy()[mask_cpu]
            target_cpu = target.detach().cpu().numpy()[mask_cpu]
            counts[f"{prefix}_decisive"] += int(len(target_cpu))
            counts[f"{prefix}_target_positive"] += int(target_cpu.sum())
            counts[f"{prefix}_predicted_positive"] += int(choose_cpu.sum())
            counts[f"{prefix}_correct"] += int((choose_cpu == target_cpu).sum())

    output = {name: value / max(batches, 1) for name, value in totals.items()}
    output.update(
        initial_translation=distribution(metric_values["initial_t"], 1000.0),
        candidate_translation=distribution(metric_values["candidate_t"], 1000.0),
        selected_translation=distribution(metric_values["selected_t"], 1000.0),
        oracle_translation=distribution(metric_values["oracle_t"], 1000.0),
        initial_rotation=distribution(metric_values["initial_r"], 180.0 / math.pi),
        candidate_rotation=distribution(metric_values["candidate_r"], 180.0 / math.pi),
        selected_rotation=distribution(metric_values["selected_r"], 180.0 / math.pi),
        oracle_rotation=distribution(metric_values["oracle_r"], 180.0 / math.pi),
    )
    for prefix in ("translation", "rotation"):
        count = counts[f"{prefix}_decisive"]
        output[f"{prefix}_selector"] = {
            "count": count,
            "target_positive_fraction": (
                counts[f"{prefix}_target_positive"] / count if count else None
            ),
            "predicted_positive_fraction": (
                counts[f"{prefix}_predicted_positive"] / count if count else None
            ),
            "accuracy": counts[f"{prefix}_correct"] / count if count else None,
        }
    return output


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint_path = Path(args.base_checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base_config = Namespace(**checkpoint["args"])
    base_config.translation_noise_mm = 0.0
    base_config.rotation_noise_deg = 0.0
    base_config.initial_pose_dropout = 0.0
    object_names = list(checkpoint["object_names"])
    object_to_index = {name: index for index, name in enumerate(object_names)}
    train_data = ObjectFrameWindowDataset(
        Path(args.train_windows).expanduser().resolve(),
        base_config,
        augment=False,
        object_to_index=object_to_index,
        pi3x_root=Path(args.pi3x_train_root).expanduser().resolve(),
    )
    val_data = ObjectFrameWindowDataset(
        Path(args.val_windows).expanduser().resolve(),
        base_config,
        augment=False,
        object_to_index=object_to_index,
        pi3x_root=Path(args.pi3x_val_root).expanduser().resolve(),
    )
    device = torch.device(args.device)
    base = build_base(checkpoint).to(device).eval()
    if args.data_parallel_base:
        base = nn.DataParallel(base)
    selector = PoseCandidateSelector(
        base_config.hidden_dim,
        args.hidden_dim,
        len(object_names),
        args.object_embedding_dim,
        args.side_embedding_dim,
        args.dropout,
    ).to(device)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_selection_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== selector epoch {epoch} =====", flush=True)
        train_metrics = run_epoch(
            base, selector, train_loader, device, args, optimizer
        )
        val_metrics = run_epoch(base, selector, val_loader, device, args, None)
        val_selection_score = (
            val_metrics["selected_translation"].get("median", float("inf"))
            / 100.0
            + val_metrics["selected_rotation"].get("median", float("inf"))
            / 90.0
        )
        val_metrics["selection_score"] = val_selection_score
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "epoch": epoch,
            "model_version": MODEL_VERSION,
            "selector_state": selector.state_dict(),
            "base_checkpoint": str(checkpoint_path),
            "base_model_version": checkpoint["model_version"],
            "object_names": object_names,
            "args": vars(args),
            "val_total": val_metrics["total"],
            "val_selection_score": val_selection_score,
        }
        torch.save(payload, out_dir / "last.pt")
        if val_selection_score < best_selection_score:
            best_selection_score = val_selection_score
            torch.save(payload, out_dir / "best.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
