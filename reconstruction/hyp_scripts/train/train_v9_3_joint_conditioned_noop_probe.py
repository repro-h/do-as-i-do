#!/usr/bin/env python3
"""Train a joint-conditioned V9.3 ray-depth refiner with a no-op probe."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import RelativeCrossAttention
from train_v9_2_pi3x_feature_trajectory_depth import (
    FeatureTrajectoryDataset,
    distribution,
    finite_float,
    temporal_differences,
)
from train_v9_camera_hand_residual import (
    KEY_JOINTS,
    load_npz,
    masked_mean,
    scalar_text,
    smooth_l1,
    temporal_loss,
)


MODEL_VERSION = "v9_3_joint_conditioned_noop_probe_v1"
JOINT_IDS = np.concatenate((np.asarray([0], dtype=np.int64), KEY_JOINTS))


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
    parser.add_argument("--joint-dim", type=int, default=64)
    parser.add_argument("--max-joint-token-distance", type=float, default=0.2)
    parser.add_argument("--max-ray-correction-mm", type=float, default=120.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--noop-threshold-mm", type=float, default=5.0)
    parser.add_argument("--noop-positive-weight", type=float, default=5.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-trajectory", type=float, default=0.2)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-residual", type=float, default=0.001)
    parser.add_argument("--w-noop", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


class JointConditionedDataset(FeatureTrajectoryDataset):
    """Associate projected HandFlow joints with nearby compact Pi3X tokens."""

    def __init__(
        self,
        windows: Path,
        global_root: Path,
        pi3x_root: Path,
        max_distance: float,
    ):
        super().__init__(windows, global_root, pi3x_root)
        self.max_distance = max_distance

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
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

        joints = np.asarray(
            glob["pred_joints_3d"], dtype=np.float32
        )[start:end].copy()
        normalized_left = bool(
            np.asarray(glob.get("normalized_left", False)).item()
        )
        if normalized_left:
            joints[..., 0] *= -1.0
        joints = finite_float(joints[:, JOINT_IDS])

        cache_path = (
            self.pi3x_root
            / stream_id
            / "pi3x_geometry_features_compact.npz"
        )
        cache = load_npz(str(cache_path.resolve()))
        frame_indices = np.asarray(cache["frame_indices"], dtype=np.int64)
        expected = np.arange(start, end, dtype=np.int64)
        positions = np.searchsorted(frame_indices, expected)
        if (
            np.any(positions >= len(frame_indices))
            or not np.array_equal(frame_indices[positions], expected)
        ):
            raise ValueError(f"{cache_path} does not cover [{start}, {end})")

        intrinsics = np.asarray(
            cache["intrinsics_resized"][positions], dtype=np.float32
        )
        resized_wh = np.asarray(cache["resized_wh"], dtype=np.float32)
        if resized_wh.ndim > 1:
            resized_wh = resized_wh[0]
        z = joints[..., 2]
        safe_z = np.maximum(z, 1e-6)
        projected_uv = np.stack(
            (
                intrinsics[:, None, 0, 0] * joints[..., 0] / safe_z
                + intrinsics[:, None, 0, 2],
                intrinsics[:, None, 1, 1] * joints[..., 1] / safe_z
                + intrinsics[:, None, 1, 2],
            ),
            axis=-1,
        )
        projected_uv[..., 0] /= max(float(resized_wh[0] - 1), 1.0)
        projected_uv[..., 1] /= max(float(resized_wh[1] - 1), 1.0)
        projected_valid = (
            np.isfinite(projected_uv).all(axis=-1)
            & (z > 1e-5)
            & (projected_uv[..., 0] >= 0.0)
            & (projected_uv[..., 0] <= 1.0)
            & (projected_uv[..., 1] >= 0.0)
            & (projected_uv[..., 1] <= 1.0)
        )
        projected_uv = finite_float(projected_uv)

        grid = np.asarray(cache["geometry_feature_grid_hw"], dtype=np.float32)
        candidate_features = []
        candidate_metadata = []
        candidate_valid = []
        candidate_types = []
        for type_index, prefix in enumerate(("hand", "object", "context")):
            features = finite_float(cache[f"{prefix}_features"][positions])
            points = finite_float(cache[f"{prefix}_points"][positions])
            indices = np.asarray(
                cache[f"{prefix}_indices"][positions], dtype=np.float32
            ).copy()
            uv = np.stack(
                (
                    indices[..., 1] / max(float(grid[1] - 1), 1.0),
                    indices[..., 0] / max(float(grid[0] - 1), 1.0),
                ),
                axis=-1,
            )
            coverage = finite_float(cache[f"{prefix}_coverage"][positions])
            confidence = finite_float(
                cache[f"{prefix}_confidence"][positions]
            )
            metadata = np.concatenate(
                (points, uv, coverage[..., None], confidence[..., None]),
                axis=-1,
            )
            valid = np.asarray(cache[f"{prefix}_valid"][positions], dtype=bool)
            candidate_features.append(features)
            candidate_metadata.append(finite_float(metadata))
            candidate_valid.append(valid)
            candidate_types.append(
                np.full(valid.shape, type_index, dtype=np.int64)
            )

        features = np.concatenate(candidate_features, axis=1)
        metadata = np.concatenate(candidate_metadata, axis=1)
        token_valid = np.concatenate(candidate_valid, axis=1)
        token_types = np.concatenate(candidate_types, axis=1)
        token_uv = metadata[..., 3:5]
        distance = np.linalg.norm(
            projected_uv[:, :, None] - token_uv[:, None], axis=-1
        )
        distance = np.where(token_valid[:, None], distance, np.inf)
        nearest = np.argmin(distance, axis=-1)
        frame = np.arange(end - start)[:, None]
        selected_features = features[frame, nearest]
        selected_metadata = metadata[frame, nearest]
        selected_types = token_types[frame, nearest]
        selected_distance = distance[frame, np.arange(len(JOINT_IDS))[None], nearest]
        selected_valid = (
            projected_valid
            & np.isfinite(selected_distance)
            & (selected_distance <= self.max_distance)
        )
        selected_distance = np.nan_to_num(
            selected_distance, nan=1.0, posinf=1.0, neginf=1.0
        ).astype(np.float32)
        selected_uv = selected_metadata[..., 3:5]
        delta_uv = projected_uv - selected_uv
        type_one_hot = np.eye(3, dtype=np.float32)[selected_types]

        if "valid_translation" in se3:
            sequence_length = len(se3["valid_translation"])
        elif "valid_rotation" in se3:
            sequence_length = len(se3["valid_rotation"])
        else:
            sequence_length = len(glob["frame_ids"])
        observed_source = (
            se3["hand_observed"]
            if "hand_observed" in se3
            else np.ones(sequence_length, dtype=bool)
        )
        observed = np.asarray(observed_source, dtype=bool)[start:end]
        presence_source = (
            se3["hand_presence"]
            if "hand_presence" in se3
            else observed_source
        )
        presence = np.asarray(presence_source, dtype=bool)[start:end]
        hand_token_present = np.asarray(
            cache["hand_valid"][positions], dtype=bool
        ).any(axis=-1)
        flags = np.stack(
            (observed, presence, hand_token_present), axis=-1
        ).astype(np.float32)
        flags = np.broadcast_to(
            flags[:, None], (end - start, len(JOINT_IDS), 3)
        )
        joint_metadata = np.concatenate(
            (
                selected_metadata,
                projected_uv,
                delta_uv,
                selected_distance[..., None],
                type_one_hot,
                flags,
            ),
            axis=-1,
        )
        sample.update({
            "joint_token_features": torch.from_numpy(
                finite_float(selected_features)
            ),
            "joint_token_metadata": torch.from_numpy(
                finite_float(joint_metadata)
            ),
            "joint_token_valid": torch.from_numpy(selected_valid),
            "hand_observed": torch.from_numpy(observed),
            "hand_presence": torch.from_numpy(presence),
            "joint_token_distance": torch.from_numpy(selected_distance),
        })
        return sample


class JointConditionedNoopModel(nn.Module):
    def __init__(
        self,
        local_dim: int,
        feature_dim: int,
        metadata_dim: int,
        joint_metadata_dim: int,
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
        self.joint_feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, args.joint_dim),
        )
        self.joint_metadata_encoder = nn.Sequential(
            nn.LayerNorm(joint_metadata_dim),
            nn.Linear(joint_metadata_dim, args.joint_dim),
            nn.GELU(),
            nn.Linear(args.joint_dim, args.joint_dim),
        )
        fusion_dim = (
            args.pi3x_relation_dim * 3
            + args.joint_dim * 3
            + args.hidden_dim // 2
            + 3
        )
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
        self.noop_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        for head in (self.anchor_head, self.trajectory_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def masked_pool(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weight = valid.to(value.dtype)[..., None]
        return (value * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        joint = self.joint_feature_encoder(batch["joint_token_features"])
        joint = joint + self.joint_metadata_encoder(
            batch["joint_token_metadata"]
        )
        joint = self.masked_pool(joint, batch["joint_token_valid"])
        joint_velocity, joint_acceleration = temporal_differences(joint)

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
                joint,
                joint_velocity,
                joint_acceleration,
                self.local_encoder(batch["local_hand_features"]),
                depth_trajectory,
            ),
            dim=-1,
        )
        temporal, _ = self.temporal(self.frame_encoder(frame))
        anchor = self.anchor_head(temporal.mean(dim=1)).squeeze(-1)
        trajectory = self.trajectory_head(temporal).squeeze(-1)
        trajectory = trajectory - trajectory.mean(dim=1, keepdim=True)
        residual = torch.tanh(anchor[:, None] + trajectory) * self.max_correction
        noop_logit = self.noop_head(temporal).squeeze(-1)
        return residual, noop_logit


def binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = target.astype(bool)
    positive = int(target.sum())
    negative = int((~target).sum())
    if not positive or not negative:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (ranks[target].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def binary_metrics(target: np.ndarray, score: np.ndarray) -> dict:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    if not len(target):
        return {"count": 0}
    predicted = score >= 0.5
    true_positive = int((predicted & target).sum())
    false_positive = int((predicted & ~target).sum())
    false_negative = int((~predicted & target).sum())
    true_negative = int((~predicted & ~target).sum())
    positive_recall = true_positive / max(true_positive + false_negative, 1)
    negative_recall = true_negative / max(true_negative + false_positive, 1)
    return {
        "count": int(len(target)),
        "target_positive_fraction": float(target.mean()),
        "predicted_positive_fraction": float(predicted.mean()),
        "accuracy": float((predicted == target).mean()),
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": positive_recall,
        "specificity": negative_recall,
        "auc": binary_auc(target, score),
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
    loss_names = ("total", "depth", "trajectory", "acceleration", "residual", "noop")
    sums = {name: 0.0 for name in loss_names}
    ray_before_chunks, ray_after_chunks = [], []
    full_before_chunks, full_after_chunks = [], []
    noop_targets, noop_scores, noop_sides = [], [], []
    joint_valid_sum = joint_count = batches = 0
    improved = degraded = evaluated = 0
    iterator = tqdm(loader, desc="train" if training else "val")
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        bad = [
            key for key, value in batch.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if bad:
            raise RuntimeError(f"non-finite batch inputs: {bad}")
        initial_t, target_t, valid = (
            batch["initial_t"], batch["target_t"], batch["valid"]
        )
        ray = initial_t / torch.linalg.norm(
            initial_t, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        target_ray = ((target_t - initial_t) * ray).sum(dim=-1)
        noop_target = (
            target_ray.abs() < args.noop_threshold_mm / 1000.0
        )
        noop_valid = valid & batch["hand_observed"]

        with torch.set_grad_enabled(training):
            predicted_ray, noop_logit = model(batch)
            corrected_t = initial_t + predicted_ray[..., None] * ray
            depth_loss = masked_mean(
                smooth_l1(
                    predicted_ray - target_ray,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                valid,
            )
            trajectory_loss = temporal_loss(
                predicted_ray[..., None], target_ray[..., None], valid, 1,
                args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration_loss = temporal_loss(
                predicted_ray[..., None], target_ray[..., None], valid, 2,
                args.smooth_l1_beta_mm / 1000.0,
            )
            residual_loss = masked_mean(
                smooth_l1(predicted_ray, 0.02), valid
            )
            positive_weight = torch.as_tensor(
                args.noop_positive_weight, device=device
            )
            noop_values = F.binary_cross_entropy_with_logits(
                noop_logit,
                noop_target.to(noop_logit.dtype),
                reduction="none",
                pos_weight=positive_weight,
            )
            noop_loss = masked_mean(noop_values, noop_valid)
            total = (
                args.w_depth * depth_loss
                + args.w_trajectory * trajectory_loss
                + args.w_acceleration * acceleration_loss
                + args.w_residual * residual_loss
                + args.w_noop * noop_loss
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for name, value in zip(
            loss_names,
            (total, depth_loss, trajectory_loss, acceleration_loss, residual_loss, noop_loss),
        ):
            sums[name] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        full_before = torch.linalg.norm(initial_t - target_t, dim=-1)
        full_after = torch.linalg.norm(corrected_t - target_t, dim=-1)
        valid_np = valid.detach().cpu().numpy().astype(bool)
        before_np = full_before.detach().cpu().numpy()[valid_np]
        after_np = full_after.detach().cpu().numpy()[valid_np]
        full_before_chunks.append(before_np)
        full_after_chunks.append(after_np)
        ray_before_chunks.append(target_ray.abs().detach().cpu().numpy()[valid_np])
        ray_after_chunks.append(
            (predicted_ray - target_ray).abs().detach().cpu().numpy()[valid_np]
        )
        improved += int((after_np < before_np).sum())
        degraded += int((after_np > before_np).sum())
        evaluated += len(before_np)
        noop_valid_np = noop_valid.detach().cpu().numpy().astype(bool)
        noop_targets.append(noop_target.detach().cpu().numpy()[noop_valid_np])
        noop_scores.append(torch.sigmoid(noop_logit).detach().cpu().numpy()[noop_valid_np])
        noop_sides.append(batch["side"].detach().cpu().numpy()[noop_valid_np])
        joint_valid_sum += int(batch["joint_token_valid"].sum().item())
        joint_count += int(batch["joint_token_valid"].numel())

    target = np.concatenate(noop_targets) if noop_targets else np.empty(0, bool)
    score = np.concatenate(noop_scores) if noop_scores else np.empty(0)
    sides = np.concatenate(noop_sides) if noop_sides else np.empty(0, int)
    noop_metrics = binary_metrics(target, score)
    noop_by_side = {
        name: binary_metrics(target[sides == value], score[sides == value])
        for name, value in (("left", 0), ("right", 1))
    }
    return {
        **{name: value / max(batches, 1) for name, value in sums.items()},
        "initial_translation": distribution(full_before_chunks),
        "corrected_translation": distribution(full_after_chunks),
        "initial_ray_depth": distribution(ray_before_chunks),
        "corrected_ray_depth": distribution(ray_after_chunks),
        "evaluated": evaluated,
        "improved": improved,
        "degraded": degraded,
        "degraded_fraction": degraded / max(evaluated, 1),
        "joint_token_valid_fraction": joint_valid_sum / max(joint_count, 1),
        "noop_probe": noop_metrics,
        "noop_probe_by_side": noop_by_side,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_data = JointConditionedDataset(
        Path(args.train_windows), Path(args.global_train_root),
        Path(args.pi3x_train_root), args.max_joint_token_distance,
    )
    val_data = JointConditionedDataset(
        Path(args.val_windows), Path(args.global_val_root),
        Path(args.pi3x_val_root), args.max_joint_token_distance,
    )
    sample = train_data[0]
    model = JointConditionedNoopModel(
        int(sample["local_hand_features"].shape[-1]),
        int(sample["hand_token_features"].shape[-1]),
        int(sample["hand_token_metadata"].shape[-1]),
        int(sample["joint_token_metadata"].shape[-1]),
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
    best_total = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====", flush=True)
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
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
            "joint_metadata_dim": int(sample["joint_token_metadata"].shape[-1]),
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
