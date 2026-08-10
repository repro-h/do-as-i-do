#!/usr/bin/env python3
"""Train a per-joint Pi3X observer with rigid hand-depth aggregation."""

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

from train_v10_pi3x_hand_neighborhood_depth import disable_mha_fastpath
from train_v11_2_handflow_latent_pi3x_ray_residual import (
    FEATURE_MODES,
    HandFlowLatentPi3XRayResidual,
    make_dataset,
)
from train_v9_2_pi3x_feature_trajectory_depth import distribution
from train_v9_camera_hand_residual import (
    masked_mean,
    smooth_l1,
    temporal_loss,
)


MODEL_VERSION = "v11_4_per_joint_rigid_pi3x_depth_v1"


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
    parser.add_argument("--epochs", type=int, default=15)
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
    parser.add_argument("--max-ray-correction-mm", type=float, default=120.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--small-anchor-mm", type=float, default=5.0)
    parser.add_argument("--joint-inlier-mm", type=float, default=15.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-joint-observation", type=float, default=0.5)
    parser.add_argument("--w-rigid-consistency", type=float, default=0.05)
    parser.add_argument("--w-reliability", type=float, default=0.1)
    parser.add_argument("--w-noop", type=float, default=0.2)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-small-anchor", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--feature-mode", choices=FEATURE_MODES, default="normal"
    )
    return parser.parse_args()


class PerJointRigidPi3XDepth(HandFlowLatentPi3XRayResidual):
    """Predict joint observations, their reliability, and a no-op gate."""

    def __init__(
        self,
        decoder_dim: int,
        metric_dim: int,
        metadata_dim: int,
        latent_dim: int,
        num_joints: int,
        args: argparse.Namespace,
    ):
        super().__init__(
            decoder_dim,
            metric_dim,
            metadata_dim,
            latent_dim,
            num_joints,
            args,
        )
        self.joint_frame_encoder = nn.Sequential(
            nn.LayerNorm(args.token_dim * 4),
            nn.Linear(args.token_dim * 4, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.joint_temporal = nn.GRU(
            args.hidden_dim,
            args.hidden_dim // 2,
            num_layers=args.layers,
            batch_first=True,
            bidirectional=True,
            dropout=args.dropout if args.layers > 1 else 0.0,
        )
        self.joint_depth_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        self.joint_reliability_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        self.noop_head = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.joint_depth_head[-1].weight)
        nn.init.zeros_(self.joint_depth_head[-1].bias)
        nn.init.zeros_(self.joint_reliability_head[-1].weight)
        nn.init.zeros_(self.joint_reliability_head[-1].bias)
        nn.init.zeros_(self.noop_head[-1].weight)
        nn.init.zeros_(self.noop_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        decoder = batch["neighborhood_features"]
        points = batch["neighborhood_points"]
        metadata = batch["neighborhood_metadata"]
        metric = batch["metric_window_features"]
        latent = batch["handflow_translation_latent"]
        mode = self.feature_mode
        if mode in ("pi3x_zero", "all_zero"):
            decoder = torch.zeros_like(decoder)
            points = torch.zeros_like(points)
            metadata = torch.zeros_like(metadata)
            metric = torch.zeros_like(metric)
        if mode in ("latent_zero", "all_zero"):
            latent = torch.zeros_like(latent)
        if mode == "pi3x_time_reverse":
            decoder = torch.flip(decoder, dims=(1,))
            points = torch.flip(points, dims=(1,))
            metadata = torch.flip(metadata, dims=(1,))
            metric = torch.flip(metric, dims=(1,))
        if mode == "latent_time_reverse":
            latent = torch.flip(latent, dims=(1,))

        point_input = torch.cat(
            (points, torch.linalg.norm(points, dim=-1, keepdim=True)), dim=-1
        )
        token = (
            self.decoder_encoder(decoder)
            + self.point_encoder(point_input)
            + self.metadata_encoder(metadata)
        )
        joint_ids = torch.arange(
            token.shape[2], device=token.device
        ).view(1, 1, -1, 1)
        token = token + self.joint_embedding(joint_ids)

        neighbor_valid = batch["neighborhood_valid"]
        score = self.local_score(token).squeeze(-1)
        score = score.masked_fill(~neighbor_valid, -1e4)
        neighbor_weight = torch.softmax(score, dim=3)
        neighbor_weight = neighbor_weight * neighbor_valid.to(score.dtype)
        neighbor_weight = neighbor_weight / neighbor_weight.sum(
            dim=3, keepdim=True
        ).clamp_min(1e-6)
        joint = (token * neighbor_weight[..., None]).sum(dim=3)

        batch_size, time, joints, dim = joint.shape
        joint_valid = neighbor_valid.any(dim=3)
        flat_valid = joint_valid.reshape(batch_size * time, joints)
        safe_valid = flat_valid.clone()
        safe_valid[~safe_valid.any(dim=1), 0] = True
        encoded = self.joint_encoder(
            joint.reshape(batch_size * time, joints, dim),
            src_key_padding_mask=~safe_valid,
        ).reshape(batch_size, time, joints, dim)

        valid_weight = joint_valid.to(encoded.dtype)
        pooled = (encoded * valid_weight[..., None]).sum(dim=2)
        pooled = pooled / valid_weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        metric_token = self.metric_encoder(metric)
        latent_token = self.latent_encoder(latent)
        context = torch.cat((
            encoded,
            pooled[:, :, None].expand(-1, -1, joints, -1),
            metric_token[:, :, None].expand(-1, -1, joints, -1),
            latent_token[:, :, None].expand(-1, -1, joints, -1),
        ), dim=-1)
        context = self.joint_frame_encoder(context)
        temporal_input = context.permute(0, 2, 1, 3).reshape(
            batch_size * joints, time, -1
        )
        temporal, _ = self.joint_temporal(temporal_input)
        temporal = temporal.reshape(
            batch_size, joints, time, -1
        ).permute(0, 2, 1, 3)

        joint_correction = torch.tanh(
            self.joint_depth_head(temporal).squeeze(-1)
        ) * self.max_correction
        reliability_logits = self.joint_reliability_head(
            temporal
        ).squeeze(-1)
        reliability = torch.sigmoid(reliability_logits)
        reliability = reliability * joint_valid.to(reliability.dtype)
        denominator = reliability.sum(dim=2)
        rigid_correction = (
            joint_correction * reliability
        ).sum(dim=2) / denominator.clamp_min(1e-6)
        rigid_correction = torch.where(
            denominator > 0,
            rigid_correction,
            torch.zeros_like(rigid_correction),
        )

        temporal_weight = joint_valid.to(temporal.dtype)
        frame_feature = (temporal * temporal_weight[..., None]).sum(dim=2)
        frame_feature = frame_feature / temporal_weight.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        noop_logits = self.noop_head(frame_feature).squeeze(-1)
        correction_gate = 1.0 - torch.sigmoid(noop_logits)
        predicted_ray = rigid_correction * correction_gate
        return {
            "predicted_ray": predicted_ray,
            "rigid_correction": rigid_correction,
            "joint_correction": joint_correction,
            "reliability_logits": reliability_logits,
            "reliability": reliability,
            "noop_logits": noop_logits,
            "correction_gate": correction_gate,
            "joint_observation_valid": joint_valid,
        }


def binary_stats(
    logits: list[np.ndarray], targets: list[np.ndarray]
) -> dict[str, float | int]:
    if not logits:
        return {"count": 0, "accuracy": 0.0, "positive_fraction": 0.0}
    score = np.concatenate(logits)
    target = np.concatenate(targets).astype(bool)
    predicted = score >= 0.0
    return {
        "count": int(len(target)),
        "accuracy": float((predicted == target).mean()),
        "positive_fraction": float(target.mean()),
        "predicted_positive_fraction": float(predicted.mean()),
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
    names = (
        "total", "depth", "joint_observation", "rigid_consistency",
        "reliability", "noop", "velocity", "acceleration", "small_anchor",
    )
    sums = {name: 0.0 for name in names}
    metric_names = ("initial_full", "corrected_full", "initial_ray", "corrected_ray")
    metrics = {name: [] for name in metric_names}
    side_metrics = {
        side: {name: [] for name in metric_names} for side in ("left", "right")
    }
    noop_logits_all: list[np.ndarray] = []
    noop_targets_all: list[np.ndarray] = []
    reliability_logits_all: list[np.ndarray] = []
    reliability_targets_all: list[np.ndarray] = []
    improved = degraded = evaluated = valid_tokens = total_tokens = batches = 0
    iterator = tqdm(loader, desc="train" if training else "val")

    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
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
        target_ray = ((target_t - initial_t) * ray).sum(dim=-1)
        joint_target_ray = (
            (batch["target_joints"] - batch["pred_joints"])
            * ray[:, :, None]
        ).sum(dim=-1)
        joint_mask = valid[:, :, None] & batch["joint_valid"]
        reliability_target = (
            (joint_target_ray - target_ray[:, :, None]).abs()
            <= args.joint_inlier_mm / 1000.0
        )
        noop_target = target_ray.abs() <= args.small_anchor_mm / 1000.0

        with torch.set_grad_enabled(training):
            output = model(batch)
            predicted_ray = output["predicted_ray"]
            corrected_t = initial_t + predicted_ray[..., None] * ray
            depth_loss = masked_mean(
                smooth_l1(
                    predicted_ray - target_ray,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                valid,
            )
            observation_mask = joint_mask & output["joint_observation_valid"]
            joint_observation_loss = masked_mean(
                smooth_l1(
                    output["joint_correction"] - joint_target_ray,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                observation_mask,
            )
            rigid_consistency_loss = masked_mean(
                smooth_l1(
                    output["joint_correction"]
                    - output["rigid_correction"][:, :, None],
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                observation_mask,
            )
            reliability_loss = masked_mean(
                F.binary_cross_entropy_with_logits(
                    output["reliability_logits"],
                    reliability_target.to(output["reliability_logits"].dtype),
                    reduction="none",
                ),
                observation_mask,
            )
            noop_loss = masked_mean(
                F.binary_cross_entropy_with_logits(
                    output["noop_logits"],
                    noop_target.to(output["noop_logits"].dtype),
                    reduction="none",
                ),
                valid,
            )
            velocity_loss = temporal_loss(
                predicted_ray[..., None], target_ray[..., None], valid, 1,
                args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration_loss = temporal_loss(
                predicted_ray[..., None], target_ray[..., None], valid, 2,
                args.smooth_l1_beta_mm / 1000.0,
            )
            small_anchor = masked_mean(
                smooth_l1(predicted_ray, 0.005), valid & noop_target
            )
            total = (
                args.w_depth * depth_loss
                + args.w_joint_observation * joint_observation_loss
                + args.w_rigid_consistency * rigid_consistency_loss
                + args.w_reliability * reliability_loss
                + args.w_noop * noop_loss
                + args.w_velocity * velocity_loss
                + args.w_acceleration * acceleration_loss
                + args.w_small_anchor * small_anchor
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        values = (
            total, depth_loss, joint_observation_loss, rigid_consistency_loss,
            reliability_loss, noop_loss, velocity_loss, acceleration_loss,
            small_anchor,
        )
        for name, value in zip(names, values):
            sums[name] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        initial_full = torch.linalg.norm(initial_t - target_t, dim=-1)
        corrected_full = torch.linalg.norm(corrected_t - target_t, dim=-1)
        metric_values = {
            "initial_full": initial_full,
            "corrected_full": corrected_full,
            "initial_ray": target_ray.abs(),
            "corrected_ray": (predicted_ray - target_ray).abs(),
        }
        valid_np = valid.detach().cpu().numpy().astype(bool)
        side_np = batch["side"].detach().cpu().numpy()
        for name, value in metric_values.items():
            array = value.detach().cpu().numpy()
            metrics[name].append(array[valid_np])
            for side_name, side_value in (("left", 0), ("right", 1)):
                mask = valid_np & (side_np == side_value)
                side_metrics[side_name][name].append(array[mask])
        observation_np = observation_mask.detach().cpu().numpy().astype(bool)
        reliability_logits_all.append(
            output["reliability_logits"].detach().cpu().numpy()[observation_np]
        )
        reliability_targets_all.append(
            reliability_target.detach().cpu().numpy()[observation_np]
        )
        noop_logits_all.append(
            output["noop_logits"].detach().cpu().numpy()[valid_np]
        )
        noop_targets_all.append(noop_target.detach().cpu().numpy()[valid_np])
        before = initial_full.detach().cpu().numpy()[valid_np]
        after = corrected_full.detach().cpu().numpy()[valid_np]
        improved += int((after < before).sum())
        degraded += int((after > before + 1e-6).sum())
        evaluated += len(before)
        valid_tokens += int(batch["neighborhood_valid"].sum())
        total_tokens += int(batch["neighborhood_valid"].numel())

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
        "noop": binary_stats(noop_logits_all, noop_targets_all),
        "joint_reliability": binary_stats(
            reliability_logits_all, reliability_targets_all
        ),
        "evaluated": evaluated,
        "improved": improved,
        "degraded": degraded,
        "degraded_fraction": degraded / max(evaluated, 1),
        "neighborhood_valid_fraction": valid_tokens / max(total_tokens, 1),
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
        "latent_dim": int(sample["handflow_translation_latent"].shape[-1]),
        "num_joints": int(sample["neighborhood_features"].shape[1]),
        "output": "per_joint_observation_then_rigid_ray_correction",
        "initial_pose_usage": "2d_sampling_and_output_composition_only",
        "explicit_hand_depth_input": False,
        "val_total": val["total"],
        "val_ray_median_mm": val["corrected_ray_depth"]["median_mm"],
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
        "metric_window_features", "neighborhood_metadata",
        "handflow_translation_latent",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"Training input lacks {missing}")
    audit = {
        "model": MODEL_VERSION,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "joint_count": int(sample["neighborhood_features"].shape[1]),
        "decoder_feature_shape": list(sample["neighborhood_features"].shape),
        "handflow_latent_shape": list(
            sample["handflow_translation_latent"].shape
        ),
        "sample_cache_mirrored": bool(sample["cache_mirrored"].all()),
        "explicit_hand_depth_input": False,
    }
    print(json.dumps(audit, indent=2), flush=True)
    model = PerJointRigidPi3XDepth(
        int(sample["neighborhood_features"].shape[-1]),
        int(sample["metric_window_features"].shape[-1]),
        int(sample["neighborhood_metadata"].shape[-1]),
        int(sample["handflow_translation_latent"].shape[-1]),
        int(sample["neighborhood_features"].shape[1]),
        args,
    )
    if args.audit_only:
        model.eval()
        batch = {
            key: value.unsqueeze(0)
            for key, value in sample.items()
        }
        with torch.no_grad():
            output = model(batch)
        forward_audit = {
            key: {
                "shape": list(value.shape),
                "finite": bool(torch.isfinite(value).all()),
            }
            for key, value in output.items()
            if value.is_floating_point()
        }
        print(json.dumps({"forward": forward_audit}, indent=2), flush=True)
        if not all(row["finite"] for row in forward_audit.values()):
            raise RuntimeError("Non-finite audit-only model output")
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
        ray = val["corrected_ray_depth"]["median_mm"]
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
