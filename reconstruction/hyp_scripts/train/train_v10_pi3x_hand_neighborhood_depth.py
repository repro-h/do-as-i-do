#!/usr/bin/env python3
"""Train a hand-only camera-ray refiner from local Pi3X neighborhoods."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_v9_2_pi3x_feature_trajectory_depth import distribution
from train_v9_3_joint_conditioned_noop_probe import JOINT_IDS
from train_v9_4_dense_joint_pi3x_noop_probe import (
    bilinear_sample,
    load_dense_npz,
    patch_uv,
)
from train_v9_camera_hand_residual import (
    load_jsonl,
    load_npz,
    masked_mean,
    scalar_text,
    smooth_l1,
    temporal_loss,
)


MODEL_VERSION = "v10_pi3x_hand_neighborhood_ray_residual_v1"


def disable_mha_fastpath() -> None:
    """Avoid the fused eval path failing on padded DataParallel batches."""
    mha_backend = getattr(torch.backends, "mha", None)
    if mha_backend is not None and hasattr(
        mha_backend, "set_fastpath_enabled"
    ):
        mha_backend.set_fastpath_enabled(False)


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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-ray-correction-mm", type=float, default=120.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--small-anchor-mm", type=float, default=5.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=0.2)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-small-anchor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--feature-mode",
        choices=("normal", "feature_zero", "spatial_shuffle"),
        default="normal",
    )
    return parser.parse_args()


def finite_float(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        value, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def dense_path(row: dict, root: Path, stream_id: str) -> Path:
    start, end = int(row["start"]), int(row["end"])
    return Path(row.get(
        "dense_pi3x_npz",
        root / stream_id / "windows" / f"window_{start:06d}_{end:06d}.npz",
    )).expanduser().resolve()


def neighborhood_offsets(size: int) -> np.ndarray:
    if size < 1 or size % 2 != 1:
        raise ValueError("neighborhood-size must be a positive odd number")
    radius = size // 2
    y, x = np.meshgrid(
        np.arange(-radius, radius + 1, dtype=np.float32),
        np.arange(-radius, radius + 1, dtype=np.float32),
        indexing="ij",
    )
    return np.stack((x, y), axis=-1).reshape(-1, 2)


class HandNeighborhoodDataset(Dataset):
    """Sample dense Pi3X features around projected HandFlow joints."""

    def __init__(
        self,
        windows: Path,
        global_root: Path,
        dense_root: Path,
        neighborhood_size: int,
        min_confidence: float,
    ):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.global_root = global_root
        self.dense_root = dense_root
        self.offsets = neighborhood_offsets(neighborhood_size)
        self.min_confidence = min_confidence
        stream_ids = sorted({str(row["stream_id"]) for row in self.rows})
        self.stream_indices = {
            stream_id: index for index, stream_id in enumerate(stream_ids)
        }

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

        pred = np.asarray(
            glob["pred_joints_3d"], dtype=np.float32
        )[start:end].copy()
        target = np.asarray(
            glob["gt_joints_3d"], dtype=np.float32
        )[start:end].copy()
        if bool(np.asarray(glob.get("normalized_left", False)).item()):
            # Pi3X cache uses the original RGB camera frame.
            pred[..., 0] *= -1.0
            target[..., 0] *= -1.0
        pred = pred[:, JOINT_IDS]
        target = target[:, JOINT_IDS]
        valid = (
            np.asarray(glob["hand_valid"], dtype=bool)[start:end]
            & np.asarray(glob["gt_valid"], dtype=bool)[start:end]
            & np.asarray(glob["supervision_valid"], dtype=bool)[start:end]
            & np.isfinite(pred[:, 0]).all(axis=-1)
            & np.isfinite(target[:, 0]).all(axis=-1)
        )
        joint_valid = (
            np.isfinite(pred).all(axis=-1)
            & np.isfinite(target).all(axis=-1)
        )
        pred = finite_float(pred)
        target = finite_float(target)

        dense_file = dense_path(row, self.dense_root, stream_id)
        dense = load_dense_npz(dense_file)
        frame_indices = np.asarray(dense["frame_indices"], dtype=np.int64)
        if not np.array_equal(
            frame_indices, np.arange(start, end, dtype=np.int64)
        ):
            raise ValueError(f"Dense frame mismatch: {dense_file}")
        intrinsics = np.asarray(
            dense["intrinsics_resized"], dtype=np.float32
        )
        if intrinsics.ndim == 2:
            intrinsics = np.broadcast_to(
                intrinsics[None], (end - start, 3, 3)
            )
        image_wh = np.asarray(
            dense["resized_wh"], dtype=np.float32
        ).reshape(2)
        z = pred[..., 2]
        safe_z = np.maximum(z, 1e-6)
        pixels = np.stack((
            intrinsics[:, None, 0, 0] * pred[..., 0] / safe_z
            + intrinsics[:, None, 0, 2],
            intrinsics[:, None, 1, 1] * pred[..., 1] / safe_z
            + intrinsics[:, None, 1, 2],
        ), axis=-1)
        projected_valid = (
            np.isfinite(pixels).all(axis=-1) & (z > 1e-5)
            & (pixels[..., 0] >= 0) & (pixels[..., 0] < image_wh[0])
            & (pixels[..., 1] >= 0) & (pixels[..., 1] < image_wh[1])
        )
        patch_hw = tuple(int(value) for value in np.asarray(
            dense["geometry_feature_grid_hw"]
        ).reshape(2))
        center_uv = patch_uv(pixels, image_wh, patch_hw)
        patch_xy = np.empty_like(center_uv)
        patch_xy[..., 0] = center_uv[..., 0] * max(patch_hw[1] - 1, 1)
        patch_xy[..., 1] = center_uv[..., 1] * max(patch_hw[0] - 1, 1)
        sample_xy = patch_xy[:, :, None] + self.offsets[None, None]
        sample_valid = (
            projected_valid[:, :, None]
            & (sample_xy[..., 0] >= 0) & (sample_xy[..., 0] <= patch_hw[1] - 1)
            & (sample_xy[..., 1] >= 0) & (sample_xy[..., 1] <= patch_hw[0] - 1)
        )
        sample_feature_uv = np.empty_like(sample_xy)
        sample_feature_uv[..., 0] = sample_xy[..., 0] / max(
            patch_hw[1] - 1, 1
        )
        sample_feature_uv[..., 1] = sample_xy[..., 1] / max(
            patch_hw[0] - 1, 1
        )
        time, joints, neighbors = sample_feature_uv.shape[:3]
        flat_feature_uv = sample_feature_uv.reshape(time, joints * neighbors, 2)
        features = bilinear_sample(
            dense["geometry_patch_features"], flat_feature_uv
        ).reshape(time, joints, neighbors, -1)

        sample_pixels = np.empty_like(sample_xy)
        sample_pixels[..., 0] = (
            (sample_xy[..., 0] + 0.5) * image_wh[0] / patch_hw[1] - 0.5
        )
        sample_pixels[..., 1] = (
            (sample_xy[..., 1] + 0.5) * image_wh[1] / patch_hw[0] - 0.5
        )
        sample_image_uv = np.empty_like(sample_pixels)
        sample_image_uv[..., 0] = sample_pixels[..., 0] / max(
            image_wh[0] - 1, 1
        )
        sample_image_uv[..., 1] = sample_pixels[..., 1] / max(
            image_wh[1] - 1, 1
        )
        confidence = bilinear_sample(
            dense["confidence"],
            sample_image_uv.reshape(time, joints * neighbors, 2),
        ).reshape(time, joints, neighbors)
        sample_valid &= np.isfinite(features).all(axis=-1)
        sample_valid &= np.isfinite(confidence)
        sample_valid &= confidence >= self.min_confidence
        offset_scale = max(float(self.offsets[:, 0].max()), 1.0)
        offset_metadata = np.broadcast_to(
            self.offsets[None, None] / offset_scale,
            (time, joints, neighbors, 2),
        )
        metadata = np.concatenate((
            offset_metadata,
            finite_float(confidence)[..., None],
        ), axis=-1)

        observed_source = se3.get(
            "hand_observed", np.asarray(glob["hand_valid"], dtype=bool)
        )
        observed = np.asarray(observed_source, dtype=bool)[start:end]
        side = 0 if scalar_text(glob["hand_side"]) == "left" else 1
        return {
            "neighborhood_features": torch.from_numpy(finite_float(features)),
            "neighborhood_metadata": torch.from_numpy(finite_float(metadata)),
            "neighborhood_valid": torch.from_numpy(sample_valid),
            "pred_joints": torch.from_numpy(pred),
            "target_joints": torch.from_numpy(target),
            "joint_valid": torch.from_numpy(joint_valid),
            "initial_t": torch.from_numpy(pred[:, 0]),
            "target_t": torch.from_numpy(target[:, 0]),
            "valid": torch.from_numpy(valid),
            "observed": torch.from_numpy(observed),
            "side": torch.full((end - start,), side, dtype=torch.long),
            "stream_index": torch.full(
                (end - start,), self.stream_indices[stream_id], dtype=torch.long
            ),
            "frame_index": torch.arange(start, end, dtype=torch.long),
        }


class HandNeighborhoodDepthModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        metadata_dim: int,
        num_joints: int,
        args: argparse.Namespace | SimpleNamespace,
    ):
        super().__init__()
        if args.hidden_dim % 2:
            raise ValueError("hidden-dim must be even")
        self.max_correction = args.max_ray_correction_mm / 1000.0
        self.feature_mode = args.feature_mode
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, args.token_dim),
        )
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, args.token_dim),
            nn.GELU(),
            nn.Linear(args.token_dim, args.token_dim),
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
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["neighborhood_features"]
        if self.feature_mode == "feature_zero":
            features = torch.zeros_like(features)
        elif self.feature_mode == "spatial_shuffle":
            features = torch.roll(features, shifts=7, dims=3)
        token = self.feature_encoder(features)
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
        weight = torch.softmax(score, dim=3)
        weight = weight * valid.to(weight.dtype)
        weight = weight / weight.sum(dim=3, keepdim=True).clamp_min(1e-6)
        joint = (token * weight[..., None]).sum(dim=3)

        batch_size, time, joints, dim = joint.shape
        joint_flat = joint.reshape(batch_size * time, joints, dim)
        joint_valid = valid.any(dim=3).reshape(batch_size * time, joints)
        safe_joint_valid = joint_valid.clone()
        safe_joint_valid[~safe_joint_valid.any(dim=1), 0] = True
        joint_flat = self.joint_encoder(
            joint_flat, src_key_padding_mask=~safe_joint_valid
        )
        joint = joint_flat.reshape(batch_size, time, joints, dim)
        joint_weight = joint_valid.reshape(
            batch_size, time, joints
        ).to(joint.dtype)
        pooled = (joint * joint_weight[..., None]).sum(dim=2)
        pooled = pooled / joint_weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        wrist = joint[:, :, 0]
        temporal, _ = self.temporal(
            self.frame_encoder(torch.cat((wrist, pooled), dim=-1))
        )
        return torch.tanh(self.head(temporal).squeeze(-1)) * self.max_correction


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace | SimpleNamespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    names = ("total", "depth", "joint", "velocity", "acceleration", "small_anchor")
    sums = {name: 0.0 for name in names}
    metric_names = ("initial_full", "corrected_full", "initial_ray", "corrected_ray")
    metrics = {name: [] for name in metric_names}
    side_metrics = {
        side: {name: [] for name in metric_names} for side in ("left", "right")
    }
    valid_tokens = total_tokens = improved = degraded = evaluated = batches = 0
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
            joint_loss = masked_mean(
                smooth_l1(
                    predicted_ray[:, :, None] - joint_target_ray,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                joint_mask,
            )
            velocity_loss = temporal_loss(
                predicted_ray[..., None], target_ray[..., None], valid, 1,
                args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration_loss = temporal_loss(
                predicted_ray[..., None], target_ray[..., None], valid, 2,
                args.smooth_l1_beta_mm / 1000.0,
            )
            small = valid & (
                target_ray.abs() <= args.small_anchor_mm / 1000.0
            )
            small_anchor = masked_mean(
                smooth_l1(predicted_ray, 0.005), small
            )
            total = (
                args.w_depth * depth_loss
                + args.w_joint * joint_loss
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
            total, depth_loss, joint_loss, velocity_loss,
            acceleration_loss, small_anchor,
        )
        for name, value in zip(names, values):
            sums[name] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        initial_full = torch.linalg.norm(initial_t - target_t, dim=-1)
        corrected_full = torch.linalg.norm(corrected_t - target_t, dim=-1)
        values_for_metrics = {
            "initial_full": initial_full,
            "corrected_full": corrected_full,
            "initial_ray": target_ray.abs(),
            "corrected_ray": (predicted_ray - target_ray).abs(),
        }
        valid_np = valid.detach().cpu().numpy().astype(bool)
        side_np = batch["side"].detach().cpu().numpy()
        for name, value in values_for_metrics.items():
            array = value.detach().cpu().numpy()
            metrics[name].append(array[valid_np])
            for side_name, side_value in (("left", 0), ("right", 1)):
                mask = valid_np & (side_np == side_value)
                side_metrics[side_name][name].append(array[mask])
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
        "feature_dim": int(sample["neighborhood_features"].shape[-1]),
        "metadata_dim": int(sample["neighborhood_metadata"].shape[-1]),
        "num_joints": int(sample["neighborhood_features"].shape[1]),
        "val_total": val["total"],
        "val_ray_median_mm": val["corrected_ray_depth"]["median_mm"],
        "val_degraded_fraction": val["degraded_fraction"],
    }


def make_dataset(
    windows: str,
    global_root: str,
    dense_root: str,
    args: argparse.Namespace | SimpleNamespace,
) -> HandNeighborhoodDataset:
    return HandNeighborhoodDataset(
        Path(windows), Path(global_root), Path(dense_root),
        args.neighborhood_size, args.min_confidence,
    )


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
        args.val_windows, args.global_val_root, args.dense_val_root, args,
    )
    sample = train_data[0]
    audit = {
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "feature_shape": list(sample["neighborhood_features"].shape),
        "metadata_shape": list(sample["neighborhood_metadata"].shape),
        "valid_tokens": int(sample["neighborhood_valid"].sum()),
        "total_tokens": int(sample["neighborhood_valid"].numel()),
        "valid_frames": int(sample["valid"].sum()),
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    model = HandNeighborhoodDepthModel(
        int(sample["neighborhood_features"].shape[-1]),
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
        ray = val["corrected_ray_depth"]["median_mm"]
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
