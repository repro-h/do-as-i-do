#!/usr/bin/env python3
"""Train V9.4 with exact projected-joint sampling from dense Pi3X grids."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_v9_2_pi3x_feature_trajectory_depth import (
    FeatureTrajectoryDataset,
    finite_float,
)
from train_v9_3_joint_conditioned_noop_probe import (
    JOINT_IDS,
    JointConditionedNoopModel,
    run_epoch,
)
from train_v9_camera_hand_residual import load_npz, scalar_text


MODEL_VERSION = "v9_4_dense_joint_pi3x_noop_probe_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--dense-train-root", required=True)
    parser.add_argument("--dense-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pi3x-relation-dim", type=int, default=128)
    parser.add_argument("--pi3x-heads", type=int, default=8)
    parser.add_argument("--joint-dim", type=int, default=64)
    parser.add_argument("--max-ray-correction-mm", type=float, default=120.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--noop-threshold-mm", type=float, default=5.0)
    parser.add_argument("--noop-positive-weight", type=float, default=5.0)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--min-object-coverage", type=float, default=0.25)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-trajectory", type=float, default=0.2)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-residual", type=float, default=0.001)
    parser.add_argument("--w-noop", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def bilinear_sample(grid: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample a [T,H,W,...] grid at normalized [T,J,2] UV positions."""
    grid = np.asarray(grid)
    uv = np.asarray(uv, dtype=np.float32)
    if grid.ndim < 3 or grid.shape[0] != uv.shape[0]:
        raise ValueError(f"Incompatible grid {grid.shape} and uv {uv.shape}")
    height, width = grid.shape[1:3]
    x = np.clip(uv[..., 0], 0.0, 1.0) * max(width - 1, 0)
    y = np.clip(uv[..., 1], 0.0, 1.0) * max(height - 1, 0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    frame = np.arange(grid.shape[0])[:, None]
    v00, v01 = grid[frame, y0, x0], grid[frame, y0, x1]
    v10, v11 = grid[frame, y1, x0], grid[frame, y1, x1]
    while wx.ndim < v00.ndim:
        wx = wx[..., None]
        wy = wy[..., None]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v01 * wx * (1.0 - wy)
        + v10 * (1.0 - wx) * wy
        + v11 * wx * wy
    )


def load_dense_npz(path: Path) -> dict[str, np.ndarray]:
    # Dense windows are too large for the shared 64-entry load_npz cache.
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def image_uv(uv_pixels: np.ndarray, image_wh: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv_pixels, dtype=np.float32).copy()
    uv[..., 0] /= max(float(image_wh[0] - 1), 1.0)
    uv[..., 1] /= max(float(image_wh[1] - 1), 1.0)
    return uv


def patch_uv(
    uv_pixels: np.ndarray,
    image_wh: np.ndarray,
    patch_hw: tuple[int, int],
) -> np.ndarray:
    """Map image pixel centers to patch-token center coordinates."""
    patch_h, patch_w = patch_hw
    uv = np.asarray(uv_pixels, dtype=np.float32).copy()
    x = (uv[..., 0] + 0.5) * patch_w / float(image_wh[0]) - 0.5
    y = (uv[..., 1] + 0.5) * patch_h / float(image_wh[1]) - 0.5
    uv[..., 0] = x / max(float(patch_w - 1), 1.0)
    uv[..., 1] = y / max(float(patch_h - 1), 1.0)
    return uv


def patch_center_points(
    points: np.ndarray,
    patch_hw: tuple[int, int],
    image_wh: np.ndarray,
) -> np.ndarray:
    """Sample a dense pointmap at the centers of the Pi3X patch grid."""
    time, patch_h, patch_w = points.shape[0], *patch_hw
    pixel_y = (
        (np.arange(patch_h, dtype=np.float32) + 0.5)
        * float(image_wh[1]) / patch_h - 0.5
    )
    pixel_x = (
        (np.arange(patch_w, dtype=np.float32) + 0.5)
        * float(image_wh[0]) / patch_w - 0.5
    )
    x, y = np.meshgrid(pixel_x, pixel_y)
    uv = image_uv(
        np.stack((x, y), axis=-1).reshape(1, -1, 2), image_wh
    )
    uv = np.broadcast_to(uv, (time, len(uv[0]), 2))
    return bilinear_sample(points, uv).reshape(time, patch_h, patch_w, 3)


class DenseJointDataset(FeatureTrajectoryDataset):
    """Add exact dense-grid Pi3X samples at projected HandFlow joints."""

    def __init__(
        self,
        windows: Path,
        global_root: Path,
        pi3x_root: Path,
        dense_root: Path,
        min_confidence: float,
        min_object_coverage: float,
    ):
        super().__init__(windows, global_root, pi3x_root)
        self.dense_root = dense_root
        self.min_confidence = min_confidence
        self.min_object_coverage = min_object_coverage

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

        joints = np.asarray(glob["pred_joints_3d"], dtype=np.float32)[
            start:end
        ].copy()
        normalized_left = bool(
            np.asarray(glob.get("normalized_left", False)).item()
        )
        if normalized_left:
            joints[..., 0] *= -1.0
        joints = finite_float(joints[:, JOINT_IDS])

        dense_path = Path(row.get(
            "dense_pi3x_npz",
            self.dense_root / stream_id / "windows"
            / f"window_{start:06d}_{end:06d}.npz",
        )).expanduser().resolve()
        dense = load_dense_npz(dense_path)
        dense_frames = np.asarray(dense["frame_indices"], dtype=np.int64)
        expected = np.arange(start, end, dtype=np.int64)
        if not np.array_equal(dense_frames, expected):
            raise ValueError(
                f"Dense cache {dense_path} has frames {dense_frames[[0, -1]]}, "
                f"expected [{start}, {end})"
            )

        intrinsics = np.asarray(dense["intrinsics_resized"], dtype=np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.broadcast_to(
                intrinsics[None], (end - start, 3, 3)
            )
        resized_wh = np.asarray(dense["resized_wh"], dtype=np.float32).reshape(-1)
        z = joints[..., 2]
        safe_z = np.maximum(z, 1e-6)
        uv_pixels = np.stack((
            intrinsics[:, None, 0, 0] * joints[..., 0] / safe_z
            + intrinsics[:, None, 0, 2],
            intrinsics[:, None, 1, 1] * joints[..., 1] / safe_z
            + intrinsics[:, None, 1, 2],
        ), axis=-1)
        uv = image_uv(uv_pixels, resized_wh)
        projected_valid = (
            np.isfinite(uv).all(axis=-1)
            & (z > 1e-5)
            & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0)
            & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
        )
        uv = finite_float(uv)

        patch_hw = tuple(
            int(value) for value in np.asarray(
                dense["geometry_feature_grid_hw"]
            ).reshape(2)
        )
        feature_uv = patch_uv(uv_pixels, resized_wh, patch_hw)
        features = finite_float(bilinear_sample(
            dense["geometry_patch_features"], feature_uv
        ))
        points = finite_float(bilinear_sample(dense["local_points"], uv))
        confidence = finite_float(bilinear_sample(
            dense["confidence"], uv
        ))
        hand_coverage = finite_float(bilinear_sample(
            dense["hand_patch_coverage"], feature_uv
        ))
        object_coverage = finite_float(bilinear_sample(
            dense["object_patch_coverage"], feature_uv
        ))

        object_points = patch_center_points(
            np.asarray(dense["local_points"], dtype=np.float32),
            patch_hw,
            resized_wh,
        )
        object_mask = (
            np.asarray(dense["object_patch_coverage"], dtype=np.float32)
            >= self.min_object_coverage
        )
        patch_pixels_x = (
            (np.arange(patch_hw[1], dtype=np.float32) + 0.5)
            * float(resized_wh[0]) / patch_hw[1] - 0.5
        )
        patch_pixels_y = (
            (np.arange(patch_hw[0], dtype=np.float32) + 0.5)
            * float(resized_wh[1]) / patch_hw[0] - 0.5
        )
        patch_x, patch_y = np.meshgrid(patch_pixels_x, patch_pixels_y)
        center_uv = image_uv(
            np.stack((patch_x, patch_y), axis=-1).reshape(1, -1, 2),
            resized_wh,
        ).repeat(end - start, axis=0)
        object_confidence = bilinear_sample(
            np.asarray(dense["confidence"], dtype=np.float32), center_uv
        ).reshape(end - start, *patch_hw)
        object_mask &= object_confidence >= self.min_confidence
        object_center = np.zeros((end - start, 3), dtype=np.float32)
        object_scale = np.ones(end - start, dtype=np.float32)
        object_valid = np.zeros(end - start, dtype=bool)
        for frame in range(end - start):
            values = object_points[frame][object_mask[frame]]
            values = values[np.isfinite(values).all(axis=-1)]
            if len(values) < 3:
                continue
            center = np.median(values, axis=0)
            scale = np.median(np.linalg.norm(values - center, axis=-1))
            object_center[frame] = center
            object_scale[frame] = max(float(scale), 1e-4)
            object_valid[frame] = True
        point_relative = (
            points - object_center[:, None]
        ) / object_scale[:, None, None]

        observed_source = se3.get(
            "hand_observed", np.asarray(glob["hand_valid"], dtype=bool)
        )
        presence_source = se3.get("hand_presence", observed_source)
        observed = np.asarray(observed_source, dtype=bool)[start:end]
        presence = np.asarray(presence_source, dtype=bool)[start:end]
        joint_valid = (
            projected_valid
            & np.isfinite(points).all(axis=-1)
            & (confidence >= self.min_confidence)
        )
        flags = np.stack((observed, presence, object_valid), axis=-1)
        flags = np.broadcast_to(
            flags[:, None], (end - start, len(JOINT_IDS), 3)
        ).astype(np.float32)
        local_joint = (joints - joints[:, :1]) / 0.1
        metadata = np.concatenate((
            point_relative,
            confidence[..., None],
            hand_coverage[..., None],
            object_coverage[..., None],
            uv,
            finite_float(local_joint),
            flags,
        ), axis=-1)
        sample.update({
            "joint_token_features": torch.from_numpy(features),
            "joint_token_metadata": torch.from_numpy(finite_float(metadata)),
            "joint_token_valid": torch.from_numpy(joint_valid),
            "hand_observed": torch.from_numpy(observed),
            "hand_presence": torch.from_numpy(presence),
        })
        return sample


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    args: argparse.Namespace,
    sample: dict[str, torch.Tensor],
    val_metrics: dict,
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
        "local_hand_dim": int(sample["local_hand_features"].shape[-1]),
        "pi3x_feature_dim": int(sample["hand_token_features"].shape[-1]),
        "pi3x_metadata_dim": int(sample["hand_token_metadata"].shape[-1]),
        "joint_metadata_dim": int(sample["joint_token_metadata"].shape[-1]),
        "val_total": val_metrics["total"],
        "val_corrected_ray_median_mm": val_metrics[
            "corrected_ray_depth"
        ]["median_mm"],
        "val_degraded_fraction": val_metrics["degraded_fraction"],
    }


def make_dataset(
    windows: str,
    global_root: str,
    compact_root: str,
    dense_root: str,
    args: argparse.Namespace,
) -> DenseJointDataset:
    return DenseJointDataset(
        Path(windows), Path(global_root), Path(compact_root), Path(dense_root),
        args.min_confidence, args.min_object_coverage,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_data = make_dataset(
        args.train_windows, args.global_train_root, args.pi3x_train_root,
        args.dense_train_root, args,
    )
    val_data = make_dataset(
        args.val_windows, args.global_val_root, args.pi3x_val_root,
        args.dense_val_root, args,
    )
    sample = train_data[0]
    audit = {
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "joint_feature_shape": list(sample["joint_token_features"].shape),
        "joint_metadata_shape": list(sample["joint_token_metadata"].shape),
        "joint_valid": int(sample["joint_token_valid"].sum()),
        "joint_total": int(sample["joint_token_valid"].numel()),
        "hand_observed": int(sample["hand_observed"].sum()),
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

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
    history: list[dict] = []
    best_total = best_ray = best_degraded = float("inf")
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
        checkpoint = checkpoint_payload(
            model, epoch, args, sample, val_metrics
        )
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["total"] < best_total:
            best_total = val_metrics["total"]
            torch.save(checkpoint, out_dir / "best.pt")
        ray = val_metrics["corrected_ray_depth"]["median_mm"]
        if ray < best_ray:
            best_ray = ray
            torch.save(checkpoint, out_dir / "best_ray.pt")
        degraded = val_metrics["degraded_fraction"]
        if degraded < best_degraded:
            best_degraded = degraded
            torch.save(checkpoint, out_dir / "best_degraded.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
