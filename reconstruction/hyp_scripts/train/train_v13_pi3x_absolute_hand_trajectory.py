#!/usr/bin/env python3
"""Train an absolute camera-ray hand trajectory from dense Pi3X tokens.

HandFlow contributes only 2D joint locations used as cross-attention queries.
Its 3D translation/depth is retained for baseline evaluation and output-ray
composition, but is never passed to the model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
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
from train_v10_pi3x_hand_neighborhood_depth import disable_mha_fastpath


MODEL_VERSION = "v13_pi3x_absolute_hand_trajectory_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--dense-train-root", required=True)
    parser.add_argument("--dense-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial-bias", type=float, default=6.0)
    parser.add_argument("--max-window-size", type=int, default=64)
    parser.add_argument("--max-depth-m", type=float, default=2.5)
    parser.add_argument("--initial-depth-m", type=float, default=0.85)
    parser.add_argument("--max-relative-offset-m", type=float, default=0.45)
    parser.add_argument("--query-dropout", type=float, default=0.2)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--w-absolute", type=float, default=1.0)
    parser.add_argument("--w-relative", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-overlap", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--feature-mode",
        choices=(
            "normal",
            "point_zero",
            "metric_zero",
            "all_zero",
            "spatial_shuffle",
            "time_reverse",
            "joint_query_zero",
            "global_query_zero",
        ),
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


def interpolate_uv(
    uv: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate 2D tracks for output rays while preserving query validity."""
    output = finite_float(uv)
    original_valid = valid.copy()
    interpolated_valid = valid.copy()
    time, joints = valid.shape
    frames = np.arange(time, dtype=np.float32)
    for joint in range(joints):
        indices = np.flatnonzero(valid[:, joint])
        if len(indices) == 0:
            continue
        interpolated_valid[:, joint] = True
        for axis in range(2):
            output[:, joint, axis] = np.interp(
                frames, indices.astype(np.float32), output[indices, joint, axis]
            )
    return output, original_valid, interpolated_valid


class DenseTrajectoryDataset(Dataset):
    """Expose full Pi3X grids and HandFlow-derived 2D query locations."""

    def __init__(
        self,
        windows: Path,
        global_root: Path,
        dense_root: Path,
        query_dropout: float = 0.0,
    ):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.global_root = global_root
        self.dense_root = dense_root
        self.query_dropout = float(query_dropout)
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
        time = end - start
        se3 = load_npz(str(Path(row["supervision_npz"]).resolve()))
        global_path = Path(
            scalar_text(se3["source_global_supervision"])
        ).expanduser().resolve()
        if not global_path.is_file():
            global_path = self.global_root / f"{stream_id}.npz"
        glob = load_npz(str(global_path))

        pred_all = np.asarray(glob["pred_joints_3d"], dtype=np.float32)[start:end]
        target_all = np.asarray(glob["gt_joints_3d"], dtype=np.float32)[start:end]
        pred = pred_all[:, JOINT_IDS].copy()
        target = target_all[:, JOINT_IDS].copy()
        normalized_left = bool(
            np.asarray(glob.get("normalized_left", False)).item()
        )

        dense_file = dense_path(row, self.dense_root, stream_id)
        dense = load_dense_npz(dense_file)
        frame_indices = np.asarray(dense["frame_indices"], dtype=np.int64)
        if not np.array_equal(
            frame_indices, np.arange(start, end, dtype=np.int64)
        ):
            raise ValueError(f"Dense frame mismatch: {dense_file}")
        cache_mirrored = bool(
            np.asarray(dense.get("horizontal_mirror", False)).item()
        )
        if cache_mirrored and not normalized_left:
            raise ValueError(
                f"Mirrored cache used with non-normalized data: {dense_file}"
            )
        if normalized_left and not cache_mirrored:
            pred[..., 0] *= -1.0
            target[..., 0] *= -1.0

        intrinsics = np.asarray(dense["intrinsics_resized"], dtype=np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.broadcast_to(intrinsics[None], (time, 3, 3)).copy()
        image_wh = np.asarray(dense["resized_wh"], dtype=np.float32).reshape(2)
        z = pred[..., 2]
        safe_z = np.maximum(z, 1e-6)
        pixels = np.stack((
            intrinsics[:, None, 0, 0] * pred[..., 0] / safe_z
            + intrinsics[:, None, 0, 2],
            intrinsics[:, None, 1, 1] * pred[..., 1] / safe_z
            + intrinsics[:, None, 1, 2],
        ), axis=-1)
        query_valid = (
            np.isfinite(pred).all(axis=-1)
            & np.isfinite(pixels).all(axis=-1)
            & (z > 1e-5)
            & (pixels[..., 0] >= 0)
            & (pixels[..., 0] < image_wh[0])
            & (pixels[..., 1] >= 0)
            & (pixels[..., 1] < image_wh[1])
        )
        pixels, query_valid, interpolated_valid = interpolate_uv(
            pixels, query_valid
        )
        ray_valid = interpolated_valid[:, 0]
        patch_hw = tuple(int(value) for value in np.asarray(
            dense["geometry_feature_grid_hw"]
        ).reshape(2))
        query_uv = patch_uv(pixels, image_wh, patch_hw) * 2.0 - 1.0

        # Randomly hide entire frames. Their targets remain valid so temporal
        # and global queries learn to bridge occlusion instead of dropping it.
        if self.query_dropout > 0.0:
            dropped = np.random.random(time) < self.query_dropout
            query_valid[dropped] = False
        query_uv = finite_float(query_uv)

        features = finite_float(dense["geometry_patch_features"])
        if features.shape[:3] != (time, *patch_hw):
            raise ValueError(
                f"Unexpected feature shape {features.shape}: {dense_file}"
            )
        grid_y, grid_x = np.meshgrid(
            np.linspace(-1.0, 1.0, patch_hw[0], dtype=np.float32),
            np.linspace(-1.0, 1.0, patch_hw[1], dtype=np.float32),
            indexing="ij",
        )
        grid_uv = np.stack((grid_x, grid_y), axis=-1)
        grid_uv = np.broadcast_to(grid_uv[None], (time, *grid_uv.shape)).copy()

        confidence = np.asarray(dense.get("confidence", []), dtype=np.float32)
        if confidence.size:
            image_uv = (grid_uv + 1.0) * 0.5
            patch_confidence = bilinear_sample(confidence, image_uv.reshape(time, -1, 2))
            patch_confidence = patch_confidence.reshape(time, *patch_hw)
        else:
            patch_confidence = np.ones((time, *patch_hw), dtype=np.float32)
        grid_valid = (
            np.isfinite(features).all(axis=-1)
            & np.isfinite(patch_confidence)
        )

        metric = np.asarray(
            dense.get("metric_window_features", []), dtype=np.float32
        )
        if metric.size == 0:
            raise KeyError(
                f"{dense_file} lacks metric_window_features; re-export with "
                "--export-metric-features"
            )
        metric = finite_float(metric.reshape(-1, metric.shape[-1]).mean(axis=0))

        initial_t = finite_float(pred[:, 0])
        target_t = finite_float(target[:, 0])
        gt_valid = np.asarray(glob["gt_valid"], dtype=bool)[start:end]
        valid = (
            gt_valid
            & np.isfinite(target[:, 0]).all(axis=-1)
            & np.isfinite(initial_t).all(axis=-1)
            & ray_valid
        )
        observed_source = se3.get("hand_observed", glob["hand_valid"])
        observed = np.asarray(observed_source, dtype=bool)[start:end]
        side = 0 if scalar_text(glob["hand_side"]) == "left" else 1

        root_pixels = pixels[:, 0]
        root_rays = np.stack((
            (root_pixels[:, 0] - intrinsics[:, 0, 2]) / intrinsics[:, 0, 0],
            (root_pixels[:, 1] - intrinsics[:, 1, 2]) / intrinsics[:, 1, 1],
            np.ones(time, dtype=np.float32),
        ), axis=-1)
        root_rays /= np.linalg.norm(root_rays, axis=-1, keepdims=True).clip(1e-6)

        return {
            "point_features": torch.from_numpy(features),
            "grid_uv": torch.from_numpy(grid_uv),
            "grid_confidence": torch.from_numpy(finite_float(patch_confidence)),
            "grid_valid": torch.from_numpy(grid_valid),
            "metric_window_features": torch.from_numpy(metric),
            "joint_uv": torch.from_numpy(query_uv),
            "joint_query_valid": torch.from_numpy(query_valid),
            "output_ray": torch.from_numpy(finite_float(root_rays)),
            "initial_t": torch.from_numpy(initial_t),
            "target_t": torch.from_numpy(target_t),
            "valid": torch.from_numpy(valid),
            "observed": torch.from_numpy(observed),
            "side": torch.full((time,), side, dtype=torch.long),
            "stream_index": torch.tensor(
                self.stream_indices[stream_id], dtype=torch.long
            ),
            "frame_index": torch.arange(start, end, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "end": torch.tensor(end, dtype=torch.long),
            "cache_mirrored": torch.tensor(cache_mirrored, dtype=torch.bool),
        }


class OverlapPairDataset(Dataset):
    """Pair adjacent windows from the same stream for overlap consistency."""

    def __init__(self, base: DenseTrajectoryDataset):
        self.base = base
        by_stream: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for index, row in enumerate(base.rows):
            by_stream[str(row["stream_id"])].append(
                (int(row["start"]), int(row["end"]), index)
            )
        self.pairs: list[tuple[int, int]] = []
        for rows in by_stream.values():
            rows.sort()
            for first, second in zip(rows, rows[1:]):
                if second[0] < first[1]:
                    self.pairs.append((first[2], second[2]))
        if not self.pairs:
            raise RuntimeError("No overlapping adjacent windows were found")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        first, second = self.pairs[index]
        return {"first": self.base[first], "second": self.base[second]}


class Pi3XAbsoluteTrajectoryModel(nn.Module):
    def __init__(
        self,
        point_dim: int,
        metric_dim: int,
        num_joints: int,
        args: argparse.Namespace | SimpleNamespace,
    ):
        super().__init__()
        if args.token_dim % args.heads:
            raise ValueError("token-dim must be divisible by heads")
        if not 0.0 < args.initial_depth_m < args.max_depth_m:
            raise ValueError("initial-depth-m must be in (0, max-depth-m)")
        self.max_depth = float(args.max_depth_m)
        self.max_relative_offset = float(args.max_relative_offset_m)
        self.spatial_bias = float(args.spatial_bias)
        self.feature_mode = str(args.feature_mode)
        self.heads = int(args.heads)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_dim),
            nn.Linear(point_dim, args.token_dim),
        )
        self.grid_encoder = nn.Sequential(
            nn.Linear(3, args.token_dim),
            nn.GELU(),
            nn.Linear(args.token_dim, args.token_dim),
        )
        self.metric_encoder = nn.Sequential(
            nn.LayerNorm(metric_dim),
            nn.Linear(metric_dim, args.token_dim),
            nn.GELU(),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(5, args.token_dim),
            nn.GELU(),
            nn.Linear(args.token_dim, args.token_dim),
        )
        self.joint_embedding = nn.Embedding(num_joints, args.token_dim)
        self.missing_embedding = nn.Parameter(torch.zeros(args.token_dim))
        self.global_query = nn.Parameter(torch.randn(args.token_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            args.token_dim, args.heads, dropout=args.dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(args.token_dim)
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(args.token_dim * 4 + 2),
            nn.Linear(args.token_dim * 4 + 2, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.temporal_position = nn.Parameter(
            torch.randn(args.max_window_size, args.hidden_dim) * 0.01
        )
        layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=args.heads,
            dim_feedforward=args.hidden_dim * 4,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=args.temporal_layers
        )
        self.window_depth_head = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        self.relative_head = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim // 2, 1),
        )
        ratio = args.initial_depth_m / args.max_depth_m
        nn.init.normal_(self.window_depth_head[-1].weight, std=1e-3)
        nn.init.constant_(
            self.window_depth_head[-1].bias,
            math.log(ratio / (1.0 - ratio)),
        )
        nn.init.normal_(self.relative_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.relative_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        point = batch["point_features"]
        metric = batch["metric_window_features"]
        joint_uv = batch["joint_uv"]
        grid_uv = batch["grid_uv"]
        mode = self.feature_mode
        if mode in ("point_zero", "all_zero"):
            point = torch.zeros_like(point)
        if mode in ("metric_zero", "all_zero"):
            metric = torch.zeros_like(metric)
        if mode == "spatial_shuffle":
            point = torch.roll(point, shifts=(3, 7), dims=(2, 3))
        if mode == "time_reverse":
            point = torch.flip(point, dims=(1,))
            metric = metric

        batch_size, time, height, width, _ = point.shape
        if time > self.temporal_position.shape[0]:
            raise ValueError(f"Window length {time} exceeds max-window-size")
        point = point.reshape(batch_size * time, height * width, -1)
        grid_uv = grid_uv.reshape(batch_size * time, height * width, 2)
        confidence = batch["grid_confidence"].reshape(
            batch_size * time, height * width, 1
        )
        key = self.point_encoder(point) + self.grid_encoder(
            torch.cat((grid_uv, confidence), dim=-1)
        )

        valid = batch["joint_query_valid"]
        root_uv = joint_uv[:, :, :1]
        local_uv = joint_uv - root_uv
        query_metadata = torch.cat((
            joint_uv,
            local_uv,
            valid.to(joint_uv.dtype)[..., None],
        ), dim=-1)
        joint_query = self.joint_encoder(query_metadata)
        joint_ids = torch.arange(
            joint_query.shape[2], device=joint_query.device
        ).view(1, 1, -1)
        joint_query = joint_query + self.joint_embedding(joint_ids)
        joint_query = torch.where(
            valid[..., None], joint_query,
            self.missing_embedding.view(1, 1, 1, -1),
        )
        if mode == "joint_query_zero":
            joint_query = torch.zeros_like(joint_query)
        global_query = self.global_query.view(1, 1, 1, -1).expand(
            batch_size, time, 1, -1
        )
        if mode == "global_query_zero":
            global_query = torch.zeros_like(global_query)
        query = torch.cat((joint_query, global_query), dim=2)
        query = query.reshape(batch_size * time, query.shape[2], -1)

        # Joint queries favor nearby patches but retain access to the full grid.
        joint_distance = (
            joint_uv.reshape(batch_size * time, -1, 1, 2)
            - grid_uv[:, None]
        ).square().sum(dim=-1)
        joint_bias = -self.spatial_bias * joint_distance
        joint_bias = torch.where(
            valid.reshape(batch_size * time, -1, 1),
            joint_bias,
            torch.zeros_like(joint_bias),
        )
        global_bias = torch.zeros(
            batch_size * time, 1, height * width,
            device=joint_bias.device, dtype=joint_bias.dtype,
        )
        attention_bias = torch.cat((joint_bias, global_bias), dim=1)
        attention_bias = attention_bias.repeat_interleave(self.heads, dim=0)
        key_padding = ~batch["grid_valid"].reshape(
            batch_size * time, height * width
        )
        all_invalid = key_padding.all(dim=1)
        key_padding[all_invalid, 0] = False
        attended, _ = self.cross_attention(
            query, key, key,
            key_padding_mask=key_padding,
            attn_mask=attention_bias,
            need_weights=False,
        )
        attended = self.cross_norm(attended + query).reshape(
            batch_size, time, -1, attended.shape[-1]
        )
        joint = attended[:, :, :-1]
        global_token = attended[:, :, -1]
        joint_weight = valid.to(joint.dtype)
        pooled = (joint * joint_weight[..., None]).sum(dim=2)
        pooled = pooled / joint_weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        wrist = joint[:, :, 0]
        metric_token = self.metric_encoder(metric)[:, None].expand(-1, time, -1)
        observed_fraction = valid.to(joint.dtype).mean(dim=2, keepdim=True)
        observed = batch["observed"].to(joint.dtype)[..., None]
        frame = self.frame_encoder(torch.cat((
            wrist, pooled, global_token, metric_token,
            observed_fraction, observed,
        ), dim=-1))
        frame = frame + self.temporal_position[:time][None]
        frame = self.temporal(frame)

        window_feature = frame.mean(dim=1)
        base_depth = torch.sigmoid(
            self.window_depth_head(window_feature).squeeze(-1)
        ) * self.max_depth
        relative = torch.tanh(
            self.relative_head(frame).squeeze(-1)
        ) * self.max_relative_offset
        relative = relative - relative.mean(dim=1, keepdim=True)
        return (base_depth[:, None] + relative).clamp(1e-4, self.max_depth)


def move_to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value.to(device)


def combine_pair(batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat((batch["first"][key], batch["second"][key]), dim=0)
        for key in batch["first"]
    }


def centered_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    weight = valid.to(prediction.dtype)
    denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    pred_center = (prediction * weight).sum(dim=1, keepdim=True) / denominator
    target_center = (target * weight).sum(dim=1, keepdim=True) / denominator
    return masked_mean(
        smooth_l1(
            (prediction - pred_center) - (target - target_center), beta
        ),
        valid,
    )


def overlap_loss(
    prediction: torch.Tensor,
    batch: dict[str, dict[str, torch.Tensor]],
    beta: float,
) -> tuple[torch.Tensor, int]:
    pair_count = prediction.shape[0] // 2
    first, second = prediction[:pair_count], prediction[pair_count:]
    losses = []
    count = 0
    for index in range(pair_count):
        start_a = int(batch["first"]["start"][index])
        end_a = int(batch["first"]["end"][index])
        start_b = int(batch["second"]["start"][index])
        end_b = int(batch["second"]["end"][index])
        begin, end = max(start_a, start_b), min(end_a, end_b)
        if end <= begin:
            continue
        a = first[index, begin - start_a:end - start_a]
        b = second[index, begin - start_b:end - start_b]
        losses.append(smooth_l1(a - b, beta).mean())
        count += end - begin
    if not losses:
        return prediction.sum() * 0.0, 0
    return torch.stack(losses).mean(), count


def summarize_metrics(source: dict[str, list[np.ndarray]]) -> dict:
    return {
        "initial_translation": distribution(source["initial_full"]),
        "predicted_translation": distribution(source["predicted_full"]),
        "initial_ray_depth": distribution(source["initial_ray"]),
        "predicted_ray_depth": distribution(source["predicted_ray"]),
        "absolute_target_depth": distribution(source["target_depth"]),
        "absolute_predicted_depth": distribution(source["predicted_depth"]),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace | SimpleNamespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    names = (
        "total", "absolute", "relative", "velocity",
        "acceleration", "overlap",
    )
    sums = {name: 0.0 for name in names}
    metric_names = (
        "initial_full", "predicted_full", "initial_ray", "predicted_ray",
        "target_depth", "predicted_depth",
    )
    metrics = {name: [] for name in metric_names}
    side_metrics = {
        side: {name: [] for name in metric_names}
        for side in ("left", "right")
    }
    stitched: dict[tuple[int, int], dict[str, object]] = {}
    improved = degraded = evaluated = overlap_frames = batches = 0
    iterator = tqdm(loader, desc="train" if training else "val")
    for raw_batch in iterator:
        paired = "first" in raw_batch
        raw_batch = move_to_device(raw_batch, device)
        batch = combine_pair(raw_batch) if paired else raw_batch
        bad = [
            key for key, value in batch.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if bad:
            raise RuntimeError(f"non-finite batch inputs: {bad}")

        initial_t, target_t = batch["initial_t"], batch["target_t"]
        ray = batch["output_ray"]
        valid = batch["valid"]
        target_depth = (target_t * ray).sum(dim=-1)
        valid &= target_depth > 1e-5
        initial_depth = (initial_t * ray).sum(dim=-1)

        with torch.set_grad_enabled(training):
            predicted_depth = model(batch)
            predicted_t = predicted_depth[..., None] * ray
            absolute = masked_mean(
                smooth_l1(
                    predicted_depth - target_depth,
                    args.smooth_l1_beta_mm / 1000.0,
                ),
                valid,
            )
            relative = centered_depth_loss(
                predicted_depth, target_depth, valid,
                args.smooth_l1_beta_mm / 1000.0,
            )
            velocity = temporal_loss(
                predicted_depth[..., None], target_depth[..., None],
                valid, 1, args.smooth_l1_beta_mm / 1000.0,
            )
            acceleration = temporal_loss(
                predicted_depth[..., None], target_depth[..., None],
                valid, 2, args.smooth_l1_beta_mm / 1000.0,
            )
            if paired:
                overlap, overlap_count = overlap_loss(
                    predicted_depth, raw_batch,
                    args.smooth_l1_beta_mm / 1000.0,
                )
            else:
                overlap = predicted_depth.sum() * 0.0
                overlap_count = 0
            total = (
                args.w_absolute * absolute
                + args.w_relative * relative
                + args.w_velocity * velocity
                + args.w_acceleration * acceleration
                + args.w_overlap * overlap
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for name, value in zip(
            names, (total, absolute, relative, velocity, acceleration, overlap)
        ):
            sums[name] += float(value.detach())
        overlap_frames += overlap_count
        batches += 1
        iterator.set_postfix(loss=f"{sums['total'] / batches:.5f}")

        initial_full = torch.linalg.norm(initial_t - target_t, dim=-1)
        predicted_full = torch.linalg.norm(predicted_t - target_t, dim=-1)
        values = {
            "initial_full": initial_full,
            "predicted_full": predicted_full,
            "initial_ray": (initial_depth - target_depth).abs(),
            "predicted_ray": (predicted_depth - target_depth).abs(),
            "target_depth": target_depth,
            "predicted_depth": predicted_depth,
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
        after = predicted_full.detach().cpu().numpy()[valid_np]
        improved += int((after < before).sum())
        degraded += int((after > before + 1e-6).sum())
        evaluated += len(before)

        if not training and not paired:
            pred_np = predicted_depth.detach().cpu().numpy()
            ray_np = ray.detach().cpu().numpy()
            initial_np = initial_t.detach().cpu().numpy()
            target_np = target_t.detach().cpu().numpy()
            stream_np = batch["stream_index"].detach().cpu().numpy()
            frame_np = batch["frame_index"].detach().cpu().numpy()
            length = pred_np.shape[1]
            center_weight = 1.0 - np.abs(
                np.linspace(-1.0, 1.0, length, dtype=np.float32)
            )
            center_weight = np.maximum(center_weight, 0.1)
            for batch_index in range(pred_np.shape[0]):
                for local in range(length):
                    if not valid_np[batch_index, local]:
                        continue
                    key = (int(stream_np[batch_index]), int(frame_np[batch_index, local]))
                    item = stitched.setdefault(key, {
                        "sum": 0.0, "weight": 0.0,
                        "ray": ray_np[batch_index, local],
                        "initial": initial_np[batch_index, local],
                        "target": target_np[batch_index, local],
                        "side": int(side_np[batch_index, local]),
                    })
                    weight = float(center_weight[local])
                    item["sum"] = float(item["sum"]) + weight * float(pred_np[batch_index, local])
                    item["weight"] = float(item["weight"]) + weight

    result = {
        **{name: value / max(batches, 1) for name, value in sums.items()},
        **summarize_metrics(metrics),
        "by_side": {
            side: summarize_metrics(source)
            for side, source in side_metrics.items()
        },
        "evaluated": evaluated,
        "improved": improved,
        "degraded": degraded,
        "degraded_fraction": degraded / max(evaluated, 1),
        "overlap_frames": overlap_frames,
    }
    if stitched:
        stitched_metrics = {name: [] for name in metric_names}
        stitched_sides = {
            side: {name: [] for name in metric_names}
            for side in ("left", "right")
        }
        stitched_degraded = 0
        for item in stitched.values():
            depth = float(item["sum"]) / max(float(item["weight"]), 1e-8)
            ray_value = np.asarray(item["ray"])
            initial_value = np.asarray(item["initial"])
            target_value = np.asarray(item["target"])
            target_depth_value = float(np.dot(target_value, ray_value))
            initial_depth_value = float(np.dot(initial_value, ray_value))
            prediction_value = depth * ray_value
            row = {
                "initial_full": np.linalg.norm(initial_value - target_value),
                "predicted_full": np.linalg.norm(prediction_value - target_value),
                "initial_ray": abs(initial_depth_value - target_depth_value),
                "predicted_ray": abs(depth - target_depth_value),
                "target_depth": target_depth_value,
                "predicted_depth": depth,
            }
            side_name = "left" if int(item["side"]) == 0 else "right"
            for name, value in row.items():
                array = np.asarray([value], dtype=np.float32)
                stitched_metrics[name].append(array)
                stitched_sides[side_name][name].append(array)
            stitched_degraded += int(row["predicted_full"] > row["initial_full"] + 1e-6)
        result["stitched"] = {
            **summarize_metrics(stitched_metrics),
            "by_side": {
                side: summarize_metrics(source)
                for side, source in stitched_sides.items()
            },
            "evaluated": len(stitched),
            "degraded_fraction": stitched_degraded / max(len(stitched), 1),
        }
    return result


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    args: argparse.Namespace,
    sample: dict[str, torch.Tensor],
    val: dict,
) -> dict:
    selected = val.get("stitched", val)
    return {
        "epoch": epoch,
        "model_version": MODEL_VERSION,
        "model_state": (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        ),
        "args": vars(args),
        "point_feature_dim": int(sample["point_features"].shape[-1]),
        "metric_feature_dim": int(sample["metric_window_features"].shape[-1]),
        "num_joints": int(sample["joint_uv"].shape[1]),
        "initial_pose_usage": "2d_query_and_output_ray_composition_only",
        "explicit_hand_depth_input": False,
        "uses_metric_scalar": False,
        "val_total": val["total"],
        "val_ray_median_mm": selected["predicted_ray_depth"]["median_mm"],
        "val_degraded_fraction": selected["degraded_fraction"],
        "val": val,
    }


def model_args_from_checkpoint(
    checkpoint: dict, cli: argparse.Namespace
) -> SimpleNamespace:
    values = dict(checkpoint["args"])
    values["feature_mode"] = cli.feature_mode
    return SimpleNamespace(**values)


def main() -> None:
    args = parse_args()
    disable_mha_fastpath()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data = DenseTrajectoryDataset(
        Path(args.train_windows), Path(args.global_train_root),
        Path(args.dense_train_root), args.query_dropout,
    )
    val_data = DenseTrajectoryDataset(
        Path(args.val_windows), Path(args.global_val_root),
        Path(args.dense_val_root), 0.0,
    )
    train_pairs = OverlapPairDataset(train_data)
    sample = train_data[0]
    audit = {
        "model": MODEL_VERSION,
        "train_windows": len(train_data),
        "train_overlap_pairs": len(train_pairs),
        "val_windows": len(val_data),
        "point_feature_shape": list(sample["point_features"].shape),
        "metric_feature_shape": list(sample["metric_window_features"].shape),
        "joint_query_shape": list(sample["joint_uv"].shape),
        "valid_grid_fraction": round(float(sample["grid_valid"].float().mean()), 6),
        "valid_query_fraction": round(
            float(sample["joint_query_valid"].float().mean()), 6
        ),
        "initial_pose_usage": "2d_query_and_output_ray_composition_only",
        "explicit_hand_depth_input": False,
        "uses_metric_scalar": False,
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    checkpoint = None
    model_args: argparse.Namespace | SimpleNamespace = args
    if args.checkpoint:
        checkpoint = torch.load(
            Path(args.checkpoint).expanduser().resolve(), map_location="cpu"
        )
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError(f"Unexpected checkpoint: {checkpoint.get('model_version')}")
        model_args = model_args_from_checkpoint(checkpoint, args)
    if args.eval_only and checkpoint is None:
        raise ValueError("--eval-only requires --checkpoint")
    model = Pi3XAbsoluteTrajectoryModel(
        int(sample["point_features"].shape[-1]),
        int(sample["metric_window_features"].shape[-1]),
        int(sample["joint_uv"].shape[1]),
        model_args,
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device)
    if args.data_parallel:
        model = nn.DataParallel(model)

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    if args.eval_only:
        result = run_epoch(model, val_loader, device, model_args)
        print(json.dumps({
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "feature_mode": args.feature_mode,
            "val": result,
        }), flush=True)
        return

    train_loader = DataLoader(train_pairs, shuffle=True, **loader_args)
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
        payload = checkpoint_payload(model, epoch, args, sample, val)
        torch.save(payload, out_dir / "last.pt")
        selected = val.get("stitched", val)
        if val["total"] < best_total:
            best_total = val["total"]
            torch.save(payload, out_dir / "best.pt")
        ray = selected["predicted_ray_depth"]["median_mm"]
        if ray < best_ray:
            best_ray = ray
            torch.save(payload, out_dir / "best_ray.pt")
        degraded_value = selected["degraded_fraction"]
        if degraded_value < best_degraded:
            best_degraded = degraded_value
            torch.save(payload, out_dir / "best_degraded.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
