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
CAMERA_POSITION_SCALE_M = 1.0
PALM_OFFSET_SCALE_M = 0.2
RELATIVE_POSITION_SCALE_M = 0.3
VELOCITY_SCALE_M_PER_FRAME = 0.05
ACCELERATION_SCALE_M_PER_FRAME2 = 0.05
OBJECT_EXTENT_SCALE_M = 0.2
MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


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
    parser.add_argument(
        "--model-variant",
        choices=("full", "single_depth"),
        default="full",
        help=(
            "full keeps the legacy motion/anomaly/carry branch; "
            "single_depth predicts directly from the spatial and temporal "
            "encoders with one depth head."
        ),
    )
    parser.add_argument("--max-correction-mm", type=float, default=60.0)
    parser.add_argument("--max-motion-residual-mm", type=float, default=40.0)
    parser.add_argument("--max-target-mm", type=float, default=120.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument(
        "--objective",
        choices=("full", "ray_depth_only", "ray_depth_guarded"),
        default="full",
        help=(
            "Training objective. ray_depth_only supervises only the final "
            "camera-ray depth and its temporal derivatives; "
            "ray_depth_guarded adds direct no-regression, direction, "
            "overshoot, and near-zero penalties."
        ),
    )
    parser.add_argument(
        "--depth-mse-scale-mm",
        type=float,
        default=20.0,
        help="Normalization scale for ray_depth_only MSE terms.",
    )
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-wrist", type=float, default=0.25)
    parser.add_argument("--w-projection", type=float, default=0.1)
    parser.add_argument("--w-velocity", type=float, default=0.5)
    parser.add_argument("--w-acceleration", type=float, default=1.0)
    parser.add_argument("--w-residual", type=float, default=0.05)
    parser.add_argument("--guard-anchor-mm", type=float, default=5.0)
    parser.add_argument("--guard-sign-min-mm", type=float, default=5.0)
    parser.add_argument("--guard-overshoot-margin-mm", type=float, default=3.0)
    parser.add_argument("--w-degradation-guard", type=float, default=0.5)
    parser.add_argument("--w-direction", type=float, default=0.2)
    parser.add_argument("--w-overshoot", type=float, default=0.2)
    parser.add_argument("--w-near-zero-anchor", type=float, default=1.0)
    parser.add_argument("--accurate-anchor-mm", type=float, default=15.0)
    parser.add_argument("--w-accurate-anchor", type=float, default=1.0)
    parser.add_argument("--anomaly-zero-mm", type=float, default=3.0)
    parser.add_argument("--anomaly-full-mm", type=float, default=10.0)
    parser.add_argument("--carry-zero-mm", type=float, default=5.0)
    parser.add_argument("--carry-full-mm", type=float, default=15.0)
    parser.add_argument("--carry-gate-start", type=float, default=0.5)
    parser.add_argument("--carry-gate-full", type=float, default=0.9)
    parser.add_argument("--anomaly-depth-boost", type=float, default=3.0)
    parser.add_argument("--w-anomaly", type=float, default=0.5)
    parser.add_argument("--w-boundary", type=float, default=1.0)
    parser.add_argument("--w-motion-anchor", type=float, default=0.25)
    parser.add_argument("--w-carry-depth", type=float, default=1.0)
    parser.add_argument("--error-weight-reference-mm", type=float, default=20.0)
    parser.add_argument("--error-weight-min", type=float, default=0.5)
    parser.add_argument("--error-weight-max", type=float, default=2.0)
    parser.add_argument(
        "--mirror-left-token-indices",
        action="store_true",
        help=(
            "Mirror Pi3X patch x coordinates for left-hand streams after "
            "their supervision is normalized into the mirrored camera frame."
        ),
    )
    parser.add_argument(
        "--left-coordinate-mode",
        choices=("normalized", "original"),
        default="normalized",
        help=(
            "Coordinate frame used for left-hand streams. original restores "
            "the supervision to the unmirrored camera frame so it matches "
            "the original Pi3X cache."
        ),
    )
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
        mirror_left_token_indices: bool = False,
        left_coordinate_mode: str = "normalized",
    ):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.pi3x_root = pi3x_root
        self.max_target_m = max_target_m
        self.mirror_left_token_indices = mirror_left_token_indices
        self.left_coordinate_mode = left_coordinate_mode

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
        intrinsics = np.asarray(
            supervision["intrinsics"], dtype=np.float32
        ).copy()
        normalized_left = bool(
            np.asarray(supervision["normalized_left"]).item()
        )
        if normalized_left and self.left_coordinate_mode == "original":
            pred = pred.copy()
            gt3d = gt3d.copy()
            gt2d = gt2d.copy()
            object_pose = object_pose.copy()
            pred[..., 0] *= -1.0
            gt3d[..., 0] *= -1.0
            width = int(np.asarray(pi3x["original_wh"])[0])
            gt2d[..., 0] = (width - 1) - gt2d[..., 0]
            intrinsics[0, 2] = (width - 1) - intrinsics[0, 2]
            object_pose[:, :3, :3] = np.einsum(
                "ij,tjk,kl->til",
                MIRROR_X,
                object_pose[:, :3, :3],
                MIRROR_X,
            )
            object_pose[:, 0, 3] *= -1.0
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
        hand_acceleration = np.zeros_like(wrist)
        relative_acceleration = np.zeros_like(relative_velocity)
        hand_acceleration[1:] = hand_velocity[1:] - hand_velocity[:-1]
        relative_acceleration[1:] = (
            relative_velocity[1:] - relative_velocity[:-1]
        )
        relative_velocity_baseline = np.median(
            relative_velocity, axis=0, keepdims=True
        )
        relative_velocity_deviation = (
            relative_velocity - relative_velocity_baseline
        )
        cumulative_relative_deviation = np.cumsum(
            relative_velocity_deviation, axis=0
        )
        hand_speed = np.linalg.norm(hand_velocity, axis=-1, keepdims=True)
        object_speed = np.linalg.norm(
            object_velocity, axis=-1, keepdims=True
        )
        relative_speed = np.linalg.norm(
            relative_velocity, axis=-1, keepdims=True
        )
        deviation_speed = np.linalg.norm(
            relative_velocity_deviation, axis=-1, keepdims=True
        )
        speed_ratio = np.clip(
            hand_speed / np.maximum(object_speed, 0.001), 0.0, 10.0
        ) / 10.0
        object_low_speed = (
            object_speed < 0.002
        ).astype(np.float32)
        motion = np.concatenate(
            [
                hand_velocity / VELOCITY_SCALE_M_PER_FRAME,
                object_velocity / VELOCITY_SCALE_M_PER_FRAME,
                relative_velocity / VELOCITY_SCALE_M_PER_FRAME,
                relative_acceleration / ACCELERATION_SCALE_M_PER_FRAME2,
                (
                    relative_velocity_deviation
                    / VELOCITY_SCALE_M_PER_FRAME
                ),
                (
                    cumulative_relative_deviation
                    / RELATIVE_POSITION_SCALE_M
                ),
                hand_speed / VELOCITY_SCALE_M_PER_FRAME,
                object_speed / VELOCITY_SCALE_M_PER_FRAME,
                relative_speed / VELOCITY_SCALE_M_PER_FRAME,
                deviation_speed / VELOCITY_SCALE_M_PER_FRAME,
                speed_ratio,
                object_low_speed,
            ],
            axis=-1,
        ).astype(np.float32)
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
                wrist / CAMERA_POSITION_SCALE_M,
                (
                    palm_local / PALM_OFFSET_SCALE_M
                ).reshape(len(wrist), -1),
                hand_velocity / VELOCITY_SCALE_M_PER_FRAME,
                hand_acceleration / ACCELERATION_SCALE_M_PER_FRAME2,
                object_center / CAMERA_POSITION_SCALE_M,
                object_velocity / VELOCITY_SCALE_M_PER_FRAME,
                relative / RELATIVE_POSITION_SCALE_M,
                relative_velocity / VELOCITY_SCALE_M_PER_FRAME,
                relative_acceleration / ACCELERATION_SCALE_M_PER_FRAME2,
                rotation_6d(object_rotation),
                camera_ray,
                np.broadcast_to(
                    object_extents / OBJECT_EXTENT_SCALE_M,
                    (len(wrist), 3),
                ),
                object_local_wrist / safe_extents,
                (object_local_palm / safe_extents).reshape(len(wrist), -1),
                valid[:, None],
            ],
            axis=-1,
        ).astype(np.float32)

        token_features, token_metadata, token_valid, token_types = [], [], [], []
        mirrored = normalized_left and self.left_coordinate_mode == "normalized"
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
            if mirrored and self.mirror_left_token_indices:
                indices = indices.copy()
                indices[..., 1] = (float(grid[1]) - 1.0) - indices[..., 1]
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
            "motion": torch.from_numpy(motion),
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
                intrinsics
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
        motion_dim: int,
        hidden_dim: int,
        spatial_layers: int,
        temporal_layers: int,
        heads: int,
        dropout: float,
        model_variant: str = "full",
    ):
        super().__init__()
        self.model_variant = model_variant
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
        if model_variant == "full":
            motion_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.motion_projection = nn.Sequential(
                nn.Linear(motion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.motion_encoder = nn.TransformerEncoder(
                motion_layer, num_layers=2
            )
            self.motion_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.anomaly_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        elif model_variant != "single_depth":
            raise ValueError(f"Unknown model variant: {model_variant}")
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.depth_head[-1].weight)
        nn.init.zeros_(self.depth_head[-1].bias)
        if model_variant == "full":
            nn.init.zeros_(self.motion_head[-1].weight)
            nn.init.zeros_(self.motion_head[-1].bias)
            nn.init.zeros_(self.anomaly_head[-1].weight)
            nn.init.constant_(self.anomaly_head[-1].bias, -2.0)

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
        motion: torch.Tensor,
        max_correction: float,
        max_motion_residual: float,
        carry_gate_start: float,
        carry_gate_full: float,
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
        geometry_prediction = (
            torch.tanh(self.depth_head(temporal).squeeze(-1))
            * max_correction
        )
        if self.model_variant == "single_depth":
            prediction = geometry_prediction
            if return_aux:
                zeros = torch.zeros_like(prediction)
                return {
                    "prediction": prediction,
                    "geometry_prediction": prediction,
                    "motion_residual": zeros,
                    "motion_innovation": zeros,
                    "anomaly_logits": torch.full_like(prediction, -20.0),
                    "carry_gate": zeros,
                    "magnitude": torch.abs(prediction),
                }
            return prediction
        motion_encoded = self.motion_encoder(
            self.motion_projection(motion) + self.position[:, :frames]
        )
        motion_innovation = (
            torch.tanh(
                self.motion_head(
                    torch.cat([temporal, motion_encoded], dim=-1)
                ).squeeze(-1)
            )
            * max_motion_residual
        )
        anomaly_logits = self.anomaly_head(motion_encoded).squeeze(-1)
        carry_probability = torch.sigmoid(anomaly_logits)
        carry_gate = (
            (carry_probability - carry_gate_start)
            / max(carry_gate_full - carry_gate_start, 1e-6)
        ).clamp(0.0, 1.0)
        motion_steps = [motion_innovation[:, 0]]
        for frame_index in range(1, frames):
            carried = (
                carry_gate[:, frame_index]
                * motion_steps[-1]
            )
            motion_steps.append(
                torch.clamp(
                    carried + motion_innovation[:, frame_index],
                    -max_motion_residual,
                    max_motion_residual,
                )
            )
        motion_residual = torch.stack(motion_steps, dim=1)
        prediction = torch.clamp(
            geometry_prediction + motion_residual,
            -max_correction,
            max_correction,
        )
        if return_aux:
            return {
                "prediction": prediction,
                "geometry_prediction": geometry_prediction,
                "motion_residual": motion_residual,
                "motion_innovation": motion_innovation,
                "anomaly_logits": anomaly_logits,
                "carry_gate": carry_gate,
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


def normalized_mse(value, target, mask, scale):
    error = (value - target) / scale
    return masked_mean(error.square(), mask)


def temporal_mse(value, target, valid, order, scale):
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
        target = target[:, 1:] - target[:, :-1]
        valid = valid[:, 1:] & valid[:, :-1]
    return normalized_mse(value, target, valid, scale)


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
        batch["motion"],
        args.max_correction_mm / 1000.0,
        args.max_motion_residual_mm / 1000.0,
        args.carry_gate_start,
        args.carry_gate_full,
        return_aux=True,
    )
    ray_depth = model_output["prediction"]
    motion_residual = model_output["motion_residual"]
    anomaly_logits = model_output["anomaly_logits"]
    pred = batch["pred_joints"]
    gt = batch["gt_joints"]
    valid = batch["valid"]
    camera_ray = F.normalize(pred[:, :, 0], dim=-1, eps=1e-8)
    target_translation = gt[:, :, 0] - pred[:, :, 0]
    target_depth = torch.sum(target_translation * camera_ray, dim=-1)
    if args.objective in ("ray_depth_only", "ray_depth_guarded"):
        mse_scale = args.depth_mse_scale_mm / 1000.0
        if mse_scale <= 0.0:
            raise ValueError("--depth-mse-scale-mm must be positive")
        if args.objective == "ray_depth_only":
            depth_loss = normalized_mse(
                ray_depth, target_depth, valid, mse_scale
            )
        else:
            beta = args.smooth_l1_beta_mm / args.depth_mse_scale_mm
            depth_loss = smooth_l1(
                ray_depth / mse_scale,
                target_depth / mse_scale,
                valid,
                beta,
            )
        losses = {
            "depth": depth_loss,
            "velocity": temporal_mse(
                ray_depth, target_depth, valid, 1, mse_scale
            ),
            "acceleration": temporal_mse(
                ray_depth, target_depth, valid, 2, mse_scale
            ),
        }
        total = (
            args.w_depth * losses["depth"]
            + args.w_velocity * losses["velocity"]
            + args.w_acceleration * losses["acceleration"]
        )
        if args.objective == "ray_depth_guarded":
            corrected_error = torch.abs(target_depth - ray_depth)
            initial_error = torch.abs(target_depth)
            degradation = F.relu(corrected_error - initial_error)
            losses["degradation_guard"] = masked_mean(
                (degradation / mse_scale).square(), valid
            )

            direction_valid = valid & (
                initial_error >= args.guard_sign_min_mm / 1000.0
            )
            wrong_direction = F.relu(-ray_depth * target_depth)
            losses["direction"] = masked_mean(
                wrong_direction / (mse_scale * mse_scale), direction_valid
            )

            overshoot_margin = args.guard_overshoot_margin_mm / 1000.0
            overshoot = F.relu(
                torch.abs(ray_depth) - initial_error - overshoot_margin
            )
            losses["overshoot"] = masked_mean(
                (overshoot / mse_scale).square(), valid
            )

            near_zero = valid & (
                initial_error < args.guard_anchor_mm / 1000.0
            )
            losses["near_zero_anchor"] = masked_mean(
                (ray_depth / mse_scale).square(), near_zero
            )
            total = (
                total
                + args.w_degradation_guard
                * losses["degradation_guard"]
                + args.w_direction * losses["direction"]
                + args.w_overshoot * losses["overshoot"]
                + args.w_near_zero_anchor * losses["near_zero_anchor"]
            )
        translation = ray_depth.unsqueeze(-1) * camera_ray
        corrected = pred + translation[:, :, None]
        before = torch.linalg.norm(pred[:, :, 0] - gt[:, :, 0], dim=-1)
        after = torch.linalg.norm(
            corrected[:, :, 0] - gt[:, :, 0], dim=-1
        )
        initial_depth_error = torch.abs(target_depth)
        depth_after = torch.abs(target_depth - ray_depth)
        return (
            total,
            losses,
            before[valid],
            after[valid],
            initial_depth_error[valid],
            depth_after[valid],
        )
    pred_velocity = torch.zeros_like(pred[:, :, 0])
    gt_velocity = torch.zeros_like(gt[:, :, 0])
    pred_velocity[:, 1:] = pred[:, 1:, 0] - pred[:, :-1, 0]
    gt_velocity[:, 1:] = gt[:, 1:, 0] - gt[:, :-1, 0]
    motion_error = torch.linalg.norm(
        pred_velocity - gt_velocity, dim=-1
    )
    cumulative_motion_error = torch.linalg.norm(
        torch.cumsum(pred_velocity - gt_velocity, dim=1),
        dim=-1,
    )
    motion_valid = valid.clone()
    motion_valid[:, 0] = False
    motion_valid[:, 1:] &= valid[:, :-1]
    anomaly_range = max(
        args.anomaly_full_mm - args.anomaly_zero_mm, 1e-6
    )
    anomaly_target = (
        (motion_error * 1000.0 - args.anomaly_zero_mm) / anomaly_range
    ).clamp(0.0, 1.0)
    carry_range = max(args.carry_full_mm - args.carry_zero_mm, 1e-6)
    carry_target = (
        (cumulative_motion_error * 1000.0 - args.carry_zero_mm)
        / carry_range
    ).clamp(0.0, 1.0)
    motion_state_target = torch.maximum(anomaly_target, carry_target)
    translation = ray_depth.unsqueeze(-1) * camera_ray
    corrected = pred + translation[:, :, None]
    beta = args.smooth_l1_beta_mm / 1000.0
    initial_depth_error = torch.abs(target_depth)
    supervision_weight = (
        initial_depth_error
        / max(args.error_weight_reference_mm / 1000.0, 1e-8)
    ).clamp(args.error_weight_min, args.error_weight_max)
    supervision_weight = supervision_weight * (
        1.0 + args.anomaly_depth_boost * motion_state_target
    )
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
    anomaly_loss = F.binary_cross_entropy_with_logits(
        anomaly_logits, motion_state_target, reduction="none"
    )
    losses["anomaly"] = masked_mean(anomaly_loss, motion_valid)
    boundary_valid = valid[:, 1:] & valid[:, :-1]
    boundary_weight = 1.0 + args.anomaly_depth_boost * torch.maximum(
        motion_state_target[:, 1:], motion_state_target[:, :-1]
    )
    losses["boundary"] = weighted_smooth_l1(
        ray_depth[:, 1:] - ray_depth[:, :-1],
        target_depth[:, 1:] - target_depth[:, :-1],
        boundary_valid,
        boundary_weight,
        beta,
    )
    non_anomaly = valid & (motion_state_target < 0.1)
    losses["motion_anchor"] = smooth_l1(
        motion_residual,
        torch.zeros_like(motion_residual),
        non_anomaly,
        beta,
    )
    carry_residual_target = (
        target_depth - model_output["geometry_prediction"].detach()
    ).clamp(
        -args.max_motion_residual_mm / 1000.0,
        args.max_motion_residual_mm / 1000.0,
    )
    losses["carry_depth"] = weighted_smooth_l1(
        motion_residual,
        carry_residual_target,
        motion_valid,
        motion_state_target,
        beta,
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
        + args.w_accurate_anchor * losses["accurate_anchor"]
        + args.w_anomaly * losses["anomaly"]
        + args.w_boundary * losses["boundary"]
        + args.w_motion_anchor * losses["motion_anchor"]
        + args.w_carry_depth * losses["carry_depth"]
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
    if args.model_variant == "single_depth" and args.objective == "full":
        raise ValueError(
            "--model-variant single_depth requires a ray-depth objective"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset = Pi3XWindowDataset(
        Path(args.train_windows).expanduser().resolve(),
        Path(args.pi3x_train_root).expanduser().resolve(),
        args.max_target_mm / 1000.0,
        args.mirror_left_token_indices,
        args.left_coordinate_mode,
    )
    val_dataset = Pi3XWindowDataset(
        Path(args.val_windows).expanduser().resolve(),
        Path(args.pi3x_val_root).expanduser().resolve(),
        args.max_target_mm / 1000.0,
        args.mirror_left_token_indices,
        args.left_coordinate_mode,
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
    motion_dim = int(sample["motion"].shape[-1])
    model = Pi3XRelativeDepthRefiner(
        scalar_dim,
        feature_dim,
        metadata_dim,
        motion_dim,
        args.hidden_dim,
        args.spatial_layers,
        args.temporal_layers,
        args.heads,
        args.dropout,
        args.model_variant,
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
            "motion_dim": motion_dim,
            "model_variant": args.model_variant,
            "scalar_feature_version": (
                "v10_single_depth"
                if args.model_variant == "single_depth"
                else (
                    "v9_ray_depth_guarded"
                    if args.objective == "ray_depth_guarded"
                    else (
                        "v8_ray_depth_only"
                        if args.objective == "ray_depth_only"
                        else "v7_thresholded_recurrent_carry"
                    )
                )
            ),
            "scalar_feature_scales": {
                "camera_position_m": CAMERA_POSITION_SCALE_M,
                "palm_offset_m": PALM_OFFSET_SCALE_M,
                "relative_position_m": RELATIVE_POSITION_SCALE_M,
                "velocity_m_per_frame": VELOCITY_SCALE_M_PER_FRAME,
                "acceleration_m_per_frame2":
                    ACCELERATION_SCALE_M_PER_FRAME2,
                "object_extent_m": OBJECT_EXTENT_SCALE_M,
            },
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
