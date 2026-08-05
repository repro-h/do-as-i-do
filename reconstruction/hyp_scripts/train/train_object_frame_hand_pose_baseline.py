#!/usr/bin/env python3
"""Train an absolute hand-root pose predictor in the filtered object frame."""

from __future__ import annotations

import argparse
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


KEY_JOINTS = np.asarray([4, 5, 8, 9, 12, 13, 16, 17, 20], dtype=np.int64)
MODEL_VERSION = "object_frame_absolute_pose_mlp_bigru_v1"
OBJECT_CONDITIONED_MODEL_VERSION = (
    "object_frame_absolute_pose_object_embedding_mlp_bigru_v2"
)
PI3X_RELATION_MODEL_VERSION = (
    "object_frame_absolute_pose_pi3x_relative_cross_attention_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--pi3x-train-root")
    parser.add_argument("--pi3x-val-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--object-embedding-dim", type=int, default=0)
    parser.add_argument("--pi3x-relation-dim", type=int, default=128)
    parser.add_argument("--pi3x-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-normalized-translation", type=float, default=3.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--translation-noise-mm", type=float, default=15.0)
    parser.add_argument("--rotation-noise-deg", type=float, default=10.0)
    parser.add_argument("--initial-pose-dropout", type=float, default=0.05)
    parser.add_argument("--w-translation", type=float, default=1.0)
    parser.add_argument("--w-rotation", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-degradation-guard", type=float, default=0.25)
    parser.add_argument(
        "--w-rotation-degradation-guard", type=float, default=0.25
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=32)
def load_npz(path_text: str) -> dict[str, np.ndarray]:
    with np.load(path_text, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def rotation_to_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def axis_angle_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = rotvec / angle
    x, y, z = axis
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3) + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    ).astype(np.float32)


class ObjectFrameWindowDataset(Dataset):
    def __init__(
        self,
        windows: Path,
        args: argparse.Namespace,
        augment: bool,
        object_to_index: dict[str, int],
        pi3x_root: Path | None,
    ):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.augment = augment
        self.translation_noise = args.translation_noise_mm / 1000.0
        self.rotation_noise = math.radians(args.rotation_noise_deg)
        self.initial_pose_dropout = args.initial_pose_dropout
        self.object_to_index = object_to_index
        self.pi3x_root = pi3x_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        stream_id = str(row["stream_id"])
        start, end = int(row["start"]), int(row["end"])
        path = Path(row["supervision_npz"]).expanduser().resolve()
        data = load_npz(str(path))

        joints = np.asarray(data["pred_joints_object"], dtype=np.float32)[start:end]
        initial_t = np.asarray(
            data["initial_translation_object"], dtype=np.float32
        )[start:end]
        target_t = np.asarray(
            data["target_translation_object"], dtype=np.float32
        )[start:end]
        initial_r = np.asarray(
            data["initial_rotation_object"], dtype=np.float32
        )[start:end]
        target_r = np.asarray(
            data["target_rotation_object"], dtype=np.float32
        )[start:end]
        filtered_pose = np.asarray(
            data["filtered_object_pose"], dtype=np.float32
        )[start:end]
        valid_t = np.asarray(data["valid_translation"], dtype=bool)[start:end]
        valid_r = np.asarray(data["valid_rotation"], dtype=bool)[start:end]
        observed = np.asarray(data["hand_observed"], dtype=bool)[start:end]
        presence = np.asarray(data["hand_presence"], dtype=bool)[start:end]
        extents = np.asarray(data["object_extents_metric"], dtype=np.float32)
        rotation_weight = float(
            np.asarray(data["rotation_supervision_weight"]).item()
        )
        object_name = scalar_text(data["object_name"])
        normalized_left = bool(np.asarray(data["normalized_left"]).item())
        if object_name not in self.object_to_index:
            raise KeyError(f"Unknown object {object_name} in {path}")

        length = end - start
        arrays = [
            joints,
            initial_t,
            target_t,
            initial_r,
            target_r,
            filtered_pose,
        ]
        if any(len(value) != length for value in arrays):
            raise ValueError(f"Window exceeds supervision for {path}")
        finite_t = (
            np.isfinite(initial_t).all(axis=-1)
            & np.isfinite(target_t).all(axis=-1)
        )
        finite_r = (
            np.isfinite(initial_r).all(axis=(1, 2))
            & np.isfinite(target_r).all(axis=(1, 2))
        )
        valid_t &= finite_t
        valid_r &= valid_t & finite_r

        scale = max(float(np.max(extents)), 0.03)
        clean_initial_t = np.nan_to_num(initial_t).copy()
        clean_initial_r = np.nan_to_num(initial_r).copy()
        input_t = clean_initial_t.copy()
        input_r = clean_initial_r.copy()
        pose_available = np.isfinite(initial_t).all(axis=-1) & np.isfinite(
            initial_r
        ).all(axis=(1, 2))

        if self.augment:
            input_t += np.random.normal(
                0.0, self.translation_noise, size=input_t.shape
            ).astype(np.float32)
            if self.rotation_noise > 0:
                for frame in range(length):
                    axis = np.random.normal(size=3)
                    axis /= max(float(np.linalg.norm(axis)), 1e-8)
                    angle = np.random.normal(0.0, self.rotation_noise)
                    input_r[frame] = (
                        axis_angle_matrix(axis * angle) @ input_r[frame]
                    )
            dropped = np.random.random(length) < self.initial_pose_dropout
            input_t[dropped] = 0.0
            input_r[dropped] = np.eye(3, dtype=np.float32)
            pose_available[dropped] = False

        safe_joints = np.nan_to_num(joints)
        wrist_relative = safe_joints[:, KEY_JOINTS] - safe_joints[:, 0:1]
        wrist_local = np.einsum(
            "tji,tik->tjk", wrist_relative, clean_initial_r
        )
        input_t_normalized = input_t / scale
        target_t_normalized = np.nan_to_num(target_t) / scale
        clean_initial_t_normalized = clean_initial_t / scale

        features = np.concatenate(
            [
                input_t_normalized,
                rotation_to_6d(input_r),
                wrist_local.reshape(length, -1) / 0.1,
                np.broadcast_to(extents / 0.2, (length, 3)),
                observed[:, None].astype(np.float32),
                presence[:, None].astype(np.float32),
                pose_available[:, None].astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        sample = {
            "dataset_index": torch.tensor(index, dtype=torch.long),
            "features": torch.from_numpy(features),
            "initial_translation": torch.from_numpy(
                clean_initial_t_normalized.astype(np.float32)
            ),
            "target_translation": torch.from_numpy(
                target_t_normalized.astype(np.float32)
            ),
            "initial_rotation": torch.from_numpy(clean_initial_r.astype(np.float32)),
            "target_rotation": torch.from_numpy(np.nan_to_num(target_r)),
            "valid_translation": torch.from_numpy(valid_t),
            "valid_rotation": torch.from_numpy(valid_r),
            "object_scale": torch.full((length,), scale, dtype=torch.float32),
            "rotation_weight": torch.full(
                (length,), rotation_weight, dtype=torch.float32
            ),
            "object_index": torch.tensor(
                self.object_to_index[object_name], dtype=torch.long
            ),
        }
        if self.pi3x_root is not None:
            pi3x_path = (
                self.pi3x_root
                / stream_id
                / "pi3x_geometry_features_compact.npz"
            ).resolve()
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

            grid = np.asarray(
                pi3x["geometry_feature_grid_hw"], dtype=np.float32
            )

            def token_group(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                token_features = np.asarray(
                    pi3x[f"{prefix}_features"][positions], dtype=np.float32
                )
                points = np.asarray(
                    pi3x[f"{prefix}_points"][positions], dtype=np.float32
                )
                coverage = np.asarray(
                    pi3x[f"{prefix}_coverage"][positions], dtype=np.float32
                )
                confidence = np.asarray(
                    pi3x[f"{prefix}_confidence"][positions], dtype=np.float32
                )
                indices = np.asarray(
                    pi3x[f"{prefix}_indices"][positions], dtype=np.float32
                )
                if normalized_left:
                    points = points.copy()
                    points[..., 0] *= -1.0
                    indices = indices.copy()
                    indices[..., 1] = (float(grid[1]) - 1.0) - indices[..., 1]
                indices[..., 0] /= max(float(grid[0] - 1), 1.0)
                indices[..., 1] /= max(float(grid[1] - 1), 1.0)
                valid = np.asarray(
                    pi3x[f"{prefix}_valid"][positions], dtype=bool
                )
                metadata = np.concatenate(
                    (
                        points,
                        indices,
                        coverage[..., None],
                        confidence[..., None],
                    ),
                    axis=-1,
                )
                return token_features, metadata, valid

            hand_features, hand_metadata, hand_token_valid = token_group("hand")
            object_features, object_metadata, object_token_valid = token_group(
                "object"
            )
            context_features, context_metadata, context_token_valid = token_group(
                "context"
            )

            object_points = object_metadata[..., :3]
            object_center = np.zeros((length, 3), dtype=np.float32)
            object_point_scale = np.ones(length, dtype=np.float32)
            for frame in range(length):
                valid_points = object_points[frame, object_token_valid[frame]]
                valid_points = valid_points[np.isfinite(valid_points).all(axis=-1)]
                if len(valid_points):
                    center = np.median(valid_points, axis=0)
                    distances = np.linalg.norm(valid_points - center, axis=-1)
                    point_scale = max(
                        float(np.sqrt(np.mean(distances ** 2))), 1e-4
                    )
                    object_center[frame] = center
                    object_point_scale[frame] = point_scale
            for metadata in (hand_metadata, object_metadata, context_metadata):
                centered_points = (
                    metadata[..., :3] - object_center[:, None]
                )
                object_frame_points = np.einsum(
                    "tni,tij->tnj",
                    centered_points,
                    filtered_pose[:, :3, :3],
                )
                metadata[..., :3] = (
                    object_frame_points / object_point_scale[:, None, None]
                )
                metadata[:] = np.nan_to_num(metadata)

            key_features = np.concatenate(
                (object_features, context_features), axis=1
            )
            key_metadata = np.concatenate(
                (object_metadata, context_metadata), axis=1
            )
            key_valid = np.concatenate(
                (object_token_valid, context_token_valid), axis=1
            )
            pose_frame_valid = np.isfinite(
                filtered_pose[:, :3, :3]
            ).all(axis=(1, 2))
            relation_frame_valid = (
                pose_frame_valid & object_token_valid.any(axis=1)
            )
            hand_token_valid &= relation_frame_valid[:, None]
            key_valid &= relation_frame_valid[:, None]
            key_types = np.concatenate(
                (
                    np.zeros_like(object_token_valid, dtype=np.int64),
                    np.ones_like(context_token_valid, dtype=np.int64),
                ),
                axis=1,
            )
            sample.update(
                hand_token_features=torch.from_numpy(hand_features),
                hand_token_metadata=torch.from_numpy(hand_metadata),
                hand_token_valid=torch.from_numpy(hand_token_valid),
                key_token_features=torch.from_numpy(key_features),
                key_token_metadata=torch.from_numpy(key_metadata),
                key_token_valid=torch.from_numpy(key_valid),
                key_token_types=torch.from_numpy(key_types),
            )
        return sample


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first = F.normalize(value[..., :3], dim=-1, eps=1e-6)
    second_raw = value[..., 3:6]
    second = second_raw - (first * second_raw).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


class RelativeCrossAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        metadata_dim: int,
        relation_dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        if relation_dim % heads:
            raise ValueError("pi3x-relation-dim must be divisible by pi3x-heads")
        self.heads = heads
        self.head_dim = relation_dim // heads
        self.scale = self.head_dim ** -0.5
        self.hand_feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, relation_dim)
        )
        self.key_feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, relation_dim)
        )
        self.hand_metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, relation_dim),
        )
        self.key_metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, relation_dim),
        )
        self.key_type_embedding = nn.Embedding(2, relation_dim)
        self.query = nn.Linear(relation_dim, relation_dim)
        self.key = nn.Linear(relation_dim, relation_dim)
        self.value = nn.Linear(relation_dim, relation_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(6, 32), nn.GELU(), nn.Linear(32, heads)
        )
        self.output = nn.Linear(relation_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hand_features: torch.Tensor,
        hand_metadata: torch.Tensor,
        hand_valid: torch.Tensor,
        key_features: torch.Tensor,
        key_metadata: torch.Tensor,
        key_valid: torch.Tensor,
        key_types: torch.Tensor,
    ) -> torch.Tensor:
        hand = self.hand_feature_projection(hand_features)
        hand = hand + self.hand_metadata_projection(hand_metadata)
        key_token = self.key_feature_projection(key_features)
        key_token = key_token + self.key_metadata_projection(key_metadata)
        key_token = key_token + self.key_type_embedding(key_types)

        batch, frames, hand_count, _ = hand.shape
        key_count = key_token.shape[2]
        query = self.query(hand).reshape(
            batch, frames, hand_count, self.heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        key = self.key(key_token).reshape(
            batch, frames, key_count, self.heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        value = self.value(key_token).reshape(
            batch, frames, key_count, self.heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)

        delta_xyz = (
            hand_metadata[:, :, :, None, :3]
            - key_metadata[:, :, None, :, :3]
        )
        distance = torch.linalg.norm(delta_xyz, dim=-1, keepdim=True)
        delta_uv = (
            hand_metadata[:, :, :, None, 3:5]
            - key_metadata[:, :, None, :, 3:5]
        )
        relative = torch.cat((delta_xyz, distance, delta_uv), dim=-1)
        bias = self.relative_bias(relative).permute(0, 1, 4, 2, 3)
        logits = torch.einsum("bthid,bthjd->bthij", query, key)
        logits = logits * self.scale + bias
        logits = logits.masked_fill(
            ~key_valid[:, :, None, None, :], -1e4
        )
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        attended = torch.einsum("bthij,bthjd->bthid", attention, value)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(
            batch, frames, hand_count, -1
        )
        has_key = key_valid.any(dim=-1)[..., None, None]
        attended = attended * has_key.to(attended.dtype)
        updated = self.norm(hand + self.output(attended))

        confidence = hand_metadata[..., 6].clamp(0.0, 1.0)
        pool_weight = confidence * hand_valid.to(confidence.dtype)
        fallback = hand_valid.to(confidence.dtype)
        use_fallback = pool_weight.sum(-1, keepdim=True) <= 1e-6
        pool_weight = torch.where(use_fallback, fallback, pool_weight)
        pooled = (
            updated * pool_weight[..., None]
        ).sum(dim=2) / pool_weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        return pooled


class AbsoluteObjectFramePoseModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        max_normalized_translation: float,
        num_objects: int,
        object_embedding_dim: int,
        pi3x_feature_dim: int = 0,
        pi3x_metadata_dim: int = 0,
        pi3x_relation_dim: int = 128,
        pi3x_heads: int = 8,
    ):
        super().__init__()
        self.max_normalized_translation = max_normalized_translation
        self.object_embedding = (
            nn.Embedding(num_objects, object_embedding_dim)
            if object_embedding_dim > 0
            else None
        )
        self.pi3x_relation = (
            RelativeCrossAttention(
                pi3x_feature_dim,
                pi3x_metadata_dim,
                pi3x_relation_dim,
                pi3x_heads,
                dropout,
            )
            if pi3x_feature_dim > 0
            else None
        )
        encoder_input_dim = input_dim + object_embedding_dim
        if self.pi3x_relation is not None:
            encoder_input_dim += pi3x_relation_dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        gru_hidden = hidden_dim // 2
        if hidden_dim % 2:
            raise ValueError("hidden-dim must be even")
        self.temporal = nn.GRU(
            hidden_dim,
            gru_hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6)
        )
        nn.init.zeros_(self.translation_head[-1].weight)
        nn.init.zeros_(self.translation_head[-1].bias)
        nn.init.zeros_(self.rotation_head[-1].weight)
        self.rotation_head[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )

    def forward(
        self,
        features: torch.Tensor,
        object_index: torch.Tensor,
        hand_token_features: torch.Tensor | None = None,
        hand_token_metadata: torch.Tensor | None = None,
        hand_token_valid: torch.Tensor | None = None,
        key_token_features: torch.Tensor | None = None,
        key_token_metadata: torch.Tensor | None = None,
        key_token_valid: torch.Tensor | None = None,
        key_token_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.object_embedding is not None:
            object_feature = self.object_embedding(object_index)
            object_feature = object_feature[:, None].expand(
                -1, features.shape[1], -1
            )
            features = torch.cat((features, object_feature), dim=-1)
        if self.pi3x_relation is not None:
            token_values = (
                hand_token_features,
                hand_token_metadata,
                hand_token_valid,
                key_token_features,
                key_token_metadata,
                key_token_valid,
                key_token_types,
            )
            if any(value is None for value in token_values):
                raise ValueError("Pi3X relation tokens are required")
            relation = self.pi3x_relation(*token_values)
            features = torch.cat((features, relation), dim=-1)
        encoded = self.frame_encoder(features)
        temporal, _ = self.temporal(encoded)
        translation = (
            torch.tanh(self.translation_head(temporal))
            * self.max_normalized_translation
        )
        rotation = rotation_6d_to_matrix(self.rotation_head(temporal))
        return translation, rotation


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def smooth_l1_vector(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    difference = torch.abs(prediction - target)
    beta = beta[..., None].clamp_min(1e-6)
    loss = torch.where(
        difference < beta,
        0.5 * difference.square() / beta,
        difference - 0.5 * beta,
    )
    return loss.mean(dim=-1)


def rotation_angle(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    relative = prediction.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    return torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def temporal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    order: int,
    beta_m: float,
) -> torch.Tensor:
    pred, truth, mask = prediction, target, valid
    for _ in range(order):
        pred = pred[:, 1:] - pred[:, :-1]
        truth = truth[:, 1:] - truth[:, :-1]
        mask = mask[:, 1:] & mask[:, :-1]
    beta = torch.full_like(mask, beta_m, dtype=prediction.dtype)
    return masked_mean(smooth_l1_vector(pred, truth, beta), mask)


def metric_distribution(values: list[np.ndarray], scale: float) -> dict:
    if not values:
        return {"count": 0}
    array = np.concatenate(values).astype(np.float64) * scale
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict_float()
    batches = 0
    initial_translation_errors = []
    predicted_translation_errors = []
    initial_rotation_errors = []
    predicted_rotation_errors = []
    iterator = tqdm(loader, desc="train" if training else "val")

    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            predicted_t_normalized, predicted_r = model(
                batch["features"],
                batch["object_index"],
                batch.get("hand_token_features"),
                batch.get("hand_token_metadata"),
                batch.get("hand_token_valid"),
                batch.get("key_token_features"),
                batch.get("key_token_metadata"),
                batch.get("key_token_valid"),
                batch.get("key_token_types"),
            )
            scale = batch["object_scale"][..., None]
            predicted_t = predicted_t_normalized * scale
            target_t = batch["target_translation"] * scale
            initial_t = batch["initial_translation"] * scale
            beta_normalized = (
                args.smooth_l1_beta_mm / 1000.0
                / batch["object_scale"]
            )
            translation_frame = smooth_l1_vector(
                predicted_t_normalized,
                batch["target_translation"],
                beta_normalized,
            )
            translation = masked_mean(
                translation_frame, batch["valid_translation"]
            )
            angle = rotation_angle(predicted_r, batch["target_rotation"])
            initial_angle = rotation_angle(
                batch["initial_rotation"], batch["target_rotation"]
            )
            rotation_mask = batch["valid_rotation"].to(angle.dtype)
            rotation_mask = rotation_mask * batch["rotation_weight"]
            rotation = (
                (angle * rotation_mask).sum()
                / rotation_mask.sum().clamp_min(1.0)
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
            predicted_error = torch.linalg.norm(predicted_t - target_t, dim=-1)
            initial_error = torch.linalg.norm(initial_t - target_t, dim=-1)
            degradation_guard = masked_mean(
                F.relu(predicted_error - initial_error) / 0.02,
                batch["valid_translation"],
            )
            rotation_degradation_guard = (
                (F.relu(angle - initial_angle) * rotation_mask).sum()
                / rotation_mask.sum().clamp_min(1.0)
            )
            total = (
                args.w_translation * translation
                + args.w_rotation * rotation
                + args.w_velocity * velocity
                + args.w_acceleration * acceleration
                + args.w_degradation_guard * degradation_guard
                + args.w_rotation_degradation_guard
                * rotation_degradation_guard
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        values = {
            "total": total,
            "translation": translation,
            "rotation": rotation,
            "velocity": velocity,
            "acceleration": acceleration,
            "degradation_guard": degradation_guard,
            "rotation_degradation_guard": rotation_degradation_guard,
        }
        for key, value in values.items():
            totals[key] += float(value.detach())
        batches += 1
        iterator.set_postfix(loss=f"{totals['total'] / batches:.5f}")

        valid_t = batch["valid_translation"].detach().cpu().numpy().astype(bool)
        valid_r = batch["valid_rotation"].detach().cpu().numpy().astype(bool)
        initial_t_error = initial_error.detach().cpu().numpy()
        predicted_t_error = predicted_error.detach().cpu().numpy()
        initial_r_error = initial_angle.detach().cpu().numpy()
        predicted_r_error = angle.detach().cpu().numpy()
        initial_translation_errors.append(initial_t_error[valid_t])
        predicted_translation_errors.append(predicted_t_error[valid_t])
        initial_rotation_errors.append(initial_r_error[valid_r])
        predicted_rotation_errors.append(predicted_r_error[valid_r])

    output = {key: value / max(batches, 1) for key, value in totals.items()}
    output.update(
        initial_translation=metric_distribution(
            initial_translation_errors, 1000.0
        ),
        predicted_translation=metric_distribution(
            predicted_translation_errors, 1000.0
        ),
        initial_rotation=metric_distribution(
            initial_rotation_errors, 180.0 / math.pi
        ),
        predicted_rotation=metric_distribution(
            predicted_rotation_errors, 180.0 / math.pi
        ),
    )
    return output


def defaultdict_float() -> dict[str, float]:
    return {
        "total": 0.0,
        "translation": 0.0,
        "rotation": 0.0,
        "velocity": 0.0,
        "acceleration": 0.0,
        "degradation_guard": 0.0,
        "rotation_degradation_guard": 0.0,
    }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    args = parse_args()
    if args.object_embedding_dim < 0:
        raise ValueError("object-embedding-dim must be non-negative")
    if args.pi3x_relation_dim <= 0 or args.pi3x_heads <= 0:
        raise ValueError("Pi3X relation dimensions must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_windows = Path(args.train_windows).expanduser().resolve()
    val_windows = Path(args.val_windows).expanduser().resolve()
    if bool(args.pi3x_train_root) != bool(args.pi3x_val_root):
        raise ValueError(
            "pi3x-train-root and pi3x-val-root must be provided together"
        )
    pi3x_train_root = (
        Path(args.pi3x_train_root).expanduser().resolve()
        if args.pi3x_train_root
        else None
    )
    pi3x_val_root = (
        Path(args.pi3x_val_root).expanduser().resolve()
        if args.pi3x_val_root
        else None
    )
    for path in (train_windows, val_windows):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (pi3x_train_root, pi3x_val_root):
        if path is not None and not path.is_dir():
            raise NotADirectoryError(path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(train_windows)
    object_names = sorted({str(row["object_name"]) for row in train_rows})
    val_object_names = {
        str(row["object_name"]) for row in load_jsonl(val_windows)
    }
    unknown_objects = sorted(val_object_names - set(object_names))
    if unknown_objects:
        raise KeyError(f"Validation contains unknown objects: {unknown_objects}")
    object_to_index = {
        name: index for index, name in enumerate(object_names)
    }
    train_data = ObjectFrameWindowDataset(
        train_windows,
        args,
        augment=True,
        object_to_index=object_to_index,
        pi3x_root=pi3x_train_root,
    )
    val_data = ObjectFrameWindowDataset(
        val_windows,
        args,
        augment=False,
        object_to_index=object_to_index,
        pi3x_root=pi3x_val_root,
    )
    sample = train_data[0]
    input_dim = int(sample["features"].shape[-1])
    pi3x_feature_dim = (
        int(sample["hand_token_features"].shape[-1])
        if "hand_token_features" in sample
        else 0
    )
    pi3x_metadata_dim = (
        int(sample["hand_token_metadata"].shape[-1])
        if "hand_token_metadata" in sample
        else 0
    )
    device = torch.device(args.device)
    model = AbsoluteObjectFramePoseModel(
        input_dim,
        args.hidden_dim,
        args.layers,
        args.dropout,
        args.max_normalized_translation,
        len(object_names),
        args.object_embedding_dim,
        pi3x_feature_dim,
        pi3x_metadata_dim,
        args.pi3x_relation_dim,
        args.pi3x_heads,
    ).to(device)
    print(
        json.dumps(
            {
                "model_version": (
                    PI3X_RELATION_MODEL_VERSION
                    if pi3x_feature_dim > 0
                    else (
                        OBJECT_CONDITIONED_MODEL_VERSION
                        if args.object_embedding_dim > 0
                        else MODEL_VERSION
                    )
                ),
                "windows": {
                    "train": len(train_data),
                    "val": len(val_data),
                },
                "input_dim": input_dim,
                "pi3x_feature_dim": pi3x_feature_dim,
                "pi3x_metadata_dim": pi3x_metadata_dim,
                "pi3x_relation_dim": (
                    args.pi3x_relation_dim if pi3x_feature_dim > 0 else 0
                ),
                "objects": len(object_names),
            }
        ),
        flush=True,
    )
    if args.data_parallel:
        model = nn.DataParallel(model)

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_data, shuffle=True, drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    history = []
    best_total = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====", flush=True)
        train_metrics = run_epoch(
            model, train_loader, device, args, optimizer
        )
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
            "model_version": (
                PI3X_RELATION_MODEL_VERSION
                if pi3x_feature_dim > 0
                else (
                    OBJECT_CONDITIONED_MODEL_VERSION
                    if args.object_embedding_dim > 0
                    else MODEL_VERSION
                )
            ),
            "model_state": (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "input_dim": input_dim,
            "object_embedding_dim": args.object_embedding_dim,
            "object_names": object_names,
            "pi3x_feature_dim": pi3x_feature_dim,
            "pi3x_metadata_dim": pi3x_metadata_dim,
            "pi3x_relation_dim": (
                args.pi3x_relation_dim if pi3x_feature_dim > 0 else 0
            ),
            "pi3x_heads": args.pi3x_heads if pi3x_feature_dim > 0 else 0,
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
