#!/usr/bin/env python3
"""Train a compact translation-equivariant hand/object relative refiner."""

from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PALM = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
TOKEN_GROUPS = ("hand", "object", "context")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--relative-train-root")
    parser.add_argument("--relative-val-root")
    parser.add_argument("--pi3x-train-root")
    parser.add_argument("--pi3x-val-root")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--spatial-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--pi3x-gate-start-mm", type=float, default=0.0)
    parser.add_argument("--pi3x-gate-full-mm", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-correction-mm", type=float, default=120.0)
    parser.add_argument("--max-object-center-error-mm", type=float, default=30.0)
    parser.add_argument("--max-target-projection-shift-px", type=float, default=20.0)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--w-relative", type=float, default=1.0)
    parser.add_argument("--w-velocity", type=float, default=0.1)
    parser.add_argument("--w-acceleration", type=float, default=0.05)
    parser.add_argument("--w-residual", type=float, default=0.001)
    parser.add_argument("--target-weight-lt5", type=float, default=1.0)
    parser.add_argument("--target-weight-5-15", type=float, default=1.0)
    parser.add_argument("--target-weight-15-30", type=float, default=1.0)
    parser.add_argument("--target-weight-ge30", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=16)
def load_npz(path_text: str) -> dict[str, np.ndarray]:
    with np.load(path_text, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.maximum(points[..., 2], 1e-4)
    return np.stack(
        (
            intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2],
            intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2],
        ),
        axis=-1,
    )


def temporal_difference(value: np.ndarray, order: int) -> np.ndarray:
    output = value.copy()
    for _ in range(order):
        difference = np.zeros_like(output)
        difference[1:] = output[1:] - output[:-1]
        output = difference
    return output


def decode_text(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


class RelativeWindowDataset(Dataset):
    def __init__(
        self,
        windows: Path,
        global_root: Path,
        args: argparse.Namespace,
        relative_root: Optional[Path] = None,
        pi3x_root: Optional[Path] = None,
    ):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.global_root = global_root
        self.relative_root = relative_root
        self.pi3x_root = pi3x_root
        self.max_correction = args.max_correction_mm / 1000.0
        self.max_object_error = args.max_object_center_error_mm / 1000.0
        self.max_projection_shift = args.max_target_projection_shift_px

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        stream_id = row["stream_id"]
        start, end = int(row["start"]), int(row["end"])
        rigid_path = (
            self.relative_root / f"{stream_id}.npz"
            if self.relative_root is not None
            else Path(row["supervision_npz"])
        ).resolve()
        rigid = load_npz(str(rigid_path))
        required = {
            "pred_hand_center",
            "pred_object_center",
            "gt_hand_center",
            "gt_object_center",
            "relative_supervision_valid",
        }
        missing = required.difference(rigid)
        if missing:
            raise KeyError(
                f"{rigid_path} is not relative-translation supervision; "
                f"missing {sorted(missing)}"
            )
        global_path = self.global_root / f"{stream_id}.npz"
        global_data = load_npz(str(global_path.resolve()))

        rigid_frames = np.asarray(rigid["frame_ids"])
        global_frames = np.asarray(global_data["frame_ids"])
        if len(rigid_frames) != len(global_frames) or not np.array_equal(
            rigid_frames.astype(str), global_frames.astype(str)
        ):
            raise ValueError(f"Frame mismatch for {stream_id}")

        pred_hand = np.asarray(rigid["pred_hand_center"], dtype=np.float32)[start:end]
        pred_object = np.asarray(
            rigid["pred_object_center"], dtype=np.float32
        )[start:end]
        gt_hand = np.asarray(rigid["gt_hand_center"], dtype=np.float32)[start:end]
        gt_object = np.asarray(rigid["gt_object_center"], dtype=np.float32)[start:end]
        object_rot6d = np.asarray(
            rigid["pred_object_rot6d"], dtype=np.float32
        )[start:end]
        loss_valid = np.asarray(
            rigid["relative_supervision_valid"], dtype=bool
        )[start:end]
        observed_valid = (
            np.asarray(rigid["pred_hand_valid"], dtype=bool)[start:end]
            & np.asarray(rigid["pred_object_valid"], dtype=bool)[start:end]
        )
        intrinsics = np.asarray(rigid["intrinsics"], dtype=np.float32)

        joints = np.asarray(
            global_data["pred_joints_3d"], dtype=np.float32
        )[start:end].copy()
        normalized_left = bool(
            np.asarray(global_data.get("normalized_left", False)).item()
        )
        if normalized_left:
            joints[..., 0] *= -1.0
        hand_side = decode_text(global_data["hand_side"])
        extents = np.asarray(
            global_data.get("object_extents_metric", np.zeros(3)),
            dtype=np.float32,
        ).reshape(3)

        arrays = [pred_hand, pred_object, gt_hand, gt_object, joints, object_rot6d]
        finite = np.logical_and.reduce(
            [np.isfinite(value).all(axis=tuple(range(1, value.ndim))) for value in arrays]
        )
        loss_valid &= finite
        observed_valid &= finite

        relative_initial = pred_hand - pred_object
        relative_target = gt_hand - gt_object
        target_delta = relative_target - relative_initial
        object_delta = gt_object - pred_object
        target_hand = gt_hand - object_delta
        projection_shift = np.linalg.norm(
            project(target_hand, intrinsics) - project(gt_hand, intrinsics), axis=-1
        )
        loss_valid &= np.linalg.norm(target_delta, axis=-1) <= self.max_correction
        loss_valid &= np.linalg.norm(object_delta, axis=-1) <= self.max_object_error
        loss_valid &= projection_shift <= self.max_projection_shift

        joints = np.nan_to_num(joints)
        pred_hand = np.nan_to_num(pred_hand)
        pred_object = np.nan_to_num(pred_object)
        relative_initial = np.nan_to_num(relative_initial)
        relative_target = np.nan_to_num(relative_target)
        object_rot6d = np.nan_to_num(object_rot6d)

        articulation = joints - joints[:, 0:1]
        palm_relative = joints[:, PALM] - pred_object[:, None]
        relative_velocity = temporal_difference(relative_initial, 1)
        relative_acceleration = temporal_difference(relative_initial, 2)
        side_feature = np.asarray(
            [1.0, 0.0] if hand_side == "left" else [0.0, 1.0],
            dtype=np.float32,
        )
        features = np.concatenate(
            [
                articulation.reshape(len(joints), -1),
                palm_relative.reshape(len(joints), -1),
                relative_initial,
                relative_velocity,
                relative_acceleration,
                object_rot6d,
                np.broadcast_to(extents, (len(joints), 3)),
                np.broadcast_to(side_feature, (len(joints), 2)),
                observed_valid[:, None].astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
        sample = {
            "features": torch.from_numpy(features),
            "relative_initial": torch.from_numpy(relative_initial),
            "relative_target": torch.from_numpy(relative_target),
            "valid": torch.from_numpy(loss_valid),
            "target_magnitude": torch.from_numpy(
                np.linalg.norm(target_delta, axis=-1).astype(np.float32)
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
            token_features = []
            token_metadata = []
            token_valid = []
            token_types = []
            grid = np.asarray(
                pi3x["geometry_feature_grid_hw"], dtype=np.float32
            )
            for type_index, prefix in enumerate(TOKEN_GROUPS):
                group_features = np.asarray(
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
                indices[..., 0] /= max(float(grid[0] - 1), 1.0)
                indices[..., 1] /= max(float(grid[1] - 1), 1.0)
                group_metadata = np.concatenate(
                    [
                        points,
                        coverage[..., None],
                        confidence[..., None],
                        indices,
                    ],
                    axis=-1,
                )
                group_valid = np.asarray(
                    pi3x[f"{prefix}_valid"][positions], dtype=bool
                )
                token_features.append(group_features)
                token_metadata.append(group_metadata)
                token_valid.append(group_valid)
                token_types.append(
                    np.full(group_valid.shape, type_index, dtype=np.int64)
                )
            sample.update(
                {
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
                }
            )
        return sample


class RelativeTranslationRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        pi3x_feature_dim: int = 0,
        pi3x_metadata_dim: int = 0,
        spatial_layers: int = 1,
        heads: int = 8,
    ):
        super().__init__()
        self.pi3x_enabled = pi3x_feature_dim > 0
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if self.pi3x_enabled:
            self.token_feature_projection = nn.Sequential(
                nn.LayerNorm(pi3x_feature_dim),
                nn.Linear(pi3x_feature_dim, hidden_dim),
            )
            self.token_metadata_projection = nn.Sequential(
                nn.Linear(pi3x_metadata_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.token_type_embedding = nn.Embedding(
                len(TOKEN_GROUPS), hidden_dim
            )
            spatial_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.spatial_encoder = nn.TransformerEncoder(
                spatial_layer, num_layers=spatial_layers
            )
            self.pi3x_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.pi3x_fusion[-1].weight)
            nn.init.zeros_(self.pi3x_fusion[-1].bias)
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def masked_pool(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(value.dtype).unsqueeze(-1)
        return (value * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)

    def forward(
        self,
        features: torch.Tensor,
        max_correction: float,
        token_features: Optional[torch.Tensor] = None,
        token_metadata: Optional[torch.Tensor] = None,
        token_valid: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        pi3x_gate_start: float = 0.0,
        pi3x_gate_full: float = 0.0,
    ) -> torch.Tensor:
        encoded = self.frame_encoder(features)
        base_temporal, _ = self.temporal(encoded)
        base_correction = torch.tanh(self.head(base_temporal)) * max_correction
        if self.pi3x_enabled:
            if any(
                value is None
                for value in (
                    token_features,
                    token_metadata,
                    token_valid,
                    token_types,
                )
            ):
                raise ValueError("Pi3X tokens are required by this checkpoint")
            batch, frames, tokens, _ = token_features.shape
            token = self.token_feature_projection(token_features)
            token = token + self.token_metadata_projection(token_metadata)
            token = token + self.token_type_embedding(token_types)
            flat = token.reshape(batch * frames, tokens, -1)
            flat_valid = token_valid.reshape(batch * frames, tokens).clone()
            missing = ~flat_valid.any(dim=1)
            flat_valid[missing, 0] = True
            spatial = self.spatial_encoder(
                flat, src_key_padding_mask=~flat_valid
            ).reshape(batch, frames, tokens, -1)
            pooled = []
            for type_index in range(len(TOKEN_GROUPS)):
                mask = token_valid & (token_types == type_index)
                pooled.append(self.masked_pool(spatial, mask))
            adapted = encoded + self.pi3x_fusion(torch.cat(pooled, dim=-1))
            adapted_temporal, _ = self.temporal(adapted)
            adapted_correction = (
                torch.tanh(self.head(adapted_temporal)) * max_correction
            )
            if pi3x_gate_full > pi3x_gate_start:
                magnitude = torch.linalg.norm(base_correction, dim=-1)
                gate = (
                    (magnitude - pi3x_gate_start)
                    / (pi3x_gate_full - pi3x_gate_start)
                ).clamp(0.0, 1.0)
                gate = gate.square() * (3.0 - 2.0 * gate)
                return base_correction + gate.unsqueeze(-1) * (
                    adapted_correction - base_correction
                )
            return adapted_correction
        return base_correction


def masked_smooth_l1(value, target, mask, beta, sample_weight=None):
    error = F.smooth_l1_loss(value, target, reduction="none", beta=beta).mean(-1)
    weight = mask.float()
    if sample_weight is not None:
        weight = weight * sample_weight
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def temporal_loss(value, target, mask, order, beta):
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
        target = target[:, 1:] - target[:, :-1]
        mask = mask[:, 1:] & mask[:, :-1]
    return masked_smooth_l1(value, target, mask, beta)


def compute_loss(model, batch, args):
    batch = {key: value.to(args.device) for key, value in batch.items()}
    correction = model(
        batch["features"],
        args.max_correction_mm / 1000.0,
        batch.get("token_features"),
        batch.get("token_metadata"),
        batch.get("token_valid"),
        batch.get("token_types"),
        args.pi3x_gate_start_mm / 1000.0,
        args.pi3x_gate_full_mm / 1000.0,
    )
    corrected = batch["relative_initial"] + correction
    beta = args.smooth_l1_beta_mm / 1000.0
    magnitude_mm = batch["target_magnitude"] * 1000.0
    target_weight = torch.where(
        magnitude_mm < 5.0,
        torch.full_like(magnitude_mm, args.target_weight_lt5),
        torch.where(
            magnitude_mm < 15.0,
            torch.full_like(magnitude_mm, args.target_weight_5_15),
            torch.where(
                magnitude_mm < 30.0,
                torch.full_like(magnitude_mm, args.target_weight_15_30),
                torch.full_like(magnitude_mm, args.target_weight_ge30),
            ),
        ),
    )
    losses = {
        "relative": masked_smooth_l1(
            corrected,
            batch["relative_target"],
            batch["valid"],
            beta,
            target_weight,
        ),
        "velocity": temporal_loss(
            corrected, batch["relative_target"], batch["valid"], 1, beta
        ),
        "acceleration": temporal_loss(
            corrected, batch["relative_target"], batch["valid"], 2, beta
        ),
        "residual": (correction.square().mean(-1) * batch["valid"].float()).sum()
        / batch["valid"].float().sum().clamp_min(1.0),
    }
    total = (
        args.w_relative * losses["relative"]
        + args.w_velocity * losses["velocity"]
        + args.w_acceleration * losses["acceleration"]
        + args.w_residual * losses["residual"]
    )
    return total, losses, batch, corrected


def summarize(values: list[torch.Tensor]) -> dict:
    if not values:
        return {"count": 0}
    value = torch.cat(values)
    if not len(value):
        return {"count": 0}
    return {
        "count": int(len(value)),
        "median_mm": float(torch.quantile(value, 0.5) * 1000.0),
        "p90_mm": float(torch.quantile(value, 0.9) * 1000.0),
        "max_mm": float(value.max() * 1000.0),
    }


def run_epoch(model, loader, args, optimizer=None, split="train"):
    training = optimizer is not None
    model.train(training)
    sums, batches = {}, 0
    initial_errors, corrected_errors = [], []
    bin_edges = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, float("inf")))
    binned_initial = {edge: [] for edge in bin_edges}
    binned_corrected = {edge: [] for edge in bin_edges}
    valid_frames = 0
    progress = tqdm(loader, desc=split, dynamic_ncols=True)
    for batch in progress:
        with torch.set_grad_enabled(training):
            total, losses, values, corrected = compute_loss(model, batch, args)
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        mask = values["valid"]
        valid_frames += int(mask.sum())
        initial_errors.append(
            torch.linalg.norm(
                values["relative_initial"][mask] - values["relative_target"][mask],
                dim=-1,
            ).detach().cpu()
        )
        corrected_errors.append(
            torch.linalg.norm(
                corrected[mask] - values["relative_target"][mask], dim=-1
            ).detach().cpu()
        )
        initial_error = torch.linalg.norm(
            values["relative_initial"] - values["relative_target"], dim=-1
        )
        corrected_error = torch.linalg.norm(
            corrected - values["relative_target"], dim=-1
        )
        magnitude_mm = values["target_magnitude"] * 1000.0
        for lower, upper in bin_edges:
            bin_mask = mask & (magnitude_mm >= lower) & (magnitude_mm < upper)
            binned_initial[(lower, upper)].append(
                initial_error[bin_mask].detach().cpu()
            )
            binned_corrected[(lower, upper)].append(
                corrected_error[bin_mask].detach().cpu()
            )
        for key, value in {"total": total, **losses}.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach())
        batches += 1
        progress.set_postfix(loss=f"{sums['total'] / batches:.5f}")
    metrics = {key: value / max(batches, 1) for key, value in sums.items()}
    metrics["valid_frames"] = valid_frames
    metrics["initial_relative"] = summarize(initial_errors)
    metrics["corrected_relative"] = summarize(corrected_errors)
    metrics["target_bins"] = {}
    for lower, upper in bin_edges:
        label = f"{int(lower)}_{'inf' if not np.isfinite(upper) else int(upper)}mm"
        metrics["target_bins"][label] = {
            "initial": summarize(binned_initial[(lower, upper)]),
            "corrected": summarize(binned_corrected[(lower, upper)]),
        }
    return metrics


def main() -> None:
    args = parse_args()
    if bool(args.pi3x_train_root) != bool(args.pi3x_val_root):
        raise ValueError(
            "--pi3x-train-root and --pi3x-val-root must be used together"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = RelativeWindowDataset(
        Path(args.train_windows).expanduser().resolve(),
        Path(args.global_train_root).expanduser().resolve(),
        args,
        (
            Path(args.relative_train_root).expanduser().resolve()
            if args.relative_train_root
            else None
        ),
        (
            Path(args.pi3x_train_root).expanduser().resolve()
            if args.pi3x_train_root
            else None
        ),
    )
    val_data = RelativeWindowDataset(
        Path(args.val_windows).expanduser().resolve(),
        Path(args.global_val_root).expanduser().resolve(),
        args,
        (
            Path(args.relative_val_root).expanduser().resolve()
            if args.relative_val_root
            else None
        ),
        (
            Path(args.pi3x_val_root).expanduser().resolve()
            if args.pi3x_val_root
            else None
        ),
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
    input_dim = int(train_data[0]["features"].shape[-1])
    sample = train_data[0]
    pi3x_feature_dim = (
        int(sample["token_features"].shape[-1])
        if "token_features" in sample
        else 0
    )
    pi3x_metadata_dim = (
        int(sample["token_metadata"].shape[-1])
        if "token_metadata" in sample
        else 0
    )
    model = RelativeTranslationRefiner(
        input_dim,
        args.hidden_dim,
        args.layers,
        args.dropout,
        pi3x_feature_dim,
        pi3x_metadata_dim,
        args.spatial_layers,
        args.heads,
    ).to(args.device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        init_checkpoint = torch.load(init_path, map_location="cpu")
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in init_checkpoint["model"].items()
            if key in current and current[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            json.dumps(
                {
                    "init_checkpoint": str(init_path),
                    "loaded_tensors": len(compatible),
                    "new_tensors": len(missing),
                    "unexpected_tensors": len(unexpected),
                }
            ),
            flush=True,
        )
    if args.freeze_base:
        pi3x_prefixes = (
            "token_feature_projection.",
            "token_metadata_projection.",
            "token_type_embedding.",
            "spatial_encoder.",
            "pi3x_fusion.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(pi3x_prefixes)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    print(
        json.dumps(
            {
                "freeze_base": bool(args.freeze_base),
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in trainable_parameters)
                ),
                "total_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, optimizer, "train")
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
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "input_dim": input_dim,
            "pi3x_feature_dim": pi3x_feature_dim,
            "pi3x_metadata_dim": pi3x_metadata_dim,
            "epoch": epoch,
            "val_total": val_metrics["total"],
            "model_version": (
                (
                    "relative_translation_pi3x_gated_spatial_bigru_v4"
                    if args.pi3x_gate_full_mm > args.pi3x_gate_start_mm
                    else "relative_translation_pi3x_spatial_bigru_weighted_v3"
                )
                if pi3x_feature_dim > 0
                else "relative_translation_mlp_bigru_weighted_v2"
            ),
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
