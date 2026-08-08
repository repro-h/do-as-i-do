#!/usr/bin/env python3
"""Train a camera-frame HandFlow residual refiner without explicit object inputs."""

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
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import RelativeCrossAttention


MODEL_VERSION = "v9_camera_hand_residual_observation_only_v1"
KEY_JOINTS = np.asarray([4, 5, 8, 9, 12, 13, 16, 17, 20], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-windows", required=True)
    p.add_argument("--val-windows", required=True)
    p.add_argument("--global-train-root", required=True)
    p.add_argument("--global-val-root", required=True)
    p.add_argument("--pi3x-train-root", required=True)
    p.add_argument("--pi3x-val-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pi3x-relation-dim", type=int, default=128)
    p.add_argument("--pi3x-heads", type=int, default=8)
    p.add_argument("--max-correction-mm", type=float, default=250.0)
    p.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    p.add_argument("--w-translation", type=float, default=1.0)
    p.add_argument("--w-rotation", type=float, default=0.5)
    p.add_argument("--w-velocity", type=float, default=0.05)
    p.add_argument("--w-acceleration", type=float, default=0.02)
    p.add_argument("--w-residual", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-parallel", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


@lru_cache(maxsize=64)
def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def scalar_text(value: np.ndarray) -> str:
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def rotation_to_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first = F.normalize(value[..., :3], dim=-1, eps=1e-6)
    second_raw = value[..., 3:6]
    second = second_raw - (first * second_raw).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def rotation_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    relative = a.transpose(-1, -2) @ b
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def smooth_l1(value: torch.Tensor, beta: float) -> torch.Tensor:
    absolute = value.abs()
    quadratic = torch.minimum(absolute, torch.as_tensor(beta, device=value.device))
    linear = absolute - quadratic
    return 0.5 * quadratic.square() / max(beta, 1e-8) + linear


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(value.dtype)
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


def temporal_loss(value: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, order: int, beta: float) -> torch.Tensor:
    for _ in range(order):
        value = value.diff(dim=1)
        target = target.diff(dim=1)
        valid = valid[:, 1:] & valid[:, :-1]
    if value.shape[1] == 0:
        return value.new_zeros(())
    return masked_mean(smooth_l1(value - target, beta), valid[..., None])


class CameraWindowDataset(Dataset):
    def __init__(self, windows: Path, global_root: Path, pi3x_root: Path):
        self.rows = load_jsonl(windows)
        self.global_root = global_root
        self.pi3x_root = pi3x_root
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        stream_id = str(row["stream_id"])
        start, end = int(row["start"]), int(row["end"])
        se3_path = Path(row["supervision_npz"]).expanduser().resolve()
        se3 = load_npz(str(se3_path))
        global_path = Path(
            scalar_text(se3["source_global_supervision"])
        ).expanduser().resolve()
        if not global_path.is_file():
            global_path = self.global_root / f"{stream_id}.npz"
        glob = load_npz(str(global_path))

        pred_joints = np.asarray(glob["pred_joints_3d"], dtype=np.float32)[start:end]
        gt_joints = np.asarray(glob["gt_joints_3d"], dtype=np.float32)[start:end]
        initial_rotvec = np.asarray(
            glob["initial_root_rotvec"], dtype=np.float64
        )[start:end]
        target_rotvec = np.asarray(
            glob["gt_root_rotvec"], dtype=np.float64
        )[start:end]
        initial_r = Rotation.from_rotvec(
            np.nan_to_num(initial_rotvec, nan=0.0, posinf=0.0, neginf=0.0)
        ).as_matrix().astype(np.float32)
        target_r = Rotation.from_rotvec(
            np.nan_to_num(target_rotvec, nan=0.0, posinf=0.0, neginf=0.0)
        ).as_matrix().astype(np.float32)

        initial_t = pred_joints[:, 0]
        target_t = gt_joints[:, 0]
        delta_t = target_t - initial_t
        delta_r = np.einsum("tij,tkj->tik", target_r, initial_r)
        valid = (
            np.asarray(glob["hand_valid"], dtype=bool)[start:end]
            & np.asarray(glob["gt_valid"], dtype=bool)[start:end]
            & np.asarray(glob["supervision_valid"], dtype=bool)[start:end]
            & np.isfinite(delta_t).all(axis=-1)
            & np.isfinite(delta_r).all(axis=(1, 2))
        )

        # Invalid supervision must be made finite before masking. Multiplying
        # NaN by a zero mask still produces NaN in PyTorch.
        initial_t = np.nan_to_num(
            initial_t, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        target_t = np.nan_to_num(
            target_t, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        delta_t = np.nan_to_num(
            delta_t, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        initial_r = np.nan_to_num(
            initial_r, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        target_r = np.nan_to_num(
            target_r, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        delta_r = np.nan_to_num(
            delta_r, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        relative_joints = pred_joints[:, KEY_JOINTS] - pred_joints[:, 0, None]
        local_hand = (relative_joints / 0.1).reshape(len(relative_joints), -1)
        local_hand = np.nan_to_num(
            local_hand, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        pi3x_path = self.pi3x_root / stream_id / "pi3x_geometry_features_compact.npz"
        pi3x = load_npz(str(pi3x_path.resolve()))
        frame_indices = np.asarray(pi3x["frame_indices"], dtype=np.int64)
        expected = np.arange(start, end, dtype=np.int64)
        positions = np.searchsorted(frame_indices, expected)
        if np.any(positions >= len(frame_indices)) or not np.array_equal(frame_indices[positions], expected):
            raise ValueError(f"{pi3x_path} does not cover [{start}, {end})")
        grid = np.asarray(pi3x["geometry_feature_grid_hw"], dtype=np.float32)

        def tokens(prefix: str):
            features = np.asarray(
                pi3x[f"{prefix}_features"][positions], dtype=np.float32
            )
            points = np.asarray(pi3x[f"{prefix}_points"][positions], dtype=np.float32)
            coverage = np.asarray(pi3x[f"{prefix}_coverage"][positions], dtype=np.float32)
            confidence = np.asarray(pi3x[f"{prefix}_confidence"][positions], dtype=np.float32)
            indices = np.asarray(pi3x[f"{prefix}_indices"][positions], dtype=np.float32)
            indices[..., 0] /= max(float(grid[0] - 1), 1.0)
            indices[..., 1] /= max(float(grid[1] - 1), 1.0)
            metadata = np.concatenate((points, indices, coverage[..., None], confidence[..., None]), axis=-1)
            valid_tokens = np.asarray(pi3x[f"{prefix}_valid"][positions], dtype=bool)
            return (
                np.nan_to_num(
                    features, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32),
                np.nan_to_num(
                    metadata, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32),
                valid_tokens,
            )

        hand_f, hand_m, hand_v = tokens("hand")
        object_f, object_m, object_v = tokens("object")
        context_f, context_m, context_v = tokens("context")
        key_f = np.concatenate((object_f, context_f), axis=1)
        key_m = np.concatenate((object_m, context_m), axis=1)
        key_v = np.concatenate((object_v, context_v), axis=1)
        key_types = np.concatenate((np.zeros_like(object_v, dtype=np.int64), np.ones_like(context_v, dtype=np.int64)), axis=1)
        return {
            "local_hand_features": torch.from_numpy(local_hand),
            "hand_token_features": torch.from_numpy(hand_f),
            "hand_token_metadata": torch.from_numpy(hand_m),
            "hand_token_valid": torch.from_numpy(hand_v),
            "key_token_features": torch.from_numpy(key_f),
            "key_token_metadata": torch.from_numpy(key_m),
            "key_token_valid": torch.from_numpy(key_v),
            "key_token_types": torch.from_numpy(key_types),
            "initial_t": torch.from_numpy(initial_t),
            "target_t": torch.from_numpy(target_t),
            "delta_t": torch.from_numpy(delta_t),
            "initial_r": torch.from_numpy(initial_r),
            "target_r": torch.from_numpy(target_r),
            "delta_r": torch.from_numpy(delta_r),
            "valid": torch.from_numpy(valid),
        }


class V9Model(nn.Module):
    def __init__(self, local_dim: int, feature_dim: int, metadata_dim: int, args: argparse.Namespace):
        super().__init__()
        self.max_correction = args.max_correction_mm / 1000.0
        self.relation = RelativeCrossAttention(feature_dim, metadata_dim, args.pi3x_relation_dim, args.pi3x_heads, args.dropout)
        self.local = nn.Sequential(nn.LayerNorm(local_dim), nn.Linear(local_dim, args.hidden_dim), nn.GELU(), nn.Dropout(args.dropout))
        self.frame = nn.Sequential(nn.Linear(args.hidden_dim + args.pi3x_relation_dim, args.hidden_dim), nn.LayerNorm(args.hidden_dim), nn.GELU(), nn.Dropout(args.dropout))
        self.temporal = nn.GRU(args.hidden_dim, args.hidden_dim // 2, args.layers, batch_first=True, bidirectional=True, dropout=args.dropout if args.layers > 1 else 0.0)
        self.t_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.GELU(), nn.Linear(args.hidden_dim, 3))
        self.r_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.GELU(), nn.Linear(args.hidden_dim, 6))
        nn.init.zeros_(self.t_head[-1].weight); nn.init.zeros_(self.t_head[-1].bias)
        nn.init.zeros_(self.r_head[-1].weight); self.r_head[-1].bias.data.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, batch):
        relation = self.relation(batch["hand_token_features"], batch["hand_token_metadata"], batch["hand_token_valid"], batch["key_token_features"], batch["key_token_metadata"], batch["key_token_valid"], batch["key_token_types"])
        frame = self.frame(torch.cat((self.local(batch["local_hand_features"]), relation), dim=-1))
        temporal, _ = self.temporal(frame)
        delta_t = torch.tanh(self.t_head(temporal)) * self.max_correction
        delta_r = rotation_6d_to_matrix(self.r_head(temporal))
        return delta_t, delta_r


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {k: 0.0 for k in ("total", "translation", "rotation", "velocity", "acceleration", "residual")}
    batches = 0; metrics = {k: [] for k in ("initial_t", "predicted_t", "initial_r", "predicted_r")}
    iterator = tqdm(loader, desc="train" if training else "val")
    for batch in iterator:
        batch = {k: v.to(device) for k, v in batch.items()}
        bad = [
            key for key, value in batch.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if bad:
            raise RuntimeError(f"non-finite batch inputs: {bad}")
        with torch.set_grad_enabled(training):
            delta_t, delta_r = model(batch)
            pred_t = batch["initial_t"] + delta_t
            pred_r = delta_r @ batch["initial_r"]
            valid = batch["valid"]
            translation = masked_mean(smooth_l1(pred_t - batch["target_t"], args.smooth_l1_beta_mm / 1000.0), valid[..., None])
            rotation = masked_mean(rotation_angle(pred_r, batch["target_r"]), valid)
            velocity = temporal_loss(pred_t, batch["target_t"], valid, 1, args.smooth_l1_beta_mm / 1000.0)
            acceleration = temporal_loss(pred_t, batch["target_t"], valid, 2, args.smooth_l1_beta_mm / 1000.0)
            residual = masked_mean(smooth_l1(delta_t, 0.02), valid[..., None])
            total = args.w_translation * translation + args.w_rotation * rotation + args.w_velocity * velocity + args.w_acceleration * acceleration + args.w_residual * residual
            if not torch.isfinite(total):
                raise RuntimeError(
                    "non-finite loss: "
                    f"translation={translation.item()} "
                    f"rotation={rotation.item()} "
                    f"velocity={velocity.item()} "
                    f"acceleration={acceleration.item()} "
                    f"residual={residual.item()}"
                )
            if training:
                optimizer.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        for name, value in (("total", total), ("translation", translation), ("rotation", rotation), ("velocity", velocity), ("acceleration", acceleration), ("residual", residual)):
            sums[name] += float(value.detach())
        initial_t_error = torch.linalg.norm(batch["initial_t"] - batch["target_t"], dim=-1)
        predicted_t_error = torch.linalg.norm(pred_t - batch["target_t"], dim=-1)
        initial_r_error = rotation_angle(batch["initial_r"], batch["target_r"])
        predicted_r_error = rotation_angle(pred_r, batch["target_r"])
        for name, value in (("initial_t", initial_t_error), ("predicted_t", predicted_t_error), ("initial_r", initial_r_error), ("predicted_r", predicted_r_error)):
            metrics[name].append(value.detach().cpu().numpy()[batch["valid"].cpu().numpy()])
        batches += 1; iterator.set_postfix(loss=f"{sums['total']/batches:.5f}")

    def dist(values, scale):
        values = np.concatenate(values) * scale if values else np.empty(0)
        return {"count": int(values.size), "median_mm": float(np.median(values)) if values.size else None, "p90_mm": float(np.percentile(values, 90)) if values.size else None, "max_mm": float(np.max(values)) if values.size else None}
    return {**{k: v / max(batches, 1) for k, v in sums.items()}, "initial_translation": dist(metrics["initial_t"], 1000.0), "predicted_translation": dist(metrics["predicted_t"], 1000.0), "initial_rotation": dist(metrics["initial_r"], 180.0 / math.pi), "predicted_rotation": dist(metrics["predicted_r"], 180.0 / math.pi)}


def main():
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train = CameraWindowDataset(Path(args.train_windows), Path(args.global_train_root), Path(args.pi3x_train_root))
    val = CameraWindowDataset(Path(args.val_windows), Path(args.global_val_root), Path(args.pi3x_val_root))
    sample = train[0]
    model = V9Model(sample["local_hand_features"].shape[-1], sample["hand_token_features"].shape[-1], sample["hand_token_metadata"].shape[-1], args)
    device = torch.device(args.device); model.to(device)
    if args.data_parallel: model = nn.DataParallel(model)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, **kwargs); val_loader = DataLoader(val, shuffle=False, **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True); history=[]; best=float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====", flush=True)
        train_m = run_epoch(model, train_loader, device, args, optimizer); val_m = run_epoch(model, val_loader, device, args); scheduler.step()
        row={"epoch":epoch,"lr":optimizer.param_groups[0]["lr"],"train":train_m,"val":val_m}; history.append(row); print(json.dumps(row), flush=True)
        ckpt={"epoch":epoch,"model_version":MODEL_VERSION,"model_state":model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),"args":vars(args),"local_hand_dim":sample["local_hand_features"].shape[-1],"pi3x_feature_dim":sample["hand_token_features"].shape[-1],"pi3x_metadata_dim":sample["hand_token_metadata"].shape[-1],"val_total":val_m["total"]}
        torch.save(ckpt, out / "last.pt")
        if val_m["total"] < best: best=val_m["total"]; torch.save(ckpt, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
