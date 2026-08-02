#!/usr/bin/env python3
"""Train a compact translation-equivariant hand/object relative refiner."""

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


PALM = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
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


@lru_cache(maxsize=256)
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
    def __init__(self, windows: Path, global_root: Path, args: argparse.Namespace):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.global_root = global_root
        self.max_correction = args.max_correction_mm / 1000.0
        self.max_object_error = args.max_object_center_error_mm / 1000.0
        self.max_projection_shift = args.max_target_projection_shift_px

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        stream_id = row["stream_id"]
        start, end = int(row["start"]), int(row["end"])
        rigid = load_npz(str(Path(row["supervision_npz"]).resolve()))
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
        return {
            "features": torch.from_numpy(features),
            "relative_initial": torch.from_numpy(relative_initial),
            "relative_target": torch.from_numpy(relative_target),
            "valid": torch.from_numpy(loss_valid),
            "target_magnitude": torch.from_numpy(
                np.linalg.norm(target_delta, axis=-1).astype(np.float32)
            ),
        }


class RelativeTranslationRefiner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
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

    def forward(self, features: torch.Tensor, max_correction: float) -> torch.Tensor:
        encoded = self.frame_encoder(features)
        temporal, _ = self.temporal(encoded)
        return torch.tanh(self.head(temporal)) * max_correction


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
    correction = model(batch["features"], args.max_correction_mm / 1000.0)
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


def run_epoch(model, loader, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums, batches = {}, 0
    initial_errors, corrected_errors = [], []
    bin_edges = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, float("inf")))
    binned_initial = {edge: [] for edge in bin_edges}
    binned_corrected = {edge: [] for edge in bin_edges}
    valid_frames = 0
    for batch in loader:
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = RelativeWindowDataset(
        Path(args.train_windows).expanduser().resolve(),
        Path(args.global_train_root).expanduser().resolve(),
        args,
    )
    val_data = RelativeWindowDataset(
        Path(args.val_windows).expanduser().resolve(),
        Path(args.global_val_root).expanduser().resolve(),
        args,
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
    model = RelativeTranslationRefiner(
        input_dim, args.hidden_dim, args.layers, args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, args)
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
            "epoch": epoch,
            "val_total": val_metrics["total"],
            "model_version": "relative_translation_mlp_bigru_weighted_v2",
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
