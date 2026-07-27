#!/usr/bin/env python3
"""Train a temporal hand-only SE(3) residual refiner."""

from __future__ import annotations

import argparse
import copy
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Load model weights from this checkpoint, but start a fresh "
            "optimizer and learning-rate schedule."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-translation-mm", type=float, default=100.0)
    parser.add_argument("--max-rotation-deg", type=float, default=30.0)
    parser.add_argument("--w-center", type=float, default=1.0)
    parser.add_argument("--w-mesh", type=float, default=1.0)
    parser.add_argument("--w-projection", type=float, default=0.2)
    parser.add_argument("--w-relative", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.15)
    parser.add_argument("--w-acceleration", type=float, default=0.25)
    parser.add_argument("--w-residual", type=float, default=0.02)
    parser.add_argument(
        "--w-init-anchor",
        type=float,
        default=0.0,
        help=(
            "Keep fine-tuned translation/rotation residuals near the "
            "--init-checkpoint predictions."
        ),
    )
    parser.add_argument("--w-rotation-smooth", type=float, default=0.05)
    parser.add_argument("--w-penetration", type=float, default=0.5)
    parser.add_argument("--w-contact", type=float, default=0.15)
    parser.add_argument("--penetration-start-epoch", type=int, default=3)
    parser.add_argument("--contact-start-epoch", type=int, default=6)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--penetration-clip-mm", type=float, default=40.0)
    parser.add_argument("--contact-target-mm", type=float, default=1.5)
    parser.add_argument("--contact-topk", type=int, default=24)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--max-target-center-mm", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="do-as-i-do-hand-rigid")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=128)
def load_npz(path_text: str) -> dict[str, np.ndarray]:
    with np.load(path_text, allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :3, :2].swapaxes(-1, -2).reshape(*matrix.shape[:-2], 6)


class HandRigidWindowDataset(Dataset):
    def __init__(self, path: Path, max_target_center_m: float):
        self.rows = load_jsonl(path)
        if not self.rows:
            raise RuntimeError(f"No windows in {path}")
        self.max_target_center_m = max_target_center_m

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        raw = load_npz(row["supervision_npz"])
        pred_vertices = np.asarray(
            raw["pred_hand_vertices"][start:end], dtype=np.float32
        )
        pred_center = np.asarray(raw["pred_hand_center"][start:end], dtype=np.float32)
        gt_vertices = np.asarray(raw["gt_hand_vertices"][start:end], dtype=np.float32)
        gt_center = np.asarray(raw["gt_hand_center"][start:end], dtype=np.float32)
        object_pose = np.asarray(raw["object_pose"][start:end], dtype=np.float32)
        valid = np.asarray(raw["supervision_valid"][start:end]).astype(bool)
        contact = np.asarray(
            raw["gt_contact_candidates"][start:end]
        ).astype(bool)
        center_error = np.linalg.norm(gt_center - pred_center, axis=-1)
        valid &= np.isfinite(center_error) & (center_error <= self.max_target_center_m)

        pred_vertices = np.nan_to_num(pred_vertices)
        pred_center = np.nan_to_num(pred_center)
        gt_vertices = np.nan_to_num(gt_vertices)
        gt_center = np.nan_to_num(gt_center)
        object_pose = np.nan_to_num(object_pose)
        object_center = object_pose[:, :3, 3]
        local = pred_vertices - pred_center[:, None]
        key_indices = np.linspace(0, local.shape[1] - 1, 16, dtype=np.int64)
        hand_velocity = np.zeros_like(pred_center)
        object_velocity = np.zeros_like(object_center)
        hand_velocity[1:] = pred_center[1:] - pred_center[:-1]
        object_velocity[1:] = object_center[1:] - object_center[:-1]
        features = np.concatenate(
            [
                pred_center,
                object_center,
                pred_center - object_center,
                hand_velocity,
                object_velocity,
                rotation_6d(object_pose[:, :3, :3]),
                local[:, key_indices].reshape(len(local), -1),
                valid[:, None],
            ],
            axis=-1,
        ).astype(np.float32)
        return {
            "features": torch.from_numpy(features),
            "pred_vertices": torch.from_numpy(pred_vertices),
            "pred_center": torch.from_numpy(pred_center),
            "gt_vertices": torch.from_numpy(gt_vertices),
            "gt_center": torch.from_numpy(gt_center),
            "object_pose": torch.from_numpy(object_pose),
            "object_anchors": torch.from_numpy(
                np.asarray(raw["object_anchors_local"], dtype=np.float32)
            ),
            "object_normals": torch.from_numpy(
                np.asarray(raw["object_normals_local"], dtype=np.float32)
            ),
            "contact": torch.from_numpy(contact),
            "valid": torch.from_numpy(valid),
            "intrinsics": torch.from_numpy(
                np.asarray(raw["intrinsics"], dtype=np.float32)
            ),
        }


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(vector, dim=-1, keepdim=True)
    axis = vector / angle.clamp_min(1e-8)
    matrix = skew(axis)
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    identity = identity.reshape(*([1] * (vector.ndim - 1)), 3, 3)
    sine = torch.sin(angle)[..., None]
    cosine = torch.cos(angle)[..., None]
    result = identity + sine * matrix + (1.0 - cosine) * (matrix @ matrix)
    small = angle[..., 0] < 1e-6
    return torch.where(small[..., None, None], identity + skew(vector), result)


class HandRigidTemporalRefiner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.position = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features, max_translation, max_rotation):
        tokens = self.input(features) + self.position[:, : features.shape[1]]
        output = self.output(self.encoder(tokens))
        translation = torch.tanh(output[..., :3]) * max_translation
        rotation = torch.tanh(output[..., 3:]) * max_rotation
        return translation, rotation


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(value.dtype).expand_as(value)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def smooth_l1_points(value, target, mask, beta):
    loss = F.smooth_l1_loss(value, target, reduction="none", beta=beta)
    return masked_mean(loss, mask)


def temporal_loss(value, target, mask, order, beta):
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
        target = target[:, 1:] - target[:, :-1]
        mask = mask[:, 1:] & mask[:, :-1]
    return smooth_l1_points(value, target, mask, beta)


def temporal_zero_loss(value, mask, order):
    for _ in range(order):
        value = value[:, 1:] - value[:, :-1]
        mask = mask[:, 1:] & mask[:, :-1]
    return masked_mean(value.square(), mask)


def project(points, intrinsics):
    z = points[..., 2].clamp_min(1e-4)
    u = intrinsics[:, None, None, 0, 0] * points[..., 0] / z
    u = u + intrinsics[:, None, None, 0, 2]
    v = intrinsics[:, None, None, 1, 1] * points[..., 1] / z
    v = v + intrinsics[:, None, None, 1, 2]
    return torch.stack([u, v], dim=-1)


def transform_object_surface(batch):
    rotation = batch["object_pose"][..., :3, :3]
    translation = batch["object_pose"][..., :3, 3]
    anchors = torch.einsum(
        "btij,baj->btai", rotation, batch["object_anchors"]
    ) + translation[:, :, None]
    normals = torch.einsum(
        "btij,baj->btai", rotation, batch["object_normals"]
    )
    return anchors, F.normalize(normals, dim=-1, eps=1e-8)


def contact_and_penetration(corrected, anchors, normals, candidate, valid, args):
    batch_size, frames, hand_count, _ = corrected.shape
    object_count = anchors.shape[2]
    distance = torch.cdist(
        corrected.reshape(batch_size * frames, hand_count, 3),
        anchors.reshape(batch_size * frames, object_count, 3),
    ).reshape(batch_size, frames, hand_count, object_count)
    nearest_distance, nearest_index = distance.min(dim=-1)
    nearest_anchor = torch.gather(
        anchors,
        2,
        nearest_index[..., None].expand(-1, -1, -1, 3),
    )
    nearest_normal = torch.gather(
        normals,
        2,
        nearest_index[..., None].expand(-1, -1, -1, 3),
    )
    signed_penetration = (
        (nearest_anchor - corrected) * nearest_normal
    ).sum(dim=-1)
    penetration = torch.relu(
        signed_penetration - args.penetration_tolerance_mm / 1000.0
    ).clamp_max(args.penetration_clip_mm / 1000.0)
    penetration_loss = masked_mean(penetration.square(), valid[:, :, None])
    penetrating = signed_penetration.detach() > 0.0

    safe_candidate = candidate & ~penetrating & valid[:, :, None]
    contact_values = []
    for batch_index in range(batch_size):
        for frame_index in range(frames):
            values = nearest_distance[batch_index, frame_index][
                safe_candidate[batch_index, frame_index]
            ]
            if not values.numel():
                continue
            count = min(args.contact_topk, int(values.numel()))
            selected = torch.topk(values, k=count, largest=False).values
            contact_values.append(
                (selected - args.contact_target_mm / 1000.0).square().mean()
            )
    contact_loss = (
        torch.stack(contact_values).mean()
        if contact_values
        else corrected.sum() * 0.0
    )
    diagnostics = {
        "penetration_ratio": masked_mean(
            penetrating.to(corrected.dtype), valid[:, :, None]
        ),
        "contact_active": corrected.new_tensor(float(len(contact_values))),
    }
    return contact_loss, penetration_loss, diagnostics


def compute_loss(model, batch, args, epoch, reference_model=None):
    batch = {key: value.to(args.device) for key, value in batch.items()}
    translation, rotation_vector = model(
        batch["features"],
        args.max_translation_mm / 1000.0,
        np.deg2rad(args.max_rotation_deg),
    )
    rotation = axis_angle_to_matrix(rotation_vector)
    local = batch["pred_vertices"] - batch["pred_center"][:, :, None]
    corrected_vertices = torch.einsum("btij,bthj->bthi", rotation, local)
    corrected_vertices = (
        corrected_vertices
        + batch["pred_center"][:, :, None]
        + translation[:, :, None]
    )
    corrected_center = batch["pred_center"] + translation
    object_center = batch["object_pose"][..., :3, 3]
    valid = batch["valid"]
    beta = args.smooth_l1_beta_mm / 1000.0
    projection_mask = (
        valid[:, :, None]
        & (corrected_vertices[..., 2] > 1e-4)
        & (batch["gt_vertices"][..., 2] > 1e-4)
    )
    losses = {
        "center": smooth_l1_points(
            corrected_center, batch["gt_center"], valid, beta
        ),
        "mesh": smooth_l1_points(
            corrected_vertices, batch["gt_vertices"], valid[:, :, None], beta
        ),
        "projection": smooth_l1_points(
            project(corrected_vertices, batch["intrinsics"]) / 100.0,
            project(batch["gt_vertices"], batch["intrinsics"]) / 100.0,
            projection_mask,
            beta,
        ),
        "relative": smooth_l1_points(
            corrected_center - object_center,
            batch["gt_center"] - object_center,
            valid,
            beta,
        ),
        "velocity": temporal_loss(
            corrected_center, batch["gt_center"], valid, 1, beta
        ),
        "acceleration": temporal_loss(
            corrected_center, batch["gt_center"], valid, 2, beta
        ),
        "residual": masked_mean(
            translation.square(), valid
        ) + 0.1 * masked_mean(rotation_vector.square(), valid),
        "rotation_smooth": temporal_zero_loss(rotation_vector, valid, 2),
    }
    if reference_model is not None:
        with torch.no_grad():
            reference_translation, reference_rotation = reference_model(
                batch["features"],
                args.max_translation_mm / 1000.0,
                np.deg2rad(args.max_rotation_deg),
            )
        losses["init_anchor"] = masked_mean(
            (translation - reference_translation).square(), valid
        ) + 0.1 * masked_mean(
            (rotation_vector - reference_rotation).square(), valid
        )
    else:
        losses["init_anchor"] = translation.sum() * 0.0
    anchors, normals = transform_object_surface(batch)
    contact, penetration, diagnostics = contact_and_penetration(
        corrected_vertices,
        anchors,
        normals,
        batch["contact"],
        valid,
        args,
    )
    losses["contact"] = contact
    losses["penetration"] = penetration
    penetration_scale = 0.0
    if epoch >= args.penetration_start_epoch:
        penetration_scale = min(
            1.0, (epoch - args.penetration_start_epoch + 1) / 2.0
        )
    contact_scale = 0.0
    if epoch >= args.contact_start_epoch:
        contact_scale = min(1.0, (epoch - args.contact_start_epoch + 1) / 3.0)
    total = (
        args.w_center * losses["center"]
        + args.w_mesh * losses["mesh"]
        + args.w_projection * losses["projection"]
        + args.w_relative * losses["relative"]
        + args.w_velocity * losses["velocity"]
        + args.w_acceleration * losses["acceleration"]
        + args.w_residual * losses["residual"]
        + args.w_init_anchor * losses["init_anchor"]
        + args.w_rotation_smooth * losses["rotation_smooth"]
        + penetration_scale * args.w_penetration * losses["penetration"]
        + contact_scale * args.w_contact * losses["contact"]
    )
    return total, losses, diagnostics, corrected_center


def run_epoch(
    model,
    loader,
    args,
    epoch,
    optimizer=None,
    split="train",
    reference_model=None,
):
    training = optimizer is not None
    model.train(training)
    sums, count = {}, 0
    initial_errors, corrected_errors = [], []
    progress = tqdm(
        loader,
        desc=f"epoch {epoch:03d} {split}",
        dynamic_ncols=True,
        leave=True,
    )
    for batch in progress:
        with torch.set_grad_enabled(training):
            total, losses, diagnostics, corrected = compute_loss(
                model, batch, args, epoch, reference_model
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                valid = batch["valid"].to(args.device)
                initial = batch["pred_center"].to(args.device)
                target = batch["gt_center"].to(args.device)
                initial_errors.append(
                    torch.linalg.norm(initial[valid] - target[valid], dim=-1).cpu()
                )
                corrected_errors.append(
                    torch.linalg.norm(corrected[valid] - target[valid], dim=-1).cpu()
                )
        values = {"total": total, **losses, **diagnostics}
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach())
        count += 1
        progress.set_postfix(
            total=f"{sums['total'] / count:.5f}",
            center=f"{sums['center'] / count:.5f}",
            mesh=f"{sums['mesh'] / count:.5f}",
            pen=f"{sums['penetration'] / count:.5f}",
            contact=f"{sums['contact'] / count:.5f}",
        )
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    if not training and initial_errors:
        initial = torch.cat(initial_errors)
        corrected = torch.cat(corrected_errors)
        metrics["center_error"] = {
            "initial_median_mm": float(torch.quantile(initial, 0.5) * 1000.0),
            "corrected_median_mm": float(torch.quantile(corrected, 0.5) * 1000.0),
            "initial_p90_mm": float(torch.quantile(initial, 0.9) * 1000.0),
            "corrected_p90_mm": float(torch.quantile(corrected, 0.9) * 1000.0),
        }
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = HandRigidWindowDataset(
        Path(args.train_windows).expanduser().resolve(),
        args.max_target_center_mm / 1000.0,
    )
    val_data = HandRigidWindowDataset(
        Path(args.val_windows).expanduser().resolve(),
        args.max_target_center_mm / 1000.0,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    input_dim = int(train_data[0]["features"].shape[-1])
    model = HandRigidTemporalRefiner(
        input_dim, args.hidden_dim, args.layers, args.heads, args.dropout
    ).to(args.device)
    initialized_from = None
    reference_model = None
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        initial = torch.load(init_path, map_location="cpu")
        initial_input_dim = int(initial.get("input_dim", input_dim))
        if initial_input_dim != input_dim:
            raise ValueError(
                f"Checkpoint input_dim={initial_input_dim}, "
                f"current input_dim={input_dim}"
            )
        model.load_state_dict(initial["model"], strict=True)
        initialized_from = {
            "checkpoint": str(init_path),
            "epoch": int(initial.get("epoch", -1)),
            "val_total": float(initial.get("val_total", float("nan"))),
        }
        print(
            "Initialized model from "
            f"{initialized_from['checkpoint']} "
            f"(epoch={initialized_from['epoch']}, "
            f"val_total={initialized_from['val_total']})",
            flush=True,
        )
        if args.w_init_anchor > 0.0:
            reference_model = copy.deepcopy(model).eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
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
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "--wandb was requested but wandb is not installed"
            ) from error
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            dir=str(out_dir),
        )
        wandb.watch(model, log="gradients", log_freq=200)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            args,
            epoch,
            optimizer,
            split="train",
            reference_model=reference_model,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                args,
                epoch,
                split="val",
                reference_model=reference_model,
            )
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
            flat_metrics = {"epoch": epoch, "lr": row["lr"]}
            for split_name, metrics in (
                ("train", train_metrics),
                ("val", val_metrics),
            ):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        flat_metrics[f"{split_name}/{key}"] = value
                for key, value in metrics.get("center_error", {}).items():
                    flat_metrics[f"{split_name}/center_error/{key}"] = value
            wandb_run.log(flat_metrics, step=epoch)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "input_dim": input_dim,
            "initialized_from": initialized_from,
            "epoch": epoch,
            "val_total": val_metrics["total"],
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
