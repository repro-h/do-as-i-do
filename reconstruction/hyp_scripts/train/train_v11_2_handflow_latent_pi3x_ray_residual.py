#!/usr/bin/env python3
"""Refine HandFlow ray depth from its latent and local Pi3X geometry."""

from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_v10_pi3x_hand_neighborhood_depth import (
    HandNeighborhoodDataset,
    disable_mha_fastpath,
    run_epoch,
)
from train_v11_1_pi3x_metric_point_ray_residual import (
    Pi3XMetricPointRayResidual,
)


MODEL_VERSION = "v11_2_handflow_latent_pi3x_ray_residual_v1"
FEATURE_MODES = (
    "normal",
    "latent_zero",
    "pi3x_zero",
    "all_zero",
    "latent_time_reverse",
    "pi3x_time_reverse",
)


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
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=0.2)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-small-anchor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--feature-mode", choices=FEATURE_MODES, default="normal"
    )
    return parser.parse_args()


@lru_cache(maxsize=128)
def load_handflow_latent(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "handflow_translation_latent" not in data.files:
            raise KeyError(f"Missing handflow_translation_latent: {path}")
        latent = np.asarray(
            data["handflow_translation_latent"], dtype=np.float32
        )
    if latent.ndim != 2 or latent.shape[1] != 512:
        raise ValueError(f"Unexpected HandFlow latent shape {latent.shape}: {path}")
    if not np.isfinite(latent).all():
        raise ValueError(f"Non-finite HandFlow latent: {path}")
    return latent


class HandFlowLatentNeighborhoodDataset(HandNeighborhoodDataset):
    def __init__(self, *args, handflow_root: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.handflow_root = handflow_root

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        output = super().__getitem__(index)
        row = self.rows[index]
        stream_id = str(row["stream_id"])
        start, end = int(row["start"]), int(row["end"])
        path = self.handflow_root / stream_id / "handflow_camera_result.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        latent = load_handflow_latent(str(path.resolve()))
        if end > len(latent):
            raise ValueError(
                f"Window {start}:{end} exceeds latent length {len(latent)}: {path}"
            )
        output["handflow_translation_latent"] = torch.from_numpy(
            latent[start:end].copy()
        )
        return output


class HandFlowLatentPi3XRayResidual(Pi3XMetricPointRayResidual):
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
            decoder_dim, metric_dim, metadata_dim, num_joints, args
        )
        self.latent_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, args.token_dim),
            nn.GELU(),
            nn.Linear(args.token_dim, args.token_dim),
        )
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(args.token_dim * 4),
            nn.Linear(args.token_dim * 4, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

        valid = batch["neighborhood_valid"]
        score = self.local_score(token).squeeze(-1).masked_fill(~valid, -1e4)
        weight = torch.softmax(score, dim=3) * valid.to(score.dtype)
        weight = weight / weight.sum(dim=3, keepdim=True).clamp_min(1e-6)
        joint = (token * weight[..., None]).sum(dim=3)

        batch_size, time, joints, dim = joint.shape
        joint_valid = valid.any(dim=3).reshape(batch_size * time, joints)
        safe_valid = joint_valid.clone()
        safe_valid[~safe_valid.any(dim=1), 0] = True
        encoded = self.joint_encoder(
            joint.reshape(batch_size * time, joints, dim),
            src_key_padding_mask=~safe_valid,
        ).reshape(batch_size, time, joints, dim)
        joint_valid = joint_valid.reshape(batch_size, time, joints)
        joint_weight = joint_valid.to(encoded.dtype)
        pooled = (encoded * joint_weight[..., None]).sum(dim=2)
        pooled = pooled / joint_weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        wrist = encoded[:, :, 0]
        metric_token = self.metric_encoder(metric)
        latent_token = self.latent_encoder(latent)
        temporal, _ = self.temporal(self.frame_encoder(torch.cat(
            (wrist, pooled, metric_token, latent_token), dim=-1
        )))
        return torch.tanh(self.head(temporal).squeeze(-1)) * self.max_correction


def make_dataset(
    windows: str,
    global_root: str,
    dense_root: str,
    handflow_root: str,
    args: argparse.Namespace,
) -> HandFlowLatentNeighborhoodDataset:
    return HandFlowLatentNeighborhoodDataset(
        Path(windows), Path(global_root), Path(dense_root),
        args.neighborhood_size, args.min_confidence,
        handflow_root=Path(handflow_root),
    )


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
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "decoder_feature_shape": list(sample["neighborhood_features"].shape),
        "point_shape": list(sample["neighborhood_points"].shape),
        "metric_feature_shape": list(sample["metric_window_features"].shape),
        "handflow_latent_shape": list(
            sample["handflow_translation_latent"].shape
        ),
        "valid_tokens": int(sample["neighborhood_valid"].sum()),
        "valid_frames": int(sample["valid"].sum()),
        "explicit_hand_depth_input": False,
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    model = HandFlowLatentPi3XRayResidual(
        int(sample["neighborhood_features"].shape[-1]),
        int(sample["metric_window_features"].shape[-1]),
        int(sample["neighborhood_metadata"].shape[-1]),
        int(sample["handflow_translation_latent"].shape[-1]),
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
        degraded = val["degraded_fraction"]
        if degraded < best_degraded:
            best_degraded = degraded
            torch.save(checkpoint, out_dir / "best_degraded.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
