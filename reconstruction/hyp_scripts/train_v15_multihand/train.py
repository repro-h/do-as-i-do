#!/usr/bin/env python3
"""Train V15 side-free multi-hand absolute camera translation."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from model import MultiHandPi3XTrajectoryModel  # noqa: E402


MODEL_VERSION = "v15_side_free_multihand_pi3x_translation_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--global-noise-px", type=float, default=4.0)
    parser.add_argument("--temporal-noise-px", type=float, default=0.5)
    parser.add_argument("--joint-noise-px", type=float, default=2.0)
    parser.add_argument("--outlier-probability", type=float, default=0.03)
    parser.add_argument("--query-dropout", type=float, default=0.1)
    parser.add_argument("--w-depth", type=float, default=0.5)
    parser.add_argument("--w-relative", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def masked_mean(value, valid):
    weight = valid.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1.0)


def smooth_l1(value, beta):
    absolute = value.abs()
    return torch.where(absolute < beta, 0.5 * absolute.square() / beta, absolute - 0.5 * beta)


def temporal_loss(prediction, target, valid, order, beta):
    pred, truth, mask = prediction, target, valid
    for _ in range(order):
        pred = pred[:, 1:] - pred[:, :-1]
        truth = truth[:, 1:] - truth[:, :-1]
        mask = mask[:, 1:] & mask[:, :-1]
    return masked_mean(smooth_l1(pred - truth, beta), mask)


def distribution(values):
    if not values:
        return {"count": 0, "median_mm": None, "p90_mm": None, "max_mm": None}
    array = np.concatenate(values) * 1000.0
    return {
        "count": int(array.size),
        "median_mm": float(np.median(array)),
        "p90_mm": float(np.percentile(array, 90)),
        "max_mm": float(np.max(array)),
    }


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "absolute", "depth", "relative", "velocity", "acceleration")}
    translation_errors, depth_errors = [], []
    batches = evaluated = 0
    for raw in tqdm(loader, desc="train" if training else "val"):
        batch = move(raw, device)
        valid = batch["target_valid"] & batch["hand_slot_valid"]
        target = batch["target_t"]
        with torch.set_grad_enabled(training):
            prediction = model(batch)
            beta = args.smooth_l1_beta_mm / 1000.0
            absolute = masked_mean(smooth_l1(prediction - target, beta), valid)
            depth = masked_mean(smooth_l1(prediction[..., 2] - target[..., 2], beta), valid)
            weight = valid.to(target.dtype)[..., None]
            denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            pred_center = (prediction * weight).sum(dim=1, keepdim=True) / denominator
            target_center = (target * weight).sum(dim=1, keepdim=True) / denominator
            relative = masked_mean(
                smooth_l1((prediction - pred_center) - (target - target_center), beta), valid
            )
            velocity = temporal_loss(prediction, target, valid, 1, beta)
            acceleration = temporal_loss(prediction, target, valid, 2, beta)
            total = (
                absolute + args.w_depth * depth + args.w_relative * relative
                + args.w_velocity * velocity + args.w_acceleration * acceleration
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key, value in (
            ("total", total), ("absolute", absolute), ("depth", depth),
            ("relative", relative), ("velocity", velocity), ("acceleration", acceleration),
        ):
            totals[key] += float(value.detach())
        mask = valid.detach().cpu().numpy().astype(bool)
        error = (prediction - target).detach().cpu().numpy()
        translation_errors.append(np.linalg.norm(error, axis=-1)[mask])
        depth_errors.append(np.abs(error[..., 2])[mask])
        evaluated += int(mask.sum())
        batches += 1
    return {
        **{key: value / max(batches, 1) for key, value in totals.items()},
        "translation_error": distribution(translation_errors),
        "depth_error": distribution(depth_errors),
        "evaluated_hands": evaluated,
    }


def make_dataset(args, split, training):
    noise = QueryNoise(
        global_sigma_px=args.global_noise_px,
        temporal_sigma_px=args.temporal_noise_px,
        joint_sigma_px=args.joint_noise_px,
        outlier_probability=args.outlier_probability,
        dropout_probability=args.query_dropout,
    )
    return DexYCBMultiHandWindowDataset(
        getattr(args, f"{split}_windows"),
        getattr(args, f"pi3x_{split}_root"),
        max_hands=args.max_hands,
        training=training,
        noise=noise,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = make_dataset(args, "train", True)
    val_data = make_dataset(args, "val", False)
    sample = train_data[0]
    audit = {
        "model_version": MODEL_VERSION,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "point_feature_shape": list(sample["point_features"].shape),
        "joint_query_shape": list(sample["joint_uv"].shape),
        "target_shape": list(sample["target_t"].shape),
        "valid_hand_slots": int(sample["hand_slot_valid"].sum()),
        "side_input": False,
        "coordinate_frame": "original_camera",
        "query_source": "dexycb_gt_2d_with_train_only_noise",
    }
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        return

    device = torch.device(args.device)
    model = MultiHandPi3XTrajectoryModel(
        point_dim=sample["point_features"].shape[-1],
        metric_dim=sample["metric_window_features"].shape[-1],
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        temporal_layers=args.temporal_layers,
        dropout=args.dropout,
    ).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    if args.eval_only:
        print(json.dumps(run_epoch(model, val_loader, device, args), indent=2))
        return

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch} =====")
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row))
        payload = {
            "epoch": epoch,
            "model_version": MODEL_VERSION,
            "model": (model.module if hasattr(model, "module") else model).state_dict(),
            "args": vars(args),
            "val": val_metrics,
        }
        torch.save(payload, out_dir / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(payload, out_dir / "best.pt")
        with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()

