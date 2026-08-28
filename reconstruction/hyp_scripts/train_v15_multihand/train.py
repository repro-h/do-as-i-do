#!/usr/bin/env python3
"""Train V15 side-free multi-hand absolute camera translation."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from model import MultiHandPi3XTrajectoryModel  # noqa: E402


MODEL_VERSION = "v15_2_visibility_aware_multihand_pi3x_ray_translation_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--pi3x-train-root", required=True)
    parser.add_argument("--pi3x-val-root", required=True)
    parser.add_argument("--visibility-train-root")
    parser.add_argument("--visibility-val-root")
    parser.add_argument("--track-train-root")
    parser.add_argument("--track-val-root")
    parser.add_argument(
        "--visibility-source", choices=("detector", "mask", "ones"),
        default="detector",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-windows-per-dataset", type=int, default=0,
        help="Per-epoch dataset budget; zero keeps ordinary shuffled sampling",
    )
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--max-window-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--global-noise-px", type=float, default=4.0)
    parser.add_argument("--temporal-noise-px", type=float, default=0.5)
    parser.add_argument("--joint-noise-px", type=float, default=2.0)
    parser.add_argument("--outlier-probability", type=float, default=0.03)
    parser.add_argument("--query-dropout", type=float, default=0.1)
    parser.add_argument("--near-anchor-frames", type=int, default=4)
    parser.add_argument("--max-anchor-frames", type=int, default=8)
    parser.add_argument("--near-missing-weight", type=float, default=0.5)
    parser.add_argument("--far-missing-weight", type=float, default=0.2)
    parser.add_argument("--w-depth", type=float, default=0.5)
    parser.add_argument("--w-relative", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=0.05)
    parser.add_argument("--w-acceleration", type=float, default=0.02)
    parser.add_argument("--w-reprojection", type=float, default=0.1)
    parser.add_argument("--reprojection-beta-px", type=float, default=2.0)
    parser.add_argument(
        "--translation-parameterization",
        choices=("ray_depth_uv", "direct_xyz"),
        default="ray_depth_uv",
    )
    parser.add_argument("--max-image-offset-fraction", type=float, default=0.15)
    parser.add_argument("--smooth-l1-beta-mm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="do-as-i-do-v15-multihand")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def row_dataset(row):
    if row.get("dataset"):
        return str(row["dataset"])
    schema = str(row.get("schema_version", "unknown"))
    return schema.split("_", 1)[0]


class DatasetStreamBalancedSampler(Sampler):
    """Draw an equal per-dataset budget while weighting streams equally."""

    def __init__(self, rows, samples_per_dataset, seed=0):
        self.samples_per_dataset = int(samples_per_dataset)
        if self.samples_per_dataset <= 0:
            raise ValueError("samples_per_dataset must be positive")
        grouped = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            grouped[row_dataset(row)][str(row["stream_id"])].append(index)
        self.grouped = {
            dataset: dict(streams) for dataset, streams in grouped.items()
        }
        self.seed = int(seed)
        self.iteration = 0

    def __len__(self):
        return self.samples_per_dataset * len(self.grouped)

    def __iter__(self):
        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        selected = []
        for streams in self.grouped.values():
            names = sorted(streams)
            for _ in range(self.samples_per_dataset):
                stream = rng.choice(names)
                selected.append(rng.choice(streams[stream]))
        rng.shuffle(selected)
        return iter(selected)


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def masked_mean(value, valid):
    weight = valid.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1.0)


def weighted_mean(value, weight):
    weight = weight.to(value.dtype)
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


def temporal_weighted_loss(prediction, target, weight, order, beta):
    pred, truth, temporal_weight = prediction, target, weight
    for _ in range(order):
        pred = pred[:, 1:] - pred[:, :-1]
        truth = truth[:, 1:] - truth[:, :-1]
        temporal_weight = torch.minimum(
            temporal_weight[:, 1:], temporal_weight[:, :-1]
        )
    return weighted_mean(smooth_l1(pred - truth, beta), temporal_weight)


def distribution(values):
    if not values:
        return {"count": 0, "median_mm": None, "p90_mm": None, "max_mm": None}
    arrays = [
        np.asarray(value).reshape(-1)
        for value in values
        if np.asarray(value).size
    ]
    if not arrays:
        return {"count": 0, "median_mm": None, "p90_mm": None, "max_mm": None}
    array = np.concatenate(arrays)
    array = array[np.isfinite(array)] * 1000.0
    if array.size == 0:
        return {"count": 0, "median_mm": None, "p90_mm": None, "max_mm": None}
    return {
        "count": int(array.size),
        "median_mm": float(np.median(array)),
        "p90_mm": float(np.percentile(array, 90)),
        "max_mm": float(np.max(array)),
    }


def wandb_metrics(split, metrics):
    result = {
        f"{split}/{key}": metrics[key]
        for key in (
            "total", "absolute", "depth", "relative", "velocity",
            "acceleration", "reprojection", "evaluated_hands",
            "observed_hands", "missing_supervised_hands",
            "unsupervised_target_hands",
        )
    }
    for name in ("translation_error", "depth_error"):
        for statistic in ("median_mm", "p90_mm", "max_mm"):
            value = metrics[name].get(statistic)
            if value is not None:
                result[f"{split}/{name}/{statistic}"] = value
    for group, values in metrics.get("by_observability", {}).items():
        for name in ("translation_error", "depth_error"):
            for statistic in ("median_mm", "p90_mm"):
                value = values[name].get(statistic)
                if value is not None:
                    result[
                        f"{split}/by_observability/{group}/{name}/{statistic}"
                    ] = value
    stitched = metrics.get("stitched")
    if stitched:
        result[f"{split}/stitched/unique_hands"] = stitched["unique_hands"]
        for name in ("translation_error", "depth_error", "overlap_disagreement"):
            for statistic in ("median_mm", "p90_mm"):
                value = stitched[name].get(statistic)
                if value is not None:
                    result[f"{split}/stitched/{name}/{statistic}"] = value
    return result


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in (
        "total", "absolute", "depth", "relative", "velocity",
        "acceleration", "reprojection",
    )}
    translation_errors, depth_errors = [], []
    grouped_errors = {
        name: {"translation": [], "depth": []}
        for name in ("observed", "missing_supervised", "unsupervised_target")
    }
    axis_errors = {axis: [] for axis in ("x", "y", "z")}
    stitched = defaultdict(list)
    batches = evaluated = 0
    observed_hands = missing_supervised_hands = unsupervised_targets = 0
    for raw in tqdm(loader, desc="train" if training else "val"):
        batch = move(raw, device)
        valid = batch["target_valid"] & batch["hand_slot_valid"]
        supervision_weight = batch["supervision_weight"] * valid.to(torch.float32)
        supervised = supervision_weight > 0
        observed = batch["observation_valid"] & valid
        target = batch["target_t"]
        with torch.set_grad_enabled(training):
            prediction, predicted_pixels = model(batch)
            beta = args.smooth_l1_beta_mm / 1000.0
            absolute = weighted_mean(
                smooth_l1(prediction - target, beta), supervision_weight
            )
            depth = weighted_mean(
                smooth_l1(prediction[..., 2] - target[..., 2], beta),
                supervision_weight,
            )
            weight = supervision_weight[..., None]
            denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            pred_center = (prediction * weight).sum(dim=1, keepdim=True) / denominator
            target_center = (target * weight).sum(dim=1, keepdim=True) / denominator
            relative = weighted_mean(
                smooth_l1((prediction - pred_center) - (target - target_center), beta),
                supervision_weight,
            )
            velocity = temporal_weighted_loss(
                prediction, target, supervision_weight, 1, beta
            )
            acceleration = temporal_weighted_loss(
                prediction, target, supervision_weight, 2, beta
            )
            target_depth = target[..., 2].clamp_min(1e-6)
            intrinsics = batch["intrinsics"][:, :, None]
            target_pixels = torch.stack((
                target[..., 0] / target_depth * intrinsics[..., 0, 0]
                + intrinsics[..., 0, 2],
                target[..., 1] / target_depth * intrinsics[..., 1, 1]
                + intrinsics[..., 1, 2],
            ), dim=-1)
            if predicted_pixels is None:
                predicted_pixels = torch.stack((
                    prediction[..., 0] / prediction[..., 2].clamp_min(1e-6)
                    * intrinsics[..., 0, 0] + intrinsics[..., 0, 2],
                    prediction[..., 1] / prediction[..., 2].clamp_min(1e-6)
                    * intrinsics[..., 1, 1] + intrinsics[..., 1, 2],
                ), dim=-1)
            reprojection = weighted_mean(
                smooth_l1(
                    predicted_pixels - target_pixels,
                    args.reprojection_beta_px,
                ), supervision_weight,
            )
            total = (
                absolute + args.w_depth * depth + args.w_relative * relative
                + args.w_velocity * velocity + args.w_acceleration * acceleration
                + args.w_reprojection * reprojection / 1000.0
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
            ("reprojection", reprojection),
        ):
            totals[key] += float(value.detach())
        mask = supervised.detach().cpu().numpy().astype(bool)
        valid_np = valid.detach().cpu().numpy().astype(bool)
        observed_np = observed.detach().cpu().numpy().astype(bool)
        group_masks = {
            "observed": mask & observed_np,
            "missing_supervised": mask & ~observed_np,
            "unsupervised_target": valid_np & ~mask,
        }
        error = (prediction - target).detach().cpu().numpy()
        translation_error = np.linalg.norm(error, axis=-1)
        depth_error = np.abs(error[..., 2])
        translation_errors.append(translation_error[mask])
        depth_errors.append(depth_error[mask])
        for name, group_mask in group_masks.items():
            grouped_errors[name]["translation"].append(
                translation_error[group_mask]
            )
            grouped_errors[name]["depth"].append(depth_error[group_mask])
        for axis, axis_index in (("x", 0), ("y", 1), ("z", 2)):
            axis_errors[axis].append(np.abs(error[..., axis_index])[mask])
        if not training:
            prediction_np = prediction.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            stream_np = batch["stream_index"].detach().cpu().numpy()
            frame_np = batch["frame_index"].detach().cpu().numpy()
            track_np = batch["track_id"].detach().cpu().numpy()
            length = prediction_np.shape[1]
            center_weight = 1.0 - np.abs(
                np.linspace(-1.0, 1.0, length, dtype=np.float32)
            )
            center_weight = np.maximum(center_weight, 0.1)
            for batch_index in range(prediction_np.shape[0]):
                for local in range(length):
                    for hand in range(prediction_np.shape[2]):
                        if not mask[batch_index, local, hand]:
                            continue
                        key = (
                            int(stream_np[batch_index]),
                            int(frame_np[batch_index, local]),
                            int(track_np[batch_index, local, hand]),
                        )
                        stitched[key].append((
                            prediction_np[batch_index, local, hand],
                            target_np[batch_index, local, hand],
                            float(center_weight[local]),
                        ))
        evaluated += int(mask.sum())
        observed_hands += int(observed.sum().item())
        missing_supervised_hands += int((supervised & ~observed).sum().item())
        unsupervised_targets += int((valid & ~supervised).sum().item())
        batches += 1
    result = {
        **{key: value / max(batches, 1) for key, value in totals.items()},
        "translation_error": distribution(translation_errors),
        "depth_error": distribution(depth_errors),
        "axis_error": {
            axis: distribution(values) for axis, values in axis_errors.items()
        },
        "by_observability": {
            name: {
                "translation_error": distribution(values["translation"]),
                "depth_error": distribution(values["depth"]),
            }
            for name, values in grouped_errors.items()
        },
        "evaluated_hands": evaluated,
        "observed_hands": observed_hands,
        "missing_supervised_hands": missing_supervised_hands,
        "unsupervised_target_hands": unsupervised_targets,
    }
    if stitched:
        stitched_translation = []
        stitched_depth = []
        stitched_axes = {axis: [] for axis in ("x", "y", "z")}
        overlap_disagreement = []
        overlap_keys = 0
        for samples in stitched.values():
            weights = np.asarray([sample[2] for sample in samples], dtype=np.float64)
            predictions = np.stack([sample[0] for sample in samples]).astype(np.float64)
            target_value = np.asarray(samples[0][1], dtype=np.float64)
            fused = (predictions * weights[:, None]).sum(axis=0) / weights.sum()
            error_value = fused - target_value
            stitched_translation.append(
                np.asarray([np.linalg.norm(error_value)], dtype=np.float32)
            )
            stitched_depth.append(
                np.asarray([abs(error_value[2])], dtype=np.float32)
            )
            for axis, axis_index in (("x", 0), ("y", 1), ("z", 2)):
                stitched_axes[axis].append(
                    np.asarray([abs(error_value[axis_index])], dtype=np.float32)
                )
            if len(samples) > 1:
                overlap_keys += 1
                disagreement = np.linalg.norm(predictions - fused[None], axis=-1)
                overlap_disagreement.append(disagreement.astype(np.float32))
        result["stitched"] = {
            "unique_hands": len(stitched),
            "translation_error": distribution(stitched_translation),
            "depth_error": distribution(stitched_depth),
            "axis_error": {
                axis: distribution(values)
                for axis, values in stitched_axes.items()
            },
            "overlap_frames": overlap_keys,
            "overlap_disagreement": distribution(overlap_disagreement),
        }
    return result


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
        visibility_source=args.visibility_source,
        visibility_root=getattr(args, f"visibility_{split}_root"),
        track_root=getattr(args, f"track_{split}_root"),
        near_anchor_frames=args.near_anchor_frames,
        max_anchor_frames=args.max_anchor_frames,
        near_missing_weight=args.near_missing_weight,
        far_missing_weight=args.far_missing_weight,
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
        "visibility_source": args.visibility_source,
        "track_source": (
            "multihand_cache" if args.track_train_root else "label_order_fallback"
        ),
        "translation_parameterization": args.translation_parameterization,
        "max_window_size": args.max_window_size,
        "train_datasets": sorted({row_dataset(row) for row in train_data.rows}),
        "train_windows_per_dataset": args.train_windows_per_dataset,
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
        max_window_size=args.max_window_size,
        dropout=args.dropout,
        translation_parameterization=args.translation_parameterization,
        max_image_offset_fraction=args.max_image_offset_fraction,
    ).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    train_sampler = (
        DatasetStreamBalancedSampler(
            train_data.rows, args.train_windows_per_dataset, args.seed
        )
        if args.train_windows_per_dataset > 0 else None
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler,
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
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(out_dir),
        )
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
        if wandb_run is not None:
            logged = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            logged.update(wandb_metrics("train", train_metrics))
            logged.update(wandb_metrics("val", val_metrics))
            wandb_run.log(logged, step=epoch)
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
    if wandb_run is not None:
        wandb_run.summary["best_val_total"] = best
        wandb_run.finish()


if __name__ == "__main__":
    main()
