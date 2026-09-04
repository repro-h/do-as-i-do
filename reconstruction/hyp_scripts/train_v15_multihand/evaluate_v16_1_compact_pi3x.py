#!/usr/bin/env python3
"""Evaluate a V16.1 checkpoint against precomputed compact Pi3X caches."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_dataset import CompactWindowDataset  # noqa: E402
from compact_model import CompactMultiHandPi3XTrajectoryModel  # noqa: E402
from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from online_pi3x import DiskCompactFeatureProvider, DummyDenseProvider  # noqa: E402
from train import run_epoch  # noqa: E402
from train_v16_1_compact_pi3x import load_model_state  # noqa: E402


FEATURE_ABLATIONS = (
    "full",
    "no-local",
    "shuffled-local",
    "no-global",
    "no-metric",
    "prior-only",
)


class FeatureAblationWrapper(torch.nn.Module):
    """Apply inference-only Pi3X feature ablations without rewriting caches."""

    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode

    @staticmethod
    def _shuffle_local(feature):
        if feature.shape[0] > 1:
            return feature.roll(shifts=1, dims=0)
        if feature.shape[1] > 1:
            return feature.roll(shifts=max(1, feature.shape[1] // 2), dims=1)
        return feature.flip(dims=(-2,))

    def forward(self, batch):
        if self.mode == "full":
            return self.model(batch)

        batch = dict(batch)
        if self.mode == "shuffled-local":
            batch["joint_patch_features"] = self._shuffle_local(
                batch["joint_patch_features"]
            )
        if self.mode in ("no-local", "prior-only"):
            batch["joint_patch_features"] = torch.zeros_like(
                batch["joint_patch_features"]
            )
        if self.mode in ("no-global", "prior-only"):
            batch["global_features"] = torch.zeros_like(batch["global_features"])
        if self.mode in ("no-metric", "prior-only"):
            batch["metric_window_features"] = torch.zeros_like(
                batch["metric_window_features"]
            )
        return self.model(batch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--visibility-root")
    parser.add_argument("--track-root")
    parser.add_argument("--compact-cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--joint-patch-radius", type=int, default=1)
    parser.add_argument("--global-grid-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument(
        "--feature-ablation",
        choices=FEATURE_ABLATIONS,
        default="full",
        help="Inference-only Pi3X feature ablation; does not retrain the model.",
    )
    return parser.parse_args()


def value(config, name, default):
    return config[name] if name in config else default


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    config = dict(checkpoint.get("args", {}))

    metadata = DexYCBMultiHandWindowDataset(
        args.windows,
        None,
        max_hands=args.max_hands,
        training=False,
        noise=QueryNoise(),
        visibility_source="detector",
        visibility_root=args.visibility_root,
        track_root=args.track_root,
        near_anchor_frames=value(config, "near_anchor_frames", 4),
        max_anchor_frames=value(config, "max_anchor_frames", 8),
        near_missing_weight=value(config, "near_missing_weight", 0.5),
        far_missing_weight=value(config, "far_missing_weight", 0.2),
        dense_provider=DummyDenseProvider(),
    )
    provider = DiskCompactFeatureProvider(
        args.compact_cache_root,
        patch_radius=args.joint_patch_radius,
        global_grid_size=args.global_grid_size,
    )
    dataset = CompactWindowDataset(metadata, provider)
    sample = dataset[0]
    compact_thj = tuple(sample["joint_patch_features"].shape[:3])
    query_thj = tuple(sample["joint_uv"].shape[:3])
    if compact_thj != query_thj:
        raise ValueError(
            f"Cache [T,H,J]={compact_thj} != query [T,H,J]={query_thj}"
        )

    device = torch.device(args.device)
    model = CompactMultiHandPi3XTrajectoryModel(
        point_dim=sample["joint_patch_features"].shape[-1],
        metric_dim=sample["metric_window_features"].shape[-1],
        token_dim=value(config, "token_dim", 128),
        hidden_dim=value(config, "hidden_dim", 192),
        heads=value(config, "heads", 4),
        temporal_layers=value(config, "temporal_layers", 2),
        dropout=value(config, "dropout", 0.1),
        max_window_size=value(config, "max_window_size", 128),
        translation_parameterization=value(
            config, "translation_parameterization", "ray_depth_uv"
        ),
        max_image_offset_fraction=value(
            config, "max_image_offset_fraction", 0.15
        ),
    ).to(device)
    load_model_state(model, checkpoint["model"])
    model = FeatureAblationWrapper(model, args.feature_ablation).to(device)
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two CUDA devices")
        model = torch.nn.DataParallel(model)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    loss_args = SimpleNamespace(
        smooth_l1_beta_mm=value(config, "smooth_l1_beta_mm", 5.0),
        reprojection_beta_px=value(config, "reprojection_beta_px", 2.0),
        w_depth=value(config, "w_depth", 0.5),
        w_relative=value(config, "w_relative", 0.5),
        w_velocity=value(config, "w_velocity", 0.05),
        w_acceleration=value(config, "w_acceleration", 0.02),
        w_reprojection=value(config, "w_reprojection", 0.1),
    )
    metrics = run_epoch(
        model,
        loader,
        device,
        loss_args,
        dataset_names=metadata.dataset_names,
        stream_names={
            index: name
            for name, index in metadata.stream_indices.items()
        },
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model_version": checkpoint.get("model_version"),
        "windows": str(Path(args.windows).expanduser().resolve()),
        "compact_cache_root": str(
            Path(args.compact_cache_root).expanduser().resolve()
        ),
        "feature_ablation": args.feature_ablation,
        "feature_ablation_scope": "inference_only_no_retraining",
        "metrics": metrics,
    }
    output = Path(args.out_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
