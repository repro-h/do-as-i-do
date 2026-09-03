#!/usr/bin/env python3
"""Freeze the existing ego splits and launch fresh full-pass compact training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DATASETS = ("h2o", "hot3d", "oakink2", "taco")


def select_and_audit(source):
    selected, report, stream_sets = {}, {}, {}
    for split in ("train", "val"):
        path = source / "manifests" / f"{split}_windows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        unknown = {r.get("dataset") for r in rows} - set(DATASETS) - {"dexycb"}
        if unknown:
            raise ValueError(f"Unknown dataset tags: {unknown}")
        rows = [r for r in rows if r["dataset"] in DATASETS]
        selected[split] = rows
        stream_sets[split] = {(r["dataset"], r["stream_id"]) for r in rows}
        report[split] = {}
        for name in DATASETS:
            subset = [r for r in rows if r["dataset"] == name]
            frames = {(r["stream_id"], int(f)) for r in subset for f in r["frame_indices"]}
            report[split][name] = dict(
                streams=len({r["stream_id"] for r in subset}),
                windows=len(subset), unique_frames=len(frames),
            )
        expected = set(DATASETS) - ({"h2o"} if split == "val" else set())
        if {r["dataset"] for r in rows} != expected:
            raise ValueError(f"Expected {split} datasets {expected}; refusing a changed split")
    if stream_sets["train"] & stream_sets["val"]:
        raise ValueError("Train/val stream overlap")
    return selected, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True,
                        help="Trusted baseline checkpoint; read config only, never weights")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpus", default="2,1,3,4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.num_workers < 0:
        parser.error("Invalid training sizes")
    gpus = args.gpus.split(",")
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must be distinct comma-separated GPU IDs")
    rows, report = select_and_audit(args.source_root)
    root = args.out_root.resolve()
    if root == args.source_root.resolve():
        raise ValueError("Use a separate output directory")
    checkpoints = root / "checkpoints"
    if checkpoints.exists() and any(checkpoints.iterdir()):
        raise FileExistsError(f"Refusing to overwrite training: {checkpoints}")
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for split, values in rows.items():
        path = manifests / f"{split}_windows.jsonl"
        content = "".join(json.dumps(row) + "\n" for row in values)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Frozen manifest differs: {path}; use a new output root")
        path.write_text(content)
    report["source_root"] = str(args.source_root.resolve())
    report["steps_per_epoch"] = (len(rows["train"]) + args.batch_size - 1) // args.batch_size
    (root / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.prepare_only:
        return
    import torch
    config = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)["args"]
    # Copy architecture, query and loss settings, not optimizer state or weights.
    keys = """hand_uni_root pi3_root pi3x_checkpoint compact_cache_root max_hands
    max_window_size pixel_limit joint_patch_radius global_grid_size token_dim hidden_dim
    heads temporal_layers dropout weight_decay global_noise_px temporal_noise_px
    joint_noise_px outlier_probability query_dropout near_anchor_frames max_anchor_frames
    near_missing_weight far_missing_weight w_depth w_relative w_velocity w_acceleration
    w_reprojection reprojection_beta_px smooth_l1_beta_mm max_image_offset_fraction seed""".split()
    for key in ("hand_uni_root", "pi3_root", "pi3x_checkpoint", "compact_cache_root"):
        if not config.get(key) or not Path(config[key]).exists():
            raise FileNotFoundError(f"Reference config {key}: {config.get(key)}")
    command = [sys.executable, "-u", str(Path(__file__).with_name("train_v16_1_compact_pi3x.py"))]
    for key in keys:
        if config.get(key) is not None:
            command += ["--" + key.replace("_", "-"), str(config[key])]
    command += ["--train-windows", str(manifests / "train_windows.jsonl"),
                "--val-windows", str(manifests / "val_windows.jsonl"),
                "--out-dir", str(checkpoints), "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
                "--train-windows-per-dataset", "0", "--lr", "0.0002", "--device", "cuda"]
    if len(gpus) > 1:
        command.append("--data-parallel")
    (root / "launch.json").write_text(json.dumps(dict(command=command, gpus=gpus,
        reference_checkpoint=str(args.reference_checkpoint), fresh_training=True), indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
