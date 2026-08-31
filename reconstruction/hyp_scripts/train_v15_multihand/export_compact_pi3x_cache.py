#!/usr/bin/env python3
"""Export compact joint/global Pi3X candidates to a resumable disk cache."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from online_pi3x import (  # noqa: E402
    DummyDenseProvider,
    Pi3XWindowMaterializer,
    compact_cache_path,
    row_key,
    valid_compact_cache,
    write_compact_cache,
)
from train_v16_online_pi3x import load_rows  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument(
        "--visibility-root",
        help="Fallback root; mixed manifests may provide visibility_npz per row",
    )
    parser.add_argument(
        "--track-root",
        help="Fallback root; mixed manifests may provide tracks_npz per row",
    )
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--pi3x-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument(
        "--feature-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--joint-patch-radius", type=int, default=1)
    parser.add_argument("--global-grid-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--status-json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metadata_dataset(args):
    return DexYCBMultiHandWindowDataset(
        args.windows,
        None,
        max_hands=args.max_hands,
        training=False,
        noise=QueryNoise(),
        visibility_source="detector",
        visibility_root=args.visibility_root,
        track_root=args.track_root,
        dense_provider=DummyDenseProvider(),
    )


def main():
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.windows)
    dataset = metadata_dataset(args)
    if len(rows) != len(dataset):
        raise RuntimeError("Manifest and metadata dataset lengths differ")
    indices = list(range(len(rows)))[args.shard_index::args.num_shards]
    if args.limit > 0:
        indices = indices[:args.limit]
    out_root = Path(args.out_root).expanduser().resolve()

    pending = []
    cached = []
    seen = set()
    for index in indices:
        row = rows[index]
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        path = compact_cache_path(out_root, row)
        if not args.overwrite and valid_compact_cache(
            path, row, args.joint_patch_radius, args.global_grid_size
        ):
            cached.append(key)
        else:
            pending.append(index)

    extractor = None
    completed = []
    failures = []
    try:
        if pending:
            extractor = Pi3XWindowMaterializer(
                args.hand_uni_root,
                args.pi3_root,
                args.pi3x_checkpoint,
                device=args.device,
                pixel_limit=args.pixel_limit,
                feature_dtype=args.feature_dtype,
            )
        for index in tqdm(pending, desc="compact Pi3X"):
            row = rows[index]
            path = compact_cache_path(out_root, row)
            try:
                metadata = dataset[index]
                joint_uv = metadata["joint_uv"].numpy()
                payload = extractor.compact(
                    row,
                    joint_uv,
                    patch_radius=args.joint_patch_radius,
                    global_grid_size=args.global_grid_size,
                )
                write_compact_cache(
                    path,
                    row,
                    payload,
                    joint_uv,
                    args.joint_patch_radius,
                    args.global_grid_size,
                )
                completed.append(row_key(row))
            except Exception as error:
                failures.append({
                    "key": row_key(row),
                    "error": repr(error),
                })
                print(f"FAILED {row_key(row)}: {type(error).__name__}: {error}")
    finally:
        if extractor is not None:
            extractor.close()

    status = {
        "cache_version": "compact_pi3x_window_v1",
        "windows": str(Path(args.windows).expanduser().resolve()),
        "out_root": str(out_root),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "requested": len(indices),
        "unique_requested": len(seen),
        "cached": len(cached),
        "completed": len(completed),
        "failed": len(failures),
        "failures": failures,
        "joint_patch_radius": args.joint_patch_radius,
        "global_grid_size": args.global_grid_size,
    }
    if args.status_json:
        status_path = Path(args.status_json).expanduser().resolve()
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
