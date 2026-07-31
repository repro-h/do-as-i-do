#!/usr/bin/env python3
"""Batch Stage1.5 occlusion-bias supervision preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--v7-prediction-root", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--pi3x-root", required=True)
    parser.add_argument("--prepare-script", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--carry-enter", type=float, default=0.8)
    parser.add_argument("--carry-exit", type=float, default=0.5)
    parser.add_argument("--occlusion-enter", type=float, default=0.25)
    parser.add_argument("--occlusion-exit", type=float, default=0.15)
    parser.add_argument("--mask-dilation-px", type=int, default=3)
    parser.add_argument("--min-core-frames", type=int, default=2)
    parser.add_argument("--gate-threshold-mm", type=float, default=5.0)
    parser.add_argument("--max-bias-mm", type=float, default=25.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "None of the candidate files exist:\n"
        + "\n".join(str(path) for path in paths)
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    prediction_root = Path(
        args.v7_prediction_root
    ).expanduser().resolve()
    supervision_root = Path(args.supervision_root).expanduser().resolve()
    pi3x_root = Path(args.pi3x_root).expanduser().resolve()
    prepare_script = Path(args.prepare_script).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards})"
        )

    rows = load_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise RuntimeError(f"No records selected from {manifest_path}")

    completed: list[str] = []
    failures: list[dict] = []
    for index, row in enumerate(rows):
        stream_id = row["stream_id"]
        stream_out = out_root / stream_id
        output_path = (
            stream_out / "stage1p5_occlusion_bias_supervision.npz"
        )
        if output_path.is_file() and not args.overwrite:
            print(
                f"[{index + 1}/{len(rows)}] cached {stream_id}",
                flush=True,
            )
            completed.append(stream_id)
            continue

        try:
            hand_path = first_existing(
                [
                    prediction_root
                    / stream_id
                    / "handflow_camera_result_pi3x_depth_refined.npz",
                    prediction_root
                    / stream_id
                    / stream_id
                    / "handflow_camera_result_pi3x_depth_refined.npz",
                ]
            )
            supervision_path = first_existing(
                [
                    supervision_root / f"{stream_id}.npz",
                    supervision_root / stream_id / f"{stream_id}.npz",
                ]
            )
            pi3x_dir_candidates = [
                pi3x_root / stream_id,
                pi3x_root / stream_id / stream_id,
            ]
            pi3x_path = first_existing(
                [
                    directory / "pi3x_geometry_features_compact.npz"
                    for directory in pi3x_dir_candidates
                ]
            )
            frame_map_path = first_existing(
                [
                    directory / "dexycb_frame_map.json"
                    for directory in pi3x_dir_candidates
                ]
            )
            command = [
                sys.executable,
                "-u",
                str(prepare_script),
                "--v7-hand-npz",
                str(hand_path),
                "--supervision-npz",
                str(supervision_path),
                "--pi3x-cache",
                str(pi3x_path),
                "--frame-map-json",
                str(frame_map_path),
                "--out-dir",
                str(stream_out),
                "--window-size",
                str(args.window_size),
                "--window-stride",
                str(args.window_stride),
                "--carry-enter",
                str(args.carry_enter),
                "--carry-exit",
                str(args.carry_exit),
                "--occlusion-enter",
                str(args.occlusion_enter),
                "--occlusion-exit",
                str(args.occlusion_exit),
                "--mask-dilation-px",
                str(args.mask_dilation_px),
                "--min-core-frames",
                str(args.min_core_frames),
                "--gate-threshold-mm",
                str(args.gate_threshold_mm),
                "--max-bias-mm",
                str(args.max_bias_mm),
                "--quiet",
            ]
            if args.overwrite:
                command.append("--overwrite")
            print(
                f"[{index + 1}/{len(rows)}] {stream_id}",
                flush=True,
            )
            subprocess.run(command, check=True)
            completed.append(stream_id)
        except Exception as error:
            failures.append(
                {"stream_id": stream_id, "error": repr(error)}
            )
            print(f"FAILED {stream_id}: {error}", flush=True)

    status = {
        "manifest": str(manifest_path),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_requested": len(rows),
        "num_completed": len(completed),
        "num_failed": len(failures),
        "completed": completed,
        "failures": failures,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    status_name = (
        "status.json"
        if args.num_shards == 1
        else (
            f"status_shard_{args.shard_index:02d}_of_"
            f"{args.num_shards:02d}.json"
        )
    )
    (out_root / status_name).write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
