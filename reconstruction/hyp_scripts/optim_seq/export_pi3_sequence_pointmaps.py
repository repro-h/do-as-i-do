#!/usr/bin/env python3
"""Export overlapping Pi3 camera-local pointmap windows for one sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_frame_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload.get("frame_map", payload))
    if isinstance(rows, dict):
        rows = list(rows.values())
    return sorted(rows, key=lambda row: int(row["output_index"]))


def window_starts(count: int, size: int, stride: int) -> list[int]:
    if count <= size:
        return [0]
    starts = list(range(0, count - size + 1, stride))
    final = count - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def main() -> None:
    args = parse_args()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    hand_uni_root = Path(args.hand_uni_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    if str(hand_uni_root) not in sys.path:
        sys.path.insert(0, str(hand_uni_root))
    from pi3_wilor_hand.pi3_runner import run_pi3

    rows = load_frame_rows(frame_map_path)
    image_paths = [str(Path(row["image_path"]).expanduser().resolve()) for row in rows]
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RGB frame: {missing[0]}")

    with np.load(hand_path, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]
    intrinsics = intrinsics.reshape(3, 3)

    size = min(max(2, args.window_size), len(rows))
    stride = max(1, min(args.window_stride, size))
    starts = window_starts(len(rows), size, stride)
    records = []
    for number, start in enumerate(starts):
        end = min(len(rows), start + size)
        output_path = windows_dir / f"window_{start:06d}_{end:06d}.npz"
        if output_path.is_file() and not args.overwrite:
            print(f"[{number + 1}/{len(starts)}] cached {start}:{end}")
        else:
            print(f"[{number + 1}/{len(starts)}] Pi3 {start}:{end}")
            result = run_pi3(
                image_paths=image_paths[start:end],
                K=intrinsics,
                pi3_root=args.pi3_root,
                ckpt=args.checkpoint,
                device=args.device,
                pixel_limit=args.pixel_limit,
                conf_thresh=args.confidence_threshold,
            )
            np.savez_compressed(
                output_path,
                start=np.int32(start),
                end=np.int32(end),
                frame_indices=np.arange(start, end, dtype=np.int32),
                local_points=result.local_points.astype(np.float16),
                confidence=result.conf.astype(np.float16),
                valid_mask=result.masks.astype(np.uint8),
                camera_poses=result.camera_poses.astype(np.float32),
                intrinsics_resized=result.K_resized.astype(np.float32),
                resized_wh=np.asarray(result.resized_wh, dtype=np.int32),
                original_wh=np.asarray(result.original_wh, dtype=np.int32),
                depth_shift=(
                    np.asarray([], dtype=np.float32)
                    if result.depth_shift is None
                    else result.depth_shift.astype(np.float32)
                ),
            )
        records.append(
            {
                "start": start,
                "end": end,
                "path": str(output_path),
            }
        )

    summary = {
        "frame_map_json": str(frame_map_path),
        "hand_npz": str(hand_path),
        "pi3_root": str(Path(args.pi3_root).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "num_frames": len(rows),
        "num_windows": len(records),
        "window_size": size,
        "window_stride": stride,
        "pixel_limit": args.pixel_limit,
        "confidence_threshold": args.confidence_threshold,
        "windows": records,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: {summary_path}")


if __name__ == "__main__":
    main()
