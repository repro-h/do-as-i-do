#!/usr/bin/env python3
"""Convert one H2O camera sequence to the V15 multi-hand window schema."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SIDES = np.asarray(["left", "right"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--stream-id")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def frame_number(path):
    return int(path.stem)


def window_ranges(length, size, stride):
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return [(start, start + size) for start in starts]


def read_intrinsics(path):
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(values) != 6:
        raise ValueError(f"Expected fx fy cx cy width height in {path}")
    fx, fy, cx, cy, width, height = values
    return (
        np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
        int(round(width)),
        int(round(height)),
    )


def read_hands(path, intrinsics):
    values = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if len(values) != 128:
        raise ValueError(f"Expected 128 hand-pose values in {path}, got {len(values)}")
    flags = np.asarray([values[0], values[64]]) > 0.5
    xyz = np.stack((values[1:64].reshape(21, 3), values[65:].reshape(21, 3)))
    valid_xyz = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 1e-8)
    uv = np.full((2, 21, 2), np.nan, dtype=np.float32)
    uv[..., 0][valid_xyz] = (
        intrinsics[0, 0] * xyz[..., 0][valid_xyz] / xyz[..., 2][valid_xyz]
        + intrinsics[0, 2]
    )
    uv[..., 1][valid_xyz] = (
        intrinsics[1, 1] * xyz[..., 1][valid_xyz] / xyz[..., 2][valid_xyz]
        + intrinsics[1, 2]
    )
    xyz[~flags] = np.nan
    uv[~flags] = np.nan
    return flags, uv, xyz


def infer_stream_id(sequence_dir):
    parts = sequence_dir.parts
    if len(parts) >= 4:
        return "h2o__" + "__".join(parts[-4:])
    return "h2o__" + sequence_dir.name


def main():
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.window_stride <= 0:
        raise ValueError("--window-stride must be positive")
    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    labels_dir = out_dir / "labels"
    manifest = out_dir / f"{args.split}_windows.jsonl"
    summary_path = out_dir / "summary.json"
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {manifest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, width, height = read_intrinsics(sequence_dir / "cam_intrinsics.txt")
    images = {frame_number(path): path for path in (sequence_dir / "rgb").glob("*.png")}
    images.update({frame_number(path): path for path in (sequence_dir / "rgb").glob("*.jpg")})
    poses = {frame_number(path): path for path in (sequence_dir / "hand_pose").glob("*.txt")}
    frames = sorted(set(images) & set(poses))
    if not frames:
        raise RuntimeError(f"No matching RGB/hand_pose frames in {sequence_dir}")

    records = []
    side_valid = np.zeros(2, dtype=np.int64)
    for frame in frames:
        image = cv2.imread(str(images[frame]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {images[frame]}")
        if image.shape[:2] != (height, width):
            raise ValueError(
                f"Image/intrinsics size mismatch for {images[frame]}: "
                f"{image.shape[1]}x{image.shape[0]} versus {width}x{height}"
            )
        flags, uv, xyz = read_hands(poses[frame], intrinsics)
        side_valid += flags.astype(np.int64)
        joint_in_frame = (
            flags[:, None]
            & np.isfinite(uv).all(axis=-1)
            & (uv[..., 0] >= 0) & (uv[..., 0] < width)
            & (uv[..., 1] >= 0) & (uv[..., 1] < height)
        )
        observation_valid = joint_in_frame.any(axis=-1)
        label = labels_dir / f"labels_{frame:06d}.npz"
        np.savez_compressed(
            label,
            joint_2d=uv,
            joint_3d=xyz,
            joint_in_frame=joint_in_frame,
            observation_valid=observation_valid,
            hand_sides=SIDES,
            is_right=np.asarray([False, True]),
            hand_valid=flags,
            # V15 reads segmentation dimensions even with detector visibility.
            seg=np.zeros((height, width), dtype=np.uint8),
            source=np.asarray(str(poses[frame])),
        )
        records.append((frame, images[frame], label, flags))

    stream_id = args.stream_id or infer_stream_id(sequence_dir)
    rows = []
    for start, end in window_ranges(len(records), args.window_size, args.window_stride):
        window = records[start:end]
        rows.append({
            "schema_version": "h2o_multihand_window_v1",
            "split": args.split,
            "stream_id": stream_id,
            "start": start,
            "end": end,
            "frame_indices": [item[0] for item in window],
            "image_paths": [str(item[1]) for item in window],
            "label_paths": [str(item[2]) for item in window],
            "intrinsics": intrinsics.tolist(),
            "hand_sides_metadata_only": ["left", "right"],
            "object_label": 0,
            "dataset": "h2o",
        })
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "h2o_multihand_window_v1",
        "sequence_dir": str(sequence_dir),
        "stream_id": stream_id,
        "frames": len(records),
        "windows": len(rows),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "left_valid_frames": int(side_valid[0]),
        "right_valid_frames": int(side_valid[1]),
        "intrinsics": intrinsics.tolist(),
        "image_wh": [width, height],
        "manifest": str(manifest),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
