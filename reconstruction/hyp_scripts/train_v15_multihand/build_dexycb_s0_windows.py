#!/usr/bin/env python3
"""Build official DexYCB S0 window manifests for V15.

The output stays in original camera/image coordinates. No left-hand mirroring
or canonical-right conversion is performed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


SUBJECTS = [f"2020{month:02d}{day:02d}-subject-{index:02d}" for month, day, index in (
    (7, 9, 1), (8, 13, 2), (8, 20, 3), (9, 3, 4), (9, 8, 5),
    (9, 18, 6), (9, 28, 7), (10, 2, 8), (10, 15, 9), (10, 22, 10),
)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexycb-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument(
        "--stream-id",
        help="Optional subject__sequence__camera stream for a fast smoke manifest",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def camera_intrinsics(root, serial):
    candidates = sorted((root / "calibration" / "intrinsics").glob(f"{serial}*.yml"))
    candidates += sorted((root / "calibration" / "intrinsics").glob(f"{serial}*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No intrinsics for camera {serial}")
    data = load_yaml(candidates[0])
    if "color" in data:
        color = data["color"]
        return [
            [float(color["fx"]), 0.0, float(color["ppx"])],
            [0.0, float(color["fy"]), float(color["ppy"])],
            [0.0, 0.0, 1.0],
        ]
    if "intrinsics" in data:
        return np.asarray(data["intrinsics"], dtype=np.float32).reshape(3, 3).tolist()
    if "K" in data:
        return np.asarray(data["K"], dtype=np.float32).reshape(3, 3).tolist()
    return np.asarray(data, dtype=np.float32).reshape(3, 3).tolist()


def frame_number(path):
    return int(path.stem.rsplit("_", 1)[-1])


def frame_is_valid(label_path):
    try:
        with np.load(str(label_path), allow_pickle=False) as data:
            joint_2d = np.asarray(data["joint_2d"])[0]
            joint_3d = np.asarray(data["joint_3d"])[0]
        return bool(
            np.isfinite(joint_2d).all()
            and np.isfinite(joint_3d).all()
            and joint_2d.shape == (21, 2)
            and joint_3d.shape == (21, 3)
        )
    except (OSError, KeyError, IndexError, ValueError):
        return False


def split_for(subject_index, sequence_index):
    if sequence_index % 5 != 4:
        return "train"
    return "val" if subject_index < 2 else "test"


def window_ranges(length, size, stride):
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if not starts or starts[-1] != final:
        starts.append(final)
    return [(start, start + size) for start in starts]


def main():
    args = parse_args()
    root = Path(args.dexycb_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: out_dir / f"{split}_windows.jsonl" for split in ("train", "val", "test")}
    if not args.overwrite:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Outputs exist; pass --overwrite: {existing}")

    rows = {split: [] for split in outputs}
    stream_counts = {split: set() for split in outputs}
    selected = None
    if args.stream_id:
        parts = args.stream_id.split("__")
        if len(parts) != 3:
            raise ValueError("stream-id must be subject__sequence__camera")
        selected = tuple(parts)
    for subject_index, subject in enumerate(SUBJECTS):
        if selected is not None and subject != selected[0]:
            continue
        subject_dir = root / subject
        if not subject_dir.is_dir():
            continue
        sequences = sorted(path for path in subject_dir.iterdir() if path.is_dir())
        for sequence_index, sequence_dir in enumerate(sequences):
            if selected is not None and sequence_dir.name != selected[1]:
                continue
            split = split_for(subject_index, sequence_index)
            meta_path = sequence_dir / "meta.yml"
            if not meta_path.is_file():
                continue
            meta = load_yaml(meta_path)
            side = str(meta.get("mano_sides", [meta.get("hand_side", "unknown")])[0]).lower()
            for camera_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
                serial = camera_dir.name
                if selected is not None and serial != selected[2]:
                    continue
                colors = sorted(camera_dir.glob("color_*.jpg"), key=frame_number)
                if not colors:
                    colors = sorted(camera_dir.glob("color_*.png"), key=frame_number)
                records = []
                for color in colors:
                    number = frame_number(color)
                    label = camera_dir / f"labels_{number:06d}.npz"
                    if label.is_file():
                        records.append((number, color, label, frame_is_valid(label)))
                if len(records) < args.window_size:
                    continue
                try:
                    intrinsics = camera_intrinsics(root, serial)
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    continue
                stream_id = f"{subject}__{sequence_dir.name}__{serial}"
                for start, end in window_ranges(len(records), args.window_size, args.window_stride):
                    window = records[start:end]
                    if sum(int(item[3]) for item in window) < args.min_valid_frames:
                        continue
                    rows[split].append({
                        "schema_version": "dexycb_s0_multihand_window_v1",
                        "split": split,
                        "stream_id": stream_id,
                        "subject": subject,
                        "sequence": sequence_dir.name,
                        "camera_serial": serial,
                        "hand_side_metadata_only": side,
                        "start": start,
                        "end": end,
                        "frame_indices": [item[0] for item in window],
                        "image_paths": [str(item[1]) for item in window],
                        "label_paths": [str(item[2]) for item in window],
                        "intrinsics": intrinsics,
                    })
                    stream_counts[split].add(stream_id)

    for split, path in outputs.items():
        with path.open("w", encoding="utf-8") as handle:
            for row in rows[split]:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(json.dumps({
            "split": split,
            "streams": len(stream_counts[split]),
            "windows": len(rows[split]),
            "output": str(path),
        }))


if __name__ == "__main__":
    main()
