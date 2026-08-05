#!/usr/bin/env python3
"""Audit exported object-frame hand SE(3) supervision and windows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-valid-frames", type=int, default=8)
    parser.add_argument(
        "--excluded-object", action="append", default=[]
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    supervision_root = Path(args.supervision_root).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    excluded = set(args.excluded_object)
    if not supervision_root.is_dir():
        raise NotADirectoryError(supervision_root)
    if not windows_path.is_file():
        raise FileNotFoundError(windows_path)

    windows = load_jsonl(windows_path)
    window_keys = [
        (str(row["stream_id"]), int(row["start"]), int(row["end"]))
        for row in windows
    ]
    duplicate_windows = len(window_keys) - len(set(window_keys))
    windows_by_stream: dict[str, list[dict]] = defaultdict(list)
    for row in windows:
        windows_by_stream[str(row["stream_id"])].append(row)

    required = {
        "frame_ids",
        "initial_translation_object",
        "target_translation_object",
        "initial_rotation_object",
        "target_rotation_object",
        "target_wrist_camera_oracle",
        "target_root_rotation_camera_oracle",
        "filtered_object_pose",
        "valid_translation",
        "valid_rotation",
        "pose_quality_valid",
        "object_name",
        "hand_side",
        "rotation_supervision_weight",
    }
    object_rows: dict[str, dict] = defaultdict(
        lambda: {
            "streams": 0,
            "zero_valid_streams": 0,
            "frames": 0,
            "valid_translation": 0,
            "valid_rotation": 0,
            "pose_gate_valid": 0,
            "windows": 0,
            "rotation_weights": set(),
        }
    )
    hand_sides: dict[str, int] = defaultdict(int)
    failures = []
    translation_roundtrip_mm = []
    rotation_roundtrip_deg = []
    rotation_det_error = []
    rotation_orthogonality_error = []
    found_streams = set()

    for path in sorted(supervision_root.glob("*.npz")):
        stream_id = path.stem
        found_streams.add(stream_id)
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = sorted(required.difference(data.files))
                if missing:
                    raise KeyError(f"missing keys: {missing}")
                object_name = scalar_text(data["object_name"])
                hand_side = scalar_text(data["hand_side"])
                frame_ids = np.asarray(data["frame_ids"])
                valid_t = np.asarray(data["valid_translation"], dtype=bool)
                valid_r = np.asarray(data["valid_rotation"], dtype=bool)
                pose_valid = np.asarray(data["pose_quality_valid"], dtype=bool)
                target_t = np.asarray(
                    data["target_translation_object"], dtype=np.float64
                )
                target_r = np.asarray(
                    data["target_rotation_object"], dtype=np.float64
                )
                object_pose = np.asarray(
                    data["filtered_object_pose"], dtype=np.float64
                )
                target_t_camera = np.asarray(
                    data["target_wrist_camera_oracle"], dtype=np.float64
                )
                target_r_camera = np.asarray(
                    data["target_root_rotation_camera_oracle"],
                    dtype=np.float64,
                )
                rotation_weight = float(
                    np.asarray(data["rotation_supervision_weight"]).item()
                )

            count = len(frame_ids)
            lengths = [
                len(valid_t), len(valid_r), len(pose_valid), len(target_t),
                len(target_r), len(object_pose), len(target_t_camera),
                len(target_r_camera),
            ]
            if any(value != count for value in lengths):
                raise ValueError(f"frame length mismatch: {lengths}")
            if np.any(valid_r & ~valid_t):
                raise ValueError("valid_rotation is not a subset of valid_translation")
            if valid_t.any() and not np.isfinite(target_t[valid_t]).all():
                raise ValueError("nonfinite target translation on valid frames")
            if valid_r.any() and not np.isfinite(target_r[valid_r]).all():
                raise ValueError("nonfinite target rotation on valid frames")

            for row in windows_by_stream.get(stream_id, []):
                start, end = int(row["start"]), int(row["end"])
                if not 0 <= start < end <= count:
                    raise ValueError(f"invalid window [{start}, {end})")
                if int(valid_r[start:end].sum()) < args.min_valid_frames:
                    raise ValueError(
                        f"window [{start}, {end}) has too few valid frames"
                    )
                if str(row.get("object_name")) != object_name:
                    raise ValueError("window object_name mismatch")

            if valid_t.any():
                predicted_camera = np.einsum(
                    "tij,tj->ti",
                    object_pose[valid_t, :3, :3],
                    target_t[valid_t],
                ) + object_pose[valid_t, :3, 3]
                translation_roundtrip_mm.extend(
                    (
                        np.linalg.norm(
                            predicted_camera - target_t_camera[valid_t], axis=-1
                        )
                        * 1000.0
                    ).tolist()
                )
            if valid_r.any():
                rotations = target_r[valid_r]
                det = np.linalg.det(rotations)
                orth = np.linalg.norm(
                    np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3),
                    axis=(1, 2),
                )
                rotation_det_error.extend(np.abs(det - 1.0).tolist())
                rotation_orthogonality_error.extend(orth.tolist())
                predicted_camera_r = (
                    object_pose[valid_r, :3, :3] @ rotations
                )
                relative = (
                    np.swapaxes(predicted_camera_r, -1, -2)
                    @ target_r_camera[valid_r]
                )
                cosine = np.clip(
                    (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
                rotation_roundtrip_deg.extend(
                    np.degrees(np.arccos(cosine)).tolist()
                )

            stats = object_rows[object_name]
            stats["streams"] += 1
            stats["zero_valid_streams"] += int(not valid_r.any())
            stats["frames"] += count
            stats["valid_translation"] += int(valid_t.sum())
            stats["valid_rotation"] += int(valid_r.sum())
            stats["pose_gate_valid"] += int(pose_valid.sum())
            stats["windows"] += len(windows_by_stream.get(stream_id, []))
            stats["rotation_weights"].add(rotation_weight)
            hand_sides[hand_side] += 1
        except Exception as error:
            failures.append({
                "stream_id": stream_id,
                "error": f"{type(error).__name__}: {error}",
            })

    missing_supervision = sorted(set(windows_by_stream) - found_streams)
    unused_supervision = sorted(found_streams - set(windows_by_stream))
    excluded_in_supervision = sorted(excluded.intersection(object_rows))
    excluded_in_windows = sorted(
        {
            str(row.get("object_name")) for row in windows
            if str(row.get("object_name")) in excluded
        }
    )
    serialized_objects = {}
    for name, row in sorted(object_rows.items()):
        serialized_objects[name] = {
            **{key: value for key, value in row.items() if key != "rotation_weights"},
            "valid_translation_fraction": (
                row["valid_translation"] / max(row["frames"], 1)
            ),
            "valid_rotation_fraction": (
                row["valid_rotation"] / max(row["frames"], 1)
            ),
            "rotation_weights": sorted(row["rotation_weights"]),
        }

    report = {
        "supervision_root": str(supervision_root),
        "windows": str(windows_path),
        "num_supervision_streams": len(found_streams),
        "num_window_streams": len(windows_by_stream),
        "num_windows": len(windows),
        "duplicate_windows": duplicate_windows,
        "missing_supervision_streams": missing_supervision,
        "unused_supervision_streams": unused_supervision,
        "excluded_objects": sorted(excluded),
        "excluded_in_supervision": excluded_in_supervision,
        "excluded_in_windows": excluded_in_windows,
        "hand_sides": dict(sorted(hand_sides.items())),
        "translation_roundtrip_mm": distribution(translation_roundtrip_mm),
        "rotation_roundtrip_deg": distribution(rotation_roundtrip_deg),
        "rotation_det_error": distribution(rotation_det_error),
        "rotation_orthogonality_error": distribution(
            rotation_orthogonality_error
        ),
        "objects": serialized_objects,
        "num_failures": len(failures),
        "failures": failures,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
