#!/usr/bin/env python3
"""Attribute hand-object errors to object pose or hand placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PALM = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--gt-object-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--worst-k", type=int, default=20)
    return parser.parse_args()


def distribution(values: np.ndarray, unit: str = "mm") -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        f"median_{unit}": float(np.median(values)),
        f"p90_{unit}": float(np.quantile(values, 0.9)),
        f"max_{unit}": float(np.max(values)),
    }


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("by_frame") or payload.get("frames") or {}
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    result = {}
    for key, row in iterator:
        if not isinstance(row, dict) or row.get("object_in_camera") is None:
            continue
        frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
        matrix = np.asarray(row["object_in_camera"], dtype=np.float64).reshape(4, 4)
        if np.isfinite(matrix).all():
            result[frame] = matrix
    return result


def frame_string(value, fallback: int) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else str(fallback)).zfill(6)


def mirror_pose(matrix: np.ndarray) -> np.ndarray:
    output = matrix.copy()
    output[:3, :3] = MIRROR_X @ matrix[:3, :3] @ MIRROR_X
    output[:3, 3] = MIRROR_X @ matrix[:3, 3]
    return output


def transform_between(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = target[:3, :3] @ source[:3, :3].T
    translation = target[:3, 3] - rotation @ source[:3, 3]
    return rotation, translation


def point_errors(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    distances = np.linalg.norm(left - right, axis=-1) * 1000.0
    return float(distances[0]), float(np.median(distances[PALM]))


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()
    gt_object_path = Path(args.gt_object_json).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()

    with np.load(prediction_path, allow_pickle=False) as archive:
        correction = np.asarray(
            archive["pi3x_translation_normalized"], dtype=np.float64
        )
        predicted = np.asarray(archive["pi3x_depth_predicted"], dtype=bool)
    with np.load(supervision_path, allow_pickle=False) as archive:
        handflow = np.asarray(archive["pred_joints_3d"], dtype=np.float64)
        gt_hand = np.asarray(archive["gt_joints_3d"], dtype=np.float64)
        fp_pose = np.asarray(archive["object_pose"], dtype=np.float64)
        valid = np.asarray(archive["supervision_valid"], dtype=bool)
        normalized_left = bool(np.asarray(archive["normalized_left"]).item())
        if "frame_ids" in archive.files:
            raw_frame_ids = np.asarray(archive["frame_ids"])
        else:
            raw_frame_ids = np.arange(len(handflow))

    count = min(
        len(handflow), len(gt_hand), len(fp_pose), len(correction),
        len(predicted), len(valid), len(raw_frame_ids)
    )
    gt_object = pose_rows(gt_object_path)
    corrected_hand = handflow[:count] + correction[:count, None, :]
    valid = valid[:count] & predicted[:count]

    rows = []
    for index in range(count):
        frame = frame_string(raw_frame_ids[index], index)
        gt_pose = gt_object.get(frame)
        if not valid[index] or gt_pose is None:
            continue
        if normalized_left:
            gt_pose = mirror_pose(gt_pose)
        current_pose = fp_pose[index]
        if not np.isfinite(current_pose).all():
            continue

        delta_rotation, delta_translation = transform_between(
            gt_pose, current_pose
        )
        target_hand = (
            gt_hand[index] @ delta_rotation.T + delta_translation
        )
        translation_only_target = (
            gt_hand[index]
            + current_pose[:3, 3]
            - gt_pose[:3, 3]
        )

        camera_wrist, camera_palm = point_errors(
            corrected_hand[index], gt_hand[index]
        )
        object_wrist, object_palm = point_errors(
            target_hand, gt_hand[index]
        )
        hand_wrist, hand_palm = point_errors(
            corrected_hand[index], target_hand
        )
        translation_object_wrist, translation_object_palm = point_errors(
            translation_only_target, gt_hand[index]
        )
        translation_hand_wrist, translation_hand_palm = point_errors(
            corrected_hand[index], translation_only_target
        )
        raw_hand_wrist, raw_hand_palm = point_errors(
            handflow[index], target_hand
        )
        translation_error = float(
            np.linalg.norm(current_pose[:3, 3] - gt_pose[:3, 3]) * 1000.0
        )
        relative_rotation = gt_pose[:3, :3].T @ current_pose[:3, :3]
        rotation_error = float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(relative_rotation).as_rotvec()
        )))

        if object_palm > 1.5 * hand_palm:
            attribution = "object_pose_dominant"
        elif hand_palm > 1.5 * object_palm:
            attribution = "hand_placement_dominant"
        else:
            attribution = "mixed"
        rows.append({
            "frame": frame,
            "attribution": attribution,
            "object_translation_error_mm": translation_error,
            "object_rotation_error_deg": rotation_error,
            "camera_hand_wrist_error_mm": camera_wrist,
            "camera_hand_palm_error_mm": camera_palm,
            "object_induced_wrist_shift_mm": object_wrist,
            "object_induced_palm_shift_mm": object_palm,
            "translation_only_object_wrist_shift_mm": (
                translation_object_wrist
            ),
            "translation_only_object_shift_mm": translation_object_palm,
            "translation_only_v8_relative_wrist_error_mm": (
                translation_hand_wrist
            ),
            "translation_only_v8_relative_palm_error_mm": translation_hand_palm,
            "v8_relative_wrist_error_mm": hand_wrist,
            "v8_relative_palm_error_mm": hand_palm,
            "raw_relative_wrist_error_mm": raw_hand_wrist,
            "raw_relative_palm_error_mm": raw_hand_palm,
        })

    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    counts = {
        name: sum(row["attribution"] == name for row in rows)
        for name in (
            "object_pose_dominant", "hand_placement_dominant", "mixed"
        )
    }
    summary = {
        "prediction_npz": str(prediction_path),
        "supervision_npz": str(supervision_path),
        "gt_object_json": str(gt_object_path),
        "normalized_left": normalized_left,
        "num_frames": len(rows),
        "attribution_counts": counts,
        "object_translation_error": distribution(
            values("object_translation_error_mm")
        ),
        "object_rotation_error_deg": distribution(
            values("object_rotation_error_deg"), "deg"
        ),
        "camera_hand_palm_error": distribution(
            values("camera_hand_palm_error_mm")
        ),
        "object_induced_palm_shift": distribution(
            values("object_induced_palm_shift_mm")
        ),
        "translation_only_object_shift": distribution(
            values("translation_only_object_shift_mm")
        ),
        "translation_only_v8_relative_palm_error": distribution(
            values("translation_only_v8_relative_palm_error_mm")
        ),
        "v8_relative_palm_error": distribution(
            values("v8_relative_palm_error_mm")
        ),
        "raw_relative_palm_error": distribution(
            values("raw_relative_palm_error_mm")
        ),
        "worst_object_frames": sorted(
            rows,
            key=lambda row: row["object_induced_palm_shift_mm"],
            reverse=True,
        )[:args.worst_k],
        "worst_hand_frames": sorted(
            rows,
            key=lambda row: row["v8_relative_palm_error_mm"],
            reverse=True,
        )[:args.worst_k],
        "frames": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"frames: {len(rows)} attribution: {counts}")
    for key in (
        "object_translation_error",
        "object_rotation_error_deg",
        "camera_hand_palm_error",
        "object_induced_palm_shift",
        "translation_only_object_shift",
        "translation_only_v8_relative_palm_error",
        "v8_relative_palm_error",
        "raw_relative_palm_error",
    ):
        print(key, summary[key])


if __name__ == "__main__":
    main()
