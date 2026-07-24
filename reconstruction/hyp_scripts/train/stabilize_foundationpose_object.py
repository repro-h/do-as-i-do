#!/usr/bin/env python3
"""Merge object-only translation corrections with SO(3)-smoothed rotations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-json", required=True)
    parser.add_argument("--stage1-prediction", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--rotation-window", type=int, default=5)
    parser.add_argument("--slow-smoothing", type=float, default=0.65)
    parser.add_argument("--fast-smoothing", type=float, default=0.15)
    parser.add_argument("--fast-motion-deg", type=float, default=8.0)
    parser.add_argument("--max-correction-mm", type=float, default=60.0)
    parser.add_argument("--max-correction-step-mm", type=float, default=3.0)
    return parser.parse_args()


def load_pose_rows(payload: dict) -> tuple[str, dict]:
    for key in ("by_frame", "frames"):
        rows = payload.get(key)
        if isinstance(rows, dict):
            return key, rows
    raise KeyError("FoundationPose JSON has no by_frame/frames pose dictionary")


def normalize_frame_id(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if text.startswith("color_"):
        text = text.split("_")[-1]
    return text.zfill(6)


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        frame_ids = [normalize_frame_id(value) for value in data["frame_ids"]]
        deltas = np.asarray(data["object_delta"], dtype=np.float64)
        predicted = np.asarray(data["predicted"], dtype=bool)
    return {
        frame_id: deltas[index]
        for index, frame_id in enumerate(frame_ids)
        if predicted[index] and np.isfinite(deltas[index]).all()
    }


def gate_translation_deltas(
    frame_ids: list[str],
    deltas: dict[str, np.ndarray],
    max_norm: float,
    max_step: float,
) -> tuple[dict[str, np.ndarray], dict]:
    result = {}
    clipped_norm = []
    clipped_step = []
    previous = None
    previous_frame_id = None
    for frame_id in frame_ids:
        if frame_id not in deltas:
            previous = None
            previous_frame_id = None
            continue
        value = deltas[frame_id].copy()
        norm = float(np.linalg.norm(value))
        if norm > max_norm:
            value *= max_norm / max(norm, 1e-12)
            clipped_norm.append(frame_id)
        if (
            previous is not None
            and previous_frame_id is not None
            and int(frame_id) == previous_frame_id + 1
        ):
            step = value - previous
            step_norm = float(np.linalg.norm(step))
            if step_norm > max_step:
                value = previous + step * (max_step / max(step_norm, 1e-12))
                clipped_step.append(frame_id)
        result[frame_id] = value
        previous = value
        previous_frame_id = int(frame_id)
    return result, {
        "num_norm_clipped": len(clipped_norm),
        "norm_clipped_frames": clipped_norm,
        "num_step_clipped": len(clipped_step),
        "step_clipped_frames": clipped_step,
    }


def angular_step_degrees(rotations: list[Rotation]) -> np.ndarray:
    if len(rotations) < 2:
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        [
            np.degrees((rotations[index].inv() * rotations[index + 1]).magnitude())
            for index in range(len(rotations) - 1)
        ],
        dtype=np.float64,
    )


def rotation_acceleration_degrees(rotations: list[Rotation]) -> np.ndarray:
    if len(rotations) < 3:
        return np.empty(0, dtype=np.float64)
    velocity = np.stack(
        [
            (rotations[index].inv() * rotations[index + 1]).as_rotvec()
            for index in range(len(rotations) - 1)
        ],
        axis=0,
    )
    return np.degrees(np.linalg.norm(velocity[1:] - velocity[:-1], axis=-1))


def smoothing_strength(
    local_speed_deg: float,
    slow_strength: float,
    fast_strength: float,
    threshold_deg: float,
) -> float:
    start = threshold_deg * 0.5
    if local_speed_deg <= start:
        return slow_strength
    if local_speed_deg >= threshold_deg:
        return fast_strength
    ratio = (local_speed_deg - start) / max(threshold_deg - start, 1e-8)
    return slow_strength * (1.0 - ratio) + fast_strength * ratio


def smooth_rotation_segment(
    rotations: list[Rotation],
    window: int,
    slow_strength: float,
    fast_strength: float,
    fast_motion_deg: float,
) -> list[Rotation]:
    if len(rotations) < 3:
        return rotations
    radius = max(1, window // 2)
    steps = angular_step_degrees(rotations)
    output = []
    for center_index, center in enumerate(rotations):
        begin = max(0, center_index - radius)
        end = min(len(rotations), center_index + radius + 1)
        offsets = np.arange(begin, end) - center_index
        sigma = max(radius / 1.5, 1.0)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        relative_vectors = np.stack(
            [
                (center.inv() * rotations[index]).as_rotvec()
                for index in range(begin, end)
            ],
            axis=0,
        )
        mean_vector = np.average(relative_vectors, axis=0, weights=weights)
        adjacent = []
        if center_index > 0:
            adjacent.append(steps[center_index - 1])
        if center_index < len(rotations) - 1:
            adjacent.append(steps[center_index])
        local_speed = max(adjacent) if adjacent else 0.0
        strength = smoothing_strength(
            local_speed,
            slow_strength,
            fast_strength,
            fast_motion_deg,
        )
        output.append(center * Rotation.from_rotvec(mean_vector * strength))
    return output


def quantiles(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if args.rotation_window < 3 or args.rotation_window % 2 == 0:
        raise ValueError("--rotation-window must be an odd integer >= 3")
    for name in ("slow_smoothing", "fast_smoothing"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")

    pose_path = Path(args.foundationpose_json).expanduser().resolve()
    prediction_path = Path(args.stage1_prediction).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    output = copy.deepcopy(payload)
    rows_key, source_rows = load_pose_rows(payload)
    _, output_rows = load_pose_rows(output)

    valid = []
    for raw_frame_id, row in source_rows.items():
        pose_value = row.get("object_in_camera")
        if pose_value is None:
            continue
        pose = np.asarray(pose_value, dtype=np.float64)
        if pose.size != 16 or not np.isfinite(pose).all():
            continue
        valid.append((normalize_frame_id(raw_frame_id), raw_frame_id, pose.reshape(4, 4)))
    valid.sort(key=lambda item: int(item[0]))
    if not valid:
        raise RuntimeError("FoundationPose JSON contains no valid object poses")

    normalized_ids = [item[0] for item in valid]
    raw_ids = [item[1] for item in valid]
    raw_deltas = load_predictions(prediction_path)
    gated_deltas, gating = gate_translation_deltas(
        normalized_ids,
        raw_deltas,
        args.max_correction_mm / 1000.0,
        args.max_correction_step_mm / 1000.0,
    )

    original_rotations = [Rotation.from_matrix(item[2][:3, :3]) for item in valid]
    smoothed_rotations = list(original_rotations)
    segment_begin = 0
    for index in range(1, len(valid) + 1):
        segment_end = index == len(valid)
        if not segment_end:
            segment_end = int(valid[index][0]) != int(valid[index - 1][0]) + 1
        if not segment_end:
            continue
        smoothed_rotations[segment_begin:index] = smooth_rotation_segment(
            original_rotations[segment_begin:index],
            args.rotation_window,
            args.slow_smoothing,
            args.fast_smoothing,
            args.fast_motion_deg,
        )
        segment_begin = index

    correction_norms = []
    correction_steps = []
    previous_delta = None
    previous_frame = None
    for index, (frame_id, raw_frame_id, source_pose) in enumerate(valid):
        pose = source_pose.copy()
        pose[:3, :3] = smoothed_rotations[index].as_matrix()
        delta = gated_deltas.get(frame_id)
        if delta is not None:
            pose[:3, 3] += delta
            correction_norms.append(np.linalg.norm(delta) * 1000.0)
            if previous_delta is not None and previous_frame == int(frame_id) - 1:
                correction_steps.append(
                    np.linalg.norm(delta - previous_delta) * 1000.0
                )
            previous_delta = delta
            previous_frame = int(frame_id)
        else:
            previous_delta = None
            previous_frame = None
        output_rows[raw_frame_id]["object_in_camera"] = pose.astype(float).tolist()

    original_speed = angular_step_degrees(original_rotations)
    smoothed_speed = angular_step_degrees(smoothed_rotations)
    original_acceleration = rotation_acceleration_degrees(original_rotations)
    smoothed_acceleration = rotation_acceleration_degrees(smoothed_rotations)
    audit = {
        "source_foundationpose_json": str(pose_path),
        "stage1_prediction": str(prediction_path),
        "settings": vars(args),
        "num_valid_poses": len(valid),
        "num_translation_corrections": len(gated_deltas),
        "translation_gating": gating,
        "translation_correction_norm_mm": quantiles(
            np.asarray(correction_norms, dtype=np.float64)
        ),
        "translation_correction_step_mm": quantiles(
            np.asarray(correction_steps, dtype=np.float64)
        ),
        "rotation_speed_deg_per_frame": {
            "before": quantiles(original_speed),
            "after": quantiles(smoothed_speed),
        },
        "rotation_acceleration_deg_per_frame2": {
            "before": quantiles(original_acceleration),
            "after": quantiles(smoothed_acceleration),
        },
    }
    output["object_stabilization"] = audit
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    audit_path = out_path.with_name(f"{out_path.stem}_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Wrote: {out_path}")
    print(f"Wrote: {audit_path}")


if __name__ == "__main__":
    main()
