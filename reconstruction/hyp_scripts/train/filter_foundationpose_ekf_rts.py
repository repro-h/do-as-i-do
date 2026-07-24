#!/usr/bin/env python3
"""Denoise FoundationPose SE(3) trajectories with CV Kalman filtering and RTS."""

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
    parser.add_argument("--out-rts-json", required=True)
    parser.add_argument("--out-ekf-json")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--translation-measurement-mm", type=float, default=4.0)
    parser.add_argument(
        "--translation-acceleration-mps2", type=float, default=0.6
    )
    parser.add_argument("--rotation-measurement-deg", type=float, default=3.0)
    parser.add_argument(
        "--angular-acceleration-deg-s2", type=float, default=90.0
    )
    return parser.parse_args()


def normalize_frame_id(value: object) -> str:
    text = str(value)
    if text.startswith("color_"):
        text = text.rsplit("_", 1)[-1]
    return text.zfill(6)


def pose_rows(payload: dict) -> tuple[str, dict]:
    for key in ("by_frame", "frames"):
        rows = payload.get(key)
        if isinstance(rows, dict):
            return key, rows
    raise KeyError("FoundationPose JSON has no by_frame/frames pose dictionary")


def cv_matrices(
    dimension: int,
    dt: float,
    measurement_sigma: float,
    acceleration_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    identity = np.eye(dimension, dtype=np.float64)
    transition = np.block(
        [[identity, dt * identity], [np.zeros_like(identity), identity]]
    )
    observation = np.concatenate(
        [identity, np.zeros_like(identity)], axis=1
    )
    process_block = np.array(
        [
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3, dt**2],
        ],
        dtype=np.float64,
    )
    process = acceleration_sigma**2 * np.kron(process_block, identity)
    measurement = measurement_sigma**2 * identity
    return transition, observation, process, measurement


def kalman_rts(
    measurements: np.ndarray,
    dt: float,
    measurement_sigma: float,
    acceleration_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    count, dimension = measurements.shape
    if count < 2:
        return measurements.copy(), measurements.copy()
    transition, observation, process, measurement = cv_matrices(
        dimension, dt, measurement_sigma, acceleration_sigma
    )
    state = np.concatenate(
        [measurements[0], (measurements[1] - measurements[0]) / dt]
    )
    position_variance = max(measurement_sigma**2, 1e-12)
    velocity_variance = max((measurement_sigma / dt) ** 2, 1e-12)
    covariance = np.diag(
        [position_variance] * dimension
        + [velocity_variance] * dimension
    )
    identity = np.eye(2 * dimension, dtype=np.float64)

    filtered_states = []
    filtered_covariances = []
    predicted_states = []
    predicted_covariances = []
    for index, value in enumerate(measurements):
        if index:
            predicted_state = transition @ state
            predicted_covariance = (
                transition @ covariance @ transition.T + process
            )
        else:
            predicted_state = state.copy()
            predicted_covariance = covariance.copy()
        innovation = value - observation @ predicted_state
        innovation_covariance = (
            observation @ predicted_covariance @ observation.T + measurement
        )
        gain = (
            predicted_covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        state = predicted_state + gain @ innovation
        covariance = (
            (identity - gain @ observation)
            @ predicted_covariance
            @ (identity - gain @ observation).T
            + gain @ measurement @ gain.T
        )
        predicted_states.append(predicted_state)
        predicted_covariances.append(predicted_covariance)
        filtered_states.append(state.copy())
        filtered_covariances.append(covariance.copy())

    smoothed_states = [value.copy() for value in filtered_states]
    smoothed_covariances = [value.copy() for value in filtered_covariances]
    for index in range(count - 2, -1, -1):
        gain = (
            filtered_covariances[index]
            @ transition.T
            @ np.linalg.inv(predicted_covariances[index + 1])
        )
        smoothed_states[index] = filtered_states[index] + gain @ (
            smoothed_states[index + 1] - predicted_states[index + 1]
        )
        smoothed_covariances[index] = filtered_covariances[index] + gain @ (
            smoothed_covariances[index + 1]
            - predicted_covariances[index + 1]
        ) @ gain.T

    filtered = np.stack(filtered_states, axis=0)[:, :dimension]
    smoothed = np.stack(smoothed_states, axis=0)[:, :dimension]
    return filtered, smoothed


def continuous_quaternions(rotations: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(rotations).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def normalize_quaternions(values: np.ndarray) -> np.ndarray:
    result = values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12
    )
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def filter_segment(
    poses: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / args.fps
    translation_ekf, translation_rts = kalman_rts(
        poses[:, :3, 3],
        dt,
        args.translation_measurement_mm / 1000.0,
        args.translation_acceleration_mps2,
    )
    quaternions = continuous_quaternions(poses[:, :3, :3])
    quaternion_measurement_sigma = np.sin(
        np.deg2rad(args.rotation_measurement_deg) * 0.5
    )
    quaternion_acceleration_sigma = np.deg2rad(
        args.angular_acceleration_deg_s2
    ) * 0.5
    quaternion_ekf, quaternion_rts = kalman_rts(
        quaternions,
        dt,
        quaternion_measurement_sigma,
        quaternion_acceleration_sigma,
    )
    quaternion_ekf = normalize_quaternions(quaternion_ekf)
    quaternion_rts = normalize_quaternions(quaternion_rts)

    ekf = poses.copy()
    rts = poses.copy()
    ekf[:, :3, 3] = translation_ekf
    rts[:, :3, 3] = translation_rts
    ekf[:, :3, :3] = Rotation.from_quat(quaternion_ekf).as_matrix()
    rts[:, :3, :3] = Rotation.from_quat(quaternion_rts).as_matrix()
    return ekf, rts


def angular_speed(rotations: np.ndarray) -> np.ndarray:
    if len(rotations) < 2:
        return np.empty(0, dtype=np.float64)
    converted = Rotation.from_matrix(rotations)
    return np.asarray(
        [
            np.degrees(
                (converted[index].inv() * converted[index + 1]).magnitude()
            )
            for index in range(len(converted) - 1)
        ]
    )


def angular_acceleration(rotations: np.ndarray) -> np.ndarray:
    if len(rotations) < 3:
        return np.empty(0, dtype=np.float64)
    converted = Rotation.from_matrix(rotations)
    velocities = np.stack(
        [
            (converted[index].inv() * converted[index + 1]).as_rotvec()
            for index in range(len(converted) - 1)
        ]
    )
    return np.degrees(np.linalg.norm(np.diff(velocities, axis=0), axis=1))


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


def trajectory_metrics(poses: np.ndarray) -> dict:
    translations = poses[:, :3, 3]
    return {
        "translation_speed_mm_per_frame": quantiles(
            np.linalg.norm(np.diff(translations, axis=0), axis=1) * 1000.0
        ),
        "translation_acceleration_mm_per_frame2": quantiles(
            np.linalg.norm(np.diff(translations, n=2, axis=0), axis=1)
            * 1000.0
        ),
        "rotation_speed_deg_per_frame": quantiles(
            angular_speed(poses[:, :3, :3])
        ),
        "rotation_acceleration_deg_per_frame2": quantiles(
            angular_acceleration(poses[:, :3, :3])
        ),
    }


def correction_metrics(source: np.ndarray, target: np.ndarray) -> dict:
    source_rotation = Rotation.from_matrix(source[:, :3, :3])
    target_rotation = Rotation.from_matrix(target[:, :3, :3])
    rotation_error = np.degrees(
        (source_rotation.inv() * target_rotation).magnitude()
    )
    translation_error = (
        np.linalg.norm(target[:, :3, 3] - source[:, :3, 3], axis=1)
        * 1000.0
    )
    return {
        "translation_mm": quantiles(translation_error),
        "rotation_deg": quantiles(rotation_error),
    }


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    source_path = Path(args.foundationpose_json).expanduser().resolve()
    rts_path = Path(args.out_rts_json).expanduser().resolve()
    ekf_path = (
        Path(args.out_ekf_json).expanduser().resolve()
        if args.out_ekf_json
        else rts_path.with_name(f"{rts_path.stem}_ekf.json")
    )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    _, rows = pose_rows(payload)

    valid = []
    for raw_frame_id, row in rows.items():
        value = row.get("object_in_camera")
        if value is None:
            continue
        pose = np.asarray(value, dtype=np.float64)
        if pose.size == 16 and np.isfinite(pose).all():
            valid.append(
                (
                    normalize_frame_id(raw_frame_id),
                    raw_frame_id,
                    pose.reshape(4, 4),
                )
            )
    valid.sort(key=lambda item: int(item[0]))
    if not valid:
        raise RuntimeError("FoundationPose JSON contains no valid object poses")

    source_poses = np.stack([item[2] for item in valid])
    ekf_poses = source_poses.copy()
    rts_poses = source_poses.copy()
    segment_start = 0
    for index in range(1, len(valid) + 1):
        segment_end = index == len(valid)
        if not segment_end:
            segment_end = int(valid[index][0]) != int(valid[index - 1][0]) + 1
        if not segment_end:
            continue
        segment_ekf, segment_rts = filter_segment(
            source_poses[segment_start:index], args
        )
        ekf_poses[segment_start:index] = segment_ekf
        rts_poses[segment_start:index] = segment_rts
        segment_start = index

    ekf_payload = copy.deepcopy(payload)
    rts_payload = copy.deepcopy(payload)
    _, ekf_rows = pose_rows(ekf_payload)
    _, rts_rows = pose_rows(rts_payload)
    for index, (_, raw_frame_id, _) in enumerate(valid):
        ekf_rows[raw_frame_id]["object_in_camera"] = ekf_poses[index].tolist()
        rts_rows[raw_frame_id]["object_in_camera"] = rts_poses[index].tolist()

    audit = {
        "source_foundationpose_json": str(source_path),
        "out_ekf_json": str(ekf_path),
        "out_rts_json": str(rts_path),
        "num_valid_poses": len(valid),
        "settings": vars(args),
        "trajectory": {
            "initial": trajectory_metrics(source_poses),
            "ekf": trajectory_metrics(ekf_poses),
            "rts": trajectory_metrics(rts_poses),
        },
        "correction_from_initial": {
            "ekf": correction_metrics(source_poses, ekf_poses),
            "rts": correction_metrics(source_poses, rts_poses),
        },
    }
    ekf_payload["object_ekf_filter"] = audit
    rts_payload["object_ekf_rts_filter"] = audit
    ekf_path.parent.mkdir(parents=True, exist_ok=True)
    rts_path.parent.mkdir(parents=True, exist_ok=True)
    ekf_path.write_text(json.dumps(ekf_payload, indent=2), encoding="utf-8")
    rts_path.write_text(json.dumps(rts_payload, indent=2), encoding="utf-8")
    audit_path = rts_path.with_name(f"{rts_path.stem}_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Wrote: {ekf_path}")
    print(f"Wrote: {rts_path}")
    print(f"Wrote: {audit_path}")


if __name__ == "__main__":
    main()
