#!/usr/bin/env python3
"""Apply segment-aware EKF and RTS smoothing to object pose trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--segmentation-audit-json", required=True)
    parser.add_argument("--out-ekf-json", required=True)
    parser.add_argument("--out-rts-json", required=True)
    parser.add_argument("--out-audit", required=True)
    parser.add_argument("--translation-measurement-mm", type=float, default=4.0)
    parser.add_argument("--translation-acceleration-mm", type=float, default=20.0)
    parser.add_argument("--rotation-measurement-deg", type=float, default=2.0)
    parser.add_argument("--rotation-acceleration-deg", type=float, default=3.0)
    parser.add_argument(
        "--initial-translation-velocity-mm", type=float, default=20.0
    )
    parser.add_argument(
        "--initial-rotation-velocity-deg", type=float, default=5.0
    )
    parser.add_argument("--boundary-blend-frames", type=int, default=4)
    parser.add_argument("--max-translation-correction-mm", type=float, default=8.0)
    parser.add_argument("--max-rotation-correction-deg", type=float, default=5.0)
    return parser.parse_args()


def normalize_frame(value: object) -> str:
    value = str(value)
    if value.startswith("color_"):
        value = value.split("_")[-1]
    return value.zfill(6)


def load_pose_rows(payload: dict) -> tuple[str, dict]:
    for key in ("by_frame", "frames"):
        if isinstance(payload.get(key), dict):
            return key, payload[key]
    raise KeyError("Pose JSON has no by_frame/frames dictionary")


def resolve_row(rows: dict, frame_id: str) -> tuple[str, dict] | None:
    for key in (frame_id, str(int(frame_id))):
        if key in rows:
            return key, rows[key]
    return None


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def constant_velocity_matrices(
    dt: float,
    acceleration_sigma: float,
    measurement_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    identity = np.eye(3)
    transition = np.block(
        [[identity, dt * identity], [np.zeros((3, 3)), identity]]
    )
    observation = np.block([identity, np.zeros((3, 3))])
    process_block = np.array(
        [
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2],
        ]
    )
    process = np.kron(process_block, identity) * acceleration_sigma**2
    measurement = identity * measurement_sigma**2
    return transition, observation, process, measurement


def ekf_rts(
    measurements: np.ndarray,
    dt: float,
    measurement_sigma: float,
    acceleration_sigma: float,
    initial_velocity_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(measurements)
    if count < 2:
        return measurements.copy(), measurements.copy()
    transition, observation, process, measurement_cov = (
        constant_velocity_matrices(
            dt, acceleration_sigma, measurement_sigma
        )
    )
    state = np.concatenate(
        [measurements[0], (measurements[1] - measurements[0]) / dt]
    )
    covariance = np.diag(
        [measurement_sigma**2] * 3 + [initial_velocity_sigma**2] * 3
    )
    filtered_states = []
    filtered_covariances = []
    predicted_states = []
    predicted_covariances = []
    identity = np.eye(6)

    for index, observation_value in enumerate(measurements):
        if index:
            predicted_state = transition @ state
            predicted_covariance = (
                transition @ covariance @ transition.T + process
            )
        else:
            predicted_state = state.copy()
            predicted_covariance = covariance.copy()
        innovation = observation_value - observation @ predicted_state
        innovation_covariance = (
            observation @ predicted_covariance @ observation.T
            + measurement_cov
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
            + gain @ measurement_cov @ gain.T
        )
        predicted_states.append(predicted_state)
        predicted_covariances.append(predicted_covariance)
        filtered_states.append(state.copy())
        filtered_covariances.append(covariance.copy())

    smoothed_states = [state.copy() for state in filtered_states]
    smoothed_covariances = [
        covariance.copy() for covariance in filtered_covariances
    ]
    for index in range(count - 2, -1, -1):
        smoother_gain = (
            filtered_covariances[index]
            @ transition.T
            @ np.linalg.inv(predicted_covariances[index + 1])
        )
        smoothed_states[index] = (
            filtered_states[index]
            + smoother_gain
            @ (
                smoothed_states[index + 1]
                - predicted_states[index + 1]
            )
        )
        smoothed_covariances[index] = (
            filtered_covariances[index]
            + smoother_gain
            @ (
                smoothed_covariances[index + 1]
                - predicted_covariances[index + 1]
            )
            @ smoother_gain.T
        )
    return (
        np.asarray(filtered_states)[:, :3],
        np.asarray(smoothed_states)[:, :3],
    )


def continuous_local_rotvec(rotations: np.ndarray) -> np.ndarray:
    anchor = rotations[0]
    relative = Rotation.from_matrix(rotations @ anchor.T)
    quaternions = relative.as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1.0
    return Rotation.from_quat(quaternions).as_rotvec()


def blend_weight(
    local_index: int,
    length: int,
    blend_frames: int,
    lock_end: bool,
) -> float:
    if blend_frames <= 0:
        return 1.0
    start_alpha = min(1.0, local_index / float(blend_frames))
    end_alpha = 1.0
    if lock_end:
        end_alpha = min(
            1.0, (length - 1 - local_index) / float(blend_frames)
        )
    alpha = min(start_alpha, end_alpha)
    return float(0.5 - 0.5 * np.cos(np.pi * np.clip(alpha, 0.0, 1.0)))


def clipped_vector(
    source: np.ndarray, target: np.ndarray, maximum: float
) -> np.ndarray:
    delta = target - source
    norm = np.linalg.norm(delta)
    if norm > maximum > 0:
        delta *= maximum / norm
    return source + delta


def blend_rotation(
    source: np.ndarray,
    target: np.ndarray,
    alpha: float,
    max_angle: float,
) -> np.ndarray:
    delta = Rotation.from_matrix(target @ source.T).as_rotvec()
    norm = np.linalg.norm(delta)
    if norm > max_angle > 0:
        delta *= max_angle / norm
    limited = Rotation.from_rotvec(delta).as_matrix() @ source
    if alpha <= 0:
        return source
    if alpha >= 1:
        return limited
    return Slerp(
        [0.0, 1.0], Rotation.from_matrix(np.stack([source, limited]))
    )([alpha]).as_matrix()[0]


def temporal_metrics(poses: np.ndarray) -> dict:
    translation_speed = (
        np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) * 1000.0
    )
    translation_acceleration = np.linalg.norm(
        np.diff(poses[:, :3, 3], n=2, axis=0), axis=1
    ) * 1000.0
    relative_rotation = (
        poses[1:, :3, :3]
        @ np.transpose(poses[:-1, :3, :3], (0, 2, 1))
    )
    rotation_speed = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(relative_rotation).as_rotvec(), axis=1
        )
    )
    rotation_acceleration = np.abs(np.diff(rotation_speed))
    return {
        "translation_speed_mm": distribution(translation_speed),
        "translation_acceleration_mm": distribution(translation_acceleration),
        "rotation_speed_deg": distribution(rotation_speed),
        "rotation_acceleration_deg": distribution(rotation_acceleration),
    }


def main() -> None:
    args = parse_args()
    source_path = Path(args.pose_json).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows_key, rows = load_pose_rows(source)
    frame_map = json.loads(
        Path(args.frame_map_json).expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    segmentation_path = (
        Path(args.segmentation_audit_json).expanduser().resolve()
    )
    segmentation = json.loads(
        segmentation_path.read_text(encoding="utf-8")
    )
    frame_ids = [
        normalize_frame(row["original_frame"]) for row in frame_map["frames"]
    ]
    poses = []
    row_keys = []
    for frame_id in frame_ids:
        resolved = resolve_row(rows, frame_id)
        if resolved is None:
            raise KeyError(f"No pose row for frame {frame_id}")
        row_key, row = resolved
        pose = np.asarray(row.get("object_in_camera"), dtype=np.float64)
        if pose.size != 16 or not np.isfinite(pose).all():
            raise ValueError(f"Invalid object pose for frame {frame_id}")
        row_keys.append(row_key)
        poses.append(pose.reshape(4, 4))
    poses = np.stack(poses)

    ekf_poses = poses.copy()
    rts_poses = poses.copy()
    segment_reports = []
    # The trajectory audits use per-frame velocity and acceleration, so the
    # state transition is deliberately expressed in frame units.
    dt = 1.0
    translation_measurement = args.translation_measurement_mm / 1000.0
    translation_acceleration = args.translation_acceleration_mm / 1000.0
    rotation_measurement = np.radians(args.rotation_measurement_deg)
    rotation_acceleration = np.radians(args.rotation_acceleration_deg)
    max_translation = args.max_translation_correction_mm / 1000.0
    max_rotation = np.radians(args.max_rotation_correction_deg)

    for segment_index, segment in enumerate(
        segmentation.get("dynamic_segments", [])
    ):
        begin, end = [int(value) for value in segment["output_frames"]]
        if not 0 <= begin <= end < len(poses):
            raise ValueError(f"Invalid dynamic segment [{begin}, {end}]")
        indices = np.arange(begin, end + 1)
        segment_poses = poses[indices]
        translations = segment_poses[:, :3, 3]
        local_rotvec = continuous_local_rotvec(
            segment_poses[:, :3, :3]
        )
        filtered_translation, smoothed_translation = ekf_rts(
            translations,
            dt,
            translation_measurement,
            translation_acceleration,
            args.initial_translation_velocity_mm / 1000.0,
        )
        filtered_rotation, smoothed_rotation = ekf_rts(
            local_rotvec,
            dt,
            rotation_measurement,
            rotation_acceleration,
            np.radians(args.initial_rotation_velocity_deg),
        )
        anchor_rotation = segment_poses[0, :3, :3]
        filtered_rotation_matrix = (
            Rotation.from_rotvec(filtered_rotation).as_matrix()
            @ anchor_rotation
        )
        smoothed_rotation_matrix = (
            Rotation.from_rotvec(smoothed_rotation).as_matrix()
            @ anchor_rotation
        )
        lock_end = end < len(poses) - 1
        for local_index, frame_index in enumerate(indices):
            alpha = blend_weight(
                local_index,
                len(indices),
                args.boundary_blend_frames,
                lock_end,
            )
            ekf_translation = clipped_vector(
                translations[local_index],
                filtered_translation[local_index],
                max_translation,
            )
            rts_translation = clipped_vector(
                translations[local_index],
                smoothed_translation[local_index],
                max_translation,
            )
            ekf_poses[frame_index, :3, 3] = (
                translations[local_index]
                + alpha * (ekf_translation - translations[local_index])
            )
            rts_poses[frame_index, :3, 3] = (
                translations[local_index]
                + alpha * (rts_translation - translations[local_index])
            )
            ekf_poses[frame_index, :3, :3] = blend_rotation(
                segment_poses[local_index, :3, :3],
                filtered_rotation_matrix[local_index],
                alpha,
                max_rotation,
            )
            rts_poses[frame_index, :3, :3] = blend_rotation(
                segment_poses[local_index, :3, :3],
                smoothed_rotation_matrix[local_index],
                alpha,
                max_rotation,
            )
        segment_reports.append(
            {
                "segment_index": segment_index,
                "output_frames": [begin, end],
                "original_frames": [frame_ids[begin], frame_ids[end]],
                "num_frames": int(len(indices)),
                "lock_end": lock_end,
            }
        )

    def write_output(path_value: str, result_poses: np.ndarray, mode: str) -> Path:
        payload = json.loads(json.dumps(source))
        output_rows = payload[rows_key]
        for index, pose in enumerate(result_poses):
            output_rows[row_keys[index]]["object_in_camera"] = pose.tolist()
        payload["segment_ekf_rts"] = {
            "source_pose_json": str(source_path),
            "segmentation_audit_json": str(segmentation_path),
            "mode": mode,
            "uses_gt_object_pose": False,
            "settings": vars(args),
        }
        path = Path(path_value).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    ekf_path = write_output(args.out_ekf_json, ekf_poses, "ekf")
    rts_path = write_output(args.out_rts_json, rts_poses, "ekf_rts")
    ekf_translation_correction = (
        np.linalg.norm(ekf_poses[:, :3, 3] - poses[:, :3, 3], axis=1)
        * 1000.0
    )
    rts_translation_correction = (
        np.linalg.norm(rts_poses[:, :3, 3] - poses[:, :3, 3], axis=1)
        * 1000.0
    )
    ekf_rotation_correction = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(
                ekf_poses[:, :3, :3]
                @ np.transpose(poses[:, :3, :3], (0, 2, 1))
            ).as_rotvec(),
            axis=1,
        )
    )
    rts_rotation_correction = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(
                rts_poses[:, :3, :3]
                @ np.transpose(poses[:, :3, :3], (0, 2, 1))
            ).as_rotvec(),
            axis=1,
        )
    )
    audit = {
        "source_pose_json": str(source_path),
        "segmentation_audit_json": str(segmentation_path),
        "out_ekf_json": str(ekf_path),
        "out_rts_json": str(rts_path),
        "settings": vars(args),
        "segments": segment_reports,
        "temporal": {
            "input": temporal_metrics(poses),
            "ekf": temporal_metrics(ekf_poses),
            "ekf_rts": temporal_metrics(rts_poses),
        },
        "correction": {
            "ekf_translation_mm": distribution(ekf_translation_correction),
            "rts_translation_mm": distribution(rts_translation_correction),
            "ekf_rotation_deg": distribution(ekf_rotation_correction),
            "rts_rotation_deg": distribution(rts_rotation_correction),
        },
    }
    audit_path = Path(args.out_audit).expanduser().resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
