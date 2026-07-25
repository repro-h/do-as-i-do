#!/usr/bin/env python3
"""Segment TAPIR motion and consolidate confirmed-static FoundationPose spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--tapir-npz", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-audit", required=True)
    parser.add_argument("--out-plot")
    parser.add_argument("--enter-translation-mm", type=float, default=6.0)
    parser.add_argument("--enter-rotation-deg", type=float, default=1.2)
    parser.add_argument("--exit-translation-mm", type=float, default=3.0)
    parser.add_argument("--exit-rotation-deg", type=float, default=0.6)
    parser.add_argument("--enter-frames", type=int, default=2)
    parser.add_argument("--exit-frames", type=int, default=4)
    parser.add_argument("--stationary-window", type=int, default=6)
    parser.add_argument("--exit-net-translation-mm", type=float, default=8.0)
    parser.add_argument("--exit-net-rotation-deg", type=float, default=1.5)
    parser.add_argument("--median-window", type=int, default=3)
    parser.add_argument("--min-static-frames", type=int, default=5)
    parser.add_argument("--static-trim-frames", type=int, default=1)
    parser.add_argument("--max-static-translation-mad-mm", type=float, default=8.0)
    parser.add_argument("--max-static-rotation-mad-deg", type=float, default=8.0)
    return parser.parse_args()


def normalize_frame(value: object) -> str:
    value = str(value)
    if value.startswith("color_"):
        value = value.split("_")[-1]
    return value.zfill(6)


def pose_rows(payload: dict) -> tuple[str, dict]:
    for key in ("by_frame", "frames"):
        if isinstance(payload.get(key), dict):
            return key, payload[key]
    raise KeyError("FoundationPose JSON has no by_frame/frames dictionary")


def resolve_row(rows: dict, frame_id: str) -> tuple[str, dict]:
    for key in (frame_id, str(int(frame_id))):
        if key in rows:
            return key, rows[key]
    raise KeyError(f"No FoundationPose pose for frame {frame_id}")


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    result = np.full_like(values, np.nan, dtype=np.float64)
    for index in range(len(values)):
        selected = values[
            max(0, index - radius) : min(len(values), index + radius + 1)
        ]
        selected = selected[np.isfinite(selected)]
        if len(selected):
            result[index] = np.median(selected)
    return result


def contiguous_segments(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    segments = []
    begin = None
    for index, current in enumerate(mask):
        if bool(current) == value and begin is None:
            begin = index
        elif bool(current) != value and begin is not None:
            segments.append((begin, index - 1))
            begin = None
    if begin is not None:
        segments.append((begin, len(mask) - 1))
    return segments


def hysteresis_motion(
    translation: np.ndarray,
    rotation: np.ndarray,
    enter_translation: float,
    enter_rotation: float,
    exit_translation: float,
    exit_rotation: float,
    enter_frames: int,
    exit_frames: int,
    stationary_ready: np.ndarray | None = None,
    stationary_backfill: int = 1,
) -> np.ndarray:
    high = (translation >= enter_translation) | (rotation >= enter_rotation)
    low = (
        stationary_ready
        if stationary_ready is not None
        else (translation <= exit_translation) & (rotation <= exit_rotation)
    )
    dynamic = np.zeros(len(translation), dtype=bool)
    state = False
    high_run = 0
    low_run = 0
    for index in range(len(translation)):
        if not state:
            high_run = high_run + 1 if high[index] else 0
            if high_run >= enter_frames:
                state = True
                dynamic[index - enter_frames + 1 : index + 1] = True
                low_run = 0
        else:
            dynamic[index] = True
            low_run = low_run + 1 if low[index] else 0
            if low_run >= exit_frames:
                backfill = max(exit_frames, stationary_backfill)
                dynamic[max(0, index - backfill + 1) : index + 1] = False
                state = False
                high_run = 0
                low_run = 0
    return dynamic


def rotation_distances_deg(
    rotations: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    relative = rotations @ reference.T
    return np.asarray(
        [rotation_angle_deg(rotation) for rotation in relative],
        dtype=np.float64,
    )


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def save_plot(
    path: Path,
    translation: np.ndarray,
    rotation: np.ndarray,
    frame_dynamic: np.ndarray,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    x = np.arange(len(translation))
    axes[0].plot(x, translation, color="#00897b")
    axes[0].axhline(args.enter_translation_mm, color="#d1495b", linestyle="--")
    axes[0].axhline(args.exit_translation_mm, color="#1976d2", linestyle=":")
    axes[0].set_ylabel("center mm/frame")
    axes[1].plot(x, rotation, color="#6a1b9a")
    axes[1].axhline(args.enter_rotation_deg, color="#d1495b", linestyle="--")
    axes[1].axhline(args.exit_rotation_deg, color="#1976d2", linestyle=":")
    axes[1].set_ylabel("rotation deg/frame")
    axes[1].set_xlabel("frame pair")
    for axis in axes:
        for begin, end in contiguous_segments(frame_dynamic, True):
            axis.axvspan(max(0, begin - 1), min(len(x) - 1, end), color="#ffcc80", alpha=0.25)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    source_path = Path(args.foundationpose_json).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows_key, rows = pose_rows(source)
    frame_map = json.loads(
        Path(args.frame_map_json).expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    frame_ids = [
        normalize_frame(row["original_frame"]) for row in frame_map["frames"]
    ]
    row_keys = []
    poses = []
    for frame_id in frame_ids:
        row_key, row = resolve_row(rows, frame_id)
        pose = np.asarray(row.get("object_in_camera"), dtype=np.float64)
        if pose.size != 16 or not np.isfinite(pose).all():
            raise ValueError(f"Invalid object pose for frame {frame_id}")
        row_keys.append(row_key)
        poses.append(pose.reshape(4, 4))
    poses = np.stack(poses)

    with np.load(
        Path(args.tapir_npz).expanduser().resolve(), allow_pickle=True
    ) as payload:
        tracks = np.asarray(payload["relative_transform_pnp"], dtype=np.float64)
        statuses = np.asarray(payload["pnp_status"]).astype(str)
        inliers = np.asarray(payload["pnp_inlier_mask"]).sum(axis=1)
    if len(tracks) != len(poses) - 1:
        raise ValueError(
            f"Track/pose mismatch: tracks={len(tracks)} poses={len(poses)}"
        )

    centers = poses[:, :3, 3]
    translation_speed = np.full(len(tracks), np.nan)
    rotation_speed = np.full(len(tracks), np.nan)
    for index, track in enumerate(tracks):
        if statuses[index] != "ok" or inliers[index] < 8:
            continue
        predicted_center = track[:3, :3] @ centers[index] + track[:3, 3]
        translation_speed[index] = (
            np.linalg.norm(predicted_center - centers[index]) * 1000.0
        )
        rotation_speed[index] = rotation_angle_deg(track[:3, :3])
    translation_filtered = median_filter(
        translation_speed, args.median_window
    )
    rotation_filtered = median_filter(rotation_speed, args.median_window)
    stationary_ready = np.zeros(len(tracks), dtype=bool)
    window_net_translation = np.full(len(tracks), np.nan)
    window_net_rotation = np.full(len(tracks), np.nan)
    window = max(2, args.stationary_window)
    for end_index in range(window - 1, len(tracks)):
        begin_index = end_index - window + 1
        if not all(
            statuses[index] == "ok" and inliers[index] >= 8
            for index in range(begin_index, end_index + 1)
        ):
            continue
        accumulated = np.eye(4, dtype=np.float64)
        for edge_index in range(begin_index, end_index + 1):
            accumulated = tracks[edge_index] @ accumulated
        begin_center = centers[begin_index]
        predicted_center = (
            accumulated[:3, :3] @ begin_center + accumulated[:3, 3]
        )
        net_translation = (
            np.linalg.norm(predicted_center - begin_center) * 1000.0
        )
        net_rotation = rotation_angle_deg(accumulated[:3, :3])
        window_net_translation[end_index] = net_translation
        window_net_rotation[end_index] = net_rotation
        local_translation = translation_filtered[
            begin_index : end_index + 1
        ]
        local_rotation = rotation_filtered[begin_index : end_index + 1]
        stationary_ready[end_index] = (
            np.nanmedian(local_translation) <= args.exit_translation_mm
            and np.nanmedian(local_rotation) <= args.exit_rotation_deg
            and net_translation <= args.exit_net_translation_mm
            and net_rotation <= args.exit_net_rotation_deg
        )
    edge_dynamic = hysteresis_motion(
        translation_filtered,
        rotation_filtered,
        args.enter_translation_mm,
        args.enter_rotation_deg,
        args.exit_translation_mm,
        args.exit_rotation_deg,
        args.enter_frames,
        1 if args.stationary_window > 1 else args.exit_frames,
        stationary_ready=stationary_ready,
        stationary_backfill=window,
    )
    frame_dynamic = np.zeros(len(poses), dtype=bool)
    frame_dynamic[0] = edge_dynamic[0]
    frame_dynamic[-1] = edge_dynamic[-1]
    frame_dynamic[1:-1] = edge_dynamic[:-1] | edge_dynamic[1:]

    output = json.loads(json.dumps(source))
    output_rows = output[rows_key]
    accepted_static_segments = []
    rejected_static_segments = []
    translation_corrections = np.zeros(len(poses))
    rotation_corrections = np.zeros(len(poses))
    for raw_begin, raw_end in contiguous_segments(frame_dynamic, False):
        begin = raw_begin + args.static_trim_frames
        end = raw_end - args.static_trim_frames
        if raw_begin == 0:
            begin = raw_begin
        if raw_end == len(poses) - 1:
            end = raw_end
        count = end - begin + 1
        segment_info = {
            "raw_output_frames": [raw_begin, raw_end],
            "output_frames": [begin, end],
            "original_frames": (
                [frame_ids[begin], frame_ids[end]] if count > 0 else None
            ),
            "num_frames": max(0, count),
        }
        if count < args.min_static_frames:
            segment_info["reason"] = "too_short_after_trim"
            rejected_static_segments.append(segment_info)
            continue
        segment_centers = centers[begin : end + 1]
        segment_rotations = poses[begin : end + 1, :3, :3]
        shared_center = np.median(segment_centers, axis=0)
        shared_rotation = Rotation.from_matrix(segment_rotations).mean().as_matrix()
        translation_distance = (
            np.linalg.norm(segment_centers - shared_center, axis=1) * 1000.0
        )
        rotation_distance = rotation_distances_deg(
            segment_rotations, shared_rotation
        )
        translation_mad = float(
            np.median(
                np.abs(translation_distance - np.median(translation_distance))
            )
        )
        rotation_mad = float(
            np.median(
                np.abs(rotation_distance - np.median(rotation_distance))
            )
        )
        translation_limit = (
            np.median(translation_distance) + 3.0 * max(translation_mad, 1.0)
        )
        rotation_limit = (
            np.median(rotation_distance) + 3.0 * max(rotation_mad, 0.5)
        )
        robust_inliers = (
            (translation_distance <= translation_limit)
            & (rotation_distance <= rotation_limit)
        )
        if robust_inliers.sum() >= max(3, count // 2):
            shared_center = np.median(
                segment_centers[robust_inliers], axis=0
            )
            shared_rotation = Rotation.from_matrix(
                segment_rotations[robust_inliers]
            ).mean().as_matrix()
            translation_distance = (
                np.linalg.norm(segment_centers - shared_center, axis=1) * 1000.0
            )
            rotation_distance = rotation_distances_deg(
                segment_rotations, shared_rotation
            )
        segment_info.update(
            {
                "robust_pose_inliers": int(robust_inliers.sum()),
                "translation_distance_mm": distribution(translation_distance),
                "rotation_distance_deg": distribution(rotation_distance),
                "translation_mad_mm": translation_mad,
                "rotation_mad_deg": rotation_mad,
            }
        )
        if (
            translation_mad > args.max_static_translation_mad_mm
            or rotation_mad > args.max_static_rotation_mad_deg
        ):
            segment_info["reason"] = "foundationpose_segment_inconsistent"
            rejected_static_segments.append(segment_info)
            continue
        for frame_index in range(begin, end + 1):
            pose = poses[frame_index].copy()
            pose[:3, 3] = shared_center
            pose[:3, :3] = shared_rotation
            output_rows[row_keys[frame_index]]["object_in_camera"] = pose.tolist()
            translation_corrections[frame_index] = (
                np.linalg.norm(shared_center - centers[frame_index]) * 1000.0
            )
            rotation_corrections[frame_index] = rotation_angle_deg(
                shared_rotation @ poses[frame_index, :3, :3].T
            )
        segment_info["reason"] = "accepted"
        accepted_static_segments.append(segment_info)

    dynamic_segments = [
        {
            "output_frames": [begin, end],
            "original_frames": [frame_ids[begin], frame_ids[end]],
            "num_frames": end - begin + 1,
        }
        for begin, end in contiguous_segments(frame_dynamic, True)
    ]
    output["tapir_motion_segmentation"] = {
        "source_foundationpose_json": str(source_path),
        "tapir_npz": str(Path(args.tapir_npz).expanduser().resolve()),
        "uses_gt_object_pose": False,
        "accepted_static_segments": accepted_static_segments,
        "dynamic_segments": dynamic_segments,
    }
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    audit = {
        "settings": vars(args),
        "source_foundationpose_json": str(source_path),
        "out_json": str(out_json),
        "num_frames": len(poses),
        "num_edges": len(tracks),
        "translation_speed_mm": distribution(translation_filtered),
        "rotation_speed_deg": distribution(rotation_filtered),
        "stationary_window_net_translation_mm": distribution(
            window_net_translation
        ),
        "stationary_window_net_rotation_deg": distribution(
            window_net_rotation
        ),
        "accepted_static_segments": accepted_static_segments,
        "rejected_static_segments": rejected_static_segments,
        "dynamic_segments": dynamic_segments,
        "translation_correction_mm": distribution(translation_corrections),
        "rotation_correction_deg": distribution(rotation_corrections),
        "edge_dynamic": edge_dynamic.astype(int).tolist(),
        "frame_dynamic": frame_dynamic.astype(int).tolist(),
        "per_edge": [
            {
                "pair": [index, index + 1],
                "original_pair": [frame_ids[index], frame_ids[index + 1]],
                "translation_mm": float(translation_speed[index]),
                "translation_filtered_mm": float(translation_filtered[index]),
                "rotation_deg": float(rotation_speed[index]),
                "rotation_filtered_deg": float(rotation_filtered[index]),
                "stationary_ready": bool(stationary_ready[index]),
                "window_net_translation_mm": (
                    float(window_net_translation[index])
                    if np.isfinite(window_net_translation[index])
                    else None
                ),
                "window_net_rotation_deg": (
                    float(window_net_rotation[index])
                    if np.isfinite(window_net_rotation[index])
                    else None
                ),
                "dynamic": bool(edge_dynamic[index]),
                "pnp_inliers": int(inliers[index]),
            }
            for index in range(len(tracks))
        ],
    }
    audit_path = Path(args.out_audit).expanduser().resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.out_plot:
        save_plot(
            Path(args.out_plot).expanduser().resolve(),
            translation_filtered,
            rotation_filtered,
            frame_dynamic,
            args,
        )
    print(json.dumps({key: value for key, value in audit.items() if key not in {"per_edge", "edge_dynamic", "frame_dynamic"}}, indent=2))


if __name__ == "__main__":
    main()
