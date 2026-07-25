#!/usr/bin/env python3
"""Locally refine FoundationPose with TAPIR PnP relative-motion constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--tapir-npz", required=True)
    parser.add_argument("--motion-audit-json", required=True)
    parser.add_argument("--segmentation-audit-json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-audit", required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--candidate-padding", type=int, default=2)
    parser.add_argument("--prior-translation-mm", type=float, default=12.0)
    parser.add_argument("--prior-rotation-deg", type=float, default=8.0)
    parser.add_argument("--edge-translation-mm", type=float, default=4.0)
    parser.add_argument("--edge-rotation-deg", type=float, default=3.0)
    parser.add_argument("--correction-smooth-mm", type=float, default=3.0)
    parser.add_argument("--correction-smooth-deg", type=float, default=2.0)
    parser.add_argument("--adaptive-smoothing", action="store_true")
    parser.add_argument("--low-speed-mm", type=float, default=4.0)
    parser.add_argument("--high-speed-mm", type=float, default=15.0)
    parser.add_argument("--low-speed-smooth-multiplier", type=float, default=2.5)
    parser.add_argument(
        "--free-end",
        action="store_true",
        help="Keep the first interval pose fixed but optimize the final pose.",
    )
    parser.add_argument("--candidate-edge-multiplier", type=float, default=2.0)
    parser.add_argument("--max-translation-mm", type=float, default=40.0)
    parser.add_argument("--max-rotation-deg", type=float, default=20.0)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--robust-scale", type=float, default=2.0)
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
    raise KeyError("FoundationPose JSON has no by_frame/frames dictionary")


def resolve_row(rows: dict, frame_id: str) -> tuple[str, dict] | None:
    for key in (frame_id, str(int(frame_id))):
        row = rows.get(key)
        if row is not None:
            return key, row
    return None


def rotation_error_vector(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = target @ prediction.T
    return Rotation.from_matrix(error).as_rotvec()


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


def evaluate_edges(
    centers: np.ndarray,
    rotations: np.ndarray,
    track_transforms: np.ndarray,
    edge_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    translation_errors = []
    rotation_errors = []
    for edge_index in edge_indices:
        track = track_transforms[edge_index]
        predicted_center = (
            track[:3, :3] @ centers[edge_index] + track[:3, 3]
        )
        predicted_rotation = track[:3, :3] @ rotations[edge_index]
        translation_errors.append(
            np.linalg.norm(centers[edge_index + 1] - predicted_center)
            * 1000.0
        )
        rotation_errors.append(
            np.degrees(
                np.linalg.norm(
                    rotation_error_vector(
                        rotations[edge_index + 1],
                        predicted_rotation,
                    )
                )
            )
        )
    return np.asarray(translation_errors), np.asarray(rotation_errors)


def main() -> None:
    args = parse_args()
    source_path = Path(args.foundationpose_json).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows_key, rows = load_pose_rows(source)
    frame_map = json.loads(
        Path(args.frame_map_json).expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    motion_audit = json.loads(
        Path(args.motion_audit_json).expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    frame_ids = [
        normalize_frame(row["original_frame"]) for row in frame_map["frames"]
    ]
    poses = []
    row_keys = []
    for frame_id in frame_ids:
        resolved = resolve_row(rows, frame_id)
        if resolved is None:
            raise KeyError(f"No FoundationPose row for frame {frame_id}")
        row_key, row = resolved
        pose = np.asarray(row.get("object_in_camera"), dtype=np.float64)
        if pose.size != 16 or not np.isfinite(pose).all():
            raise ValueError(f"Invalid object pose for frame {frame_id}")
        row_keys.append(row_key)
        poses.append(pose.reshape(4, 4))
    poses = np.stack(poses)

    with np.load(
        Path(args.tapir_npz).expanduser().resolve(), allow_pickle=True
    ) as payload:
        track_transforms = np.asarray(
            payload["relative_transform_pnp"], dtype=np.float64
        )
        pnp_status = np.asarray(payload["pnp_status"]).astype(str)
        pnp_inliers = np.asarray(payload["pnp_inlier_mask"]).sum(axis=1)
    if len(track_transforms) != len(poses) - 1:
        raise ValueError(
            f"Track/pose mismatch: tracks={len(track_transforms)} poses={len(poses)}"
        )

    candidates = motion_audit.get("boundary_candidates", [])
    candidate_indices = {
        int(row["pair_index"]) for row in candidates
    }
    segmentation = None
    if args.segmentation_audit_json:
        segmentation = json.loads(
            Path(args.segmentation_audit_json)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )
    dynamic_segments = (
        segmentation.get("dynamic_segments", []) if segmentation else []
    )
    if args.start_frame is None and dynamic_segments:
        start = max(0, int(dynamic_segments[0]["output_frames"][0]) - 1)
    elif args.start_frame is None:
        start = max(
            0,
            min(candidate_indices) - args.candidate_padding,
        ) if candidate_indices else 0
    else:
        start = args.start_frame
    if args.end_frame is None and dynamic_segments:
        end = int(dynamic_segments[-1]["output_frames"][1])
    elif args.end_frame is None:
        end = min(
            len(poses) - 1,
            max(candidate_indices) + 1 + args.candidate_padding,
        ) if candidate_indices else len(poses) - 1
    else:
        end = args.end_frame
    if not 0 <= start < end < len(poses):
        raise ValueError(
            f"Invalid optimization interval [{start}, {end}] for {len(poses)} poses"
        )

    base_centers = poses[:, :3, 3].copy()
    base_rotations = poses[:, :3, :3].copy()
    variable_indices = np.arange(
        start + 1,
        end + 1 if args.free_end else end,
    )
    if not len(variable_indices):
        raise ValueError("Optimization interval has no interior frames")
    variable_lookup = {
        int(frame_index): local_index
        for local_index, frame_index in enumerate(variable_indices)
    }
    edge_indices = np.arange(start, end)
    valid_edges = np.asarray(
        [
            index
            for index in edge_indices
            if pnp_status[index] == "ok" and pnp_inliers[index] >= 8
        ],
        dtype=np.int64,
    )
    if not len(valid_edges):
        raise RuntimeError("No valid TAPIR PnP edges in optimization interval")
    track_center_speed = np.full(len(track_transforms), np.nan)
    for edge_index in valid_edges:
        track = track_transforms[edge_index]
        predicted_center = (
            track[:3, :3] @ base_centers[edge_index]
            + track[:3, 3]
        )
        track_center_speed[edge_index] = (
            np.linalg.norm(predicted_center - base_centers[edge_index])
            * 1000.0
        )

    translation_prior_scale = args.prior_translation_mm / 1000.0
    rotation_prior_scale = np.radians(args.prior_rotation_deg)
    translation_edge_scale = args.edge_translation_mm / 1000.0
    rotation_edge_scale = np.radians(args.edge_rotation_deg)
    translation_smooth_scale = args.correction_smooth_mm / 1000.0
    rotation_smooth_scale = np.radians(args.correction_smooth_deg)

    def decode(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        corrections = parameters.reshape(len(variable_indices), 6)
        centers = base_centers.copy()
        rotations = base_rotations.copy()
        for frame_index, local_index in variable_lookup.items():
            centers[frame_index] += corrections[local_index, :3]
            delta_rotation = Rotation.from_rotvec(
                corrections[local_index, 3:]
            ).as_matrix()
            rotations[frame_index] = (
                delta_rotation @ base_rotations[frame_index]
            )
        return centers, rotations

    def residuals(parameters: np.ndarray) -> np.ndarray:
        corrections = parameters.reshape(len(variable_indices), 6)
        centers, rotations = decode(parameters)
        residual = []
        for local_index, frame_index in enumerate(variable_indices):
            residual.extend(
                corrections[local_index, :3]
                / translation_prior_scale
            )
            residual.extend(
                corrections[local_index, 3:]
                / rotation_prior_scale
            )
        for edge_index in valid_edges:
            track = track_transforms[edge_index]
            predicted_center = (
                track[:3, :3] @ centers[edge_index] + track[:3, 3]
            )
            predicted_rotation = (
                track[:3, :3] @ rotations[edge_index]
            )
            multiplier = (
                args.candidate_edge_multiplier
                if edge_index in candidate_indices
                else 1.0
            )
            residual.extend(
                (centers[edge_index + 1] - predicted_center)
                / translation_edge_scale
                * multiplier
            )
            residual.extend(
                rotation_error_vector(
                    rotations[edge_index + 1],
                    predicted_rotation,
                )
                / rotation_edge_scale
                * multiplier
            )
        interval_corrections = np.zeros((end - start + 1, 6))
        for frame_index, local_index in variable_lookup.items():
            interval_corrections[frame_index - start] = corrections[local_index]
        for local_index in range(1, len(interval_corrections) - 1):
            global_frame = start + local_index
            local_speed = np.nanmedian(
                track_center_speed[
                    max(start, global_frame - 1) : min(end, global_frame + 1)
                ]
            )
            smooth_multiplier = 1.0
            if args.adaptive_smoothing and np.isfinite(local_speed):
                denominator = max(
                    args.high_speed_mm - args.low_speed_mm,
                    1e-6,
                )
                motion_alpha = np.clip(
                    (local_speed - args.low_speed_mm) / denominator,
                    0.0,
                    1.0,
                )
                smooth_multiplier = (
                    args.low_speed_smooth_multiplier * (1.0 - motion_alpha)
                    + motion_alpha
                )
            residual.extend(
                (
                    interval_corrections[local_index + 1, :3]
                    - 2.0 * interval_corrections[local_index, :3]
                    + interval_corrections[local_index - 1, :3]
                )
                / translation_smooth_scale
                * smooth_multiplier
            )
            residual.extend(
                (
                    interval_corrections[local_index + 1, 3:]
                    - 2.0 * interval_corrections[local_index, 3:]
                    + interval_corrections[local_index - 1, 3:]
                )
                / rotation_smooth_scale
                * smooth_multiplier
            )
        return np.asarray(residual, dtype=np.float64)

    initial = np.zeros((len(variable_indices), 6), dtype=np.float64)
    translation_limit = args.max_translation_mm / 1000.0
    rotation_limit = np.radians(args.max_rotation_deg)
    lower = np.tile(
        [-translation_limit] * 3 + [-rotation_limit] * 3,
        len(variable_indices),
    )
    upper = -lower
    result = least_squares(
        residuals,
        initial.reshape(-1),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=args.robust_scale,
        max_nfev=args.max_nfev,
        verbose=1,
    )
    fitted_centers, fitted_rotations = decode(result.x)
    corrections = result.x.reshape(len(variable_indices), 6)

    output = json.loads(json.dumps(source))
    output_rows = output[rows_key]
    for frame_index in variable_indices:
        pose = poses[frame_index].copy()
        pose[:3, :3] = fitted_rotations[frame_index]
        pose[:3, 3] = fitted_centers[frame_index]
        output_rows[row_keys[frame_index]]["object_in_camera"] = pose.tolist()
    output["tapir_pose_graph"] = {
        "source_foundationpose_json": str(source_path),
        "tapir_npz": str(Path(args.tapir_npz).expanduser().resolve()),
        "motion_audit_json": str(
            Path(args.motion_audit_json).expanduser().resolve()
        ),
        "optimization_interval": [start, end],
        "free_end": args.free_end,
        "uses_gt_object_pose": False,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    before_translation, before_rotation = evaluate_edges(
        base_centers, base_rotations, track_transforms, valid_edges
    )
    after_translation, after_rotation = evaluate_edges(
        fitted_centers, fitted_rotations, track_transforms, valid_edges
    )
    correction_translation = (
        np.linalg.norm(corrections[:, :3], axis=1) * 1000.0
    )
    correction_rotation = (
        np.linalg.norm(corrections[:, 3:], axis=1) * 180.0 / np.pi
    )
    audit = {
        "settings": vars(args),
        "source_foundationpose_json": str(source_path),
        "out_json": str(out_path),
        "optimization_interval": [start, end],
        "fixed_frames": [start] if args.free_end else [start, end],
        "optimized_original_frames": [frame_ids[start], frame_ids[end]],
        "candidate_pair_indices": sorted(candidate_indices),
        "valid_edge_count": int(len(valid_edges)),
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": result.message,
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
        },
        "tapir_edge_center_error_mm": {
            "before": distribution(before_translation),
            "after": distribution(after_translation),
        },
        "tapir_edge_rotation_error_deg": {
            "before": distribution(before_rotation),
            "after": distribution(after_rotation),
        },
        "translation_correction_mm": distribution(correction_translation),
        "rotation_correction_deg": distribution(correction_rotation),
        "per_frame": [
            {
                "output_frame": int(frame_index),
                "original_frame": frame_ids[frame_index],
                "translation_correction_mm": float(
                    correction_translation[local_index]
                ),
                "rotation_correction_deg": float(
                    correction_rotation[local_index]
                ),
            }
            for local_index, frame_index in enumerate(variable_indices)
        ],
    }
    audit_path = Path(args.out_audit).expanduser().resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
