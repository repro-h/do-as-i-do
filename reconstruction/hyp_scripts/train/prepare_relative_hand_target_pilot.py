#!/usr/bin/env python3
"""Prepare one-stream hand targets relative to the current object pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PALM = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--v8-prediction-npz", required=True)
    parser.add_argument("--raw-hand-meshes", required=True)
    parser.add_argument("--v8-hand-meshes", required=True)
    parser.add_argument("--gt-hand-meshes", required=True)
    parser.add_argument("--filtered-object-json", required=True)
    parser.add_argument("--gt-object-json", required=True)
    parser.add_argument("--gt-ycb-object-mesh", default=None)
    parser.add_argument(
        "--canonical-alignment-json",
        default=None,
        help=(
            "Optional SAM-to-YCB canonical alignment. When provided, "
            "--gt-object-json is interpreted as a DexYCB YCB layout and "
            "converted into the SAM canonical frame before hand transfer."
        ),
    )
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--object-mesh-scale", type=float, required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--transform-mode",
        choices=("translation_only", "full_se3"),
        default="translation_only",
    )
    parser.add_argument(
        "--symmetry-axis",
        choices=("none", "x", "y", "z"),
        default="none",
    )
    parser.add_argument(
        "--symmetry-step-deg",
        type=float,
        default=15.0,
    )
    parser.add_argument("--symmetry-axis-flip", action="store_true")
    parser.add_argument(
        "--symmetry-selection-mode",
        choices=("sequence", "temporal"),
        default="sequence",
    )
    parser.add_argument(
        "--symmetry-yaw-transition-mm-per-deg",
        type=float,
        default=0.2,
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = load_json(path)
    if payload.get("objects") is not None:
        output = {}
        for index, row in enumerate(payload["objects"]):
            local = row.get("local_to_scene") or {}
            quaternion = local.get("quat_wxyz_camera_frame")
            translation = local.get("translation_camera_frame")
            if quaternion is None or translation is None:
                continue
            frame = str(
                row.get("frame_idx", row.get("frame_index", index))
            ).zfill(6)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = quaternion_wxyz_to_matrix(quaternion)
            matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
            if np.isfinite(matrix).all():
                output[frame] = matrix
        return output
    rows = payload.get("by_frame") or payload.get("frames") or {}
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    output = {}
    for key, row in iterator:
        if not isinstance(row, dict) or row.get("object_in_camera") is None:
            continue
        frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
        matrix = np.asarray(row["object_in_camera"], dtype=np.float64).reshape(4, 4)
        if np.isfinite(matrix).all():
            output[frame] = matrix
    return output


def quaternion_wxyz_to_matrix(value) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=np.float64).reshape(4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("Zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sam_to_ycb_rigid(path: Path) -> tuple[np.ndarray, float | None]:
    payload = load_json(path)
    similarity = payload.get("raw_sam_to_ycb_similarity")
    if not isinstance(similarity, dict):
        raise KeyError("raw_sam_to_ycb_similarity")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        similarity["rotation"], dtype=np.float64
    ).reshape(3, 3)
    matrix[:3, 3] = np.asarray(
        similarity["translation_m"], dtype=np.float64
    ).reshape(3)
    production = payload.get("production_sam_to_ycb_similarity") or {}
    residual_scale = production.get("residual_scale")
    return matrix, (
        float(residual_scale) if residual_scale is not None else None
    )


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


def relative_transform(
    source: np.ndarray, target: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "translation_only":
        return np.eye(3), target[:3, 3] - source[:3, 3]
    rotation = target[:3, :3] @ source[:3, :3].T
    translation = target[:3, 3] - rotation @ source[:3, 3]
    return rotation, translation


def apply_transform(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return points @ rotation.T + translation


def axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    index = {"x": 0, "y": 1, "z": 2}[axis]
    first = (index + 1) % 3
    second = (index + 2) % 3
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    matrix = np.eye(3, dtype=np.float64)
    matrix[first, first] = cosine
    matrix[second, second] = cosine
    matrix[first, second] = -sine
    matrix[second, first] = sine
    return matrix


def symmetry_candidates(
    axis: str, step_deg: float, allow_flip: bool
) -> list[dict]:
    if axis == "none":
        return [{"angle_deg": 0.0, "flipped": False, "matrix": np.eye(4)}]
    if not 0.0 < step_deg <= 360.0:
        raise ValueError("symmetry-step-deg must be in (0, 360]")
    count = max(1, int(round(360.0 / step_deg)))
    angles = np.linspace(0.0, 360.0, count, endpoint=False)
    flip = np.eye(3, dtype=np.float64)
    if allow_flip:
        perpendicular = "xyz"[("xyz".index(axis) + 1) % 3]
        flip = axis_rotation(perpendicular, np.pi)
    candidates = []
    for flipped in ([False, True] if allow_flip else [False]):
        for angle in angles:
            rotation = axis_rotation(axis, np.radians(angle))
            if flipped:
                rotation = rotation @ flip
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rotation
            candidates.append(
                {
                    "angle_deg": float(angle),
                    "flipped": bool(flipped),
                    "matrix": matrix,
                }
            )
    return candidates


def transform_between_poses(
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    symmetry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = target_pose @ symmetry @ np.linalg.inv(source_pose)
    return transform[:3, :3], transform[:3, 3]


def symmetry_data_costs(
    candidates: list[dict],
    frame_ids: np.ndarray,
    gt_rows: dict[str, np.ndarray],
    filtered_pose: np.ndarray,
    gt_joints: np.ndarray,
    supervision_valid: np.ndarray,
    normalized_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(frame_ids), len(filtered_pose))
    costs = np.full((count, len(candidates)), np.inf, dtype=np.float64)
    counts = np.zeros((count, len(candidates)), dtype=np.int32)
    for candidate_index, candidate in enumerate(candidates):
        symmetry = np.asarray(candidate["matrix"], dtype=np.float64)
        if normalized_left:
            symmetry = mirror_pose(symmetry)
        for index in range(count):
            if not supervision_valid[index]:
                continue
            frame = frame_string(frame_ids[index], index)
            gt_pose = gt_rows.get(frame)
            if gt_pose is None:
                continue
            if normalized_left:
                gt_pose = mirror_pose(gt_pose)
            rotation, translation = transform_between_poses(
                gt_pose, filtered_pose[index], symmetry
            )
            target = apply_transform(
                gt_joints[index, PALM], rotation, translation
            )
            errors = np.linalg.norm(
                target - gt_joints[index, PALM], axis=-1
            )
            errors = errors[np.isfinite(errors)]
            if len(errors):
                costs[index, candidate_index] = float(
                    np.median(errors) * 1000.0
                )
                counts[index, candidate_index] = len(errors)
    return costs, counts


def select_sequence_symmetry(
    candidates: list[dict], costs: np.ndarray
) -> tuple[np.ndarray, dict, list[dict]]:
    scored = []
    for candidate_index, candidate in enumerate(candidates):
        values = costs[:, candidate_index]
        values = values[np.isfinite(values)]
        score = float(np.median(values)) if len(values) else np.inf
        scored.append(
            {
                "candidate_index": candidate_index,
                "angle_deg": candidate["angle_deg"],
                "flipped": candidate["flipped"],
                "palm_3d_median_mm": score,
                "num_values": int(len(values)),
                "matrix": candidate["matrix"],
            }
        )
    scored.sort(key=lambda row: row["palm_3d_median_mm"])
    if not scored or not np.isfinite(scored[0]["palm_3d_median_mm"]):
        raise RuntimeError("No valid sequence symmetry candidate")
    selected = np.full(
        len(costs), scored[0]["candidate_index"], dtype=np.int64
    )
    audit = {
        "mode": "sequence",
        "selected_angle_deg": scored[0]["angle_deg"],
        "selected_flipped": scored[0]["flipped"],
        "selected_palm_3d_median_mm": scored[0]["palm_3d_median_mm"],
    }
    return selected, audit, scored


def circular_angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.abs(first[:, None] - second[None, :])
    return np.minimum(difference, 360.0 - difference)


def select_temporal_symmetry(
    candidates: list[dict],
    costs: np.ndarray,
    transition_mm_per_deg: float,
) -> tuple[np.ndarray, dict, list[dict]]:
    valid_frames = np.flatnonzero(np.isfinite(costs).any(axis=1))
    if not len(valid_frames):
        raise RuntimeError("No valid temporal symmetry frames")
    solutions = []
    for flipped in sorted({bool(row["flipped"]) for row in candidates}):
        states = np.asarray(
            [
                index for index, row in enumerate(candidates)
                if bool(row["flipped"]) == flipped
            ],
            dtype=np.int64,
        )
        angles = np.asarray(
            [candidates[index]["angle_deg"] for index in states],
            dtype=np.float64,
        )
        transition = (
            circular_angle_difference(angles, angles)
            * transition_mm_per_deg
        )
        back = np.full((len(valid_frames), len(states)), -1, dtype=np.int64)
        previous = costs[valid_frames[0], states].copy()
        for step, frame in enumerate(valid_frames[1:], start=1):
            total = previous[:, None] + transition
            back[step] = np.argmin(total, axis=0)
            previous = costs[frame, states] + np.min(total, axis=0)
        state = int(np.argmin(previous))
        selected_states = np.full(len(valid_frames), state, dtype=np.int64)
        for step in range(len(valid_frames) - 1, 0, -1):
            selected_states[step - 1] = back[step, selected_states[step]]
        selected_candidates = states[selected_states]
        solutions.append(
            {
                "flipped": flipped,
                "total_cost": float(previous[state]),
                "selected": selected_candidates,
            }
        )
    solution = min(solutions, key=lambda row: row["total_cost"])
    selected = np.full(len(costs), int(solution["selected"][0]), dtype=np.int64)
    selected[valid_frames] = solution["selected"]
    for frame in range(valid_frames[0] - 1, -1, -1):
        selected[frame] = selected[frame + 1]
    for frame in range(valid_frames[0] + 1, len(selected)):
        if frame not in set(valid_frames.tolist()):
            selected[frame] = selected[frame - 1]

    selected_costs = costs[valid_frames, selected[valid_frames]]
    selected_angles = np.asarray(
        [candidates[index]["angle_deg"] for index in selected],
        dtype=np.float64,
    )
    steps = np.minimum(
        np.abs(np.diff(selected_angles)),
        360.0 - np.abs(np.diff(selected_angles)),
    )
    candidate_scores = []
    for index, candidate in enumerate(candidates):
        values = costs[:, index]
        values = values[np.isfinite(values)]
        candidate_scores.append(
            {
                "candidate_index": index,
                "angle_deg": candidate["angle_deg"],
                "flipped": candidate["flipped"],
                "palm_3d_median_mm": (
                    float(np.median(values)) if len(values) else np.inf
                ),
                "num_values": int(len(values)),
                "matrix": candidate["matrix"],
            }
        )
    candidate_scores.sort(key=lambda row: row["palm_3d_median_mm"])
    audit = {
        "mode": "temporal",
        "selected_flipped": bool(solution["flipped"]),
        "selected_palm_3d_median_mm": float(np.median(selected_costs)),
        "selected_angle_min_deg": float(np.min(selected_angles)),
        "selected_angle_max_deg": float(np.max(selected_angles)),
        "selected_angle_step_median_deg": (
            float(np.median(steps)) if len(steps) else 0.0
        ),
        "selected_angle_step_max_deg": (
            float(np.max(steps)) if len(steps) else 0.0
        ),
        "total_cost": solution["total_cost"],
    }
    return selected, audit, candidate_scores


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


def main() -> None:
    args = parse_args()
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "supervision": args.supervision_npz,
            "prediction": args.v8_prediction_npz,
            "raw_hand_meshes": args.raw_hand_meshes,
            "v8_hand_meshes": args.v8_hand_meshes,
            "gt_hand_meshes": args.gt_hand_meshes,
            "filtered_object_json": args.filtered_object_json,
            "gt_object_json": args.gt_object_json,
            "gt_ycb_object_mesh": args.gt_ycb_object_mesh,
            "canonical_alignment_json": args.canonical_alignment_json,
            "object_mesh": args.object_mesh,
        }.items()
        if value is not None
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    supervision = load_npz(paths["supervision"])
    prediction = load_npz(paths["prediction"])
    raw_meshes = load_npz(paths["raw_hand_meshes"])
    v8_meshes = load_npz(paths["v8_hand_meshes"])
    gt_meshes = load_npz(paths["gt_hand_meshes"])
    filtered_rows = pose_rows(paths["filtered_object_json"])
    gt_rows = pose_rows(paths["gt_object_json"])
    canonical_transform = None
    canonical_residual_scale = None
    if "canonical_alignment_json" in paths:
        canonical_transform, canonical_residual_scale = sam_to_ycb_rigid(
            paths["canonical_alignment_json"]
        )
        gt_rows = {
            frame: pose @ canonical_transform
            for frame, pose in gt_rows.items()
        }

    side = args.hand_side
    gt_vertices = np.asarray(gt_meshes[f"{side}_vertices"], dtype=np.float64)
    gt_faces = np.asarray(gt_meshes[f"{side}_faces"], dtype=np.int64)
    gt_mesh_valid = np.asarray(gt_meshes[f"{side}_valid"], dtype=bool)
    raw_vertices = np.asarray(raw_meshes[f"{side}_vertices"], dtype=np.float64)
    v8_vertices = np.asarray(v8_meshes[f"{side}_vertices"], dtype=np.float64)

    handflow_joints = np.asarray(supervision["pred_joints_3d"], dtype=np.float64)
    gt_joints = np.asarray(supervision["gt_joints_3d"], dtype=np.float64)
    gt_joints_2d = np.asarray(supervision["gt_joints_2d"], dtype=np.float64)
    fp_pose_normalized = np.asarray(supervision["object_pose"], dtype=np.float64)
    intrinsics = np.asarray(supervision["intrinsics"], dtype=np.float64)
    supervision_valid = np.asarray(supervision["supervision_valid"], dtype=bool)
    gt_joint_valid = np.asarray(
        supervision.get("gt_valid", supervision_valid), dtype=bool
    )
    handflow_valid = np.asarray(
        supervision.get("hand_valid", supervision_valid), dtype=bool
    )
    normalized_left = bool(np.asarray(supervision["normalized_left"]).item())
    frame_ids = np.asarray(supervision["frame_ids"])
    v8_correction = np.asarray(
        prediction["pi3x_translation_normalized"], dtype=np.float64
    )
    v8_predicted = np.asarray(prediction["pi3x_depth_predicted"], dtype=bool)

    count = min(
        len(frame_ids), len(handflow_joints), len(gt_joints),
        len(fp_pose_normalized), len(v8_correction), len(gt_vertices),
        len(raw_vertices), len(v8_vertices), len(gt_mesh_valid),
        len(gt_joint_valid), len(handflow_valid),
    )
    candidates = symmetry_candidates(
        args.symmetry_axis,
        args.symmetry_step_deg,
        args.symmetry_axis_flip,
    )
    symmetry_costs, _ = symmetry_data_costs(
        candidates,
        frame_ids[:count],
        gt_rows,
        fp_pose_normalized[:count],
        gt_joints[:count],
        gt_joint_valid[:count],
        normalized_left,
    )
    if args.symmetry_selection_mode == "temporal":
        selected_symmetry_indices, symmetry_selection, symmetry_scores = (
            select_temporal_symmetry(
                candidates,
                symmetry_costs,
                args.symmetry_yaw_transition_mm_per_deg,
            )
        )
    else:
        selected_symmetry_indices, symmetry_selection, symmetry_scores = (
            select_sequence_symmetry(candidates, symmetry_costs)
        )
    selected_symmetry_angles = np.asarray(
        [
            candidates[index]["angle_deg"]
            for index in selected_symmetry_indices
        ],
        dtype=np.float32,
    )
    selected_symmetry_flipped = np.asarray(
        [
            candidates[index]["flipped"]
            for index in selected_symmetry_indices
        ],
        dtype=bool,
    )
    target_vertices = np.full_like(gt_vertices[:count], np.nan)
    target_joints = np.full_like(gt_joints[:count], np.nan)
    raw_delta_camera = np.full((count, 3), np.nan, dtype=np.float64)
    v8_delta_camera = np.full((count, 3), np.nan, dtype=np.float64)
    raw_delta_object = np.full((count, 3), np.nan, dtype=np.float64)
    v8_delta_object = np.full((count, 3), np.nan, dtype=np.float64)
    target_2d_error = np.full(count, np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    joint_metric_valid = np.zeros(count, dtype=bool)
    raw_valid = np.zeros(count, dtype=bool)
    v8_valid = np.zeros(count, dtype=bool)
    frame_rows = []

    for index in range(count):
        symmetry_original = np.asarray(
            candidates[selected_symmetry_indices[index]]["matrix"],
            dtype=np.float64,
        )
        symmetry_normalized = (
            mirror_pose(symmetry_original)
            if normalized_left
            else symmetry_original
        )
        frame = frame_string(frame_ids[index], index)
        filtered_pose = filtered_rows.get(frame)
        gt_pose_original = gt_rows.get(frame)
        invalid_reasons = []
        if filtered_pose is None:
            invalid_reasons.append("missing_filtered_object_pose")
        if gt_pose_original is None:
            invalid_reasons.append("missing_gt_object_pose")
        if not gt_mesh_valid[index]:
            invalid_reasons.append("invalid_gt_hand_mesh")
        has_joint_supervision = bool(
            gt_joint_valid[index]
            and np.isfinite(gt_joints[index]).all()
            and np.isfinite(gt_joints_2d[index]).all()
        )
        if not has_joint_supervision:
            metric_unavailable_reasons = ["invalid_gt_joint_supervision"]
        else:
            metric_unavailable_reasons = []
        if invalid_reasons:
            frame_rows.append({
                "frame": frame,
                "valid": False,
                "invalid_reasons": invalid_reasons,
                "v8_predicted": bool(v8_predicted[index]),
            })
            continue

        if args.transform_mode == "full_se3":
            mesh_rotation, mesh_translation = transform_between_poses(
                gt_pose_original, filtered_pose, symmetry_original
            )
        else:
            mesh_rotation, mesh_translation = relative_transform(
                gt_pose_original, filtered_pose, args.transform_mode
            )
        target_vertices[index] = apply_transform(
            gt_vertices[index], mesh_rotation, mesh_translation
        )
        valid[index] = True

        if has_joint_supervision:
            gt_pose_normalized = (
                mirror_pose(gt_pose_original)
                if normalized_left
                else gt_pose_original
            )
            if args.transform_mode == "full_se3":
                joint_rotation, joint_translation = transform_between_poses(
                    gt_pose_normalized,
                    fp_pose_normalized[index],
                    symmetry_normalized,
                )
            else:
                joint_rotation, joint_translation = relative_transform(
                    gt_pose_normalized,
                    fp_pose_normalized[index],
                    args.transform_mode,
                )
            target_joints[index] = apply_transform(
                gt_joints[index], joint_rotation, joint_translation
            )
            joint_metric_valid[index] = True
            rotation_fp = fp_pose_normalized[index, :3, :3]
            if handflow_valid[index] and np.isfinite(handflow_joints[index]).all():
                raw_delta_camera[index] = np.median(
                    target_joints[index, PALM] - handflow_joints[index, PALM],
                    axis=0,
                )
                raw_delta_object[index] = rotation_fp.T @ raw_delta_camera[index]
                raw_valid[index] = True
            if raw_valid[index] and v8_predicted[index]:
                v8_joints = handflow_joints[index] + v8_correction[index, None]
                v8_delta_camera[index] = np.median(
                    target_joints[index, PALM] - v8_joints[PALM], axis=0
                )
                v8_delta_object[index] = rotation_fp.T @ v8_delta_camera[index]
                v8_valid[index] = True

            projected = target_joints[index, PALM].copy()
            z = np.maximum(projected[:, 2], 1e-6)
            uv = np.stack(
                [
                    intrinsics[0, 0] * projected[:, 0] / z + intrinsics[0, 2],
                    intrinsics[1, 1] * projected[:, 1] / z + intrinsics[1, 2],
                ],
                axis=-1,
            )
            target_2d_error[index] = np.median(
                np.linalg.norm(uv - gt_joints_2d[index, PALM], axis=-1)
            )
        frame_rows.append({
            "frame": frame,
            "valid": True,
            "invalid_reasons": [],
            "joint_metrics_available": bool(joint_metric_valid[index]),
            "metric_unavailable_reasons": metric_unavailable_reasons,
            "raw_available": bool(raw_valid[index]),
            "v8_predicted": bool(v8_predicted[index]),
            "raw_target_translation_mm": (
                np.linalg.norm(raw_delta_camera[index]) * 1000.0
                if raw_valid[index]
                else None
            ),
            "v8_target_translation_mm": (
                np.linalg.norm(v8_delta_camera[index]) * 1000.0
                if v8_valid[index]
                else None
            ),
            "target_2d_palm_error_px": target_2d_error[index],
        })

    target_mesh_path = out_dir / "relative_target_hand_meshes.npz"
    inactive_vertices = np.zeros_like(target_vertices)
    inactive_valid = np.zeros(count, dtype=bool)
    mesh_payload = {
        "right_faces": gt_faces,
        "left_faces": gt_faces.copy(),
        "source": np.asarray("gt_hand_transferred_to_filtered_object"),
        "transform_mode": np.asarray(args.transform_mode),
    }
    if side == "right":
        mesh_payload.update(
            right_vertices=target_vertices.astype(np.float32),
            right_valid=valid,
            left_vertices=inactive_vertices.astype(np.float32),
            left_valid=inactive_valid,
        )
    else:
        mesh_payload.update(
            right_vertices=inactive_vertices.astype(np.float32),
            right_valid=inactive_valid,
            left_vertices=target_vertices.astype(np.float32),
            left_valid=valid,
        )
    np.savez_compressed(target_mesh_path, **mesh_payload)

    supervision_out = out_dir / "relative_hand_target_supervision.npz"
    np.savez_compressed(
        supervision_out,
        frame_ids=frame_ids[:count],
        valid=valid,
        joint_metric_valid=joint_metric_valid,
        raw_valid=raw_valid,
        v8_valid=v8_valid,
        target_joints_3d=target_joints.astype(np.float32),
        raw_target_translation_camera=raw_delta_camera.astype(np.float32),
        raw_target_translation_object=raw_delta_object.astype(np.float32),
        v8_target_translation_camera=v8_delta_camera.astype(np.float32),
        v8_target_translation_object=v8_delta_object.astype(np.float32),
        target_2d_palm_error_px=target_2d_error.astype(np.float32),
        selected_symmetry_angle_deg=selected_symmetry_angles,
        selected_symmetry_flipped=selected_symmetry_flipped,
        selected_symmetry_candidate_index=selected_symmetry_indices,
        transform_mode=np.asarray(args.transform_mode),
        normalized_left=np.asarray(normalized_left),
    )

    raw_magnitude = (
        np.linalg.norm(raw_delta_camera[raw_valid], axis=-1) * 1000.0
    )
    v8_magnitude = (
        np.linalg.norm(v8_delta_camera[v8_valid], axis=-1) * 1000.0
    )
    audit = {
        "supervision": str(paths["supervision"]),
        "prediction": str(paths["prediction"]),
        "raw_hand_meshes": str(paths["raw_hand_meshes"]),
        "v8_hand_meshes": str(paths["v8_hand_meshes"]),
        "gt_hand_meshes": str(paths["gt_hand_meshes"]),
        "target_hand_meshes": str(target_mesh_path),
        "target_supervision": str(supervision_out),
        "filtered_object_json": str(paths["filtered_object_json"]),
        "gt_object_json": str(paths["gt_object_json"]),
        "gt_ycb_object_mesh": (
            str(paths["gt_ycb_object_mesh"])
            if "gt_ycb_object_mesh" in paths
            else None
        ),
        "canonical_alignment_json": (
            str(paths["canonical_alignment_json"])
            if "canonical_alignment_json" in paths
            else None
        ),
        "sam_to_ycb_rigid": (
            canonical_transform.tolist()
            if canonical_transform is not None
            else None
        ),
        "canonical_residual_scale": canonical_residual_scale,
        "canonical_scale_warning": bool(
            canonical_residual_scale is not None
            and abs(canonical_residual_scale - 1.0) > 0.1
        ),
        "object_mesh": str(paths["object_mesh"]),
        "object_mesh_scale": args.object_mesh_scale,
        "hand_side": side,
        "transform_mode": args.transform_mode,
        "symmetry": {
            "axis": args.symmetry_axis,
            "step_deg": args.symmetry_step_deg,
            "allow_axis_flip": args.symmetry_axis_flip,
            "selection_mode": args.symmetry_selection_mode,
            "yaw_transition_mm_per_deg": (
                args.symmetry_yaw_transition_mm_per_deg
            ),
            **symmetry_selection,
            "selected_angle_deg_by_frame": (
                selected_symmetry_angles.astype(float).tolist()
            ),
            "selected_flipped_by_frame": (
                selected_symmetry_flipped.tolist()
            ),
            "top_candidates": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "matrix"
                }
                for row in symmetry_scores[:10]
            ],
        },
        "num_frames": count,
        "num_valid": int(valid.sum()),
        "num_joint_metric_valid": int(joint_metric_valid.sum()),
        "num_raw_valid": int(raw_valid.sum()),
        "num_v8_valid": int(v8_valid.sum()),
        "raw_target_translation": distribution(raw_magnitude),
        "v8_target_translation": distribution(v8_magnitude),
        "target_2d_palm_error_px": distribution(
            target_2d_error[valid], "px"
        ),
        "frames": frame_rows,
    }
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote: {audit_path}")
    print(f"valid: {int(valid.sum())}/{count}")
    print("raw target translation:", audit["raw_target_translation"])
    print("v8 target translation:", audit["v8_target_translation"])
    print("target 2D palm error:", audit["target_2d_palm_error_px"])
    print("selected symmetry:", audit["symmetry"])


if __name__ == "__main__":
    main()
