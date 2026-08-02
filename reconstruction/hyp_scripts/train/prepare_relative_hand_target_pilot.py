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
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--object-mesh-scale", type=float, required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--transform-mode",
        choices=("translation_only", "full_se3"),
        default="translation_only",
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
            "object_mesh": args.object_mesh,
        }.items()
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
    )
    target_vertices = np.full_like(gt_vertices[:count], np.nan)
    target_joints = np.full_like(gt_joints[:count], np.nan)
    raw_delta_camera = np.full((count, 3), np.nan, dtype=np.float64)
    v8_delta_camera = np.full((count, 3), np.nan, dtype=np.float64)
    raw_delta_object = np.full((count, 3), np.nan, dtype=np.float64)
    v8_delta_object = np.full((count, 3), np.nan, dtype=np.float64)
    target_2d_error = np.full(count, np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    frame_rows = []

    for index in range(count):
        frame = frame_string(frame_ids[index], index)
        filtered_pose = filtered_rows.get(frame)
        gt_pose_original = gt_rows.get(frame)
        if (
            filtered_pose is None
            or gt_pose_original is None
            or not gt_mesh_valid[index]
            or not supervision_valid[index]
            or not v8_predicted[index]
        ):
            continue

        mesh_rotation, mesh_translation = relative_transform(
            gt_pose_original, filtered_pose, args.transform_mode
        )
        target_vertices[index] = apply_transform(
            gt_vertices[index], mesh_rotation, mesh_translation
        )

        gt_pose_normalized = (
            mirror_pose(gt_pose_original)
            if normalized_left
            else gt_pose_original
        )
        joint_rotation, joint_translation = relative_transform(
            gt_pose_normalized,
            fp_pose_normalized[index],
            args.transform_mode,
        )
        target_joints[index] = apply_transform(
            gt_joints[index], joint_rotation, joint_translation
        )
        v8_joints = handflow_joints[index] + v8_correction[index, None]
        raw_delta_camera[index] = np.median(
            target_joints[index, PALM] - handflow_joints[index, PALM], axis=0
        )
        v8_delta_camera[index] = np.median(
            target_joints[index, PALM] - v8_joints[PALM], axis=0
        )
        rotation_fp = fp_pose_normalized[index, :3, :3]
        raw_delta_object[index] = rotation_fp.T @ raw_delta_camera[index]
        v8_delta_object[index] = rotation_fp.T @ v8_delta_camera[index]

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
        valid[index] = True
        frame_rows.append({
            "frame": frame,
            "raw_target_translation_mm": (
                np.linalg.norm(raw_delta_camera[index]) * 1000.0
            ),
            "v8_target_translation_mm": (
                np.linalg.norm(v8_delta_camera[index]) * 1000.0
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
        target_joints_3d=target_joints.astype(np.float32),
        raw_target_translation_camera=raw_delta_camera.astype(np.float32),
        raw_target_translation_object=raw_delta_object.astype(np.float32),
        v8_target_translation_camera=v8_delta_camera.astype(np.float32),
        v8_target_translation_object=v8_delta_object.astype(np.float32),
        target_2d_palm_error_px=target_2d_error.astype(np.float32),
        transform_mode=np.asarray(args.transform_mode),
        normalized_left=np.asarray(normalized_left),
    )

    raw_magnitude = np.linalg.norm(raw_delta_camera[valid], axis=-1) * 1000.0
    v8_magnitude = np.linalg.norm(v8_delta_camera[valid], axis=-1) * 1000.0
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
        "object_mesh": str(paths["object_mesh"]),
        "object_mesh_scale": args.object_mesh_scale,
        "hand_side": side,
        "transform_mode": args.transform_mode,
        "num_frames": count,
        "num_valid": int(valid.sum()),
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


if __name__ == "__main__":
    main()
