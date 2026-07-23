#!/usr/bin/env python3
"""Adapt FoundationPose and HandFlow outputs for Do-As-I-Do's Viser viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FoundationPose camera-space poses and a HandFlow result NPZ "
            "to the layout/hand-mesh formats consumed by scripts/visualize_3d.py."
        )
    )
    parser.add_argument("--foundationpose-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--handflow-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--invalid-hand-mode",
        choices=("keep", "nan"),
        default="keep",
        help="Keep generated HandFlow meshes on detector-invalid frames or replace them with NaN.",
    )
    parser.add_argument(
        "--hand-side",
        choices=("left", "right"),
        default="right",
        help="Destination hand side in the Do-As-I-Do hand-mesh archive.",
    )
    parser.add_argument(
        "--mirror-x",
        action="store_true",
        help=(
            "Reflect HandFlow camera-space vertices across x=0 and reverse face "
            "winding. Use this when a left-hand video was mirrored for right-hand inference."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def matrix_to_wxyz(matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    return [
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def adapt_layout(
    pose_payload: dict,
    frame_map_payload: dict,
    source_pose_path: Path,
    out_path: Path,
) -> dict:
    pose_rows = pose_payload.get("by_frame") or pose_payload.get("frames") or {}
    original_to_output = {
        str(row["original_frame"]).zfill(6): int(row["output_index"])
        for row in frame_map_payload["frames"]
    }

    source_scale = float(pose_payload.get("source_mesh_scale", 1.0))
    objects = []
    missing_pose_frames = []

    for original_frame, output_index in sorted(
        original_to_output.items(), key=lambda item: item[1]
    ):
        row = pose_rows.get(original_frame)
        if row is None:
            row = pose_rows.get(str(int(original_frame)))
        if row is None or row.get("object_in_camera") is None:
            missing_pose_frames.append(original_frame)
            continue

        pose = np.asarray(row["object_in_camera"], dtype=np.float64).reshape(4, 4)
        if not np.isfinite(pose).all():
            missing_pose_frames.append(original_frame)
            continue

        objects.append(
            {
                "frame_idx": output_index,
                "frame_index": output_index,
                "source_frame": original_frame,
                "mode": row.get("mode"),
                "local_to_scene": {
                    "quat_wxyz_camera_frame": matrix_to_wxyz(pose[:3, :3]),
                    "translation_camera_frame": pose[:3, 3].astype(float).tolist(),
                    "scale": [source_scale, source_scale, source_scale],
                },
            }
        )

    intrinsics = np.asarray(pose_payload.get("intrinsics"), dtype=np.float64)
    payload = {
        "source": "foundationpose_handflow_visualization_adapter",
        "frame": "camera_frame",
        "pose_convention": "object_model_to_camera",
        "source_pose_json": str(source_pose_path),
        "source_mesh_scale": source_scale,
        "intrinsics": intrinsics.tolist() if intrinsics.shape == (3, 3) else None,
        "num_objects": len(objects),
        "missing_pose_frames": missing_pose_frames,
        "objects": objects,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def adapt_handflow(
    source_path: Path,
    out_path: Path,
    invalid_hand_mode: str,
    hand_side: str,
    mirror_x: bool,
) -> dict:
    with np.load(source_path, allow_pickle=True) as data:
        vertices = np.asarray(data["verts_cam"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
        valid = np.asarray(
            data["pred_valid"] if "pred_valid" in data else np.ones(len(vertices)),
            dtype=bool,
        )
        intrinsics = (
            np.asarray(data["intrinsics"], dtype=np.float32)
            if "intrinsics" in data
            else np.empty((0,), dtype=np.float32)
        )
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0

    if mirror_x:
        vertices = vertices.copy()
        vertices[..., 0] *= -1.0
        faces = faces[:, [0, 2, 1]].copy()

    if invalid_hand_mode == "nan":
        vertices = vertices.copy()
        vertices[~valid] = np.nan

    inactive_vertices = np.zeros_like(vertices)
    inactive_valid = np.zeros(len(vertices), dtype=bool)
    if hand_side == "right":
        right_vertices = vertices
        right_valid = valid
        left_vertices = inactive_vertices
        left_valid = inactive_valid
    else:
        right_vertices = inactive_vertices
        right_valid = inactive_valid
        left_vertices = vertices
        left_valid = valid

    np.savez_compressed(
        out_path,
        right_vertices=right_vertices,
        right_faces=faces,
        right_valid=right_valid,
        left_vertices=left_vertices,
        left_faces=faces.copy(),
        left_valid=left_valid,
        intrinsics=intrinsics,
        fps=np.asarray(fps, dtype=np.float32),
        source=np.asarray("handflow"),
        source_npz=np.asarray(str(source_path)),
    )
    return {
        "num_frames": int(len(vertices)),
        "num_valid_frames": int(valid.sum()),
        "num_vertices": int(vertices.shape[1]),
        "num_faces": int(len(faces)),
        "invalid_hand_mode": invalid_hand_mode,
        "hand_side": hand_side,
        "mirror_x": bool(mirror_x),
    }


def main() -> None:
    args = parse_args()
    pose_path = Path(args.foundationpose_json).expanduser().resolve()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    handflow_path = Path(args.handflow_npz).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    layout_path = out_dir / "foundationpose_layout_camera_frame.json"
    hand_path = out_dir / "all_hand_meshes_handflow.npz"
    summary_path = out_dir / "visualization_adapter_summary.json"

    pose_payload = load_json(pose_path)
    frame_map_payload = load_json(frame_map_path)
    layout_summary = adapt_layout(
        pose_payload,
        frame_map_payload,
        pose_path,
        layout_path,
    )
    hand_summary = adapt_handflow(
        handflow_path,
        hand_path,
        args.invalid_hand_mode,
        args.hand_side,
        args.mirror_x,
    )

    summary = {
        "foundationpose_json": str(pose_path),
        "frame_map_json": str(frame_map_path),
        "handflow_npz": str(handflow_path),
        "layout_json": str(layout_path),
        "hand_meshes_npz": str(hand_path),
        "num_object_pose_frames": int(layout_summary["num_objects"]),
        "missing_object_pose_frames": layout_summary["missing_pose_frames"],
        "hand": hand_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
