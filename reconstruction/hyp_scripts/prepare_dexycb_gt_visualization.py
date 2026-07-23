#!/usr/bin/env python3
"""Export DexYCB GT MANO meshes and YCB poses for visualize_3d.py."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch
import yaml
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--object-model-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--object-mesh-name", default="textured_simple.obj")
    return parser.parse_args()


def load_object_names(model_root: Path) -> dict[int, str]:
    numbered = []
    for path in sorted(model_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            numbered.append((int(path.name.split("_", 1)[0]), path.name))
        except ValueError:
            continue
    numbered.sort()
    return {class_id: name for class_id, (_, name) in enumerate(numbered, start=1)}


def load_mano_pca(mano_data_dir: Path, is_left: bool) -> tuple[np.ndarray, np.ndarray]:
    path = mano_data_dir / ("MANO_LEFT.pkl" if is_left else "MANO_RIGHT.pkl")
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    return (
        np.asarray(raw["hands_components"], dtype=np.float32),
        np.asarray(raw["hands_mean"], dtype=np.float32),
    )


def axis_angle_to_matrix(axis_angle: np.ndarray) -> torch.Tensor:
    matrices = Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    return torch.from_numpy(matrices.astype(np.float32)).view(1, 16, 3, 3)


def matrix_to_wxyz(matrix: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def main() -> None:
    args = parse_args()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    mano_data_dir = Path(args.mano_data_dir).expanduser().resolve()
    model_root = Path(args.object_model_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_map = json.loads(frame_map_path.read_text(encoding="utf-8"))
    rows = frame_map["frames"]
    if not rows:
        raise RuntimeError("Frame map contains no frames")

    first_label = Path(rows[0]["label_path"]).resolve()
    stream_dir = first_label.parent
    meta_path = stream_dir.parent / "meta.yml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}

    hand_side = str((meta.get("mano_sides") or ["right"])[0]).lower()
    is_left = hand_side == "left"
    betas = np.asarray(
        meta.get("mano_betas", meta.get("betas", np.zeros(10))),
        dtype=np.float32,
    ).reshape(-1)[:10]
    if betas.shape[0] < 10:
        betas = np.pad(betas, (0, 10 - betas.shape[0]))

    pca_basis, mean_pose = load_mano_pca(mano_data_dir, is_left)
    mano_layer = smplx.MANOLayer(
        model_path=str(mano_data_dir),
        is_rhand=not is_left,
        use_pca=False,
        flat_hand_mean=True,
    )
    mano_layer.eval()

    ycb_ids = list(meta.get("ycb_ids", []) or [])
    grasp_index = int(meta.get("ycb_grasp_ind", 0))
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(f"Invalid ycb_grasp_ind={grasp_index}")
    target_id = int(ycb_ids[grasp_index])
    object_name = load_object_names(model_root)[target_id]
    object_mesh = model_root / object_name / args.object_mesh_name
    if not object_mesh.is_file():
        raise FileNotFoundError(object_mesh)

    num_frames = max(int(row["output_index"]) for row in rows) + 1
    vertices = np.full((num_frames, 778, 3), np.nan, dtype=np.float32)
    valid = np.zeros(num_frames, dtype=bool)
    object_rows = []
    missing_hand = []
    missing_object = []

    for row in rows:
        frame_index = int(row["output_index"])
        label_path = Path(row["label_path"])
        with np.load(label_path) as raw:
            pose_m = np.asarray(raw["pose_m"], dtype=np.float32).reshape(-1)
            if pose_m.shape[0] >= 51 and not np.allclose(pose_m[:51], 0.0):
                pose_aa = np.concatenate(
                    [pose_m[:3], pose_m[3:48] @ pca_basis + mean_pose],
                    axis=0,
                ).astype(np.float32)
                rotations = axis_angle_to_matrix(pose_aa)
                with torch.no_grad():
                    output = mano_layer(
                        global_orient=rotations[:, 0:1],
                        hand_pose=rotations[:, 1:],
                        betas=torch.from_numpy(betas).view(1, 10),
                        pose2rot=False,
                    )
                vertices[frame_index] = (
                    output.vertices[0].cpu().numpy().astype(np.float32)
                    + pose_m[48:51][None, :]
                )
                valid[frame_index] = True
            else:
                missing_hand.append(frame_index)

            poses = np.asarray(raw["pose_y"], dtype=np.float32)
            if poses.ndim == 2:
                poses = poses[None]
            if grasp_index >= len(poses):
                missing_object.append(frame_index)
                continue
            pose = poses[grasp_index]

        if pose.shape == (3, 4):
            rotation, translation = pose[:, :3], pose[:, 3]
        elif pose.shape == (4, 4):
            rotation, translation = pose[:3, :3], pose[:3, 3]
        else:
            missing_object.append(frame_index)
            continue
        if not np.isfinite(pose).all():
            missing_object.append(frame_index)
            continue
        object_rows.append(
            {
                "frame_idx": frame_index,
                "frame_index": frame_index,
                "source_frame": row["original_frame"],
                "local_to_scene": {
                    "quat_wxyz_camera_frame": matrix_to_wxyz(rotation),
                    "translation_camera_frame": translation.astype(float).tolist(),
                    "scale": [1.0, 1.0, 1.0],
                },
            }
        )

    faces = np.asarray(mano_layer.faces, dtype=np.int64)
    inactive_vertices = np.full_like(vertices, np.nan)
    inactive_valid = np.zeros_like(valid)
    hand_npz = out_dir / "dexycb_gt_hand_meshes.npz"
    if is_left:
        left_vertices, left_valid = vertices, valid
        right_vertices, right_valid = inactive_vertices, inactive_valid
    else:
        left_vertices, left_valid = inactive_vertices, inactive_valid
        right_vertices, right_valid = vertices, valid
    np.savez_compressed(
        hand_npz,
        left_vertices=left_vertices,
        left_faces=faces,
        left_valid=left_valid,
        right_vertices=right_vertices,
        right_faces=faces.copy(),
        right_valid=right_valid,
        source=np.asarray("dexycb_gt_pose_m"),
    )

    object_layout = out_dir / "dexycb_gt_object_layout_camera_frame.json"
    object_layout.write_text(
        json.dumps(
            {
                "source": "dexycb_gt_pose_y",
                "frame": "camera_frame",
                "pose_convention": "ycb_model_to_camera",
                "object_name": object_name,
                "object_mesh": str(object_mesh),
                "num_objects": len(object_rows),
                "missing_pose_frames": missing_object,
                "objects": object_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "frame_map_json": str(frame_map_path),
        "meta_path": str(meta_path),
        "hand_side": hand_side,
        "num_frames": num_frames,
        "num_valid_hand_frames": int(valid.sum()),
        "missing_hand_frames": missing_hand,
        "num_valid_object_frames": len(object_rows),
        "missing_object_frames": missing_object,
        "gt_hand_meshes": str(hand_npz),
        "gt_object_layout": str(object_layout),
        "gt_object_mesh": str(object_mesh),
    }
    (out_dir / "dexycb_gt_visualization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
