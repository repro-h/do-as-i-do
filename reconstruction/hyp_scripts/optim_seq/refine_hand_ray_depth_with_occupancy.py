#!/usr/bin/env python3
"""Refine an occluded hand segment with a conservative object occupancy proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument(
        "--gt-hand-npz",
        default=None,
        help="Optional Stage1 supervision NPZ with gt_hand_vertices for audit.",
    )
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mesh-scale", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--occupancy-source",
        choices=("mesh", "convex_hull"),
        default="convex_hull",
    )
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--interior-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--focus-start-frame", type=int, required=True)
    parser.add_argument("--focus-end-frame", type=int, required=True)
    parser.add_argument("--apply-start-frame", type=int, required=True)
    parser.add_argument("--apply-end-frame", type=int, required=True)
    parser.add_argument("--ramp-frames", type=int, default=4)
    parser.add_argument("--bias-min-mm", type=float, default=0.0)
    parser.add_argument("--bias-max-mm", type=float, default=20.0)
    parser.add_argument("--bias-step-mm", type=float, default=0.5)
    parser.add_argument("--w-penetration-depth", type=float, default=1.0)
    parser.add_argument("--w-penetration-fraction", type=float, default=100.0)
    parser.add_argument("--w-anchor", type=float, default=0.02)
    return parser.parse_args()


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64) * scale,
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


class Occupancy:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        pitch: float,
        interior_tolerance: float,
    ) -> None:
        voxel = mesh.voxelized(pitch).fill()
        matrix = np.asarray(voxel.matrix, dtype=bool)
        erosion_iterations = int(np.ceil(interior_tolerance / pitch))
        if erosion_iterations > 0:
            interior = ndimage.binary_erosion(
                matrix,
                iterations=erosion_iterations,
                border_value=0,
            )
        else:
            interior = matrix
        self.voxel = voxel
        self.interior = interior
        self.depth = ndimage.distance_transform_edt(interior) * pitch

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(self.voxel.points_to_indices(points), dtype=np.int64)
        shape = np.asarray(self.interior.shape, dtype=np.int64)
        in_bounds = (
            (indices >= 0).all(axis=1)
            & (indices < shape[None]).all(axis=1)
        )
        inside = np.zeros(len(points), dtype=bool)
        depth = np.zeros(len(points), dtype=np.float64)
        bounded = indices[in_bounds]
        if len(bounded):
            coordinate = tuple(bounded[:, axis] for axis in range(3))
            inside[in_bounds] = self.interior[coordinate]
            depth[in_bounds] = self.depth[coordinate]
        return inside, depth


def penetration_metrics(
    vertices: np.ndarray,
    object_pose: np.ndarray,
    occupancy: Occupancy,
) -> dict:
    rotation = object_pose[:3, :3]
    translation = object_pose[:3, 3]
    local = (vertices - translation[None]) @ rotation
    inside, depth = occupancy.query(local)
    penetrating_depth = depth[inside] * 1000.0
    return {
        "count": int(inside.sum()),
        "fraction": float(inside.mean()),
        "mean_square_depth_mm2": (
            float(np.square(penetrating_depth).mean())
            if len(penetrating_depth)
            else 0.0
        ),
        "depth_mm": distribution(penetrating_depth),
    }


def ramp_weights(
    count: int,
    start: int,
    end: int,
    ramp_frames: int,
) -> np.ndarray:
    weights = np.zeros(count, dtype=np.float64)
    weights[start : end + 1] = 1.0
    ramp = min(max(ramp_frames, 0), end - start + 1)
    if ramp > 0:
        weights[start : start + ramp] = np.linspace(
            1.0 / ramp, 1.0, ramp
        )
    return weights


def main() -> None:
    args = parse_args()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(hand_path, allow_pickle=False) as raw:
        hand = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(supervision_path, allow_pickle=False) as raw:
        supervision = {key: np.asarray(raw[key]) for key in raw.files}

    vertices = np.asarray(hand["verts_cam"], dtype=np.float64)
    object_pose = np.asarray(supervision["object_pose"], dtype=np.float64)
    pred_joints = np.asarray(
        supervision["pred_joints_3d"], dtype=np.float64
    )
    count = min(len(vertices), len(object_pose), len(pred_joints))
    vertices = vertices[:count]
    object_pose = object_pose[:count]
    pred_joints = pred_joints[:count]

    if bool(np.asarray(supervision.get("normalized_left", False)).item()):
        raise ValueError("This pilot currently expects camera-frame right-hand data")
    if not (
        0 <= args.apply_start_frame <= args.focus_start_frame
        <= args.focus_end_frame <= args.apply_end_frame < count
    ):
        raise ValueError(
            "Expected apply_start <= focus_start <= focus_end <= "
            f"apply_end < {count}"
        )
    if args.bias_step_mm <= 0 or args.bias_max_mm < args.bias_min_mm:
        raise ValueError("Invalid bias search range")

    mesh = load_mesh(mesh_path, args.mesh_scale)
    if args.occupancy_source == "convex_hull":
        mesh = mesh.convex_hull
    occupancy = Occupancy(
        mesh,
        args.voxel_mm / 1000.0,
        args.interior_tolerance_mm / 1000.0,
    )

    camera_ray = pred_joints[:, 0]
    camera_ray /= np.maximum(
        np.linalg.norm(camera_ray, axis=-1, keepdims=True), 1e-8
    )
    frame_weight = ramp_weights(
        count,
        args.apply_start_frame,
        args.apply_end_frame,
        args.ramp_frames,
    )
    focus_indices = np.arange(
        args.focus_start_frame, args.focus_end_frame + 1
    )
    candidates_mm = np.arange(
        args.bias_min_mm,
        args.bias_max_mm + args.bias_step_mm * 0.5,
        args.bias_step_mm,
        dtype=np.float64,
    )

    candidate_rows = []
    for bias_mm in candidates_mm:
        translated = vertices + (
            camera_ray * (bias_mm / 1000.0) * frame_weight[:, None]
        )[:, None]
        rows = [
            penetration_metrics(
                translated[frame], object_pose[frame], occupancy
            )
            for frame in focus_indices
        ]
        total_vertices = len(focus_indices) * vertices.shape[1]
        total_count = sum(row["count"] for row in rows)
        fraction = total_count / max(total_vertices, 1)
        depth_values = np.asarray(
            [row["mean_square_depth_mm2"] for row in rows],
            dtype=np.float64,
        )
        mean_square_depth = float(depth_values.mean())
        score = (
            args.w_penetration_depth * mean_square_depth
            + args.w_penetration_fraction * fraction
            + args.w_anchor * bias_mm * bias_mm
        )
        candidate_rows.append(
            {
                "bias_mm": float(bias_mm),
                "score": float(score),
                "penetrating_count": int(total_count),
                "penetrating_fraction": float(fraction),
                "mean_square_depth_mm2": mean_square_depth,
                "frames": {
                    f"{frame:06d}": row
                    for frame, row in zip(focus_indices, rows)
                },
            }
        )

    best = min(candidate_rows, key=lambda row: (row["score"], row["bias_mm"]))
    baseline = min(candidate_rows, key=lambda row: abs(row["bias_mm"]))
    selected_bias_m = best["bias_mm"] / 1000.0
    translation = (
        camera_ray * selected_bias_m * frame_weight[:, None]
    )
    corrected_vertices = vertices + translation[:, None]

    output = dict(hand)
    output["verts_cam"] = corrected_vertices.astype(np.float32)
    output["occupancy_shared_ray_depth"] = (
        selected_bias_m * frame_weight
    ).astype(np.float32)
    output["occupancy_translation_camera"] = translation.astype(np.float32)
    output["occupancy_refined"] = (frame_weight > 0)
    output["occupancy_source_hand"] = np.asarray(str(hand_path))
    if "hand_center_cam" in output:
        output["hand_center_cam"] = (
            np.asarray(output["hand_center_cam"])[:count] + translation
        ).astype(np.float32)

    result_path = out_dir / "hand_camera_result_occupancy_depth_refined.npz"
    np.savez_compressed(result_path, **output)
    gt_audit = None
    if args.gt_hand_npz:
        gt_path = Path(args.gt_hand_npz).expanduser().resolve()
        with np.load(gt_path, allow_pickle=False) as raw:
            gt_vertices = np.asarray(
                raw["gt_hand_vertices"], dtype=np.float64
            )
            gt_valid = (
                np.asarray(raw["gt_hand_valid"]).astype(bool)
                if "gt_hand_valid" in raw.files
                else np.ones(len(gt_vertices), dtype=bool)
            )
        gt_rows = {}
        for frame in focus_indices:
            if frame >= len(gt_vertices) or not gt_valid[frame]:
                continue
            gt_rows[f"{frame:06d}"] = penetration_metrics(
                gt_vertices[frame], object_pose[frame], occupancy
            )
        gt_audit = {
            "gt_hand_npz": str(gt_path),
            "frames": gt_rows,
            "penetrating_count": int(
                sum(row["count"] for row in gt_rows.values())
            ),
        }
    audit = {
        "hand_npz": str(hand_path),
        "supervision_npz": str(supervision_path),
        "object_mesh": str(mesh_path),
        "result": str(result_path),
        "settings": vars(args),
        "voxel_grid_shape": list(occupancy.interior.shape),
        "num_interior_voxels": int(occupancy.interior.sum()),
        "baseline": baseline,
        "selected": best,
        "gt_audit": gt_audit,
        "candidates": candidate_rows,
    }
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items()
                      if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
