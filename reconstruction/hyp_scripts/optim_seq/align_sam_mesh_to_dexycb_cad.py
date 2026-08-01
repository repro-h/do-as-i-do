#!/usr/bin/env python3
"""Align one SAM3D canonical mesh to its DexYCB YCB CAD model."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sam-mesh", default=None)
    parser.add_argument("--ycb-model-root", required=True)
    parser.add_argument("--foundationpose-json", default=None)
    parser.add_argument("--filtered-object-json", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--alignment-mode",
        choices=("pose_consensus", "surface_icp"),
        default="pose_consensus",
    )
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--trim-fraction", type=float, default=0.7)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Empty mesh: {path}")
        loaded = loaded.to_geometry()
    mesh = loaded.copy()
    mesh.remove_unreferenced_vertices()
    return mesh


def sample_surface(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def orientation_candidates(
    source: np.ndarray, target: np.ndarray
) -> list[np.ndarray]:
    source_basis = principal_axes(source)
    target_basis = principal_axes(target)
    candidates = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            signed = permutation_matrix @ np.diag(signs)
            rotation = target_basis @ signed @ source_basis.T
            if np.linalg.det(rotation) > 0.0:
                candidates.append(rotation)
    return candidates


def estimate_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def transform(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(round(len(values) * fraction)))
    return float(np.mean(np.partition(values, count - 1)[:count]))


def alignment_score(
    source: np.ndarray,
    target: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    trim_fraction: float,
) -> tuple[float, dict]:
    transformed = transform(source, scale, rotation, translation)
    source_to_target = cKDTree(target).query(transformed, workers=-1)[0]
    target_to_source = cKDTree(transformed).query(target, workers=-1)[0]
    forward = trimmed_mean(source_to_target**2, trim_fraction)
    backward = trimmed_mean(target_to_source**2, trim_fraction)
    score = np.sqrt(0.5 * (forward + backward))
    return float(score), {
        "trimmed_symmetric_rmse_mm": float(score * 1000.0),
        "source_to_target_median_mm": float(np.median(source_to_target) * 1000.0),
        "target_to_source_median_mm": float(np.median(target_to_source) * 1000.0),
        "source_to_target_p90_mm": float(np.quantile(source_to_target, 0.9) * 1000.0),
        "target_to_source_p90_mm": float(np.quantile(target_to_source, 0.9) * 1000.0),
    }


def run_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial_rotation: np.ndarray,
    iterations: int,
    trim_fraction: float,
    initial_scale: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    scale = initial_scale
    rotation = initial_rotation
    translation = target_mean - scale * (rotation @ source_mean)
    target_tree = cKDTree(target)
    minimum_scale = initial_scale * 0.5
    maximum_scale = initial_scale * 2.0
    for _ in range(iterations):
        transformed = transform(source, scale, rotation, translation)
        distances, indices = target_tree.query(transformed, workers=-1)
        count = max(32, int(round(len(source) * trim_fraction)))
        keep = np.argpartition(distances, count - 1)[:count]
        next_scale, next_rotation, next_translation = estimate_similarity(
            source[keep], target[indices[keep]]
        )
        next_scale = float(np.clip(next_scale, minimum_scale, maximum_scale))
        next_translation = target[indices[keep]].mean(axis=0) - next_scale * (
            next_rotation @ source[keep].mean(axis=0)
        )
        change = np.linalg.norm(next_translation - translation)
        change += abs(next_scale - scale)
        change += np.linalg.norm(next_rotation - rotation)
        scale, rotation, translation = next_scale, next_rotation, next_translation
        if change < 1e-8:
            break
    return scale, rotation, translation


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def load_pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("by_frame") or payload.get("frames") or {}
    output = {}
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    for key, row in iterator:
        frame = str(key).zfill(6)
        value = row
        if isinstance(row, dict):
            frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
            value = row.get("object_in_camera") or row.get("pose")
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.size == 16 and np.isfinite(matrix).all():
                output[frame] = matrix.reshape(4, 4)
    return output


def load_gt_ycb_poses(
    sequence_dir: Path, grasp_index: int
) -> tuple[dict[str, np.ndarray], list[str]]:
    poses = {}
    missing = []
    for label_path in sorted(sequence_dir.glob("labels_*.npz")):
        frame = label_path.stem.split("_")[-1].zfill(6)
        with np.load(label_path, allow_pickle=False) as payload:
            values = np.asarray(payload["pose_y"], dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if grasp_index >= len(values):
            missing.append(frame)
            continue
        value = np.asarray(values[grasp_index], dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :4] = value[:3, :4]
        if np.isfinite(matrix).all():
            poses[frame] = matrix
        else:
            missing.append(frame)
    return poses, missing


def pose_consensus_alignment(
    filtered_rows: dict[str, np.ndarray],
    gt_rows: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    transforms = []
    frames = []
    for frame, gt_pose in gt_rows.items():
        predicted = filtered_rows.get(frame)
        if predicted is None:
            continue
        transforms.append(np.linalg.inv(gt_pose) @ predicted)
        frames.append(frame)
    if len(transforms) < 3:
        raise RuntimeError("Need at least three shared GT/filtered pose frames")
    transforms = np.stack(transforms)
    rotation = Rotation.from_matrix(transforms[:, :3, :3]).mean().as_matrix()
    translation = np.median(transforms[:, :3, 3], axis=0)

    rotation_residual = np.asarray(
        [
            np.degrees(
                np.linalg.norm(
                    Rotation.from_matrix(rotation.T @ value[:3, :3]).as_rotvec()
                )
            )
            for value in transforms
        ]
    )
    translation_residual = (
        np.linalg.norm(transforms[:, :3, 3] - translation, axis=1) * 1000.0
    )
    audit = {
        "count": len(transforms),
        "frames": frames,
        "rotation_consensus_residual_deg": distribution(
            rotation_residual.tolist()
        ),
        "translation_consensus_residual_mm": distribution(
            translation_residual.tolist()
        ),
    }
    return rotation, translation, audit


def stream_id_from_path(path: Path) -> str:
    if len(path.parts) < 3:
        raise ValueError(f"Cannot derive stream ID from {path}")
    return "__".join(path.parts[-3:])


def manifest_record(path: Path, stream_id: str) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("stream_id")) == stream_id:
                return row
    raise KeyError(f"Stream {stream_id} not found in {path}")


def main() -> None:
    args = parse_args()
    selected_sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    stream_id = stream_id_from_path(selected_sequence_dir)
    record = manifest_record(
        Path(args.manifest).expanduser().resolve(), stream_id
    )
    sequence_dir = Path(record["stream_dir"]).expanduser().resolve()
    sam_path = Path(
        args.sam_mesh or record["sam3d_glb"]
    ).expanduser().resolve()
    model_root = Path(args.ycb_model_root).expanduser().resolve()
    foundationpose_path = Path(
        args.foundationpose_json or record["foundationpose_json"]
    ).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = sequence_dir.parent / "meta.yml"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    ycb_ids = list(metadata.get("ycb_ids", []) or [])
    grasp_index = int(metadata.get("ycb_grasp_ind", 0))
    object_id = int(ycb_ids[grasp_index])
    matches = sorted(model_root.glob(f"{object_id:03d}_*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one YCB model for {object_id}, got {matches}")
    object_name = matches[0].name
    ycb_mesh_path = matches[0] / "textured_simple.obj"

    sam_mesh = load_mesh(sam_path)
    ycb_mesh = load_mesh(ycb_mesh_path)
    foundationpose = json.loads(foundationpose_path.read_text(encoding="utf-8"))
    initial_source_scale = float(
        foundationpose.get(
            "source_mesh_scale", foundationpose.get("final_global_scale", 1.0)
        )
    )
    gt_ycb_poses, missing_frames = load_gt_ycb_poses(
        sequence_dir, grasp_index
    )
    rng = np.random.default_rng(args.seed)
    source = sample_surface(sam_mesh, args.samples, rng)
    target = sample_surface(ycb_mesh, args.samples, rng)
    source_radius = np.sqrt(np.mean(np.sum((source - source.mean(0)) ** 2, axis=1)))
    target_radius = np.sqrt(np.mean(np.sum((target - target.mean(0)) ** 2, axis=1)))
    initial_scale = float(target_radius / max(source_radius, 1e-12))

    candidates = []
    consensus_audit = None
    if args.alignment_mode == "pose_consensus":
        if not args.filtered_object_json:
            raise ValueError(
                "--filtered-object-json is required for pose_consensus"
            )
        filtered_rows = load_pose_rows(
            Path(args.filtered_object_json).expanduser().resolve()
        )
        rotation, translation, consensus_audit = pose_consensus_alignment(
            filtered_rows, gt_ycb_poses
        )
        scale = initial_source_scale
        score, metrics = alignment_score(
            source,
            target,
            scale,
            rotation,
            translation,
            args.trim_fraction,
        )
        candidates.append((score, scale, rotation, translation, metrics))
        print(
            f"pose consensus scale={scale:.6f} "
            f"surface_rmse={score * 1000.0:.3f}mm",
            flush=True,
        )
    else:
        rotations = orientation_candidates(source, target)[: args.max_candidates]
        for index, initial_rotation in enumerate(rotations):
            scale, rotation, translation = run_icp(
                source,
                target,
                initial_rotation,
                args.icp_iterations,
                args.trim_fraction,
                initial_scale,
            )
            score, metrics = alignment_score(
                source,
                target,
                scale,
                rotation,
                translation,
                args.trim_fraction,
            )
            candidates.append((score, scale, rotation, translation, metrics))
            print(
                f"candidate {index + 1:02d}/{len(rotations):02d} "
                f"scale={scale:.6f} rmse={score * 1000.0:.3f}mm",
                flush=True,
            )
    candidates.sort(key=lambda row: row[0])
    score, scale, rotation, translation, metrics = candidates[0]

    aligned_mesh = sam_mesh.copy()
    aligned_mesh.vertices = transform(
        np.asarray(sam_mesh.vertices, dtype=np.float64),
        scale,
        rotation,
        translation,
    )
    aligned_mesh.export(out_dir / "sam_mesh_aligned_to_ycb.obj")

    sam_to_ycb = np.eye(4, dtype=np.float64)
    sam_to_ycb[:3, :3] = rotation
    sam_to_ycb[:3, 3] = translation
    gt_rows = {}
    for frame, matrix in gt_ycb_poses.items():
        sam_pose = matrix @ sam_to_ycb
        gt_rows[frame] = {
            "frame": frame,
            "object_in_camera": sam_pose.tolist(),
            "valid": True,
        }

    gt_payload = {
        "source": "dexycb_pose_y_with_sam_to_ycb_alignment",
        "uses_gt_object_pose": True,
        "intrinsics": foundationpose.get("intrinsics"),
        "source_mesh": str(sam_path),
        "source_mesh_scale": scale,
        "object_name": object_name,
        "sam_to_ycb_rigid": sam_to_ycb.tolist(),
        "by_frame": gt_rows,
    }
    gt_json_path = out_dir / "dexycb_gt_pose_in_sam_canonical.json"
    gt_json_path.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")

    pose_audit = None
    if args.filtered_object_json:
        filtered_path = Path(args.filtered_object_json).expanduser().resolve()
        filtered_rows = load_pose_rows(filtered_path)
        translation_errors = []
        rotation_errors = []
        for frame, row in gt_rows.items():
            predicted = filtered_rows.get(frame)
            if predicted is None:
                continue
            expected = np.asarray(row["object_in_camera"], dtype=np.float64)
            translation_errors.append(
                float(np.linalg.norm(predicted[:3, 3] - expected[:3, 3]) * 1000.0)
            )
            relative = expected[:3, :3].T @ predicted[:3, :3]
            rotation_errors.append(
                float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
            )
        pose_audit = {
            "filtered_object_json": str(filtered_path),
            "translation_error_mm": distribution(translation_errors),
            "rotation_error_deg": distribution(rotation_errors),
        }

    summary = {
        "stream_id": stream_id,
        "selected_sequence_dir": str(selected_sequence_dir),
        "sequence_dir": str(sequence_dir),
        "object_id": object_id,
        "object_name": object_name,
        "sam_mesh": str(sam_path),
        "ycb_mesh": str(ycb_mesh_path),
        "foundationpose_source_mesh_scale": initial_source_scale,
        "aligned_source_mesh_scale": scale,
        "scale_ratio_to_foundationpose": scale / initial_source_scale,
        "sam_to_ycb_rigid": sam_to_ycb.tolist(),
        "surface_alignment": metrics,
        "alignment_mode": args.alignment_mode,
        "pose_consensus": consensus_audit,
        "num_gt_pose_frames": len(gt_rows),
        "missing_gt_pose_frames": missing_frames,
        "gt_pose_json": str(gt_json_path),
        "aligned_mesh": str(out_dir / "sam_mesh_aligned_to_ycb.obj"),
        "filtered_pose_audit": pose_audit,
        "top_candidates": [
            {
                "rank": index + 1,
                "scale": float(row[1]),
                **row[4],
            }
            for index, row in enumerate(candidates[:5])
        ],
    }
    summary_path = out_dir / "alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
