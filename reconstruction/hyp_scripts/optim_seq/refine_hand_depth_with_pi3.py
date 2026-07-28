#!/usr/bin/env python3
"""Refine HandFlow ray depth from calibrated Pi3 pointmaps and temporal context."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--pi3-cache", required=True)
    parser.add_argument("--object-pose-json", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mesh-scale", type=float, default=-1.0)
    parser.add_argument("--object-label", type=int, default=-1)
    parser.add_argument("--hand-label", type=int, default=255)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--mask-erode-px", type=int, default=2)
    parser.add_argument("--max-rays", type=int, default=1024)
    parser.add_argument("--min-object-hits", type=int, default=48)
    parser.add_argument("--min-hand-hits", type=int, default=32)
    parser.add_argument("--max-calibration-mad-mm", type=float, default=30.0)
    parser.add_argument("--max-observation-mad-mm", type=float, default=25.0)
    parser.add_argument("--observation-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=2.0)
    parser.add_argument("--acceleration-weight", type=float, default=12.0)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--max-correction-mm", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload.get("frame_map", payload))
    if isinstance(rows, dict):
        rows = list(rows.values())
    return sorted(rows, key=lambda row: int(row["output_index"]))


def load_segmentation(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("seg", "segmentation", "label"):
            if key in payload.files:
                return np.asarray(payload[key])
    raise KeyError(f"No segmentation in {path}")


def pose_rows(path: Path) -> tuple[dict[str, np.ndarray], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("by_frame") or payload.get("frames") or payload
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    output = {}
    for key, row in iterator:
        frame = str(key).zfill(6)
        value = row
        if isinstance(row, dict):
            frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
            value = row.get("object_in_camera") or row.get("pose") or row.get("transform")
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.size == 16 and np.isfinite(matrix).all():
                output[frame] = matrix.reshape(4, 4)
    scale = float(payload.get("source_mesh_scale", payload.get("final_global_scale", 1.0)))
    return output, scale


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    return mesh


def transform_mesh(mesh: trimesh.Trimesh, pose: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    vertices = np.asarray(result.vertices)
    result.vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
    return result


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def sampled_pixels(mask: np.ndarray, maximum: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) > maximum:
        selected = rng.choice(len(xs), size=maximum, replace=False)
        xs, ys = xs[selected], ys[selected]
    return xs, ys


def raycast_depth(mesh: trimesh.Trimesh, xs: np.ndarray, ys: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = np.stack(
        [
            (xs - K[0, 2]) / K[0, 0],
            (ys - K[1, 2]) / K[1, 1],
            np.ones(len(xs)),
        ],
        axis=1,
    )
    locations, ray_ids, _ = mesh.ray.intersects_location(
        np.zeros_like(directions), directions, multiple_hits=False
    )
    return locations[:, 2], ray_ids


def robust_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float, int]:
    keep = np.isfinite(source) & np.isfinite(target) & (source > 1e-6) & (target > 1e-6)
    source, target = source[keep], target[keep]
    if len(source) < 8:
        raise ValueError("too_few_calibration_values")
    for _ in range(4):
        matrix = np.stack([source, np.ones(len(source))], axis=1)
        scale, shift = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - (scale * source + shift)
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        threshold = max(3.0 * 1.4826 * mad, 0.003)
        inlier = np.abs(residual - median) <= threshold
        source, target = source[inlier], target[inlier]
        if len(source) < 8:
            break
    residual = target - (scale * source + shift)
    return float(scale), float(shift), float(np.median(np.abs(residual))), int(len(source))


def robust_location(values: np.ndarray) -> tuple[float, float, int]:
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no_values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    keep = np.abs(values - median) <= max(3.0 * 1.4826 * mad, 0.003)
    values = values[keep]
    return float(np.median(values)), mad, int(len(values))


def solve_trajectory(
    observations: np.ndarray,
    weights: np.ndarray,
    observation_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
    anchor_weight: float,
) -> np.ndarray:
    count = len(observations)
    identity = sparse.eye(count, format="csr")
    velocity = sparse.diags([-np.ones(count - 1), np.ones(count - 1)], [0, 1], shape=(count - 1, count))
    acceleration = sparse.diags(
        [np.ones(count - 2), -2.0 * np.ones(count - 2), np.ones(count - 2)],
        [0, 1, 2],
        shape=(count - 2, count),
    )
    obs_diag = sparse.diags(observation_weight * weights, format="csr")
    system = obs_diag + anchor_weight * identity
    if count > 1:
        system += velocity_weight * (velocity.T @ velocity)
    if count > 2:
        system += acceleration_weight * (acceleration.T @ acceleration)
    rhs = observation_weight * weights * np.nan_to_num(observations)
    return np.asarray(spsolve(system.tocsc(), rhs), dtype=np.float64)


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    cache_path = Path(args.pi3_cache).expanduser().resolve()
    pose_path = Path(args.object_pose_json).expanduser().resolve()
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(hand_path, allow_pickle=False) as payload:
        hand = {key: payload[key] for key in payload.files}
    vertices = np.asarray(hand["verts_cam"], dtype=np.float64)
    faces = np.asarray(hand["faces"], dtype=np.int64)
    valid = np.asarray(hand.get("pred_valid", np.ones(len(vertices))), dtype=bool)
    rows = load_rows(frame_map_path)
    poses, pose_scale = pose_rows(pose_path)
    mesh_scale = pose_scale if args.mesh_scale <= 0 else args.mesh_scale
    canonical_mesh = load_mesh(mesh_path, mesh_scale)

    object_label = args.object_label
    if object_label < 0:
        match = next(
            (
                candidate
                for candidate in (
                    re.match(r"(\d+)_", part)
                    for part in reversed(mesh_path.parts)
                )
                if candidate is not None
            ),
            None,
        )
        if match is None:
            raise ValueError("Cannot infer object label; pass --object-label")
        object_label = int(match.group(1))

    window_paths = sorted((cache_path / "windows").glob("window_*.npz"))
    if not window_paths:
        raise FileNotFoundError(f"No Pi3 windows in {cache_path / 'windows'}")

    rng = np.random.default_rng(args.seed)
    observations: list[list[tuple[float, float, dict]]] = [[] for _ in rows]
    window_audits = []
    for window_path in window_paths:
        with np.load(window_path, allow_pickle=False) as payload:
            frame_indices = np.asarray(payload["frame_indices"], dtype=int)
            points = np.asarray(payload["local_points"], dtype=np.float32)
            confidence = np.asarray(payload["confidence"], dtype=np.float32)
            K = np.asarray(payload["intrinsics_resized"], dtype=np.float64)
            width, height = np.asarray(payload["resized_wh"], dtype=int)

        calibration_source = []
        calibration_target = []
        prepared = []
        for local_index, frame_index in enumerate(frame_indices):
            row = rows[frame_index]
            segmentation = load_segmentation(Path(row["label_path"]).expanduser().resolve())
            segmentation = cv2.resize(segmentation, (width, height), interpolation=cv2.INTER_NEAREST)
            conf = confidence[local_index]
            object_mask = erode((segmentation == object_label) & (conf >= args.confidence_threshold), args.mask_erode_px)
            xs, ys = sampled_pixels(object_mask, args.max_rays, rng)
            frame = str(row["output_index"]).zfill(6)
            pose = poses.get(frame)
            if pose is None:
                pose = poses.get(str(row["original_frame"]).zfill(6))
            if pose is not None and len(xs):
                metric_mesh = transform_mesh(canonical_mesh, pose)
                metric_z, ray_ids = raycast_depth(metric_mesh, xs, ys, K)
                if len(ray_ids):
                    pi3_z = points[local_index, ys[ray_ids], xs[ray_ids], 2]
                    calibration_source.append(pi3_z)
                    calibration_target.append(metric_z)
            prepared.append((segmentation, conf))

        audit = {"path": str(window_path), "frames": frame_indices.tolist(), "status": "invalid"}
        try:
            if not calibration_source:
                raise ValueError("no_object_calibration_hits")
            source = np.concatenate(calibration_source)
            target = np.concatenate(calibration_target)
            scale, shift, calibration_mad, calibration_count = robust_affine(source, target)
            if calibration_count < args.min_object_hits:
                raise ValueError(f"object_hits_{calibration_count}")
            if calibration_mad * 1000.0 > args.max_calibration_mad_mm:
                raise ValueError(f"calibration_mad_mm_{calibration_mad * 1000.0:.2f}")
            if not 0.05 <= scale <= 20.0:
                raise ValueError(f"calibration_scale_{scale:.4f}")

            audit.update(
                {
                    "status": "ok",
                    "scale": scale,
                    "shift_m": shift,
                    "calibration_mad_mm": calibration_mad * 1000.0,
                    "calibration_count": calibration_count,
                }
            )
            for local_index, frame_index in enumerate(frame_indices):
                if frame_index >= len(vertices) or not valid[frame_index]:
                    continue
                segmentation, conf = prepared[local_index]
                hand_mask = erode(
                    (segmentation == args.hand_label) & (conf >= args.confidence_threshold),
                    args.mask_erode_px,
                )
                xs, ys = sampled_pixels(hand_mask, args.max_rays, rng)
                if not len(xs):
                    continue
                hand_mesh = trimesh.Trimesh(vertices=vertices[frame_index], faces=faces, process=False)
                predicted_z, ray_ids = raycast_depth(hand_mesh, xs, ys, K)
                if not len(ray_ids):
                    continue
                pi3_z = points[local_index, ys[ray_ids], xs[ray_ids], 2]
                observed_z = scale * pi3_z + shift
                residual = observed_z - predicted_z
                location, mad, count = robust_location(residual)
                if count < args.min_hand_hits or mad * 1000.0 > args.max_observation_mad_mm:
                    continue
                confidence_value = min(1.0, count / args.max_rays)
                confidence_value *= float(np.exp(-(mad * 1000.0) / 15.0))
                observations[frame_index].append(
                    (
                        location,
                        confidence_value,
                        {
                            "window": window_path.name,
                            "residual_mm": location * 1000.0,
                            "mad_mm": mad * 1000.0,
                            "count": count,
                        },
                    )
                )
        except Exception as error:
            audit["reason"] = str(error)
        window_audits.append(audit)
        print(f"{window_path.name}: {audit['status']}" + (f" scale={audit['scale']:.4f} shift={audit['shift_m']:+.4f}m" if audit["status"] == "ok" else f" reason={audit.get('reason')}"))

    observation = np.full(len(rows), np.nan, dtype=np.float64)
    weight = np.zeros(len(rows), dtype=np.float64)
    frame_details = []
    for index, values in enumerate(observations):
        if values:
            residuals = np.asarray([value[0] for value in values])
            confidences = np.asarray([value[1] for value in values])
            observation[index] = float(np.median(residuals))
            weight[index] = float(np.clip(np.median(confidences), 0.0, 1.0))
        frame_details.append(
            {
                "frame": f"{index:06d}",
                "status": "observed" if values else "interpolated",
                "observation_camera_z_mm": None if not values else observation[index] * 1000.0,
                "confidence": weight[index],
                "windows": [value[2] for value in values],
            }
        )

    correction_z = solve_trajectory(
        observation,
        weight,
        args.observation_weight,
        args.velocity_weight,
        args.acceleration_weight,
        args.anchor_weight,
    )
    limit = args.max_correction_mm / 1000.0
    correction_z = np.clip(correction_z, -limit, limit)
    centers = np.nanmean(vertices, axis=1)
    rays = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    correction_distance = correction_z / np.maximum(rays[:, 2], 1e-3)
    correction = rays * correction_distance[:, None]
    corrected_vertices = vertices + correction[:, None, :]

    output = dict(hand)
    output["verts_cam"] = corrected_vertices.astype(np.float32)
    output["pi3_depth_translation_camera"] = correction.astype(np.float32)
    output["pi3_depth_correction_z"] = correction_z.astype(np.float32)
    output["pi3_depth_observation_z"] = observation.astype(np.float32)
    output["pi3_depth_observation_weight"] = weight.astype(np.float32)
    output["pi3_depth_observed"] = (weight > 0).astype(bool)
    output["pi3_depth_source"] = np.asarray(str(hand_path))
    if "hand_center_cam" in output:
        output["hand_center_cam"] = (np.asarray(output["hand_center_cam"]) + correction).astype(np.float32)
    result_path = out_dir / "hand_camera_result_pi3_depth_refined.npz"
    np.savez_compressed(result_path, **output)

    for index, detail in enumerate(frame_details):
        detail["final_camera_z_mm"] = float(correction_z[index] * 1000.0)
        detail["translation_xyz_mm"] = (correction[index] * 1000.0).tolist()
    audit = {
        "hand_npz": str(hand_path),
        "frame_map_json": str(frame_map_path),
        "pi3_cache": str(cache_path),
        "object_pose_json": str(pose_path),
        "object_mesh": str(mesh_path),
        "mesh_scale": mesh_scale,
        "object_label": object_label,
        "result": str(result_path),
        "settings": vars(args),
        "num_frames": len(rows),
        "num_observed_frames": int((weight > 0).sum()),
        "num_interpolated_frames": int((weight <= 0).sum()),
        "observation_camera_z_mm": distribution(observation * 1000.0),
        "final_camera_z_mm": distribution(correction_z * 1000.0),
        "windows": window_audits,
        "frames": frame_details,
    }
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key not in {"windows", "frames"}}, indent=2))


if __name__ == "__main__":
    main()
