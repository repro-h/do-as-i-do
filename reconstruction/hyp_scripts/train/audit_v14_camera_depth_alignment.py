#!/usr/bin/env python3
"""Audit whether V14-to-GT hand error is predominantly camera-ray depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--gt-hand-npz", required=True)
    intrinsics = parser.add_mutually_exclusive_group()
    intrinsics.add_argument(
        "--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY")
    )
    intrinsics.add_argument("--intrinsics-file")
    parser.add_argument(
        "--dexycb-root", default="/mnt/nas/wuke/HumanData/DexYCB"
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npz")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def aligned_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {frame_id(value): index for index, value in enumerate(source)}
    missing = [frame_id(value) for value in target if frame_id(value) not in lookup]
    if missing:
        raise KeyError(f"Missing {len(missing)} frames; first: {missing[:5]}")
    return np.asarray([lookup[frame_id(value)] for value in target], dtype=np.int64)


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = points[..., 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
    return np.stack(
        [
            intrinsics[0, 0] * points[..., 0] / safe_z + intrinsics[0, 2],
            intrinsics[1, 1] * points[..., 1] / safe_z + intrinsics[1, 2],
        ],
        axis=-1,
    )


def read_intrinsics_file(path: Path) -> np.ndarray:
    color: dict[str, float] = {}
    in_color = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip() == "color:":
            in_color = True
            continue
        if in_color and raw_line and not raw_line[0].isspace():
            break
        if not in_color or ":" not in raw_line:
            continue
        key, value = raw_line.strip().split(":", 1)
        if key in {"fx", "fy", "ppx", "ppy", "cx", "cy"}:
            color[key] = float(value.strip())
    cx = color.get("ppx", color.get("cx"))
    cy = color.get("ppy", color.get("cy"))
    if not {"fx", "fy"}.issubset(color) or cx is None or cy is None:
        raise KeyError(f"No complete color intrinsics in {path}")
    return np.asarray(
        [
            [color["fx"], 0.0, cx],
            [0.0, color["fy"], cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def resolve_intrinsics(args: argparse.Namespace, stream_id: str) -> tuple[np.ndarray, str]:
    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        matrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return matrix, "command_line"
    if args.intrinsics_file:
        path = Path(args.intrinsics_file).expanduser().resolve()
    else:
        serial = stream_id.split("__")[-1]
        path = (
            Path(args.dexycb_root).expanduser().resolve()
            / "calibration"
            / "intrinsics"
            / f"{serial}_640x480.yml"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Intrinsics not found: {path}; pass --intrinsics FX FY CX CY"
        )
    return read_intrinsics_file(path), str(path)


def safe_unit(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1e-8)


def mean_vertex_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(predicted - target, axis=-1).mean(axis=-1) * 1000.0


def main() -> None:
    args = parse_args()
    trajectory_path = Path(args.trajectory_npz).expanduser().resolve()
    query_path = Path(args.query_npz).expanduser().resolve()
    gt_path = Path(args.gt_hand_npz).expanduser().resolve()
    trajectory = load_npz(trajectory_path)
    query = load_npz(query_path)
    gt = load_npz(gt_path)

    ids = np.asarray(query["frame_ids"])
    raw_stream_id = query.get("stream_id", trajectory.get("stream_id"))
    stream_id = str(np.asarray(raw_stream_id).item()) if raw_stream_id is not None else ""
    if not stream_id:
        stream_id = query_path.parent.name
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    side = str(query["hand_side"].item()).lower()

    relative = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float64
    )
    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float64
    )
    predicted = relative + wrist[:, None]
    target = np.asarray(gt[f"{side}_vertices"], dtype=np.float64)[: len(ids)]
    gt_valid = np.asarray(gt[f"{side}_valid"]).astype(bool)[: len(ids)]
    prediction_valid = np.asarray(
        trajectory.get("prediction_valid", np.ones(len(trajectory["frame_ids"])))
    ).astype(bool)[trajectory_indices]
    valid = (
        gt_valid
        & prediction_valid
        & np.isfinite(predicted).all(axis=(1, 2))
        & np.isfinite(target).all(axis=(1, 2))
        & (predicted[..., 2] > 1e-5).all(axis=1)
        & (target[..., 2] > 1e-5).all(axis=1)
    )
    if not valid.any():
        raise RuntimeError("No jointly valid V14/GT frames")

    intrinsics, intrinsics_source = resolve_intrinsics(args, stream_id)
    pred_uv = project(predicted, intrinsics)
    gt_uv = project(target, intrinsics)
    vertex_uv_error = np.linalg.norm(pred_uv - gt_uv, axis=-1)

    pred_center = predicted.mean(axis=1)
    gt_center = target.mean(axis=1)
    pred_center_uv = project(pred_center, intrinsics)
    gt_center_uv = project(gt_center, intrinsics)
    center_uv_error = np.linalg.norm(pred_center_uv - gt_center_uv, axis=-1)

    center_delta = gt_center - pred_center
    center_ray = safe_unit(gt_center)
    center_parallel_signed = np.sum(center_delta * center_ray, axis=-1)
    center_parallel = center_parallel_signed[:, None] * center_ray
    center_perpendicular = center_delta - center_parallel
    center_total_mm = np.linalg.norm(center_delta, axis=-1) * 1000.0
    center_parallel_mm = np.abs(center_parallel_signed) * 1000.0
    center_perpendicular_mm = np.linalg.norm(center_perpendicular, axis=-1) * 1000.0

    vertex_delta = target - predicted
    vertex_ray = safe_unit(target)
    vertex_parallel_signed = np.sum(vertex_delta * vertex_ray, axis=-1)
    vertex_perpendicular = vertex_delta - vertex_parallel_signed[..., None] * vertex_ray
    vertex_parallel_mm = np.abs(vertex_parallel_signed) * 1000.0
    vertex_perpendicular_mm = np.linalg.norm(vertex_perpendicular, axis=-1) * 1000.0

    ray_corrected = predicted + center_parallel[:, None]
    z_corrected = predicted.copy()
    z_corrected[..., 2] += center_delta[:, None, 2]
    center_corrected = predicted + center_delta[:, None]
    initial_mpvpe = mean_vertex_error(predicted, target)
    ray_mpvpe = mean_vertex_error(ray_corrected, target)
    z_mpvpe = mean_vertex_error(z_corrected, target)
    center_mpvpe = mean_vertex_error(center_corrected, target)

    valid_vertex_delta = vertex_delta[valid]
    valid_vertex_parallel = vertex_parallel_signed[valid]
    vertex_energy = float(np.square(valid_vertex_delta).sum())
    vertex_parallel_energy = float(np.square(valid_vertex_parallel).sum())
    center_energy = float(np.square(center_delta[valid]).sum())
    center_parallel_energy = float(np.square(center_parallel_signed[valid]).sum())
    vertex_ray_fraction = vertex_parallel_energy / max(vertex_energy, 1e-12)
    center_ray_fraction = center_parallel_energy / max(center_energy, 1e-12)
    initial_mean = float(initial_mpvpe[valid].mean())
    ray_mean = float(ray_mpvpe[valid].mean())
    ray_reduction = (initial_mean - ray_mean) / max(initial_mean, 1e-8)

    median_center_uv = float(np.median(center_uv_error[valid]))
    median_vertex_uv = float(np.median(vertex_uv_error[valid]))
    if median_center_uv <= 5.0 and center_ray_fraction >= 0.75 and ray_reduction >= 0.5:
        verdict = "depth_dominant"
    elif median_center_uv <= 10.0 and center_ray_fraction >= 0.5:
        verdict = "mixed_but_depth_is_major"
    else:
        verdict = "not_depth_dominant"

    summary = {
        "method": "v14_gt_camera_ray_depth_alignment_audit_v1",
        "stream_id": stream_id,
        "hand_side": side,
        "frames": int(len(ids)),
        "valid_frames": int(valid.sum()),
        "intrinsics": intrinsics.tolist(),
        "intrinsics_source": intrinsics_source,
        "reprojection_error_px": {
            "hand_centroid": distribution(center_uv_error[valid]),
            "vertices": distribution(vertex_uv_error[valid]),
            "frame_vertex_median": distribution(
                np.median(vertex_uv_error, axis=1)[valid]
            ),
        },
        "center_translation_error_mm": {
            "total": distribution(center_total_mm[valid]),
            "camera_ray_parallel": distribution(center_parallel_mm[valid]),
            "camera_ray_perpendicular": distribution(center_perpendicular_mm[valid]),
            "camera_xy": distribution(
                np.linalg.norm(center_delta[valid, :2], axis=-1) * 1000.0
            ),
            "camera_z_absolute": distribution(np.abs(center_delta[valid, 2]) * 1000.0),
            "ray_energy_fraction": center_ray_fraction,
        },
        "corresponding_vertex_error_mm": {
            "camera_ray_parallel": distribution(vertex_parallel_mm[valid]),
            "camera_ray_perpendicular": distribution(vertex_perpendicular_mm[valid]),
            "ray_energy_fraction": vertex_ray_fraction,
        },
        "mpvpe_mm": {
            "initial": distribution(initial_mpvpe[valid]),
            "after_center_camera_ray_translation": distribution(ray_mpvpe[valid]),
            "after_center_z_only_translation": distribution(z_mpvpe[valid]),
            "after_full_center_translation": distribution(center_mpvpe[valid]),
            "ray_translation_mean_reduction_fraction": ray_reduction,
        },
        "verdict": verdict,
        "criteria": {
            "depth_dominant": {
                "centroid_reprojection_median_px_max": 5.0,
                "center_ray_energy_fraction_min": 0.75,
                "ray_translation_mean_reduction_fraction_min": 0.5,
            }
        },
        "inputs": {
            "trajectory_npz": str(trajectory_path),
            "query_npz": str(query_path),
            "gt_hand_npz": str(gt_path),
        },
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_npz = (
        Path(args.out_npz).expanduser().resolve()
        if args.out_npz
        else out_json.with_suffix(".npz")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_npz,
        frame_ids=ids,
        valid=valid,
        center_reprojection_error_px=center_uv_error.astype(np.float32),
        vertex_reprojection_error_px=vertex_uv_error.astype(np.float32),
        center_translation_total_mm=center_total_mm.astype(np.float32),
        center_translation_ray_parallel_signed_mm=(
            center_parallel_signed * 1000.0
        ).astype(np.float32),
        center_translation_ray_perpendicular_mm=center_perpendicular_mm.astype(
            np.float32
        ),
        center_translation_camera_xyz_mm=(center_delta * 1000.0).astype(np.float32),
        initial_mpvpe_mm=initial_mpvpe.astype(np.float32),
        ray_translation_mpvpe_mm=ray_mpvpe.astype(np.float32),
        z_translation_mpvpe_mm=z_mpvpe.astype(np.float32),
        full_center_translation_mpvpe_mm=center_mpvpe.astype(np.float32),
    )

    print(json.dumps(summary, indent=2))
    print(f"Output: {out_npz}")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
