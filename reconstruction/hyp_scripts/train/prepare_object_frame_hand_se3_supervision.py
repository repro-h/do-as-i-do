#!/usr/bin/env python3
"""Export compact hand-root SE(3) supervision in the SAM object frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--global-supervision-root", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--window-jsonl", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--min-valid-frames", type=int, default=8)
    parser.add_argument("--scale-warning-threshold", type=float, default=0.1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--stream-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scalar_string(value: np.ndarray) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def frame_string(value, fallback: int) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else str(fallback)).zfill(6)


def mirror_pose(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=np.float64).copy()
    output[:3, :3] = MIRROR_X @ output[:3, :3] @ MIRROR_X
    output[:3, 3] = MIRROR_X @ output[:3, 3]
    return output


def mirror_rotations(matrices: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjk,kl->til", MIRROR_X, matrices, MIRROR_X)


def rotation_to_6d(matrix: np.ndarray) -> np.ndarray:
    """Store the first two rotation columns as [r0, r1]."""
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def canonical_alignment(path: Path) -> tuple[np.ndarray, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("valid") is False:
        raise ValueError(f"Canonical alignment is marked invalid: {path}")
    similarity = payload["raw_sam_to_ycb_similarity"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        similarity["rotation"], dtype=np.float64
    ).reshape(3, 3)
    matrix[:3, 3] = np.asarray(
        similarity["translation_m"], dtype=np.float64
    ).reshape(3)
    rotation = matrix[:3, :3]
    if (
        abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4
        or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-4
    ):
        raise ValueError(f"Invalid canonical rotation: {path}")
    production = payload.get("production_sam_to_ycb_similarity") or {}
    residual_scale = production.get("residual_scale")
    if residual_scale is not None:
        residual_scale = float(residual_scale)
    return matrix, residual_scale


def load_gt_object_pose(path: Path, grasp_index: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if "pose_y" not in payload.files:
            return None
        poses = np.asarray(payload["pose_y"], dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None]
    if poses.ndim != 3 or grasp_index >= len(poses):
        return None
    value = poses[grasp_index]
    if value.shape[0] < 3 or value.shape[1] < 4:
        return None
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = value[:3, :4]
    return matrix if np.isfinite(matrix).all() else None


def camera_point_to_object(point: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return pose[:3, :3].T @ (point - pose[:3, 3])


def camera_points_to_object(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points - pose[:3, 3]) @ pose[:3, :3]


def distribution(values: np.ndarray, scale: float = 1.0) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def load_metadata(stream_dir: Path) -> dict:
    path = stream_dir.parent / "meta.yml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def prepare_stream(
    record: dict,
    global_root: Path,
    canonical_root: Path,
    out_path: Path,
    scale_warning_threshold: float,
) -> dict:
    stream_id = str(record["stream_id"])
    object_name = str(record["object_name"])
    stream_dir = Path(record["stream_dir"]).expanduser().resolve()
    global_path = global_root / f"{stream_id}.npz"
    canonical_path = canonical_root / object_name / "canonical_alignment.json"
    for path in (global_path, canonical_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(global_path, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"])
        pred_joints = np.asarray(archive["pred_joints_3d"], dtype=np.float64)
        gt_joints = np.asarray(archive["gt_joints_3d"], dtype=np.float64)
        gt_root_rotvec = np.asarray(
            archive["gt_root_rotvec"], dtype=np.float64
        )
        initial_root_rotvec = np.asarray(
            archive["initial_root_rotvec"], dtype=np.float64
        )
        filtered_pose = np.asarray(archive["object_pose"], dtype=np.float64)
        hand_valid = np.asarray(archive["hand_valid"], dtype=bool)
        gt_valid = np.asarray(archive["gt_valid"], dtype=bool)
        object_valid = np.asarray(archive["object_valid"], dtype=bool)
        normalized_left = bool(np.asarray(archive["normalized_left"]).item())
        hand_side = scalar_string(archive["hand_side"])
        object_extents = np.asarray(
            archive["object_extents_metric"]
            if "object_extents_metric" in archive.files
            else np.zeros(3),
            dtype=np.float32,
        )

    count = min(
        len(frame_ids), len(pred_joints), len(gt_joints),
        len(gt_root_rotvec), len(initial_root_rotvec), len(filtered_pose),
        len(hand_valid), len(gt_valid), len(object_valid),
    )
    frame_ids = frame_ids[:count]
    pred_joints = pred_joints[:count]
    gt_joints = gt_joints[:count]
    gt_root_rotvec = gt_root_rotvec[:count]
    initial_root_rotvec = initial_root_rotvec[:count]
    filtered_pose = filtered_pose[:count]
    hand_valid = hand_valid[:count]
    gt_valid = gt_valid[:count]
    object_valid = object_valid[:count]

    sam_to_ycb, residual_scale = canonical_alignment(canonical_path)
    if normalized_left:
        sam_to_ycb = mirror_pose(sam_to_ycb)

    metadata = load_metadata(stream_dir)
    grasp_index = int(metadata.get("ycb_grasp_ind", 0))
    initial_root_valid = np.isfinite(initial_root_rotvec).all(axis=1)
    gt_root_valid = np.isfinite(gt_root_rotvec).all(axis=1)
    initial_root_rotation = np.full((count, 3, 3), np.nan, dtype=np.float64)
    target_root_rotation = np.full((count, 3, 3), np.nan, dtype=np.float64)
    if initial_root_valid.any():
        matrices = Rotation.from_rotvec(
            initial_root_rotvec[initial_root_valid]
        ).as_matrix()
        if normalized_left:
            matrices = mirror_rotations(matrices)
        initial_root_rotation[initial_root_valid] = matrices
    if gt_root_valid.any():
        target_root_rotation[gt_root_valid] = Rotation.from_rotvec(
            gt_root_rotvec[gt_root_valid]
        ).as_matrix()

    initial_translation_object = np.full((count, 3), np.nan, dtype=np.float64)
    target_translation_object = np.full((count, 3), np.nan, dtype=np.float64)
    initial_rotation_object = np.full((count, 3, 3), np.nan, dtype=np.float64)
    target_rotation_object = np.full((count, 3, 3), np.nan, dtype=np.float64)
    pred_joints_object = np.full_like(pred_joints, np.nan, dtype=np.float64)
    gt_object_pose = np.full((count, 4, 4), np.nan, dtype=np.float64)
    gt_sam_object_pose = np.full((count, 4, 4), np.nan, dtype=np.float64)
    gt_object_valid = np.zeros(count, dtype=bool)

    for index in range(count):
        frame = frame_string(frame_ids[index], index)
        if hand_valid[index] and object_valid[index]:
            pred_joints_object[index] = camera_points_to_object(
                pred_joints[index], filtered_pose[index]
            )
            initial_translation_object[index] = pred_joints_object[index, 0]
            if initial_root_valid[index]:
                initial_rotation_object[index] = (
                    filtered_pose[index, :3, :3].T
                    @ initial_root_rotation[index]
                )

        gt_ycb_pose = load_gt_object_pose(
            stream_dir / f"labels_{frame}.npz", grasp_index
        )
        if gt_ycb_pose is None:
            continue
        if normalized_left:
            gt_ycb_pose = mirror_pose(gt_ycb_pose)
        gt_object_pose[index] = gt_ycb_pose
        gt_sam_pose = gt_ycb_pose @ sam_to_ycb
        gt_sam_object_pose[index] = gt_sam_pose
        gt_object_valid[index] = True
        if gt_valid[index]:
            target_translation_object[index] = camera_point_to_object(
                gt_joints[index, 0], gt_sam_pose
            )
            if gt_root_valid[index]:
                target_rotation_object[index] = (
                    gt_sam_pose[:3, :3].T @ target_root_rotation[index]
                )

    valid_translation = (
        hand_valid
        & gt_valid
        & object_valid
        & gt_object_valid
        & np.isfinite(initial_translation_object).all(axis=1)
        & np.isfinite(target_translation_object).all(axis=1)
    )
    valid_rotation = (
        valid_translation
        & initial_root_valid
        & gt_root_valid
        & np.isfinite(initial_rotation_object).all(axis=(1, 2))
        & np.isfinite(target_rotation_object).all(axis=(1, 2))
    )
    translation_residual = target_translation_object - initial_translation_object
    rotation_residual = np.full((count, 3, 3), np.nan, dtype=np.float64)
    rotation_residual[valid_rotation] = np.einsum(
        "tij,tkj->tik",
        target_rotation_object[valid_rotation],
        initial_rotation_object[valid_rotation],
    )
    rotation_residual_rotvec = np.full((count, 3), np.nan, dtype=np.float64)
    if valid_rotation.any():
        rotation_residual_rotvec[valid_rotation] = Rotation.from_matrix(
            rotation_residual[valid_rotation]
        ).as_rotvec()

    initial_rotation_6d = np.full((count, 6), np.nan, dtype=np.float64)
    target_rotation_6d = np.full((count, 6), np.nan, dtype=np.float64)
    initial_rotation_6d[valid_rotation] = rotation_to_6d(
        initial_rotation_object[valid_rotation]
    )
    target_rotation_6d[valid_rotation] = rotation_to_6d(
        target_rotation_object[valid_rotation]
    )
    target_translation_camera = np.full((count, 3), np.nan, dtype=np.float64)
    target_rotation_camera = np.full((count, 3, 3), np.nan, dtype=np.float64)
    translation_correction_camera = np.full(
        (count, 3), np.nan, dtype=np.float64
    )
    for index in np.flatnonzero(valid_translation):
        target_translation_camera[index] = (
            filtered_pose[index, :3, :3]
            @ target_translation_object[index]
            + filtered_pose[index, :3, 3]
        )
        translation_correction_camera[index] = (
            target_translation_camera[index] - pred_joints[index, 0]
        )
    for index in np.flatnonzero(valid_rotation):
        target_rotation_camera[index] = (
            filtered_pose[index, :3, :3]
            @ target_rotation_object[index]
        )
    scale_warning = bool(
        residual_scale is not None
        and abs(residual_scale - 1.0) > scale_warning_threshold
    )

    np.savez_compressed(
        out_path,
        frame_ids=frame_ids,
        pred_joints_object=pred_joints_object.astype(np.float32),
        initial_translation_object=initial_translation_object.astype(np.float32),
        target_translation_object=target_translation_object.astype(np.float32),
        translation_residual_object=translation_residual.astype(np.float32),
        initial_rotation_object=initial_rotation_object.astype(np.float32),
        target_rotation_object=target_rotation_object.astype(np.float32),
        rotation_residual_object=rotation_residual.astype(np.float32),
        rotation_residual_rotvec=rotation_residual_rotvec.astype(np.float32),
        initial_rotation_6d=initial_rotation_6d.astype(np.float32),
        target_rotation_6d=target_rotation_6d.astype(np.float32),
        initial_wrist_camera=pred_joints[:, 0].astype(np.float32),
        target_wrist_camera_oracle=target_translation_camera.astype(np.float32),
        target_root_rotation_camera_oracle=target_rotation_camera.astype(
            np.float32
        ),
        translation_correction_camera=translation_correction_camera.astype(
            np.float32
        ),
        filtered_object_pose=filtered_pose.astype(np.float32),
        gt_ycb_object_pose=gt_object_pose.astype(np.float32),
        gt_sam_object_pose=gt_sam_object_pose.astype(np.float32),
        object_extents_metric=object_extents,
        hand_valid=hand_valid,
        gt_valid=gt_valid,
        filtered_object_valid=object_valid,
        gt_object_valid=gt_object_valid,
        valid_translation=valid_translation,
        valid_rotation=valid_rotation,
        normalized_left=np.asarray(normalized_left),
        hand_side=np.asarray(hand_side),
        object_name=np.asarray(object_name),
        stream_id=np.asarray(stream_id),
        canonical_sam_to_ycb=sam_to_ycb.astype(np.float32),
        canonical_residual_scale=np.asarray(
            np.nan if residual_scale is None else residual_scale,
            dtype=np.float32,
        ),
        canonical_scale_warning=np.asarray(scale_warning),
        rotation_6d_convention=np.asarray("first_two_columns_r0_r1"),
        source_global_supervision=np.asarray(str(global_path)),
        source_canonical_alignment=np.asarray(str(canonical_path)),
    )

    translation_error = np.linalg.norm(translation_residual[valid_translation], axis=1)
    rotation_error = np.degrees(
        np.linalg.norm(rotation_residual_rotvec[valid_rotation], axis=1)
    )
    return {
        "frames": count,
        "valid_translation": int(valid_translation.sum()),
        "valid_rotation": int(valid_rotation.sum()),
        "missing_fraction": float(1.0 - valid_translation.mean()),
        "scale_warning": scale_warning,
        "canonical_residual_scale": residual_scale,
        "initial_to_target_translation_mm": distribution(
            translation_error, scale=1000.0
        ),
        "initial_to_target_rotation_deg": distribution(rotation_error),
    }


def window_starts(num_frames: int, window_size: int, stride: int) -> list[int]:
    if num_frames < window_size:
        return []
    max_start = num_frames - window_size
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("Window size and stride must be positive")

    manifest = Path(args.manifest).expanduser().resolve()
    global_root = Path(args.global_supervision_root).expanduser().resolve()
    canonical_root = Path(args.canonical_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    window_path = Path(args.window_jsonl).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    window_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(manifest)
    requested_streams = set(args.stream_id)
    if requested_streams:
        records = [
            row for row in records
            if str(row.get("stream_id")) in requested_streams
        ]
        found = {str(row["stream_id"]) for row in records}
        missing = sorted(requested_streams - found)
        if missing:
            raise KeyError(f"Streams not found in manifest: {missing}")
    selected = [
        row for index, row in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        selected = selected[: args.limit]

    windows = []
    rows = []
    failures = []
    for index, record in enumerate(selected, start=1):
        stream_id = str(record["stream_id"])
        out_path = out_root / f"{stream_id}.npz"
        print(f"[{index}/{len(selected)}] {stream_id}", flush=True)
        try:
            if args.overwrite or not out_path.is_file():
                metrics = prepare_stream(
                    record,
                    global_root,
                    canonical_root,
                    out_path,
                    args.scale_warning_threshold,
                )
            else:
                with np.load(out_path, allow_pickle=False) as archive:
                    valid_translation = np.asarray(
                        archive["valid_translation"], dtype=bool
                    )
                    valid_rotation = np.asarray(
                        archive["valid_rotation"], dtype=bool
                    )
                    metrics = {
                        "frames": len(archive["frame_ids"]),
                        "valid_translation": int(valid_translation.sum()),
                        "valid_rotation": int(valid_rotation.sum()),
                        "missing_fraction": float(
                            1.0 - valid_translation.mean()
                        ),
                        "scale_warning": bool(
                            np.asarray(archive["canonical_scale_warning"]).item()
                        ),
                    }
            with np.load(out_path, allow_pickle=False) as archive:
                valid = np.asarray(archive["valid_rotation"], dtype=bool)
            for start in window_starts(
                metrics["frames"], args.window_size, args.window_stride
            ):
                end = start + args.window_size
                if int(valid[start:end].sum()) < args.min_valid_frames:
                    continue
                windows.append(
                    {
                        "stream_id": stream_id,
                        "object_name": str(record["object_name"]),
                        "hand_side": str(record["hand_side"]),
                        "supervision_npz": str(out_path),
                        "start": start,
                        "end": end,
                    }
                )
            rows.append({"stream_id": stream_id, **metrics})
        except Exception as error:
            failures.append(
                {
                    "stream_id": stream_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  failed: {type(error).__name__}: {error}", flush=True)

    with window_path.open("w", encoding="utf-8") as handle:
        for row in windows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "manifest": str(manifest),
        "split": args.split,
        "global_supervision_root": str(global_root),
        "canonical_root": str(canonical_root),
        "out_root": str(out_root),
        "window_jsonl": str(window_path),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_requested": len(selected),
        "num_completed": len(rows),
        "num_windows": len(windows),
        "num_scale_warnings": sum(row["scale_warning"] for row in rows),
        "num_failures": len(failures),
        "streams": rows,
        "failures": failures,
    }
    summary_path = window_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
