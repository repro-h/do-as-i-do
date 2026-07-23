#!/usr/bin/env python3
"""Audit HandFlow + FoundationPose initialization on a hybrid pilot."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--object-samples", type=int, default=4096)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantiles(values, scale: float = 1.0) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)] * scale
    if array.size == 0:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def motion_metrics(points: np.ndarray, valid: np.ndarray) -> tuple[list[float], list[float]]:
    speed = []
    acceleration = []
    for index in range(1, len(points)):
        if valid[index - 1] and valid[index]:
            speed.append(float(np.linalg.norm(points[index] - points[index - 1])))
    for index in range(2, len(points)):
        if valid[index - 2] and valid[index - 1] and valid[index]:
            value = points[index] - 2 * points[index - 1] + points[index - 2]
            acceleration.append(float(np.linalg.norm(value)))
    return speed, acceleration


def load_mano_support(mano_data_dir: Path, is_left: bool):
    path = mano_data_dir / ("MANO_LEFT.pkl" if is_left else "MANO_RIGHT.pkl")
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    layer = smplx.MANOLayer(
        model_path=str(mano_data_dir),
        is_rhand=not is_left,
        use_pca=False,
        flat_hand_mean=True,
    )
    layer.eval()
    return (
        layer,
        np.asarray(raw["hands_components"], dtype=np.float32),
        np.asarray(raw["hands_mean"], dtype=np.float32),
    )


def decode_gt_hand(
    label_paths: list[Path],
    meta: dict,
    mano_data_dir: Path,
    is_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    layer, pca_basis, mean_pose = load_mano_support(mano_data_dir, is_left)
    betas = np.asarray(
        meta.get("mano_betas", meta.get("betas", np.zeros(10))),
        dtype=np.float32,
    ).reshape(-1)[:10]
    if len(betas) < 10:
        betas = np.pad(betas, (0, 10 - len(betas)))
    vertices = np.full((len(label_paths), 778, 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(label_paths), dtype=bool)
    for index, path in enumerate(label_paths):
        with np.load(path) as raw:
            pose_m = np.asarray(raw["pose_m"], dtype=np.float32).reshape(-1)
        if len(pose_m) < 51 or np.allclose(pose_m[:51], 0.0):
            continue
        pose_aa = np.concatenate(
            [pose_m[:3], pose_m[3:48] @ pca_basis + mean_pose], axis=0
        ).astype(np.float32)
        matrices = Rotation.from_rotvec(pose_aa.reshape(-1, 3)).as_matrix()
        rotations = torch.from_numpy(matrices.astype(np.float32)).view(1, 16, 3, 3)
        with torch.no_grad():
            output = layer(
                global_orient=rotations[:, 0:1],
                hand_pose=rotations[:, 1:],
                betas=torch.from_numpy(betas).view(1, 10),
                pose2rot=False,
            )
        vertices[index] = (
            output.vertices[0].numpy().astype(np.float32) + pose_m[48:51][None]
        )
        valid[index] = True
    return vertices, valid


def load_object_vertices(path: Path, scale: float, count: int) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * float(scale)
    if len(vertices) > count:
        indices = np.linspace(0, len(vertices) - 1, count, dtype=np.int64)
        vertices = vertices[indices]
    return vertices


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = load_json(path)
    rows = payload.get("by_frame") or payload.get("frames") or {}
    output = {}
    if isinstance(rows, dict):
        iterator = rows.items()
    else:
        iterator = enumerate(rows)
    for key, row in iterator:
        if not isinstance(row, dict) or row.get("object_in_camera") is None:
            continue
        frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
        pose = np.asarray(row["object_in_camera"], dtype=np.float32).reshape(4, 4)
        if np.isfinite(pose).all():
            output[frame] = pose
    return output


def audit_stream(record: dict, handflow_root: Path, mano_data_dir: Path, samples: int):
    stream_id = record["stream_id"]
    handflow_path = handflow_root / stream_id / "handflow_camera_result.npz"
    if not handflow_path.is_file():
        raise FileNotFoundError(handflow_path)
    with np.load(handflow_path, allow_pickle=False) as raw:
        pred_vertices = np.asarray(raw["verts_cam"], dtype=np.float32)
        pred_valid = np.asarray(raw["pred_valid"]).astype(bool)

    stream_dir = Path(record["stream_dir"])
    color_paths = sorted(stream_dir.glob("color_*.jpg"))
    if not color_paths:
        color_paths = sorted(stream_dir.glob("color_*.png"))
    frame_ids = [path.stem.split("_", 1)[1].zfill(6) for path in color_paths]
    label_paths = [stream_dir / f"labels_{frame}.npz" for frame in frame_ids]
    count = min(len(frame_ids), len(pred_vertices))
    frame_ids = frame_ids[:count]
    label_paths = label_paths[:count]
    pred_vertices = pred_vertices[:count]
    pred_valid = pred_valid[:count] & np.isfinite(pred_vertices).all(axis=(1, 2))

    meta = yaml.safe_load(Path(record["meta_path"]).read_text(encoding="utf-8")) or {}
    gt_vertices, gt_valid = decode_gt_hand(
        label_paths,
        meta,
        mano_data_dir,
        record["hand_side"] == "left",
    )
    gt_vertices = gt_vertices[:count]
    gt_valid = gt_valid[:count]
    joint_valid = pred_valid & gt_valid
    pred_centers = np.nanmean(pred_vertices, axis=1)
    gt_centers = np.nanmean(gt_vertices, axis=1)

    mpvpe = []
    center_error = []
    for index in np.flatnonzero(joint_valid):
        mpvpe.append(
            float(np.linalg.norm(pred_vertices[index] - gt_vertices[index], axis=1).mean())
        )
        center_error.append(float(np.linalg.norm(pred_centers[index] - gt_centers[index])))

    poses = pose_rows(Path(record["foundationpose_json"]))
    object_valid = np.asarray([frame in poses for frame in frame_ids], dtype=bool)
    object_translations = np.full((count, 3), np.nan, dtype=np.float32)
    relative_centers = np.full((count, 3), np.nan, dtype=np.float32)
    contact_min = []
    contact_p10 = []
    canonical = load_object_vertices(
        Path(record["sam3d_glb"]),
        record["foundationpose_source_mesh_scale"],
        samples,
    )
    for index, frame in enumerate(frame_ids):
        if not object_valid[index]:
            continue
        pose = poses[frame]
        object_translations[index] = pose[:3, 3]
        if joint_valid[index]:
            relative_centers[index] = pred_centers[index] - pose[:3, 3]
            object_vertices = canonical @ pose[:3, :3].T + pose[:3, 3]
            distances, _ = cKDTree(object_vertices).query(pred_vertices[index], k=1)
            contact_min.append(float(np.min(distances)))
            contact_p10.append(float(np.quantile(distances, 0.1)))

    # Compare temporal hand motion only where both prediction and DexYCB hand
    # annotations are valid. HandFlow may emit plausible meshes while the hand
    # is outside the image, which must not become contact supervision.
    pred_speed, pred_acc = motion_metrics(pred_centers, joint_valid)
    gt_speed, gt_acc = motion_metrics(gt_centers, joint_valid)
    object_speed, object_acc = motion_metrics(object_translations, object_valid)
    relative_valid = joint_valid & object_valid & np.isfinite(relative_centers).all(axis=1)
    relative_speed, relative_acc = motion_metrics(relative_centers, relative_valid)

    return {
        "stream_id": stream_id,
        "hand_side": record["hand_side"],
        "object_name": record["object_name"],
        "num_frames": count,
        "pred_valid_frames": int(pred_valid.sum()),
        "gt_valid_frames": int(gt_valid.sum()),
        "joint_valid_frames": int(joint_valid.sum()),
        "object_valid_frames": int(object_valid.sum()),
        "hand_mpvpe_mm": quantiles(mpvpe, 1000.0),
        "hand_center_error_mm": quantiles(center_error, 1000.0),
        "pred_hand_speed_mm": quantiles(pred_speed, 1000.0),
        "pred_hand_acceleration_mm": quantiles(pred_acc, 1000.0),
        "gt_hand_speed_mm": quantiles(gt_speed, 1000.0),
        "gt_hand_acceleration_mm": quantiles(gt_acc, 1000.0),
        "object_speed_mm": quantiles(object_speed, 1000.0),
        "object_acceleration_mm": quantiles(object_acc, 1000.0),
        "relative_speed_mm": quantiles(relative_speed, 1000.0),
        "relative_acceleration_mm": quantiles(relative_acc, 1000.0),
        "hand_object_min_distance_mm": quantiles(contact_min, 1000.0),
        "hand_object_p10_distance_mm": quantiles(contact_p10, 1000.0),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    mano_data_dir = Path(args.mano_data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(manifest_path)

    rows = []
    failures = []
    aggregate = defaultdict(list)
    aggregate_by_side = {
        "left": defaultdict(list),
        "right": defaultdict(list),
    }
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record['stream_id']}")
        try:
            row = audit_stream(
                record, handflow_root, mano_data_dir, args.object_samples
            )
            rows.append(row)
            for key, value in row.items():
                if isinstance(value, dict) and value.get("count", 0) > 0:
                    aggregate[key].append(value["median"])
                    aggregate_by_side[row["hand_side"]][key].append(value["median"])
        except Exception as error:
            failures.append(
                {
                    "stream_id": record["stream_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    rows_path = out_dir / "stream_metrics.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "manifest": str(manifest_path),
        "handflow_root": str(handflow_root),
        "num_requested": len(records),
        "num_audited": len(rows),
        "num_failed": len(failures),
        "aggregate_stream_medians": {
            key: quantiles(values) for key, values in sorted(aggregate.items())
        },
        "aggregate_stream_medians_by_side": {
            side: {
                key: quantiles(values)
                for key, values in sorted(side_values.items())
            }
            for side, side_values in aggregate_by_side.items()
        },
        "failures": failures,
        "streams": rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "streams"}, indent=2))
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
