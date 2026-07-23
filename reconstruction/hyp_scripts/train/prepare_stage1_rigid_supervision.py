#!/usr/bin/env python3
"""Prepare compact temporal supervision for the Stage-1 rigid refiner."""

from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--dexycb-model-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--window-jsonl", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--min-valid-hand-frames", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mesh_centroid(path: Path, scale: float = 1.0) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * float(scale)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid mesh vertices: {path}")
    return vertices.mean(axis=0)


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = load_json(path)
    rows = payload.get("by_frame") or payload.get("frames") or payload.get("poses") or {}
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    output = {}
    for key, row in iterator:
        if not isinstance(row, dict) or row.get("object_in_camera") is None:
            continue
        frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
        pose = np.asarray(row["object_in_camera"], dtype=np.float32).reshape(4, 4)
        if np.isfinite(pose).all():
            output[frame] = pose
    return output


def rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, :2].T.reshape(-1).astype(np.float32)


@lru_cache(maxsize=4)
def load_mano_layer(mano_data_dir: Path, is_left: bool):
    model_name = "MANO_LEFT.pkl" if is_left else "MANO_RIGHT.pkl"
    with (mano_data_dir / model_name).open("rb") as handle:
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


def decode_gt_hand_centers(
    label_paths: list[Path],
    meta: dict,
    mano_data_dir: Path,
    is_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    layer, pca_basis, mean_pose = load_mano_layer(mano_data_dir, is_left)
    betas = np.asarray(
        meta.get("mano_betas", meta.get("betas", np.zeros(10))), dtype=np.float32
    ).reshape(-1)[:10]
    betas = np.pad(betas, (0, max(0, 10 - len(betas))))[:10]

    valid_indices = []
    pose_vectors = []
    translations = []
    for index, path in enumerate(label_paths):
        if not path.is_file():
            continue
        with np.load(path) as raw:
            pose_m = np.asarray(raw["pose_m"], dtype=np.float32).reshape(-1)
        if len(pose_m) < 51 or np.allclose(pose_m[:51], 0.0):
            continue
        pose_aa = np.concatenate(
            [pose_m[:3], pose_m[3:48] @ pca_basis + mean_pose]
        ).astype(np.float32)
        valid_indices.append(index)
        pose_vectors.append(pose_aa)
        translations.append(pose_m[48:51])

    centers = np.full((len(label_paths), 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(label_paths), dtype=bool)
    if not valid_indices:
        return centers, valid

    matrices = Rotation.from_rotvec(
        np.asarray(pose_vectors, dtype=np.float32).reshape(-1, 3)
    ).as_matrix()
    rotations = torch.from_numpy(matrices.astype(np.float32)).view(-1, 16, 3, 3)
    batch_betas = torch.from_numpy(betas).view(1, 10).expand(len(valid_indices), -1)
    with torch.no_grad():
        output = layer(
            global_orient=rotations[:, 0:1],
            hand_pose=rotations[:, 1:],
            betas=batch_betas,
            pose2rot=False,
        )
    vertices = output.vertices.numpy().astype(np.float32)
    vertices += np.asarray(translations, dtype=np.float32)[:, None]
    centers[np.asarray(valid_indices)] = vertices.mean(axis=1)
    valid[np.asarray(valid_indices)] = True
    return centers, valid


def gt_object_centers(
    label_paths: list[Path],
    grasp_index: int,
    canonical_centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.full((len(label_paths), 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(label_paths), dtype=bool)
    for index, path in enumerate(label_paths):
        if not path.is_file():
            continue
        with np.load(path) as raw:
            if "pose_y" not in raw:
                continue
            poses = np.asarray(raw["pose_y"], dtype=np.float32)
        if poses.ndim != 3 or grasp_index >= len(poses):
            continue
        pose = poses[grasp_index]
        if pose.shape == (3, 4):
            rotation, translation = pose[:, :3], pose[:, 3]
        elif pose.shape == (4, 4):
            rotation, translation = pose[:3, :3], pose[:3, 3]
        else:
            continue
        if np.isfinite(pose).all():
            centers[index] = canonical_centroid @ rotation.T + translation
            valid[index] = True
    return centers, valid


def prepare_stream(
    record: dict,
    handflow_root: Path,
    model_root: Path,
    mano_data_dir: Path,
    out_path: Path,
) -> dict:
    stream_id = record["stream_id"]
    handflow_path = handflow_root / stream_id / "handflow_camera_result.npz"
    with np.load(handflow_path, allow_pickle=False) as raw:
        pred_hand_center = np.asarray(raw["hand_center_cam"], dtype=np.float32)
        pred_hand_valid = np.asarray(raw["pred_valid"]).astype(bool)
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32).reshape(3, 3)

    stream_dir = Path(record["stream_dir"])
    color_paths = sorted(stream_dir.glob("color_*.jpg"))
    if not color_paths:
        color_paths = sorted(stream_dir.glob("color_*.png"))
    frame_ids = [path.stem.split("_", 1)[1].zfill(6) for path in color_paths]
    count = min(len(frame_ids), len(pred_hand_center))
    frame_ids = frame_ids[:count]
    pred_hand_center = pred_hand_center[:count]
    pred_hand_valid = pred_hand_valid[:count] & np.isfinite(pred_hand_center).all(axis=1)
    label_paths = [stream_dir / f"labels_{frame}.npz" for frame in frame_ids]

    meta = yaml.safe_load(Path(record["meta_path"]).read_text(encoding="utf-8")) or {}
    gt_hand_center, gt_hand_valid = decode_gt_hand_centers(
        label_paths,
        meta,
        mano_data_dir,
        record["hand_side"] == "left",
    )

    sam_centroid = mesh_centroid(
        Path(record["sam3d_glb"]), record["foundationpose_source_mesh_scale"]
    )
    poses = pose_rows(Path(record["foundationpose_json"]))
    pred_object_center = np.full((count, 3), np.nan, dtype=np.float32)
    pred_object_rot6d = np.full((count, 6), np.nan, dtype=np.float32)
    pred_object_valid = np.zeros(count, dtype=bool)
    for index, frame in enumerate(frame_ids):
        if frame not in poses:
            continue
        pose = poses[frame]
        pred_object_center[index] = sam_centroid @ pose[:3, :3].T + pose[:3, 3]
        pred_object_rot6d[index] = rotation_6d(pose[:3, :3])
        pred_object_valid[index] = True

    cad_path = model_root / record["object_name"] / "textured_simple.obj"
    cad_centroid = mesh_centroid(cad_path)
    grasp_index = int(meta["ycb_grasp_ind"])
    gt_object_center, gt_object_valid = gt_object_centers(
        label_paths, grasp_index, cad_centroid
    )

    hand_supervision_valid = pred_hand_valid & gt_hand_valid
    object_supervision_valid = pred_object_valid & gt_object_valid
    relative_supervision_valid = hand_supervision_valid & object_supervision_valid
    np.savez_compressed(
        out_path,
        frame_ids=np.asarray(frame_ids),
        pred_hand_center=pred_hand_center,
        pred_hand_valid=pred_hand_valid,
        pred_object_center=pred_object_center,
        pred_object_rot6d=pred_object_rot6d,
        pred_object_valid=pred_object_valid,
        gt_hand_center=gt_hand_center,
        gt_hand_valid=gt_hand_valid,
        gt_object_center=gt_object_center,
        gt_object_valid=gt_object_valid,
        hand_supervision_valid=hand_supervision_valid,
        object_supervision_valid=object_supervision_valid,
        relative_supervision_valid=relative_supervision_valid,
        hand_side=np.asarray(record["hand_side"]),
        object_name=np.asarray(record["object_name"]),
        stream_id=np.asarray(stream_id),
        intrinsics=intrinsics,
    )
    return {
        "num_frames": count,
        "num_hand_supervision": int(hand_supervision_valid.sum()),
        "num_object_supervision": int(object_supervision_valid.sum()),
        "num_relative_supervision": int(relative_supervision_valid.sum()),
    }


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    manifest = Path(args.manifest).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    model_root = Path(args.dexycb_model_root).expanduser().resolve()
    mano_data_dir = Path(args.mano_data_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    window_path = Path(args.window_jsonl).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    window_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(manifest)
    selected = [
        record
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    windows = []
    failures = []
    for index, record in enumerate(selected, start=1):
        stream_id = record["stream_id"]
        out_path = out_root / f"{stream_id}.npz"
        print(f"[{index}/{len(selected)}] {stream_id}")
        try:
            if args.overwrite or not out_path.is_file():
                metrics = prepare_stream(
                    record, handflow_root, model_root, mano_data_dir, out_path
                )
            else:
                with np.load(out_path, allow_pickle=False) as raw:
                    metrics = {
                        "num_frames": len(raw["frame_ids"]),
                        "num_hand_supervision": int(raw["hand_supervision_valid"].sum()),
                    }
            with np.load(out_path, allow_pickle=False) as raw:
                hand_valid = np.asarray(raw["hand_supervision_valid"]).astype(bool)
                object_valid = np.asarray(raw["object_supervision_valid"]).astype(bool)
            for start in range(
                0,
                max(0, metrics["num_frames"] - args.window_size + 1),
                args.window_stride,
            ):
                end = start + args.window_size
                if object_valid[start:end].all() and (
                    hand_valid[start:end].sum() >= args.min_valid_hand_frames
                ):
                    windows.append(
                        {
                            "stream_id": stream_id,
                            "supervision_npz": str(out_path),
                            "start": start,
                            "end": end,
                            "length": args.window_size,
                            "hand_side": record["hand_side"],
                            "object_name": record["object_name"],
                        }
                    )
        except Exception as error:
            failures.append(
                {"stream_id": stream_id, "error": f"{type(error).__name__}: {error}"}
            )

    with window_path.open("w", encoding="utf-8") as handle:
        for row in windows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "manifest": str(manifest),
        "handflow_root": str(handflow_root),
        "out_root": str(out_root),
        "window_jsonl": str(window_path),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_requested": len(selected),
        "num_windows": len(windows),
        "num_failures": len(failures),
        "failures": failures,
    }
    summary_path = window_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
