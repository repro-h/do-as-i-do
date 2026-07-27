#!/usr/bin/env python3
"""Prepare compact supervision for hand-only temporal rigid correction."""

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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--filtered-object-root", required=True)
    parser.add_argument("--dexycb-model-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--window-jsonl", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--min-valid-hand-frames", type=int, default=8)
    parser.add_argument("--hand-samples", type=int, default=128)
    parser.add_argument("--object-anchors", type=int, default=384)
    parser.add_argument("--gt-contact-threshold-mm", type=float, default=8.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        result = trimesh.util.concatenate(geometries)
    else:
        result = loaded
    if not len(result.vertices) or not len(result.faces):
        raise ValueError(f"Empty mesh: {path}")
    return result


def surface_anchors(
    path: Path, scale: float, count: int
) -> tuple[np.ndarray, np.ndarray]:
    value = mesh(path)
    vertices = np.asarray(value.vertices, dtype=np.float64) * float(scale)
    faces = np.asarray(value.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=-1)
    cdf = np.cumsum(np.maximum(area, 1e-12))
    targets = (np.arange(count, dtype=np.float64) + 0.5) / count * cdf[-1]
    selected = np.searchsorted(cdf, targets).clip(0, len(faces) - 1)
    sequence = np.arange(count, dtype=np.float64)
    u = np.mod((sequence + 0.5) * 0.7548776662466927, 1.0)
    v = np.mod((sequence + 0.5) * 0.5698402909980532, 1.0)
    sqrt_u = np.sqrt(np.maximum(u, 1e-8))
    barycentric = np.stack(
        [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v], axis=-1
    )
    anchors = (triangles[selected] * barycentric[:, :, None]).sum(axis=1)
    normals = cross[selected] / np.maximum(
        np.linalg.norm(cross[selected], axis=-1, keepdims=True), 1e-8
    )
    return anchors.astype(np.float32), normals.astype(np.float32)


@lru_cache(maxsize=4)
def mano_resources(mano_data_dir_text: str, is_left: bool):
    mano_data_dir = Path(mano_data_dir_text)
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


def decode_gt_vertices(
    label_paths: list[Path],
    meta: dict,
    mano_data_dir: Path,
    is_left: bool,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    layer, pca_basis, mean_pose = mano_resources(str(mano_data_dir), is_left)
    betas = np.asarray(
        meta.get("mano_betas", meta.get("betas", np.zeros(10))), dtype=np.float32
    ).reshape(-1)[:10]
    betas = np.pad(betas, (0, max(0, 10 - len(betas))))[:10]
    valid_indices, poses, translations = [], [], []
    for index, path in enumerate(label_paths):
        if not path.is_file():
            continue
        with np.load(path) as raw:
            pose_m = np.asarray(raw["pose_m"], dtype=np.float32).reshape(-1)
        if len(pose_m) < 51 or np.allclose(pose_m[:51], 0.0):
            continue
        poses.append(
            np.concatenate([pose_m[:3], pose_m[3:48] @ pca_basis + mean_pose])
        )
        translations.append(pose_m[48:51])
        valid_indices.append(index)

    output = np.full(
        (len(label_paths), len(sample_indices), 3), np.nan, dtype=np.float32
    )
    valid = np.zeros(len(label_paths), dtype=bool)
    if not valid_indices:
        return output, valid
    matrices = Rotation.from_rotvec(np.asarray(poses).reshape(-1, 3)).as_matrix()
    rotations = torch.from_numpy(matrices.astype(np.float32)).view(-1, 16, 3, 3)
    batch_betas = torch.from_numpy(betas).view(1, 10).expand(len(valid_indices), -1)
    with torch.no_grad():
        result = layer(
            global_orient=rotations[:, 0:1],
            hand_pose=rotations[:, 1:],
            betas=batch_betas,
            pose2rot=False,
        )
    vertices = result.vertices.numpy().astype(np.float32)
    vertices += np.asarray(translations, dtype=np.float32)[:, None]
    output[np.asarray(valid_indices)] = vertices[:, sample_indices]
    valid[np.asarray(valid_indices)] = True
    return output, valid


def gt_contact_mask(
    gt_vertices: np.ndarray,
    valid: np.ndarray,
    label_paths: list[Path],
    grasp_index: int,
    cad_vertices: np.ndarray,
    threshold_m: float,
) -> np.ndarray:
    result = np.zeros(gt_vertices.shape[:2], dtype=bool)
    for index, path in enumerate(label_paths):
        if not valid[index] or not path.is_file():
            continue
        with np.load(path) as raw:
            poses = np.asarray(raw.get("pose_y", []), dtype=np.float32)
        if poses.ndim != 3 or grasp_index >= len(poses):
            continue
        pose = poses[grasp_index]
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        posed = cad_vertices @ rotation.T + translation
        distance, _ = cKDTree(posed).query(gt_vertices[index], k=1)
        result[index] = distance < threshold_m
    return result


def prepare_stream(
    record: dict,
    args: argparse.Namespace,
    handflow_root: Path,
    filtered_root: Path,
    model_root: Path,
    mano_data_dir: Path,
    out_path: Path,
) -> dict:
    stream_id = record["stream_id"]
    handflow_path = handflow_root / stream_id / "handflow_camera_result.npz"
    with np.load(handflow_path, allow_pickle=False) as raw:
        vertices = np.asarray(raw["verts_cam"], dtype=np.float32)
        hand_valid = np.asarray(raw["pred_valid"]).astype(bool)
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32).reshape(3, 3)
    if vertices.shape[1] < args.hand_samples:
        raise ValueError(f"Too few hand vertices: {vertices.shape}")
    sample_indices = np.linspace(
        0, vertices.shape[1] - 1, args.hand_samples, dtype=np.int64
    )
    pred_vertices = vertices[:, sample_indices]
    pred_centers = np.nanmean(pred_vertices, axis=1).astype(np.float32)

    stream_dir = Path(record["stream_dir"])
    color_paths = sorted(stream_dir.glob("color_*.jpg")) or sorted(
        stream_dir.glob("color_*.png")
    )
    frame_ids = [path.stem.rsplit("_", 1)[-1].zfill(6) for path in color_paths]
    count = min(len(frame_ids), len(pred_vertices))
    frame_ids = frame_ids[:count]
    pred_vertices = pred_vertices[:count]
    pred_centers = pred_centers[:count]
    hand_valid = (
        hand_valid[:count]
        & np.isfinite(pred_vertices).all(axis=(1, 2))
        & np.isfinite(pred_centers).all(axis=1)
    )
    label_paths = [stream_dir / f"labels_{frame}.npz" for frame in frame_ids]
    meta = yaml.safe_load(Path(record["meta_path"]).read_text(encoding="utf-8")) or {}
    gt_vertices, gt_valid = decode_gt_vertices(
        label_paths,
        meta,
        mano_data_dir,
        record["hand_side"] == "left",
        sample_indices,
    )
    gt_centers = np.full((count, 3), np.nan, dtype=np.float32)
    gt_centers[gt_valid] = gt_vertices[gt_valid].mean(axis=1)

    filtered_json = (
        filtered_root
        / args.split
        / stream_id
        / "segmented_ekf_rts"
        / "foundationpose_segmented_ekf_rts.json"
    )
    poses = pose_rows(filtered_json)
    object_pose = np.full((count, 4, 4), np.nan, dtype=np.float32)
    object_valid = np.zeros(count, dtype=bool)
    for index, frame in enumerate(frame_ids):
        if frame in poses:
            object_pose[index] = poses[frame]
            object_valid[index] = True

    anchors, normals = surface_anchors(
        Path(record["sam3d_glb"]),
        float(record["foundationpose_source_mesh_scale"]),
        args.object_anchors,
    )
    cad = mesh(model_root / record["object_name"] / "textured_simple.obj")
    cad_vertices = np.asarray(cad.vertices, dtype=np.float32)
    contact = gt_contact_mask(
        gt_vertices,
        gt_valid,
        label_paths,
        int(meta["ycb_grasp_ind"]),
        cad_vertices,
        args.gt_contact_threshold_mm / 1000.0,
    )
    supervision_valid = hand_valid & gt_valid & object_valid
    np.savez_compressed(
        out_path,
        frame_ids=np.asarray(frame_ids),
        pred_hand_vertices=pred_vertices,
        pred_hand_center=pred_centers,
        pred_hand_valid=hand_valid,
        gt_hand_vertices=gt_vertices,
        gt_hand_center=gt_centers,
        gt_hand_valid=gt_valid,
        object_pose=object_pose,
        object_valid=object_valid,
        object_anchors_local=anchors,
        object_normals_local=normals,
        gt_contact_candidates=contact,
        supervision_valid=supervision_valid,
        hand_sample_indices=sample_indices,
        intrinsics=intrinsics,
        stream_id=np.asarray(stream_id),
        hand_side=np.asarray(record["hand_side"]),
        object_name=np.asarray(record["object_name"]),
        filtered_object_json=np.asarray(str(filtered_json)),
    )
    return {
        "frames": count,
        "valid": int(supervision_valid.sum()),
        "contact_frames": int((contact.any(axis=1) & supervision_valid).sum()),
    }


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    manifest = Path(args.manifest).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    filtered_root = Path(args.filtered_object_root).expanduser().resolve()
    model_root = Path(args.dexycb_model_root).expanduser().resolve()
    mano_data_dir = Path(args.mano_data_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    window_path = Path(args.window_jsonl).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    window_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(manifest)
    selected = [
        row for index, row in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    windows, failures = [], []
    for index, record in enumerate(selected, start=1):
        stream_id = record["stream_id"]
        out_path = out_root / f"{stream_id}.npz"
        print(f"[{index}/{len(selected)}] {stream_id}", flush=True)
        try:
            if args.overwrite or not out_path.is_file():
                metrics = prepare_stream(
                    record, args, handflow_root, filtered_root,
                    model_root, mano_data_dir, out_path
                )
            else:
                with np.load(out_path, allow_pickle=False) as raw:
                    metrics = {
                        "frames": len(raw["frame_ids"]),
                        "valid": int(raw["supervision_valid"].sum()),
                    }
            with np.load(out_path, allow_pickle=False) as raw:
                valid = np.asarray(raw["supervision_valid"]).astype(bool)
            max_start = max(0, metrics["frames"] - args.window_size)
            starts = list(
                range(0, max_start + 1, args.window_stride)
            )
            if starts[-1] != max_start:
                starts.append(max_start)
            for start in starts:
                end = start + args.window_size
                if valid[start:end].sum() >= args.min_valid_hand_frames:
                    windows.append(
                        {
                            "stream_id": stream_id,
                            "supervision_npz": str(out_path),
                            "start": start,
                            "end": end,
                        }
                    )
        except Exception as error:
            failures.append(
                {"stream_id": stream_id, "error": f"{type(error).__name__}: {error}"}
            )
            print(f"  failed: {type(error).__name__}: {error}", flush=True)
    with window_path.open("w", encoding="utf-8") as handle:
        for row in windows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "manifest": str(manifest),
        "split": args.split,
        "num_requested": len(selected),
        "num_windows": len(windows),
        "num_failures": len(failures),
        "failures": failures,
    }
    window_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
