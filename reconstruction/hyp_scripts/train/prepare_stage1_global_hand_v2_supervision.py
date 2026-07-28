#!/usr/bin/env python3
"""Export compact wrist/root supervision for Stage1 Global Hand v2."""

from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


PALM_JOINTS = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
DISTAL_MANO_JOINTS = {
    "thumb": 15,
    "index": 3,
    "middle": 6,
    "ring": 12,
    "pinky": 9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--filtered-object-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--window-jsonl", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--min-valid-hand-frames", type=int, default=8)
    parser.add_argument("--geometry-features", action="store_true")
    parser.add_argument("--object-surface-samples", type=int, default=1024)
    parser.add_argument("--contact-per-finger-vertices", type=int, default=32)
    parser.add_argument("--contact-palm-vertices", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_mano_joint_regressor(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    regressor = raw["J_regressor"]
    if issparse(regressor):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float32)
    if regressor.shape != (16, 778):
        raise ValueError(f"Unexpected MANO J_regressor shape: {regressor.shape}")
    return regressor


def load_mano_semantic_groups(
    path: Path, per_finger: int, palm_count: int
) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    weights = raw.get("weights", raw.get("lbs_weights"))
    if hasattr(weights, "toarray"):
        weights = weights.toarray()
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape[0] != 778 or weights.shape[1] < 16:
        raise ValueError(f"Unexpected MANO weights shape: {weights.shape}")
    groups = {}
    for name, joint_index in DISTAL_MANO_JOINTS.items():
        count = min(max(1, per_finger), len(weights))
        indices = np.argpartition(weights[:, joint_index], -count)[-count:]
        groups[name] = indices.astype(np.int64)
    count = min(max(1, palm_count), len(weights))
    groups["palm"] = np.argpartition(weights[:, 0], -count)[-count:].astype(
        np.int64
    )
    return groups


@lru_cache(maxsize=32)
def object_surface(
    path_string: str, scale: float, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path_string, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float64) * scale
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError(f"Empty object mesh: {path_string}")
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    area = np.linalg.norm(cross, axis=-1)
    cdf = np.cumsum(np.maximum(area, 1e-12))
    targets = (
        (np.arange(sample_count, dtype=np.float64) + 0.5)
        / sample_count
        * cdf[-1]
    )
    selected = np.searchsorted(cdf, targets).clip(0, len(faces) - 1)
    sequence = np.arange(sample_count, dtype=np.float64)
    u = np.mod((sequence + 0.5) * 0.7548776662466927, 1.0)
    v = np.mod((sequence + 0.5) * 0.5698402909980532, 1.0)
    sqrt_u = np.sqrt(np.maximum(u, 1e-8))
    barycentric = np.stack(
        [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v], axis=-1
    )
    points = (
        triangles[selected] * barycentric[:, :, None]
    ).sum(axis=1).astype(np.float32)
    extents = (vertices.max(axis=0) - vertices.min(axis=0)).astype(np.float32)
    return points, extents


def surface_geometry_features(
    hand_vertices: np.ndarray,
    object_pose: np.ndarray,
    hand_valid: np.ndarray,
    object_valid: np.ndarray,
    object_points: np.ndarray,
    semantic_groups: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    group_names = list(semantic_groups)
    statistic_names = [
        "distance_min",
        "distance_p10",
        "distance_median",
        "distance_p90",
        "distance_lt_5mm",
        "distance_lt_10mm",
        "distance_lt_20mm",
        "ray_direction_p10",
        "ray_direction_median",
        "ray_direction_p90",
        "ray_direction_positive_fraction",
    ]
    names = [
        f"{group}_{statistic}"
        for group in group_names
        for statistic in statistic_names
    ]
    output = np.zeros((len(hand_vertices), len(names)), dtype=np.float32)
    tree = cKDTree(object_points)
    for frame in np.flatnonzero(hand_valid & object_valid):
        rotation = object_pose[frame, :3, :3]
        translation = object_pose[frame, :3, 3]
        local_hand = (hand_vertices[frame] - translation) @ rotation
        wrist = hand_vertices[frame, 0]
        ray = wrist / max(float(np.linalg.norm(wrist)), 1e-8)
        cursor = 0
        for indices in semantic_groups.values():
            points = local_hand[indices]
            distance, nearest_index = tree.query(points, k=1)
            nearest_local = object_points[nearest_index]
            nearest_camera = nearest_local @ rotation.T + translation
            direction = np.sum(
                (nearest_camera - hand_vertices[frame, indices]) * ray,
                axis=-1,
            )
            values = [
                distance.min(),
                *np.quantile(distance, [0.1, 0.5, 0.9]),
                np.mean(distance < 0.005),
                np.mean(distance < 0.010),
                np.mean(distance < 0.020),
                *np.quantile(direction, [0.1, 0.5, 0.9]),
                np.mean(direction > 0.0),
            ]
            output[frame, cursor : cursor + len(values)] = values
            cursor += len(values)
    return output, names


def mano16_to_dexycb21(joints: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Append MANO fingertips and reorder to the common DexYCB 21-joint order."""
    fingertips = vertices[:, [744, 320, 443, 555, 672]]
    mano21 = np.concatenate([joints, fingertips], axis=1)
    # MANO order: wrist, index, middle, pinky, ring, thumb plus fingertips.
    order = np.asarray(
        [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
         10, 11, 12, 19, 7, 8, 9, 20],
        dtype=np.int64,
    )
    return mano21[:, order]


def load_label_targets(
    label_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(label_paths)
    joints_3d = np.full((count, 21, 3), np.nan, dtype=np.float32)
    joints_2d = np.full((count, 21, 2), np.nan, dtype=np.float32)
    root_rotvec = np.full((count, 3), np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for index, path in enumerate(label_paths):
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as raw:
            joint_3d = np.asarray(raw.get("joint_3d", []), dtype=np.float32)
            joint_2d = np.asarray(raw.get("joint_2d", []), dtype=np.float32)
            pose_m = np.asarray(raw.get("pose_m", []), dtype=np.float32)
        joint_3d = joint_3d.reshape(-1, 3)
        joint_2d = joint_2d.reshape(-1, 2)
        pose_m = pose_m.reshape(-1)
        if (
            joint_3d.shape != (21, 3)
            or joint_2d.shape != (21, 2)
            or pose_m.size < 51
        ):
            continue
        joint_valid = (
            np.isfinite(joint_3d).all(axis=1)
            & np.isfinite(joint_2d).all(axis=1)
            & ~np.all(np.isclose(joint_3d, -1.0), axis=1)
            & ~np.all(np.isclose(joint_2d, -1.0), axis=1)
        )
        if not joint_valid[PALM_JOINTS].all() or np.allclose(pose_m[:51], 0.0):
            continue
        joints_3d[index] = joint_3d
        joints_2d[index] = joint_2d
        root_rotvec[index] = pose_m[:3]
        valid[index] = True
    return joints_3d, joints_2d, root_rotvec, valid


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("poses", payload.get("frames", payload))
    if isinstance(rows, dict):
        iterator = rows.items()
    else:
        iterator = (
            (str(row.get("frame", row.get("frame_id", index))).zfill(6), row)
            for index, row in enumerate(rows)
        )
    output = {}
    for frame, row in iterator:
        value = row
        if isinstance(row, dict):
            value = (
                row.get("object_in_camera")
                or row.get("pose")
                or row.get("transform")
            )
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.size == 16 and np.isfinite(matrix).all():
            output[str(frame).zfill(6)] = matrix.reshape(4, 4)
    return output


def mirror_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    return MIRROR_X @ matrix @ MIRROR_X


def normalize_left(
    vertices: np.ndarray,
    pred_joints: np.ndarray,
    gt_joints_3d: np.ndarray,
    gt_joints_2d: np.ndarray,
    gt_root_rotvec: np.ndarray,
    object_pose: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
) -> None:
    vertices[..., 0] *= -1.0
    pred_joints[..., 0] *= -1.0
    gt_joints_3d[..., 0] *= -1.0
    gt_joints_2d[..., 0] = (width - 1) - gt_joints_2d[..., 0]
    intrinsics[0, 2] = (width - 1) - intrinsics[0, 2]
    root_valid = np.isfinite(gt_root_rotvec).all(axis=1)
    if root_valid.any():
        rotation = Rotation.from_rotvec(gt_root_rotvec[root_valid]).as_matrix()
        rotation = np.einsum(
            "ij,tjk,kl->til", MIRROR_X, rotation, MIRROR_X
        )
        gt_root_rotvec[root_valid] = Rotation.from_matrix(rotation).as_rotvec()
    pose_valid = np.isfinite(object_pose).all(axis=(1, 2))
    for index in np.flatnonzero(pose_valid):
        object_pose[index, :3, :3] = mirror_rotation_matrix(
            object_pose[index, :3, :3]
        )
        object_pose[index, 0, 3] *= -1.0


def prepare_stream(
    record: dict,
    args: argparse.Namespace,
    handflow_root: Path,
    filtered_root: Path,
    joint_regressor: np.ndarray,
    semantic_groups: dict[str, np.ndarray] | None,
    out_path: Path,
) -> dict:
    stream_id = record["stream_id"]
    stream_dir = Path(record["stream_dir"])
    images = sorted(stream_dir.glob("color_*.jpg")) or sorted(
        stream_dir.glob("color_*.png")
    )
    if not images:
        raise FileNotFoundError(f"No RGB frames in {stream_dir}")
    frame_ids = [path.stem.rsplit("_", 1)[-1].zfill(6) for path in images]
    label_paths = [stream_dir / f"labels_{frame}.npz" for frame in frame_ids]

    handflow_path = handflow_root / stream_id / "handflow_camera_result.npz"
    with np.load(handflow_path, allow_pickle=False) as raw:
        vertices = np.asarray(raw["verts_cam"], dtype=np.float32)
        hand_valid = np.asarray(
            raw.get("pred_valid", np.ones(len(vertices)))
        ).astype(bool)
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32).reshape(3, 3)
        raw_pose = np.asarray(
            raw.get("handflow_raw_pose", np.empty((0, 48))),
            dtype=np.float32,
        )

    count = min(len(frame_ids), len(vertices), len(hand_valid))
    frame_ids = frame_ids[:count]
    images = images[:count]
    label_paths = label_paths[:count]
    vertices = vertices[:count]
    hand_valid = hand_valid[:count] & np.isfinite(vertices).all(axis=(1, 2))
    pred_mano16 = np.einsum("jv,tvc->tjc", joint_regressor, vertices)
    pred_joints = mano16_to_dexycb21(pred_mano16, vertices)

    gt_joints_3d, gt_joints_2d, gt_root_rotvec, gt_valid = load_label_targets(
        label_paths
    )

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

    initial_root_rotvec = np.full((count, 3), np.nan, dtype=np.float32)
    if raw_pose.ndim >= 2 and raw_pose.shape[-1] >= 3:
        raw_count = min(count, len(raw_pose))
        initial_root_rotvec[:raw_count] = raw_pose[:raw_count, :3]

    is_left = record["hand_side"] == "left"
    if is_left:
        import cv2
        sample = cv2.imread(str(images[0]))
        if sample is None:
            raise ValueError(f"Cannot read {images[0]}")
        width = int(sample.shape[1])
        normalize_left(
            vertices,
            pred_joints,
            gt_joints_3d,
            gt_joints_2d,
            gt_root_rotvec,
            object_pose,
            intrinsics,
            width,
        )

    supervision_valid = hand_valid & gt_valid & object_valid
    root_valid = supervision_valid & np.isfinite(initial_root_rotvec).all(axis=1)
    geometry_features = np.empty((count, 0), dtype=np.float32)
    geometry_feature_names: list[str] = []
    object_extents = np.zeros(3, dtype=np.float32)
    if args.geometry_features:
        if semantic_groups is None:
            raise RuntimeError("Missing MANO semantic groups")
        object_points, object_extents = object_surface(
            str(Path(record["sam3d_glb"]).resolve()),
            float(record["foundationpose_source_mesh_scale"]),
            args.object_surface_samples,
        )
        if is_left:
            object_points = object_points.copy()
            object_points[:, 0] *= -1.0
        geometry_features, geometry_feature_names = surface_geometry_features(
            vertices,
            object_pose,
            hand_valid,
            object_valid,
            object_points,
            semantic_groups,
        )
    np.savez_compressed(
        out_path,
        frame_ids=np.asarray(frame_ids),
        pred_hand_vertices=vertices,
        pred_joints_3d=pred_joints,
        gt_joints_3d=gt_joints_3d,
        gt_joints_2d=gt_joints_2d,
        gt_root_rotvec=gt_root_rotvec,
        initial_root_rotvec=initial_root_rotvec,
        object_pose=object_pose,
        intrinsics=intrinsics,
        hand_valid=hand_valid,
        gt_valid=gt_valid,
        object_valid=object_valid,
        supervision_valid=supervision_valid,
        root_valid=root_valid,
        object_extents_metric=object_extents,
        surface_geometry_features=geometry_features,
        surface_geometry_feature_names=np.asarray(geometry_feature_names),
        palm_joint_indices=PALM_JOINTS,
        normalized_left=np.asarray(is_left),
        hand_side=np.asarray(record["hand_side"]),
        stream_id=np.asarray(stream_id),
        source_handflow=np.asarray(str(handflow_path)),
        filtered_object_json=np.asarray(str(filtered_json)),
    )
    return {
        "frames": count,
        "valid": int(supervision_valid.sum()),
        "root_valid": int(root_valid.sum()),
    }


def window_starts(num_frames: int, window_size: int, stride: int) -> list[int]:
    max_start = max(0, num_frames - window_size)
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    manifest = Path(args.manifest).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    filtered_root = Path(args.filtered_object_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    window_path = Path(args.window_jsonl).expanduser().resolve()
    mano_root = Path(args.mano_data_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    window_path.parent.mkdir(parents=True, exist_ok=True)
    joint_regressor = load_mano_joint_regressor(mano_root / "MANO_RIGHT.pkl")
    semantic_groups = None
    if args.geometry_features:
        semantic_groups = load_mano_semantic_groups(
            mano_root / "MANO_RIGHT.pkl",
            args.contact_per_finger_vertices,
            args.contact_palm_vertices,
        )

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
                    joint_regressor, semantic_groups, out_path
                )
            else:
                with np.load(out_path, allow_pickle=False) as raw:
                    metrics = {
                        "frames": len(raw["frame_ids"]),
                        "valid": int(raw["supervision_valid"].sum()),
                        "root_valid": int(raw["root_valid"].sum()),
                    }
            with np.load(out_path, allow_pickle=False) as raw:
                valid = np.asarray(raw["supervision_valid"]).astype(bool)
            for start in window_starts(
                metrics["frames"], args.window_size, args.window_stride
            ):
                end = min(start + args.window_size, metrics["frames"])
                if (
                    end - start == args.window_size
                    and valid[start:end].sum() >= args.min_valid_hand_frames
                ):
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
