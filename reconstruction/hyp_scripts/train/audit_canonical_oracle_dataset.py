#!/usr/bin/env python3
"""Audit object-frame oracle hand supervision over DexYCB manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--filtered-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--cluster-threshold-deg", type=float, default=30.0)
    parser.add_argument("--scale-warning", type=float, default=0.1)
    parser.add_argument("--missing-warning-fraction", type=float, default=0.05)
    return parser.parse_args()


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                stream_id = str(row["stream_id"])
                if stream_id in seen:
                    continue
                seen.add(stream_id)
                rows.append(row)
    return rows


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("by_frame") or payload.get("frames") or {}
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    output = {}
    for key, row in iterator:
        frame = str(key).zfill(6)
        value = row
        if isinstance(row, dict):
            frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
            value = row.get("object_in_camera") or row.get("pose")
        if value is None:
            continue
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.size == 16 and np.isfinite(matrix).all():
            output[frame] = matrix.reshape(4, 4)
    return output


def canonical_alignment(path: Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    similarity = payload["raw_sam_to_ycb_similarity"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(similarity["rotation"], dtype=np.float64)
    matrix[:3, 3] = np.asarray(
        similarity["translation_m"], dtype=np.float64
    )
    return matrix, payload


def rotation_angle_deg(matrix: np.ndarray) -> float:
    return float(
        np.degrees(
            np.linalg.norm(Rotation.from_matrix(matrix).as_rotvec())
        )
    )


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def load_gt_frame(
    path: Path, grasp_index: int
) -> tuple[np.ndarray, np.ndarray] | None:
    with np.load(path, allow_pickle=False) as payload:
        poses = np.asarray(payload["pose_y"], dtype=np.float64)
        pose_m = np.asarray(payload["pose_m"], dtype=np.float64).reshape(-1)
    if poses.ndim == 2:
        poses = poses[None]
    if grasp_index >= len(poses) or len(pose_m) < 51:
        return None
    if np.allclose(pose_m[:51], 0.0) or not np.isfinite(pose_m[:51]).all():
        return None
    object_value = poses[grasp_index]
    if not np.isfinite(object_value).all():
        return None
    object_pose = np.eye(4, dtype=np.float64)
    object_pose[:3, :4] = object_value[:3, :4]
    hand_pose = np.eye(4, dtype=np.float64)
    hand_pose[:3, :3] = Rotation.from_rotvec(pose_m[:3]).as_matrix()
    hand_pose[:3, 3] = pose_m[48:51]
    return object_pose, hand_pose


def pairwise_rotation_distances(rotations: list[np.ndarray]) -> np.ndarray:
    count = len(rotations)
    distances = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            value = rotation_angle_deg(
                rotations[first].T @ rotations[second]
            )
            distances[first, second] = value
            distances[second, first] = value
    return distances


def cluster_stream_rotations(
    streams: list[dict], threshold: float, surface_rotation: np.ndarray
) -> list[dict]:
    if not streams:
        return []
    rotations = [np.asarray(row.pop("_rotation")) for row in streams]
    if len(rotations) == 1:
        labels = np.ones(1, dtype=np.int64)
    else:
        distances = pairwise_rotation_distances(rotations)
        tree = linkage(squareform(distances, checks=False), method="complete")
        labels = fcluster(tree, t=threshold, criterion="distance")
    output = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        group_rotations = [rotations[index] for index in indices]
        representative = Rotation.from_matrix(
            np.stack(group_rotations)
        ).mean().as_matrix()
        residuals = [
            rotation_angle_deg(representative.T @ value)
            for value in group_rotations
        ]
        selected = []
        for index in indices:
            row = streams[index]
            row["difference_to_cluster_mean_deg"] = rotation_angle_deg(
                representative.T @ rotations[index]
            )
            selected.append(row)
        selected.sort(key=lambda row: row["stream_id"])
        output.append(
            {
                "count": int(len(indices)),
                "representative_stream": min(
                    selected,
                    key=lambda row: row["difference_to_cluster_mean_deg"],
                )["stream_id"],
                "representative_rotation": representative.tolist(),
                "difference_to_surface_deg": rotation_angle_deg(
                    surface_rotation.T @ representative
                ),
                "within_cluster_residual_deg": distribution(residuals),
                "streams": selected,
            }
        )
    output.sort(key=lambda row: (-row["count"], row["representative_stream"]))
    return output


def audit_stream(
    record: dict,
    canonical_matrix: np.ndarray,
    canonical_payload: dict,
    filtered_root: Path,
    scale_warning_threshold: float,
) -> tuple[dict, np.ndarray | None]:
    stream_id = str(record["stream_id"])
    split = str(record.get("split", "val"))
    filtered_path = (
        filtered_root
        / split
        / stream_id
        / "segmented_ekf_rts/foundationpose_segmented_ekf_rts.json"
    )
    stream_dir = Path(record["stream_dir"]).expanduser().resolve()
    metadata = yaml.safe_load(
        (stream_dir.parent / "meta.yml").read_text(encoding="utf-8")
    ) or {}
    grasp_index = int(metadata.get("ycb_grasp_ind", 0))
    labels = sorted(stream_dir.glob("labels_*.npz"))
    filtered = pose_rows(filtered_path)

    frame_rotation_relations = []
    surface_rotation_difference = []
    surface_translation_difference = []
    roundtrip_translation = []
    roundtrip_rotation = []
    transfer_translation = []
    transfer_rotation = []
    target_translation_steps = []
    target_rotation_steps = []
    previous_target = None
    valid_frames = []
    nonfinite = 0

    for label_path in labels:
        frame = label_path.stem.rsplit("_", 1)[-1].zfill(6)
        filtered_pose = filtered.get(frame)
        gt = load_gt_frame(label_path, grasp_index)
        if filtered_pose is None or gt is None:
            continue
        gt_object, gt_hand = gt
        gt_sam_pose = gt_object @ canonical_matrix
        transfer = filtered_pose @ np.linalg.inv(gt_sam_pose)
        object_target = np.linalg.inv(gt_sam_pose) @ gt_hand
        camera_target = filtered_pose @ object_target
        reconstructed = np.linalg.inv(filtered_pose) @ camera_target
        if not all(
            np.isfinite(value).all()
            for value in (transfer, object_target, camera_target, reconstructed)
        ):
            nonfinite += 1
            continue

        relative = np.linalg.inv(gt_object) @ filtered_pose
        frame_rotation_relations.append(relative[:3, :3])
        surface_rotation_difference.append(
            rotation_angle_deg(
                canonical_matrix[:3, :3].T @ relative[:3, :3]
            )
        )
        surface_translation_difference.append(
            float(
                np.linalg.norm(
                    canonical_matrix[:3, 3] - relative[:3, 3]
                )
                * 1000.0
            )
        )
        roundtrip_translation.append(
            float(
                np.linalg.norm(
                    reconstructed[:3, 3] - object_target[:3, 3]
                )
                * 1000.0
            )
        )
        roundtrip_rotation.append(
            rotation_angle_deg(
                reconstructed[:3, :3].T @ object_target[:3, :3]
            )
        )
        transfer_translation.append(
            float(np.linalg.norm(transfer[:3, 3]) * 1000.0)
        )
        transfer_rotation.append(rotation_angle_deg(transfer[:3, :3]))
        if previous_target is not None:
            target_translation_steps.append(
                float(
                    np.linalg.norm(
                        object_target[:3, 3] - previous_target[:3, 3]
                    )
                    * 1000.0
                )
            )
            target_rotation_steps.append(
                rotation_angle_deg(
                    previous_target[:3, :3].T @ object_target[:3, :3]
                )
            )
        previous_target = object_target
        valid_frames.append(frame)

    scale = canonical_payload.get("production_sam_to_ycb_similarity") or {}
    residual_scale = scale.get("residual_scale")
    missing_fraction = 1.0 - len(valid_frames) / max(len(labels), 1)
    stream_rotation = None
    stream_rotation_residual = {"count": 0}
    if frame_rotation_relations:
        stream_rotation = Rotation.from_matrix(
            np.stack(frame_rotation_relations)
        ).mean().as_matrix()
        stream_rotation_residual = distribution(
            [
                rotation_angle_deg(stream_rotation.T @ value)
                for value in frame_rotation_relations
            ]
        )

    row = {
        "stream_id": stream_id,
        "split": split,
        "hand_side": record["hand_side"],
        "object_name": record["object_name"],
        "num_label_frames": len(labels),
        "num_filtered_pose_frames": len(filtered),
        "num_valid_oracle_frames": len(valid_frames),
        "missing_fraction": missing_fraction,
        "nonfinite_frames": nonfinite,
        "canonical_residual_scale": residual_scale,
        "scale_warning": bool(
            residual_scale is not None
            and abs(float(residual_scale) - 1.0) > scale_warning_threshold
        ),
        "roundtrip_translation_mm": distribution(roundtrip_translation),
        "roundtrip_rotation_deg": distribution(roundtrip_rotation),
        "surface_rotation_difference_deg": distribution(
            surface_rotation_difference
        ),
        "surface_translation_difference_mm": distribution(
            surface_translation_difference
        ),
        "gt_to_filtered_transfer_translation_mm": distribution(
            transfer_translation
        ),
        "gt_to_filtered_transfer_rotation_deg": distribution(
            transfer_rotation
        ),
        "object_target_translation_step_mm": distribution(
            target_translation_steps
        ),
        "object_target_rotation_step_deg": distribution(target_rotation_steps),
        "stream_rotation_residual_deg": stream_rotation_residual,
    }
    if stream_rotation is not None:
        row["rotation_residual_to_stream_mean_deg"] = float(
            stream_rotation_residual.get("median", 0.0)
        )
    return row, stream_rotation


def main() -> None:
    args = parse_args()
    manifests = [Path(path).expanduser().resolve() for path in args.manifest]
    canonical_root = Path(args.canonical_root).expanduser().resolve()
    filtered_root = Path(args.filtered_root).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(manifests)
    canonical_cache: dict[str, tuple[np.ndarray, dict]] = {}
    rows = []
    failures = []
    rotations_by_object: dict[str, list[dict]] = {}
    surface_by_object: dict[str, np.ndarray] = {}

    for index, record in enumerate(records, start=1):
        stream_id = str(record["stream_id"])
        object_name = str(record["object_name"])
        print(f"[{index}/{len(records)}] {stream_id}", flush=True)
        try:
            if object_name not in canonical_cache:
                alignment_path = (
                    canonical_root / object_name / "canonical_alignment.json"
                )
                canonical_cache[object_name] = canonical_alignment(
                    alignment_path
                )
                surface_by_object[object_name] = canonical_cache[
                    object_name
                ][0][:3, :3]
            matrix, payload = canonical_cache[object_name]
            row, stream_rotation = audit_stream(
                record,
                matrix,
                payload,
                filtered_root,
                args.scale_warning,
            )
            rows.append(row)
            if stream_rotation is not None:
                cluster_row = dict(row)
                cluster_row["_rotation"] = stream_rotation
                rotations_by_object.setdefault(object_name, []).append(
                    cluster_row
                )
        except Exception as error:
            failures.append(
                {"stream_id": stream_id, "error": repr(error)}
            )
            print(f"  failed: {error!r}", flush=True)

    objects = {}
    for object_name, stream_rows in rotations_by_object.items():
        objects[object_name] = {
            "num_streams": len(stream_rows),
            "rotation_clusters": cluster_stream_rotations(
                stream_rows,
                args.cluster_threshold_deg,
                surface_by_object[object_name],
            ),
        }

    roundtrip_translation = [
        row["roundtrip_translation_mm"].get("max", np.nan) for row in rows
    ]
    roundtrip_rotation = [
        row["roundtrip_rotation_deg"].get("max", np.nan) for row in rows
    ]
    warning_streams = [
        row["stream_id"]
        for row in rows
        if row["scale_warning"]
        or row["missing_fraction"] > args.missing_warning_fraction
        or row["nonfinite_frames"] > 0
        or row["roundtrip_translation_mm"].get("max", 0.0) > 1e-4
        or row["roundtrip_rotation_deg"].get("max", 0.0) > 1e-4
    ]
    summary = {
        "manifests": [str(path) for path in manifests],
        "canonical_root": str(canonical_root),
        "filtered_root": str(filtered_root),
        "num_requested": len(records),
        "num_completed": len(rows),
        "num_failed": len(failures),
        "num_warning_streams": len(warning_streams),
        "warning_streams": warning_streams,
        "aggregate_roundtrip_translation_mm": distribution(
            roundtrip_translation
        ),
        "aggregate_roundtrip_rotation_deg": distribution(roundtrip_rotation),
        "objects": objects,
        "streams": rows,
        "failures": failures,
        "settings": vars(args),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "requested": len(records),
                "completed": len(rows),
                "failed": len(failures),
                "warnings": len(warning_streams),
                "objects": len(objects),
                "output": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
