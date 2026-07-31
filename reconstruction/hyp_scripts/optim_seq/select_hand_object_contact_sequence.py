#!/usr/bin/env python3
"""Select stable CHOIR-style hand-object contact regions for one stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optimize_hand_object_contact_sequence import (
    contact_candidates,
    contact_states,
    deterministic_surface_samples,
    load_mesh,
    mano_semantic_contact_indices,
    nearest_surface,
    sampled_vertex_normals,
    transform_surface,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    stream = parser.add_mutually_exclusive_group(required=True)
    stream.add_argument("--stream-id")
    stream.add_argument("--sequence-dir")
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object-samples", type=int, default=4096)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--contact-enter-mm", type=float, default=8.0)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.5)
    parser.add_argument(
        "--penetration-max-distance-mm", type=float, default=30.0
    )
    parser.add_argument("--normal-dot-max", type=float, default=-0.1)
    parser.add_argument("--min-contact-points", type=int, default=3)
    parser.add_argument("--contact-topk", type=int, default=12)
    parser.add_argument("--contact-per-finger-vertices", type=int, default=32)
    parser.add_argument("--contact-palm-vertices", type=int, default=64)
    parser.add_argument("--enter-patience", type=int, default=3)
    parser.add_argument("--exit-patience", type=int, default=5)
    parser.add_argument("--contact-update-frames", type=int, default=8)
    parser.add_argument(
        "--contact-persistence-mode",
        choices=("active_only", "whole_chunk"),
        default="active_only",
    )
    parser.add_argument("--contact-start-frame", type=int, default=-1)
    parser.add_argument("--contact-end-frame", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stream_id_from_sequence_dir(value: str) -> str:
    parts = Path(value).expanduser().resolve().parts
    if len(parts) < 3:
        raise ValueError(f"Cannot derive stream ID from {value}")
    return "__".join(parts[-3:])


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    stream_id = args.stream_id or stream_id_from_sequence_dir(
        args.sequence_dir
    )
    records = {
        row["stream_id"]: row
        for row in load_jsonl(Path(args.manifest).expanduser().resolve())
    }
    if stream_id not in records:
        raise KeyError(f"Stream is not in manifest: {stream_id}")
    record = records[stream_id]
    prediction_path = (
        Path(args.prediction_root).expanduser().resolve()
        / stream_id
        / "handflow_camera_result_pi3x_depth_refined.npz"
    )
    supervision_path = (
        Path(args.supervision_root).expanduser().resolve()
        / f"{stream_id}.npz"
    )
    mesh_path = Path(record["sam3d_glb"]).expanduser().resolve()
    for path in (prediction_path, supervision_path, mesh_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    out_dir = Path(args.out_root).expanduser().resolve() / stream_id
    result_path = out_dir / "contact_selection.npz"
    audit_path = out_dir / "audit.json"
    if result_path.is_file() and audit_path.is_file() and not args.overwrite:
        print(f"Existing: {audit_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(prediction_path, allow_pickle=False) as raw:
        hand = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(supervision_path, allow_pickle=False) as raw:
        supervision = {key: np.asarray(raw[key]) for key in raw.files}

    vertices = np.asarray(hand["verts_cam"], dtype=np.float32)
    faces = np.asarray(hand["faces"], dtype=np.int64)
    poses = np.asarray(supervision["object_pose"], dtype=np.float32)
    valid = (
        np.asarray(hand["pred_valid"]).astype(bool)
        & np.asarray(supervision["object_valid"]).astype(bool)
    )
    count = min(len(vertices), len(poses), len(valid))
    vertices, poses, valid = vertices[:count], poses[:count], valid[:count]
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    if normalized_left:
        vertices = vertices.copy()
        vertices[..., 0] *= -1.0

    hand_side = str(
        np.asarray(
            hand.get("hand_side", supervision.get("hand_side", "right"))
        ).item()
    ).lower()
    semantic_indices, semantic_groups = mano_semantic_contact_indices(
        Path(args.mano_data_dir).expanduser().resolve(),
        hand_side,
        args.contact_per_finger_vertices,
        args.contact_palm_vertices,
    )
    hand_normals = sampled_vertex_normals(
        vertices, faces, semantic_indices
    )

    mesh = load_mesh(
        mesh_path, float(record["foundationpose_source_mesh_scale"])
    )
    local_points, local_normals = deterministic_surface_samples(
        mesh, args.object_samples
    )
    if normalized_left:
        local_points = local_points.copy()
        local_normals = local_normals.copy()
        local_points[:, 0] *= -1.0
        local_normals[:, 0] *= -1.0

    device = torch.device(args.device)
    point_tensor = torch.from_numpy(local_points).to(device)
    normal_tensor = torch.from_numpy(local_normals).to(device)
    distance_all_chunks = []
    point_all_chunks = []
    normal_all_chunks = []
    inside_all_chunks = []
    for start in range(0, count, args.frame_batch_size):
        end = min(start + args.frame_batch_size, count)
        pose_tensor = torch.from_numpy(poses[start:end]).to(device)
        object_points, object_normals = transform_surface(
            point_tensor, normal_tensor, pose_tensor
        )
        with torch.no_grad():
            result = nearest_surface(
                torch.from_numpy(vertices[start:end]).to(device),
                object_points,
                object_normals,
            )
        distance_all_chunks.append(result[0].cpu())
        point_all_chunks.append(result[1].cpu())
        normal_all_chunks.append(result[2].cpu())
        inside_all_chunks.append(result[3].cpu())
    distance_all = torch.cat(distance_all_chunks).to(device)
    nearest_point_all = torch.cat(point_all_chunks).to(device)
    nearest_normal_all = torch.cat(normal_all_chunks).to(device)
    inside_all = torch.cat(inside_all_chunks).to(device)
    distance = distance_all[:, semantic_indices]
    nearest_point = nearest_point_all[:, semantic_indices]
    nearest_normal = nearest_normal_all[:, semantic_indices]
    inside = inside_all[:, semantic_indices]
    hand_normal_tensor = torch.from_numpy(hand_normals).to(device)
    valid_tensor = torch.from_numpy(valid).to(device)
    semantic_mask = torch.ones(
        len(semantic_indices), dtype=torch.bool, device=device
    )
    candidates = contact_candidates(
        distance,
        inside,
        hand_normal_tensor,
        nearest_normal,
        valid_tensor,
        semantic_mask,
        args,
    )
    candidate_np = candidates.cpu().numpy()
    distance_np = distance.cpu().numpy()
    selected_np, updates = contact_states(
        candidate_np, distance_np, args
    )
    normal_dot_np = (
        hand_normal_tensor * nearest_normal
    ).sum(dim=-1).cpu().numpy()
    penetration_mask = (
        (inside_all > args.penetration_tolerance_mm / 1000.0)
        & (
            distance_all
            <= args.penetration_max_distance_mm / 1000.0
        )
        & valid_tensor[:, None]
    )
    penetration_depth = torch.relu(
        inside_all - args.penetration_tolerance_mm / 1000.0
    )
    penetration_depth = penetration_depth * penetration_mask
    penetration_np = penetration_mask.cpu().numpy()
    penetration_depth_np = penetration_depth.cpu().numpy()

    candidate_full = np.zeros((count, vertices.shape[1]), dtype=bool)
    selected_full = np.zeros_like(candidate_full)
    candidate_full[:, semantic_indices] = candidate_np
    selected_full[:, semantic_indices] = selected_np
    vertex_to_group = {}
    for name, indices in semantic_groups.items():
        for index in indices:
            vertex_to_group.setdefault(int(index), []).append(name)

    mapped_updates = []
    for row in updates:
        local_ids = np.asarray(row["sample_ids"], dtype=np.int64)
        vertex_ids = semantic_indices[local_ids] if len(local_ids) else []
        mapped_updates.append(
            {
                "frames": row["frames"],
                "semantic_local_ids": local_ids.tolist(),
                "mano_vertex_ids": np.asarray(vertex_ids).astype(int).tolist(),
            }
        )

    frame_rows = []
    for frame in range(count):
        selected_vertices = np.flatnonzero(selected_full[frame])
        group_counts = {
            name: int(
                np.intersect1d(
                    selected_vertices,
                    np.asarray(indices, dtype=np.int64),
                ).size
            )
            for name, indices in semantic_groups.items()
        }
        chosen_local = np.flatnonzero(selected_np[frame])
        frame_rows.append(
            {
                "frame": frame,
                "valid": bool(valid[frame]),
                "num_candidates": int(candidate_np[frame].sum()),
                "num_selected": int(selected_np[frame].sum()),
                "selected_vertex_ids": selected_vertices.astype(int).tolist(),
                "selected_groups": {
                    key: value for key, value in group_counts.items() if value
                },
                "selected_distance_mm": distribution(
                    distance_np[frame, chosen_local] * 1000.0
                ),
                "num_penetrating": int(penetration_np[frame].sum()),
                "penetration_depth_mm": distribution(
                    penetration_depth_np[frame, penetration_np[frame]]
                    * 1000.0
                ),
            }
        )

    selected_distance = distance_np[selected_np] * 1000.0
    audit = {
        "stream_id": stream_id,
        "object_name": record["object_name"],
        "hand_side": hand_side,
        "coordinate_frame": "normalized_camera" if normalized_left else "camera",
        "prediction": str(prediction_path),
        "supervision": str(supervision_path),
        "object_mesh": str(mesh_path),
        "object_mesh_scale": float(
            record["foundationpose_source_mesh_scale"]
        ),
        "settings": vars(args),
        "num_frames": count,
        "num_valid_frames": int(valid.sum()),
        "num_semantic_vertices": int(len(semantic_indices)),
        "num_candidate_frame_vertices": int(candidate_np.sum()),
        "num_selected_frame_vertices": int(selected_np.sum()),
        "num_contact_frames": int(selected_np.any(axis=1).sum()),
        "selected_distance_mm": distribution(selected_distance),
        "num_penetrating_frame_vertices": int(penetration_np.sum()),
        "num_penetrating_frames": int(
            penetration_np.any(axis=1).sum()
        ),
        "penetration_depth_mm": distribution(
            penetration_depth_np[penetration_np] * 1000.0
        ),
        "semantic_groups": semantic_groups,
        "contact_updates": mapped_updates,
        "frames": frame_rows,
    }
    np.savez_compressed(
        result_path,
        stream_id=np.asarray(stream_id),
        semantic_vertex_indices=semantic_indices.astype(np.int64),
        candidate_mask=candidate_full,
        contact_mask=selected_full,
        penetration_mask=penetration_np,
        penetration_depth=penetration_depth_np.astype(np.float32),
        nearest_distance=distance_np.astype(np.float32),
        nearest_object_point=nearest_point.cpu().numpy().astype(np.float32),
        nearest_object_normal=nearest_normal.cpu().numpy().astype(np.float32),
        signed_inside=inside.cpu().numpy().astype(np.float32),
        normal_dot=normal_dot_np.astype(np.float32),
        valid=valid,
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key not in (
        "settings", "semantic_groups", "contact_updates", "frames"
    )}, indent=2))
    print(f"Done: {audit_path}")


if __name__ == "__main__":
    main()
