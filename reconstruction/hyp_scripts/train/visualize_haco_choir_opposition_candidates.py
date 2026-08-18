#!/usr/bin/env python3
"""Inspect CHOIR-style paired thumb/index object-surface candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from refine_v14_haco_sequence_contact_containment import mano_contact_region_ids


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--object-samples", type=int, default=16384)
    parser.add_argument("--candidate-topk", type=int, default=256)
    parser.add_argument("--choir-topk", type=int, default=8)
    parser.add_argument("--choir-sigma-mm", type=float, default=10.0)
    parser.add_argument("--w-opposition", type=float, default=20.0)
    parser.add_argument("--w-facing", type=float, default=20.0)
    parser.add_argument("--max-pair-width-mm", type=float, default=100.0)
    parser.add_argument("--out-npz")
    parser.add_argument("--out-json")
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def index_for(values: np.ndarray, target: str) -> int:
    normalized = [frame_id(value) for value in values]
    if target not in normalized:
        raise KeyError(f"Frame {target} not found")
    return normalized.index(target)


def physical_pose(pose: np.ndarray, normalized_left: bool) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    if normalized_left:
        result[:3, :3] = MIRROR_X @ result[:3, :3] @ MIRROR_X
        result[:3, 3] = MIRROR_X @ result[:3, 3]
    return result


def choir_distance(
    surface: np.ndarray,
    hand: np.ndarray,
    topk: int,
    sigma: float,
) -> np.ndarray:
    pairwise = np.linalg.norm(surface[:, None] - hand[None], axis=-1)
    count = min(max(1, topk), pairwise.shape[1])
    nearest = np.partition(pairwise, count - 1, axis=1)[:, :count]
    weights = np.exp(-nearest * nearest / (2.0 * sigma * sigma))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.sqrt((weights * nearest * nearest).sum(axis=1))


def colors(count: int, rgb: tuple[int, int, int]) -> np.ndarray:
    return np.tile(np.asarray(rgb, dtype=np.uint8), (count, 1))


def main() -> None:
    args = parse_args()
    requested = frame_id(args.frame_id)
    stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    stage_index = index_for(stage1["frame_ids"], requested)
    query_index = index_for(query["frame_ids"], requested)
    contact_index = index_for(contact["frame_ids"], requested)
    supervision_index = index_for(supervision["frame_ids"], requested)

    hand = np.asarray(
        stage1["refined_hand_vertices_camera"][stage_index], dtype=np.float32
    )
    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    probability = np.asarray(
        contact["contact_probability"][contact_index], dtype=np.float32
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    region_id, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )
    thumb_id = region_names.index("thumb")
    index_id = region_names.index("index")
    thumb_mask = (region_id == thumb_id) & (probability >= threshold)
    index_mask = (region_id == index_id) & (probability >= threshold)
    if thumb_mask.sum() < 3 or index_mask.sum() < 3:
        raise RuntimeError(
            f"Insufficient HACO contacts: thumb={thumb_mask.sum()}, "
            f"index={index_mask.sum()}"
        )

    mesh = trimesh.load(
        Path(args.object_mesh).expanduser().resolve(), process=False
    )
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    np.random.seed(0)
    surface_local, face_index = trimesh.sample.sample_surface(
        mesh, args.object_samples
    )
    normals_local = np.asarray(mesh.face_normals[face_index], dtype=np.float32)
    pose = physical_pose(
        supervision["gt_ycb_object_pose"][supervision_index],
        bool(np.asarray(supervision.get("normalized_left", False)).item()),
    )
    surface = np.asarray(surface_local, dtype=np.float32) @ pose[:3, :3].T + pose[:3, 3]
    normals = normals_local @ pose[:3, :3].T
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)

    sigma = args.choir_sigma_mm / 1000.0
    thumb_distance = choir_distance(
        surface, hand[thumb_mask], args.choir_topk, sigma
    )
    index_distance = choir_distance(
        surface, hand[index_mask], args.choir_topk, sigma
    )
    candidate_count = min(args.candidate_topk, len(surface))
    thumb_candidates = np.argpartition(
        thumb_distance, candidate_count - 1
    )[:candidate_count]
    index_candidates = np.argpartition(
        index_distance, candidate_count - 1
    )[:candidate_count]

    thumb_points = surface[thumb_candidates]
    index_points = surface[index_candidates]
    delta = index_points[None] - thumb_points[:, None]
    width = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(width[..., None], 1e-12)
    thumb_normals = normals[thumb_candidates]
    index_normals = normals[index_candidates]
    normal_dot = thumb_normals @ index_normals.T
    opposition = 0.5 * (1.0 + normal_dot)
    thumb_facing = 0.5 * (
        1.0 + (direction * thumb_normals[:, None]).sum(axis=-1)
    )
    index_facing = 0.5 * (
        1.0 - (direction * index_normals[None]).sum(axis=-1)
    )
    facing = thumb_facing + index_facing
    distance_mm = (
        thumb_distance[thumb_candidates, None]
        + index_distance[index_candidates][None]
    ) * 1000.0
    score = (
        distance_mm
        + args.w_opposition * opposition
        + args.w_facing * facing
    )
    invalid_width = (width <= 1e-4) | (
        width * 1000.0 > args.max_pair_width_mm
    )
    score[invalid_width] = np.inf
    flat = int(np.argmin(score))
    thumb_choice, index_choice = np.unravel_index(flat, score.shape)
    if not np.isfinite(score[thumb_choice, index_choice]):
        raise RuntimeError("No valid opposition pair")
    selected_thumb = thumb_points[thumb_choice]
    selected_index = index_points[index_choice]

    summary = {
        "frame_id": requested,
        "thumb_contact_vertices": int(thumb_mask.sum()),
        "index_contact_vertices": int(index_mask.sum()),
        "candidate_count_per_region": candidate_count,
        "selected_score": float(score[thumb_choice, index_choice]),
        "selected_pair_width_mm": float(width[thumb_choice, index_choice] * 1000.0),
        "selected_normal_dot": float(normal_dot[thumb_choice, index_choice]),
        "selected_thumb_choir_mm": float(
            thumb_distance[thumb_candidates[thumb_choice]] * 1000.0
        ),
        "selected_index_choir_mm": float(
            index_distance[index_candidates[index_choice]] * 1000.0
        ),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out_json:
        output_json = Path(args.out_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.out_npz:
        output_npz = Path(args.out_npz).expanduser().resolve()
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            frame_id=np.asarray(requested),
            thumb_contact_vertices=hand[thumb_mask],
            index_contact_vertices=hand[index_mask],
            thumb_candidates=thumb_points,
            index_candidates=index_points,
            thumb_candidate_choir_mm=thumb_distance[thumb_candidates] * 1000.0,
            index_candidate_choir_mm=index_distance[index_candidates] * 1000.0,
            selected_thumb=selected_thumb,
            selected_index=selected_index,
            selected_thumb_normal=thumb_normals[thumb_choice],
            selected_index_normal=index_normals[index_choice],
        )

    server = viser.ViserServer(port=args.port)
    server.scene.add_mesh_simple(
        "/object", vertices=surface_local @ pose[:3, :3].T + pose[:3, 3],
        faces=np.asarray(mesh.faces), color=(170, 180, 195), opacity=0.65,
    )
    server.scene.add_mesh_simple(
        "/stage1_hand", vertices=hand, faces=faces,
        color=(80, 175, 245), opacity=0.65,
    )
    server.scene.add_point_cloud(
        "/haco/thumb", points=hand[thumb_mask],
        colors=colors(int(thumb_mask.sum()), (255, 60, 180)), point_size=0.004,
    )
    server.scene.add_point_cloud(
        "/haco/index", points=hand[index_mask],
        colors=colors(int(index_mask.sum()), (40, 255, 100)), point_size=0.004,
    )
    server.scene.add_point_cloud(
        "/candidates/thumb", points=thumb_points,
        colors=colors(candidate_count, (255, 140, 220)), point_size=0.002,
    )
    server.scene.add_point_cloud(
        "/candidates/index", points=index_points,
        colors=colors(candidate_count, (120, 255, 145)), point_size=0.002,
    )
    server.scene.add_point_cloud(
        "/selected/pair", points=np.stack([selected_thumb, selected_index]),
        colors=np.asarray([[255, 0, 255], [0, 255, 0]], dtype=np.uint8),
        point_size=0.009,
    )
    server.scene.add_line_segments(
        "/selected/opposition",
        points=np.stack([selected_thumb, selected_index])[None],
        colors=np.asarray([[[255, 230, 40], [255, 230, 40]]], dtype=np.uint8),
        line_width=4.0,
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
