#!/usr/bin/env python3
"""Inspect CHOIR-style paired thumb/index object-surface candidates."""

from __future__ import annotations

import argparse
import heapq
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
    parser.add_argument("--stage1-npz")
    parser.add_argument(
        "--trajectory-npz",
        help="V14 trajectory used to reconstruct the initial hand directly.",
    )
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--dense-root")
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument(
        "--pair-hand-source", choices=("v14", "stage1"), default="stage1"
    )
    parser.add_argument("--object-samples", type=int, default=16384)
    parser.add_argument("--candidate-topk", type=int, default=256)
    parser.add_argument("--candidate-slack-mm", type=float, default=20.0)
    parser.add_argument("--pixel-radius", type=float, default=30.0)
    parser.add_argument("--pixel-soft-topk", type=int, default=8)
    parser.add_argument("--pixel-sigma", type=float, default=12.0)
    parser.add_argument("--w-pixel", type=float, default=0.5)
    parser.add_argument("--choir-topk", type=int, default=8)
    parser.add_argument("--choir-sigma-mm", type=float, default=10.0)
    parser.add_argument("--w-opposition", type=float, default=20.0)
    parser.add_argument("--w-facing", type=float, default=20.0)
    parser.add_argument("--max-pair-width-mm", type=float, default=100.0)
    parser.add_argument("--max-midpoint-shift-mm", type=float, default=30.0)
    parser.add_argument("--max-width-change-mm", type=float, default=40.0)
    parser.add_argument("--min-axis-cosine", type=float, default=0.35)
    parser.add_argument("--w-midpoint", type=float, default=2.0)
    parser.add_argument("--w-axis", type=float, default=20.0)
    parser.add_argument("--w-width", type=float, default=0.5)
    parser.add_argument(
        "--translation-invariant",
        action="store_true",
        help=(
            "Select the pair from relative finger geometry, 2D evidence and "
            "surface normals without using unreliable absolute hand depth."
        ),
    )
    parser.add_argument(
        "--max-pair-normal-dot",
        type=float,
        default=1.0,
        help="Reject pairs whose object surface normals are not opposed enough.",
    )
    parser.add_argument("--patch-radius-mm", type=float, default=8.0)
    parser.add_argument("--patch-normal-cosine", type=float, default=0.7)
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


def load_intrinsics(
    args: argparse.Namespace,
    query: dict[str, np.ndarray],
    stream_id: str,
) -> np.ndarray:
    if "intrinsics" in query:
        matrix = np.asarray(query["intrinsics"], dtype=np.float32)
        return matrix[0] if matrix.ndim == 3 else matrix.reshape(3, 3)
    if args.dense_root:
        windows = sorted(
            (Path(args.dense_root).expanduser().resolve() / stream_id / "windows")
            .glob("*.npz")
        )
        if not windows:
            raise FileNotFoundError(f"No Pi3X windows for {stream_id}")
        dense = load_npz(windows[0])
        matrix = np.asarray(dense["intrinsics_resized"], dtype=np.float32)
        matrix = (matrix[0] if matrix.ndim == 3 else matrix).reshape(3, 3).copy()
        resized_wh = np.asarray(dense["resized_wh"], dtype=np.float32).reshape(2)
        original_wh = np.asarray(query["image_wh"], dtype=np.float32).reshape(-1, 2)[0]
        matrix[0, (0, 2)] *= original_wh[0] / resized_wh[0]
        matrix[1, (1, 2)] *= original_wh[1] / resized_wh[1]
        return matrix
    if args.intrinsics:
        fx, fy, cx, cy = args.intrinsics
        return np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    raise KeyError("Pass --dense-root or --intrinsics FX FY CX CY")


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    projected = points @ intrinsics.T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
    uv[points[:, 2] <= 1e-6] = np.nan
    return uv


def colors(count: int, rgb: tuple[int, int, int]) -> np.ndarray:
    return np.tile(np.asarray(rgb, dtype=np.uint8), (count, 1))


def geodesic_patch(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    center: int,
    radius: float,
    normal_cosine: float,
) -> np.ndarray:
    adjacency: list[dict[int, float]] = [dict() for _ in range(len(vertices))]
    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            distance = float(np.linalg.norm(vertices[first] - vertices[second]))
            previous = adjacency[first].get(second)
            if previous is None or distance < previous:
                adjacency[first][second] = distance
                adjacency[second][first] = distance
    distances = {center: 0.0}
    queue = [(0.0, center)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances.get(vertex) or distance > radius:
            continue
        for neighbor, edge_length in adjacency[vertex].items():
            candidate = distance + edge_length
            if candidate <= radius and candidate < distances.get(neighbor, np.inf):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    selected = np.asarray(sorted(distances), dtype=np.int64)
    normal_gate = normals[selected] @ normals[center] >= normal_cosine
    selected = selected[normal_gate]
    return selected if len(selected) else np.asarray([center], dtype=np.int64)


def main() -> None:
    args = parse_args()
    requested = frame_id(args.frame_id)
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    query_index = index_for(query["frame_ids"], requested)
    contact_index = index_for(contact["frame_ids"], requested)
    supervision_index = index_for(supervision["frame_ids"], requested)

    if args.trajectory_npz:
        trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
        trajectory_index = index_for(trajectory["frame_ids"], requested)
        wrist = np.asarray(
            trajectory["predicted_wrist_camera"][trajectory_index],
            dtype=np.float32,
        )
        root_relative = np.asarray(
            query["vertices_3d_root_relative_original"][query_index],
            dtype=np.float32,
        )
        initial_hand = root_relative + wrist[None]
        hand = initial_hand.copy()
    elif args.stage1_npz:
        stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
        stage_index = index_for(stage1["frame_ids"], requested)
        hand = np.asarray(
            stage1["refined_hand_vertices_camera"][stage_index], dtype=np.float32
        )
        initial_hand = np.asarray(
            stage1["initial_hand_vertices_camera"][stage_index], dtype=np.float32
        )
    else:
        raise ValueError("Pass either --trajectory-npz or --stage1-npz")
    if args.pair_hand_source == "stage1" and not args.stage1_npz:
        raise ValueError("--pair-hand-source stage1 requires --stage1-npz")
    pair_hand = initial_hand if args.pair_hand_source == "v14" else hand
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
    object_vertices_local = np.asarray(mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(mesh.faces, dtype=np.int64)
    surface_local = object_vertices_local
    normals_local = np.asarray(mesh.vertex_normals, dtype=np.float32)
    pose = physical_pose(
        supervision["gt_ycb_object_pose"][supervision_index],
        bool(np.asarray(supervision.get("normalized_left", False)).item()),
    )
    surface = np.asarray(surface_local, dtype=np.float32) @ pose[:3, :3].T + pose[:3, 3]
    object_vertices_camera = (
        object_vertices_local @ pose[:3, :3].T + pose[:3, 3]
    )
    normals = normals_local @ pose[:3, :3].T
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)

    stream_id = str(np.asarray(query["stream_id"]).item())
    intrinsics = load_intrinsics(args, query, stream_id)
    object_uv = project(surface, intrinsics)
    initial_hand_uv = project(initial_hand, intrinsics)
    thumb_pixel_distance = choir_distance(
        object_uv, initial_hand_uv[thumb_mask],
        args.pixel_soft_topk, args.pixel_sigma,
    )
    index_pixel_distance = choir_distance(
        object_uv, initial_hand_uv[index_mask],
        args.pixel_soft_topk, args.pixel_sigma,
    )

    sigma = args.choir_sigma_mm / 1000.0
    thumb_distance = choir_distance(
        surface, pair_hand[thumb_mask], args.choir_topk, sigma
    )
    index_distance = choir_distance(
        surface, pair_hand[index_mask], args.choir_topk, sigma
    )
    thumb_eligible = np.flatnonzero(
        np.isfinite(thumb_pixel_distance)
        & (thumb_pixel_distance <= args.pixel_radius)
    )
    index_eligible = np.flatnonzero(
        np.isfinite(index_pixel_distance)
        & (index_pixel_distance <= args.pixel_radius)
    )
    thumb_candidates = thumb_eligible[
        np.argsort(thumb_pixel_distance[thumb_eligible])[:args.candidate_topk]
    ]
    index_candidates = index_eligible[
        np.argsort(index_pixel_distance[index_eligible])[:args.candidate_topk]
    ]
    if len(thumb_candidates) == 0 or len(index_candidates) == 0:
        raise RuntimeError("Local candidate filtering removed every candidate")

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
    thumb_weight = probability[thumb_mask]
    index_weight = probability[index_mask]
    thumb_center = np.average(
        pair_hand[thumb_mask], axis=0, weights=np.maximum(thumb_weight, 1e-6)
    )
    index_center = np.average(
        pair_hand[index_mask], axis=0, weights=np.maximum(index_weight, 1e-6)
    )
    hand_midpoint = 0.5 * (thumb_center + index_center)
    hand_axis = index_center - thumb_center
    hand_width = float(np.linalg.norm(hand_axis))
    hand_axis /= max(hand_width, 1e-12)
    pair_midpoint = 0.5 * (
        thumb_points[:, None] + index_points[None]
    )
    midpoint_shift = np.linalg.norm(
        pair_midpoint - hand_midpoint[None, None], axis=-1
    )
    axis_cosine = (direction * hand_axis[None, None]).sum(axis=-1)
    axis_error = 1.0 - axis_cosine
    distance_mm = (
        thumb_distance[thumb_candidates, None]
        + index_distance[index_candidates][None]
    ) * 1000.0
    pixel_distance = (
        thumb_pixel_distance[thumb_candidates, None]
        + index_pixel_distance[index_candidates][None]
    )
    width_change_mm = np.abs(width - hand_width) * 1000.0
    score = (
        args.w_pixel * pixel_distance
        + args.w_opposition * opposition
        + args.w_facing * (2.0 - facing)
        + args.w_axis * axis_error
        + args.w_width * width_change_mm
    )
    if not args.translation_invariant:
        score = (
            score
            + distance_mm
            + args.w_midpoint * midpoint_shift * 1000.0
        )
    invalid_width = (width <= 1e-4) | (
        width * 1000.0 > args.max_pair_width_mm
    )
    invalid = (
        invalid_width
        | (normal_dot > args.max_pair_normal_dot)
        | (width_change_mm > args.max_width_change_mm)
        | (axis_cosine < args.min_axis_cosine)
    )
    if not args.translation_invariant:
        invalid |= midpoint_shift * 1000.0 > args.max_midpoint_shift_mm
    score[invalid] = np.inf
    flat = int(np.argmin(score))
    thumb_choice, index_choice = np.unravel_index(flat, score.shape)
    if not np.isfinite(score[thumb_choice, index_choice]):
        raise RuntimeError("No valid opposition pair")
    selected_thumb = thumb_points[thumb_choice]
    selected_index = index_points[index_choice]
    selected_thumb_vertex = int(thumb_candidates[thumb_choice])
    selected_index_vertex = int(index_candidates[index_choice])
    patch_radius = args.patch_radius_mm / 1000.0
    thumb_patch_ids = geodesic_patch(
        object_vertices_local, object_faces, normals_local,
        selected_thumb_vertex, patch_radius, args.patch_normal_cosine,
    )
    index_patch_ids = geodesic_patch(
        object_vertices_local, object_faces, normals_local,
        selected_index_vertex, patch_radius, args.patch_normal_cosine,
    )
    thumb_patch_camera = object_vertices_camera[thumb_patch_ids]
    index_patch_camera = object_vertices_camera[index_patch_ids]

    summary = {
        "frame_id": requested,
        "pair_hand_source": args.pair_hand_source,
        "thumb_contact_vertices": int(thumb_mask.sum()),
        "index_contact_vertices": int(index_mask.sum()),
        "thumb_candidate_count": int(len(thumb_candidates)),
        "index_candidate_count": int(len(index_candidates)),
        "thumb_pixel_distance_min": float(thumb_pixel_distance.min()),
        "index_pixel_distance_min": float(index_pixel_distance.min()),
        "selected_score": float(score[thumb_choice, index_choice]),
        "selected_pair_width_mm": float(width[thumb_choice, index_choice] * 1000.0),
        "selected_thumb_vertex_id": selected_thumb_vertex,
        "selected_index_vertex_id": selected_index_vertex,
        "thumb_patch_vertices": int(len(thumb_patch_ids)),
        "index_patch_vertices": int(len(index_patch_ids)),
        "patch_radius_mm": args.patch_radius_mm,
        "patch_normal_cosine": args.patch_normal_cosine,
        "selected_normal_dot": float(normal_dot[thumb_choice, index_choice]),
        "selected_midpoint_shift_mm": float(
            midpoint_shift[thumb_choice, index_choice] * 1000.0
        ),
        "selected_axis_cosine": float(axis_cosine[thumb_choice, index_choice]),
        "translation_invariant": args.translation_invariant,
        "selected_width_change_mm": float(
            width_change_mm[thumb_choice, index_choice]
        ),
        "hand_contact_width_mm": float(hand_width * 1000.0),
        "selected_thumb_choir_mm": float(
            thumb_distance[thumb_candidates[thumb_choice]] * 1000.0
        ),
        "selected_index_choir_mm": float(
            index_distance[index_candidates[index_choice]] * 1000.0
        ),
        "selected_thumb_pixel_distance": float(
            thumb_pixel_distance[thumb_candidates[thumb_choice]]
        ),
        "selected_index_pixel_distance": float(
            index_pixel_distance[index_candidates[index_choice]]
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
            thumb_candidate_pixel_distance=thumb_pixel_distance[thumb_candidates],
            index_candidate_pixel_distance=index_pixel_distance[index_candidates],
            selected_thumb=selected_thumb,
            selected_index=selected_index,
            selected_thumb_normal=thumb_normals[thumb_choice],
            selected_index_normal=index_normals[index_choice],
            selected_thumb_vertex_id=np.asarray(selected_thumb_vertex, dtype=np.int64),
            selected_index_vertex_id=np.asarray(selected_index_vertex, dtype=np.int64),
            thumb_patch_vertex_ids=thumb_patch_ids,
            index_patch_vertex_ids=index_patch_ids,
            thumb_patch_vertices_canonical=object_vertices_local[thumb_patch_ids],
            index_patch_vertices_canonical=object_vertices_local[index_patch_ids],
            thumb_patch_normals_canonical=normals_local[thumb_patch_ids],
            index_patch_normals_canonical=normals_local[index_patch_ids],
        )

    server = viser.ViserServer(port=args.port)
    server.scene.add_mesh_simple(
        "/object", vertices=object_vertices_camera,
        faces=object_faces, color=(170, 180, 195), opacity=0.8,
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
        colors=colors(len(thumb_points), (255, 140, 220)), point_size=0.002,
    )
    server.scene.add_point_cloud(
        "/candidates/index", points=index_points,
        colors=colors(len(index_points), (120, 255, 145)), point_size=0.002,
    )
    server.scene.add_point_cloud(
        "/selected/pair", points=np.stack([selected_thumb, selected_index]),
        colors=np.asarray([[255, 0, 255], [0, 255, 0]], dtype=np.uint8),
        point_size=0.009,
    )
    server.scene.add_point_cloud(
        "/patch/thumb_fixed", points=thumb_patch_camera,
        colors=colors(len(thumb_patch_camera), (255, 40, 190)), point_size=0.006,
    )
    server.scene.add_point_cloud(
        "/patch/index_fixed", points=index_patch_camera,
        colors=colors(len(index_patch_camera), (20, 255, 80)), point_size=0.006,
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
