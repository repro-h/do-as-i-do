#!/usr/bin/env python3
"""Select stable canonical YCB contact patches from a HACO sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from audit_capped_mano_ycb_vertices import contains_points_vote
from refine_v14_haco_sequence_contact_containment import mano_contact_region_ids
from visualize_haco_choir_opposition_candidates import (
    frame_id,
    geodesic_patch,
    index_for,
    load_intrinsics,
    load_npz,
    physical_pose,
    project,
)
from visualize_haco_multiregion_object_contacts import visible_object_vertices
from visualize_capped_mano_wrist import cap_faces, directed_boundary_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--hand-npz")
    parser.add_argument("--hand-vertices-key")
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--phase-npz")
    parser.add_argument("--phase-key", default="predicted_contact_gate")
    parser.add_argument("--minimum-phase-gate", type=float, default=0.999)
    parser.add_argument(
        "--selection-frame",
        help=(
            "Select patches from exactly one frame and accept every geometrically "
            "valid region without temporal consensus"
        ),
    )
    parser.add_argument(
        "--onset-window-frames",
        type=int,
        default=0,
        help=(
            "Use only this many frames starting at the first eligible contact "
            "frame; zero keeps the full contact segment"
        ),
    )
    parser.add_argument(
        "--per-region-onset-window-frames",
        type=int,
        default=0,
        help=(
            "Use an independent first-touch window for each HACO region. This "
            "keeps later-arriving fingers without using their late penetrated frames."
        ),
    )
    parser.add_argument(
        "--per-region-onset-from-valid-observation",
        action="store_true",
        help=(
            "Start each region window at its first object candidate that "
            "passes all geometry gates, rather than its first HACO-only frame."
        ),
    )
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--dense-root")
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--minimum-contact-vertices", type=int, default=3)
    parser.add_argument(
        "--haco-topk-per-region",
        type=int,
        default=0,
        help="Use only each region's highest-probability HACO vertices; 0 uses all.",
    )
    parser.add_argument("--haco-components-per-region", type=int, default=1)
    parser.add_argument("--pixel-radius", type=float, default=30.0)
    parser.add_argument("--pixel-soft-topk", type=int, default=8)
    parser.add_argument("--pixel-sigma", type=float, default=12.0)
    parser.add_argument("--candidate-topk", type=int, default=512)
    parser.add_argument(
        "--penetration-seeded-candidates",
        action="store_true",
        help=(
            "Prioritize the first-touch object surface already contained by "
            "the capped V14 hand, grouped by HACO region. Candidate restriction "
            "uses mesh geodesic distance so nearby opposite surfaces stay separate."
        ),
    )
    parser.add_argument("--penetration-seed-min-vertices", type=int, default=3)
    parser.add_argument("--penetration-seed-radius-mm", type=float, default=15.0)
    parser.add_argument("--penetration-seed-weight", type=float, default=4.0)
    parser.add_argument("--penetration-seed-point-chunk", type=int, default=256)
    parser.add_argument("--penetration-seed-rays", type=int, choices=(1, 3), default=3)
    parser.add_argument(
        "--penetration-seed-device", choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--distance-slack-mm", type=float, default=60.0)
    parser.add_argument("--max-contact-distance-mm", type=float, default=90.0)
    parser.add_argument("--max-depth-intrusion-mm", type=float, default=12.0)
    parser.add_argument("--depth-intrusion-sigma-mm", type=float, default=4.0)
    parser.add_argument("--onset-half-life-frames", type=float, default=24.0)
    parser.add_argument("--min-facing-cosine", type=float, default=0.15)
    parser.add_argument("--max-normal-dot", type=float, default=1.0)
    parser.add_argument("--normal-fallback-max-dot", type=float, default=1.0)
    parser.add_argument(
        "--minimum-normal-compatible-candidates", type=int, default=8
    )
    parser.add_argument(
        "--require-normal-compatible",
        action="store_true",
        help="Reject a frame/region instead of falling back to ungated normals.",
    )
    parser.add_argument("--visible-surface-only", action="store_true")
    parser.add_argument("--visibility-layers", type=int, default=1)
    parser.add_argument("--visibility-bin-px", type=float, default=3.0)
    parser.add_argument(
        "--visibility-depth-tolerance-mm", type=float, default=10.0
    )
    parser.add_argument("--w-pixel", type=float, default=2.0)
    parser.add_argument("--w-distance", type=float, default=0.15)
    parser.add_argument("--w-depth-intrusion", type=float, default=8.0)
    parser.add_argument("--w-facing", type=float, default=50.0)
    parser.add_argument("--w-normal", type=float, default=60.0)
    parser.add_argument("--cluster-radius-mm", type=float, default=12.0)
    parser.add_argument("--minimum-consensus", type=float, default=0.6)
    parser.add_argument("--minimum-dominant-observations", type=int, default=3)
    parser.add_argument("--patch-radius-mm", type=float, default=6.0)
    parser.add_argument("--patch-normal-cosine", type=float, default=0.8)
    parser.add_argument("--translation-vote-cluster-mm", type=float, default=20.0)
    parser.add_argument(
        "--auto-opposition-max-normal-dot", type=float, default=-0.3
    )
    parser.add_argument(
        "--auto-opposition-max-vote-difference-mm", type=float, default=15.0
    )
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    output = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(output, faces[:, corner], face_normals)
    output /= np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1e-12)
    # A canonical-right-to-left reflection changes mesh handedness without
    # changing the stored MANO face winding. Orient the resulting normals by
    # the mesh interior so left/right caches both point out of the hand.
    center = np.median(vertices, axis=0)
    radial_alignment = np.sum(output * (vertices - center), axis=-1)
    if float(np.median(radial_alignment)) < 0.0:
        output = -output
    return output


def select_hand_vertices_key(
    data: dict[str, np.ndarray], requested: str | None
) -> str:
    if requested:
        if requested not in data:
            raise KeyError(f"Hand archive lacks {requested!r}")
        return requested
    for key in (
        "refined_hand_vertices_camera",
        "stage1_hand_vertices_camera",
        "initial_hand_vertices_camera",
    ):
        if key in data:
            return key
    raise KeyError("Could not find camera-space hand vertices in hand archive")


def layered_visible_object_vertices(
    uv: np.ndarray,
    depth: np.ndarray,
    bin_size: float,
    tolerance: float,
    layer_count: int,
) -> np.ndarray:
    if layer_count <= 1:
        return visible_object_vertices(uv, depth, bin_size, tolerance)
    valid = np.isfinite(uv).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    bins = np.zeros((len(uv), 2), dtype=np.int64)
    bins[valid] = np.floor(
        uv[valid] / max(bin_size, 1.0)
    ).astype(np.int64)
    grouped: dict[tuple[int, int], list[int]] = {}
    for index in np.flatnonzero(valid):
        key = (int(bins[index, 0]), int(bins[index, 1]))
        grouped.setdefault(key, []).append(int(index))
    selected = np.zeros(len(uv), dtype=bool)
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: float(depth[index]))
        layer_depths: list[float] = []
        for index in ordered:
            value = float(depth[index])
            if not layer_depths or value > layer_depths[-1] + tolerance:
                layer_depths.append(value)
                if len(layer_depths) > layer_count:
                    break
            if len(layer_depths) <= layer_count:
                selected[index] = True
        retained = layer_depths[:layer_count]
        if retained:
            selected[indices] = np.asarray([
                any(abs(float(depth[index]) - value) <= tolerance for value in retained)
                for index in indices
            ])
    return selected


def adjacency(faces: np.ndarray, vertex_count: int) -> list[set[int]]:
    output: list[set[int]] = [set() for _ in range(vertex_count)]
    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            output[first].add(second)
            output[second].add(first)
    return output


def strongest_components(
    mask: np.ndarray,
    graph: list[set[int]],
    probability: np.ndarray,
    count: int,
) -> np.ndarray:
    remaining = set(np.flatnonzero(mask).tolist())
    components: list[np.ndarray] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        stack = [seed]
        while stack:
            vertex = stack.pop()
            neighbors = graph[vertex].intersection(remaining)
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
            component.extend(neighbors)
        components.append(np.asarray(component, dtype=np.int64))
    components.sort(
        key=lambda vertices: float(probability[vertices].sum()), reverse=True
    )
    output = np.zeros_like(mask, dtype=bool)
    selected = components[: max(1, count)]
    if selected:
        output[np.concatenate(selected)] = True
    return output


def soft_pixel_distance(
    object_uv: np.ndarray,
    hand_uv: np.ndarray,
    topk: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite_object = np.isfinite(object_uv).all(axis=-1)
    finite_hand = np.isfinite(hand_uv).all(axis=-1)
    distance = np.full(len(object_uv), np.inf, dtype=np.float32)
    nearest = np.full(len(object_uv), -1, dtype=np.int64)
    if not finite_object.any() or not finite_hand.any():
        return distance, nearest
    valid_hand_indices = np.flatnonzero(finite_hand)
    tree = cKDTree(hand_uv[finite_hand])
    count = min(max(1, topk), len(valid_hand_indices))
    values, indices = tree.query(object_uv[finite_object], k=count)
    if count == 1:
        values = values[:, None]
        indices = indices[:, None]
    weights = np.exp(-values * values / (2.0 * sigma * sigma))
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
    distance[finite_object] = np.sqrt((weights * values * values).sum(axis=-1))
    nearest[finite_object] = valid_hand_indices[indices[:, 0]]
    return distance, nearest


def mesh_graph(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
    edges: dict[tuple[int, int], float] = {}
    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            key = (min(first, second), max(first, second))
            length = float(np.linalg.norm(vertices[first] - vertices[second]))
            edges[key] = min(edges.get(key, np.inf), length)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (first, second), length in edges.items():
        rows.extend((first, second))
        columns.extend((second, first))
        values.extend((length, length))
    return csr_matrix((values, (rows, columns)), shape=(len(vertices), len(vertices)))


def clusters_from_distances(distances: np.ndarray, radius: float) -> list[np.ndarray]:
    parent = np.arange(len(distances), dtype=np.int64)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(distances)):
        for second in range(first + 1, len(distances)):
            if distances[first, second] <= radius:
                union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(distances)):
        groups.setdefault(root(index), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def main() -> None:
    args = parse_args()
    if args.selection_frame and (
        args.onset_window_frames > 0
        or args.per_region_onset_window_frames > 0
    ):
        raise ValueError(
            "--selection-frame and --onset-window-frames are mutually exclusive"
        )
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.depth_intrusion_sigma_mm <= 0:
        raise ValueError("--depth-intrusion-sigma-mm must be positive")
    if args.onset_half_life_frames <= 0:
        raise ValueError("--onset-half-life-frames must be positive")
    if args.minimum_dominant_observations <= 0:
        raise ValueError("--minimum-dominant-observations must be positive")
    if args.penetration_seed_min_vertices <= 0:
        raise ValueError("--penetration-seed-min-vertices must be positive")
    if args.penetration_seed_radius_mm <= 0:
        raise ValueError("--penetration-seed-radius-mm must be positive")
    penetration_device = torch.device(args.penetration_seed_device)
    if penetration_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--penetration-seed-device cuda requested but unavailable")
    if args.visibility_layers <= 0:
        raise ValueError("--visibility-layers must be positive")
    if args.onset_window_frames < 0:
        raise ValueError("--onset-window-frames must be non-negative")
    if args.per_region_onset_window_frames < 0:
        raise ValueError(
            "--per-region-onset-window-frames must be non-negative"
        )
    if args.onset_window_frames and args.per_region_onset_window_frames:
        raise ValueError(
            "Global and per-region onset windows are mutually exclusive"
        )
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    phase = (
        load_npz(Path(args.phase_npz).expanduser().resolve())
        if args.phase_npz else None
    )

    ids = np.asarray(query["frame_ids"])
    trajectory_indices = np.asarray(
        [index_for(trajectory["frame_ids"], frame_id(value)) for value in ids]
    )
    contact_indices = np.asarray(
        [index_for(contact["frame_ids"], frame_id(value)) for value in ids]
    )
    supervision_indices = np.asarray(
        [index_for(supervision["frame_ids"], frame_id(value)) for value in ids]
    )
    gate = np.ones(len(ids), dtype=np.float32)
    if phase is not None:
        if args.phase_key not in phase:
            raise KeyError(f"Phase archive lacks {args.phase_key!r}")
        phase_indices = np.asarray(
            [index_for(phase["frame_ids"], frame_id(value)) for value in ids]
        )
        gate = np.asarray(phase[args.phase_key][phase_indices], dtype=np.float32)
    valid = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
        & (gate >= args.minimum_phase_gate)
    )
    valid_indices = np.flatnonzero(valid)
    if args.onset_window_frames > 0 and len(valid_indices):
        onset = int(valid_indices[0])
        valid &= np.arange(len(ids)) < onset + args.onset_window_frames
    if args.selection_frame:
        selected_index = index_for(ids, frame_id(args.selection_frame))
        eligible = (
            np.asarray([selected_index], dtype=np.int64)
            if valid[selected_index] else np.empty(0, dtype=np.int64)
        )
    else:
        eligible = np.flatnonzero(valid)[:: args.frame_stride]
    if not len(eligible):
        raise RuntimeError("No valid stable-contact frame was selected")

    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    mano_graph = adjacency(faces, 778)
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    probability_all = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    wrists = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    root_relative = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    )
    hand_source = str(Path(args.trajectory_npz).expanduser().resolve())
    hand_vertices_key = "query_root_relative_plus_predicted_wrist"
    hand_vertices = root_relative + wrists[:, None]
    hand_valid = np.isfinite(hand_vertices).all(axis=(1, 2))
    if args.hand_npz:
        hand_data = load_npz(Path(args.hand_npz).expanduser().resolve())
        if "frame_ids" not in hand_data:
            raise KeyError("Hand archive lacks 'frame_ids'")
        hand_indices = np.asarray([
            index_for(hand_data["frame_ids"], frame_id(value)) for value in ids
        ])
        hand_vertices_key = select_hand_vertices_key(
            hand_data, args.hand_vertices_key
        )
        hand_vertices = np.asarray(
            hand_data[hand_vertices_key][hand_indices], dtype=np.float32
        )
        if hand_vertices.shape != root_relative.shape:
            raise ValueError(
                f"Hand shape mismatch: {hand_vertices.shape} != "
                f"{root_relative.shape}"
            )
        hand_valid = np.isfinite(hand_vertices).all(axis=(1, 2))
        hand_source = str(Path(args.hand_npz).expanduser().resolve())
    valid &= hand_valid
    if args.selection_frame:
        selected_index = index_for(ids, frame_id(args.selection_frame))
        eligible = (
            np.asarray([selected_index], dtype=np.int64)
            if valid[selected_index] else np.empty(0, dtype=np.int64)
        )
    else:
        eligible = np.flatnonzero(valid)[:: args.frame_stride]
    if not len(eligible):
        raise RuntimeError("No valid stable-contact frame was selected")

    mesh = trimesh.load(Path(args.object_mesh).expanduser().resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    object_local = np.asarray(mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(mesh.faces, dtype=np.int64)
    object_normals_local = np.asarray(mesh.vertex_normals, dtype=np.float32)
    object_normals_local /= np.maximum(
        np.linalg.norm(object_normals_local, axis=-1, keepdims=True), 1e-12
    )
    object_sparse_graph = mesh_graph(object_local, object_faces)
    wrist_boundary = directed_boundary_loop(faces)
    capped_faces = np.concatenate(
        (faces, cap_faces(wrist_boundary, hand_vertices.shape[1])), axis=0
    )
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    intrinsics = load_intrinsics(
        args, query, str(np.asarray(query["stream_id"]).item())
    )

    observations: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    penetration_seed_audit: list[dict[str, object]] = []
    frozen_penetration_seed_ids: dict[str, np.ndarray] = {}
    penetration_seed_source_frames: dict[str, str] = {}
    contact_onset_index = int(np.flatnonzero(valid)[0])
    region_onset_indices: dict[str, int] = {}
    for progress, index in enumerate(eligible, start=1):
        requested = frame_id(ids[index])
        hand = hand_vertices[index]
        hand_normals = vertex_normals(hand, faces)
        probability = probability_all[index]
        pose = physical_pose(
            supervision["gt_ycb_object_pose"][supervision_indices[index]],
            normalized_left,
        )
        object_vertices = object_local @ pose[:3, :3].T + pose[:3, 3]
        object_normals = object_normals_local @ pose[:3, :3].T
        object_normals /= np.maximum(
            np.linalg.norm(object_normals, axis=-1, keepdims=True), 1e-12
        )
        penetration_labels = np.full(len(object_vertices), -1, dtype=np.int16)
        if args.penetration_seeded_candidates:
            cap_center = hand[wrist_boundary].mean(axis=0)
            capped_vertices = np.concatenate((hand, cap_center[None]), axis=0)
            lower = capped_vertices.min(axis=0) - 1e-4
            upper = capped_vertices.max(axis=0) + 1e-4
            broadphase = np.all(
                (object_vertices >= lower) & (object_vertices <= upper), axis=-1
            )
            broadphase_ids = np.flatnonzero(broadphase)
            if len(broadphase_ids):
                contained, _ = contains_points_vote(
                    object_vertices[broadphase_ids],
                    capped_vertices,
                    capped_faces,
                    args.penetration_seed_point_chunk,
                    ray_count=args.penetration_seed_rays,
                    device=penetration_device,
                )
                inside_ids = broadphase_ids[contained]
                if len(inside_ids):
                    nearest_hand = cKDTree(hand).query(
                        object_vertices[inside_ids], k=1
                    )[1]
                    penetration_labels[inside_ids] = region_ids[
                        nearest_hand
                    ].astype(np.int16)
        object_uv = project(object_vertices, intrinsics)
        hand_uv = project(hand, intrinsics)
        visible = np.ones(len(object_vertices), dtype=bool)
        if args.visible_surface_only:
            visible = layered_visible_object_vertices(
                object_uv,
                object_vertices[:, 2],
                args.visibility_bin_px,
                args.visibility_depth_tolerance_mm / 1000.0,
                args.visibility_layers,
            )

        selected_this_frame: list[str] = []
        for region_index, region_name in enumerate(region_names):
            raw_mask = (
                (region_ids == region_index) & (probability >= threshold)
            )
            if int(raw_mask.sum()) < args.minimum_contact_vertices:
                continue
            if args.per_region_onset_window_frames > 0:
                region_onset = region_onset_indices.get(region_name)
                if region_onset is None:
                    if not args.per_region_onset_from_valid_observation:
                        region_onset = region_onset_indices.setdefault(
                            region_name, int(index)
                        )
                elif int(index) >= (
                    region_onset + args.per_region_onset_window_frames
                ):
                    continue
            mask = strongest_components(
                raw_mask,
                mano_graph,
                probability,
                args.haco_components_per_region,
            )
            if int(mask.sum()) < args.minimum_contact_vertices:
                continue
            if args.haco_topk_per_region > 0 and int(mask.sum()) > args.haco_topk_per_region:
                component_ids = np.flatnonzero(mask)
                keep = component_ids[np.argsort(
                    probability[component_ids]
                )[-args.haco_topk_per_region:]]
                mask = np.zeros_like(mask)
                mask[keep] = True
            region_hand = hand[mask]
            region_hand_normals = hand_normals[mask]
            region_uv = hand_uv[mask]
            pixel_distance, nearest_contact = soft_pixel_distance(
                object_uv,
                region_uv,
                args.pixel_soft_topk,
                args.pixel_sigma,
            )
            eligible_object = np.flatnonzero(
                np.isfinite(pixel_distance)
                & (pixel_distance <= args.pixel_radius)
                & visible
                & (nearest_contact >= 0)
            )
            if not len(eligible_object):
                skipped.append({
                    "frame_id": requested,
                    "region": region_name,
                    "reason": "no_2d_candidate",
                })
                continue
            candidates = eligible_object[
                np.argsort(pixel_distance[eligible_object])[: args.candidate_topk]
            ]
            current_penetration_seed_ids = np.flatnonzero(
                penetration_labels == region_index
            )
            if (
                region_name not in frozen_penetration_seed_ids
                and len(current_penetration_seed_ids)
                >= args.penetration_seed_min_vertices
            ):
                frozen_penetration_seed_ids[region_name] = (
                    current_penetration_seed_ids.copy()
                )
                penetration_seed_source_frames[region_name] = requested
            penetration_seed_ids = frozen_penetration_seed_ids.get(
                region_name, np.empty(0, dtype=np.int64)
            )
            seed_geodesic_mm = np.full(len(candidates), np.inf, dtype=np.float32)
            seed_mode = "fallback_no_seed"
            if len(penetration_seed_ids) >= args.penetration_seed_min_vertices:
                seed_distance = dijkstra(
                    object_sparse_graph,
                    directed=False,
                    indices=penetration_seed_ids,
                    limit=args.penetration_seed_radius_mm / 1000.0,
                    min_only=True,
                )
                seed_geodesic_mm = (
                    np.asarray(seed_distance[candidates], dtype=np.float32) * 1000.0
                )
                seeded = np.isfinite(seed_geodesic_mm)
                if seeded.any():
                    candidates = candidates[seeded]
                    seed_geodesic_mm = seed_geodesic_mm[seeded]
                    seed_mode = "penetration_seeded"
                else:
                    seed_mode = "fallback_no_candidate_on_seed_surface"
            penetration_seed_audit.append({
                "frame_id": requested,
                "region": region_name,
                "seed_vertices": int(len(penetration_seed_ids)),
                "current_inside_vertices": int(
                    len(current_penetration_seed_ids)
                ),
                "seed_source_frame": penetration_seed_source_frames.get(
                    region_name
                ),
                "mode": seed_mode,
                "seeded_candidates": int(
                    np.isfinite(seed_geodesic_mm).sum()
                ),
            })
            matched = nearest_contact[candidates]
            matched_hand = region_hand[matched]
            matched_normals = region_hand_normals[matched]
            points = object_vertices[candidates]
            displacement = matched_hand - points
            distance_mm = np.linalg.norm(displacement, axis=-1) * 1000.0
            # A matched hand point behind the nearest visible object surface is
            # a local penetration signal. Unlike mesh containment, this remains
            # cheap and does not require the object mesh to be watertight.
            depth_intrusion_mm = np.maximum(
                (matched_hand[:, 2] - points[:, 2]) * 1000.0,
                0.0,
            )
            direction = displacement / np.maximum(
                np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-12
            )
            facing = np.sum(object_normals[candidates] * direction, axis=-1)
            normal_dot = np.sum(
                object_normals[candidates] * matched_normals, axis=-1
            )
            minimum_distance = float(distance_mm.min())
            candidate_base_valid = (
                (distance_mm <= args.max_contact_distance_mm)
                & (distance_mm <= minimum_distance + args.distance_slack_mm)
                & (depth_intrusion_mm <= args.max_depth_intrusion_mm)
                & (facing >= args.min_facing_cosine)
            )
            primary_normal = (
                candidate_base_valid & (normal_dot <= args.max_normal_dot)
            )
            fallback_normal = (
                candidate_base_valid
                & (normal_dot <= args.normal_fallback_max_dot)
            )
            if args.require_normal_compatible:
                candidate_valid = primary_normal
                normal_filter_mode = "strict"
            elif (
                int(primary_normal.sum())
                >= args.minimum_normal_compatible_candidates
            ):
                candidate_valid = primary_normal
                normal_filter_mode = "primary"
            elif (
                int(fallback_normal.sum())
                >= args.minimum_normal_compatible_candidates
            ):
                candidate_valid = fallback_normal
                normal_filter_mode = "fallback"
            else:
                candidate_valid = candidate_base_valid
                normal_filter_mode = "ungated"
            if not candidate_valid.any():
                skipped.append({
                    "frame_id": requested,
                    "region": region_name,
                    "reason": "distance_or_direction_gate",
                    "minimum_distance_mm": minimum_distance,
                    "minimum_depth_intrusion_mm": float(
                        depth_intrusion_mm.min()
                    ),
                    "maximum_facing_cosine": float(facing.max()),
                    "minimum_normal_dot": float(normal_dot.min()),
                })
                continue
            candidates = candidates[candidate_valid]
            matched = matched[candidate_valid]
            distance_mm = distance_mm[candidate_valid]
            depth_intrusion_mm = depth_intrusion_mm[candidate_valid]
            facing = facing[candidate_valid]
            normal_dot = normal_dot[candidate_valid]
            candidate_pixel = pixel_distance[candidates]
            score = (
                args.w_pixel * candidate_pixel
                + args.w_distance * distance_mm
                + args.w_depth_intrusion * depth_intrusion_mm
                + args.w_facing * (1.0 - facing)
                + args.w_normal * (1.0 + normal_dot)
            )
            if seed_mode == "penetration_seeded":
                score += args.penetration_seed_weight * seed_geodesic_mm[
                    candidate_valid
                ]
            selected_offset = int(np.argmin(score))
            selected_id = int(candidates[selected_offset])
            if (
                args.per_region_onset_window_frames > 0
                and args.per_region_onset_from_valid_observation
            ):
                region_onset_indices.setdefault(region_name, int(index))
            contact_vertices = np.flatnonzero(mask)
            contact_weights = np.maximum(probability[contact_vertices], 1e-6)
            hand_region_center = np.average(
                hand[contact_vertices], axis=0, weights=contact_weights
            )
            selected_object_point = object_vertices[selected_id]
            translation_vote = selected_object_point - hand_region_center
            haco_probability = float(probability[contact_vertices].mean())
            onset_reference = region_onset_indices.get(
                region_name, contact_onset_index
            )
            onset_weight = 0.5 ** (
                max(int(index) - onset_reference, 0)
                / args.onset_half_life_frames
            )
            intrusion_weight = np.exp(
                -float(depth_intrusion_mm[selected_offset])
                / args.depth_intrusion_sigma_mm
            )
            quality = (
                haco_probability
                * np.sqrt(len(contact_vertices))
                * np.exp(-float(candidate_pixel[selected_offset]) / 20.0)
                * np.exp(-float(distance_mm[selected_offset]) / 160.0)
                * intrusion_weight
                * onset_weight
                * max(float(facing[selected_offset]), 0.05)
                * np.clip((1.0 - float(normal_dot[selected_offset])) * 0.5, 0.1, 1.0)
            )
            observations.append({
                "frame_id": requested,
                "frame_index": int(index),
                "region": region_name,
                "selected_vertex_id": selected_id,
                "score": float(score[selected_offset]),
                "pixel_distance": float(candidate_pixel[selected_offset]),
                "distance_mm": float(distance_mm[selected_offset]),
                "depth_intrusion_mm": float(
                    depth_intrusion_mm[selected_offset]
                ),
                "facing_cosine": float(facing[selected_offset]),
                "normal_dot": float(normal_dot[selected_offset]),
                "normal_filter_mode": normal_filter_mode,
                "penetration_seed_mode": seed_mode,
                "penetration_seed_vertices": int(len(penetration_seed_ids)),
                "penetration_seed_geodesic_mm": float(
                    seed_geodesic_mm[candidate_valid][selected_offset]
                    if seed_mode == "penetration_seeded" else np.nan
                ),
                "primary_normal_candidates": int(primary_normal.sum()),
                "fallback_normal_candidates": int(fallback_normal.sum()),
                "haco_vertices": int(len(contact_vertices)),
                "haco_probability": haco_probability,
                "onset_weight": float(onset_weight),
                "intrusion_weight": float(intrusion_weight),
                "vote_weight": float(quality),
                "hand_region_center_camera": hand_region_center.tolist(),
                "selected_object_point_camera": selected_object_point.tolist(),
                "translation_vote_camera": translation_vote.tolist(),
            })
            selected_this_frame.append(region_name)
        print(
            f"[{progress}/{len(eligible)}] {requested} "
            f"selected={','.join(selected_this_frame) or '-'}",
            flush=True,
        )

    if not observations:
        raise RuntimeError("No frame/region produced a valid object contact")

    selected_regions: list[dict[str, object]] = []
    output_arrays: dict[str, np.ndarray] = {
        "frame_ids": ids,
        "sampled_frame_ids": ids[eligible],
        "intrinsics": intrinsics.astype(np.float32),
        "hand_source": np.asarray(hand_source),
        "hand_vertices_key": np.asarray(hand_vertices_key),
        "observation_frame_ids": np.asarray(
            [row["frame_id"] for row in observations]
        ),
        "observation_frame_indices": np.asarray(
            [row["frame_index"] for row in observations], dtype=np.int32
        ),
        "observation_region_names": np.asarray(
            [row["region"] for row in observations]
        ),
        "observation_selected_vertex_ids": np.asarray(
            [row["selected_vertex_id"] for row in observations], dtype=np.int64
        ),
        "observation_scores": np.asarray(
            [row["score"] for row in observations], dtype=np.float32
        ),
        "observation_distance_mm": np.asarray(
            [row["distance_mm"] for row in observations], dtype=np.float32
        ),
        "observation_depth_intrusion_mm": np.asarray(
            [row["depth_intrusion_mm"] for row in observations],
            dtype=np.float32,
        ),
        "observation_penetration_seed_modes": np.asarray(
            [row["penetration_seed_mode"] for row in observations]
        ),
        "observation_penetration_seed_vertices": np.asarray(
            [row["penetration_seed_vertices"] for row in observations],
            dtype=np.int32,
        ),
        "observation_penetration_seed_geodesic_mm": np.asarray(
            [row["penetration_seed_geodesic_mm"] for row in observations],
            dtype=np.float32,
        ),
        "observation_onset_weights": np.asarray(
            [row["onset_weight"] for row in observations], dtype=np.float32
        ),
        "observation_vote_weights": np.asarray(
            [row["vote_weight"] for row in observations], dtype=np.float32
        ),
        "observation_hand_region_centers_camera": np.asarray(
            [row["hand_region_center_camera"] for row in observations],
            dtype=np.float32,
        ),
        "observation_selected_object_points_camera": np.asarray(
            [row["selected_object_point_camera"] for row in observations],
            dtype=np.float32,
        ),
        "observation_translation_votes_camera": np.asarray(
            [row["translation_vote_camera"] for row in observations],
            dtype=np.float32,
        ),
    }
    for region_name, seed_ids in frozen_penetration_seed_ids.items():
        output_arrays[
            f"{region_name}_penetration_seed_vertex_ids"
        ] = seed_ids.astype(np.int64)

    region_translation_votes: dict[str, np.ndarray] = {}
    region_vote_weights: dict[str, float] = {}
    region_center_ids: dict[str, int] = {}
    for region_name in region_names:
        region_observations = [
            row for row in observations if row["region"] == region_name
        ]
        if not region_observations:
            continue
        center_ids = np.asarray(
            [row["selected_vertex_id"] for row in region_observations],
            dtype=np.int64,
        )
        weights = np.asarray(
            [row["vote_weight"] for row in region_observations],
            dtype=np.float64,
        )
        distances_to_mesh = dijkstra(
            object_sparse_graph,
            directed=False,
            indices=center_ids,
            limit=args.cluster_radius_mm / 1000.0,
        )
        pairwise = distances_to_mesh[:, center_ids]
        clusters = clusters_from_distances(
            pairwise, args.cluster_radius_mm / 1000.0
        )
        cluster_weights = np.asarray(
            [weights[cluster].sum() for cluster in clusters], dtype=np.float64
        )
        dominant = clusters[int(np.argmax(cluster_weights))]
        consensus = float(weights[dominant].sum() / max(weights.sum(), 1e-12))
        dominant_ids = center_ids[dominant]
        medoid_distances = dijkstra(
            object_sparse_graph,
            directed=False,
            indices=dominant_ids,
        )[:, dominant_ids]
        medoid_cost = (medoid_distances * weights[dominant][None]).sum(axis=1)
        center_id = int(dominant_ids[int(np.argmin(medoid_cost))])
        dominant_votes = np.asarray([
            region_observations[int(index)]["translation_vote_camera"]
            for index in dominant
        ], dtype=np.float32)
        dominant_weights = weights[dominant]
        translation_vote = np.average(
            dominant_votes,
            axis=0,
            weights=np.maximum(dominant_weights, 1e-12),
        ).astype(np.float32)
        region_translation_votes[region_name] = translation_vote
        region_vote_weights[region_name] = float(dominant_weights.sum())
        region_center_ids[region_name] = center_id
        patch_ids = geodesic_patch(
            object_local,
            object_faces,
            object_normals_local,
            center_id,
            args.patch_radius_mm / 1000.0,
            args.patch_normal_cosine,
        )
        selected_regions.append({
            "region": region_name,
            "observations": len(region_observations),
            "clusters": len(clusters),
            "dominant_observations": int(len(dominant)),
            "consensus_fraction": consensus,
            "stable": bool(
                args.selection_frame
                or (
                    consensus >= args.minimum_consensus
                    and len(dominant) >= args.minimum_dominant_observations
                )
            ),
            "selected_vertex_id": center_id,
            "patch_vertices": int(len(patch_ids)),
            "translation_vote_mm": (translation_vote * 1000.0).tolist(),
        })
        output_arrays[f"{region_name}_selected_vertex_id"] = np.asarray(
            center_id, dtype=np.int64
        )
        output_arrays[f"{region_name}_patch_vertex_ids"] = patch_ids
        output_arrays[f"{region_name}_patch_vertices_canonical"] = object_local[patch_ids]
        output_arrays[f"{region_name}_patch_normals_canonical"] = object_normals_local[patch_ids]
        output_arrays[f"{region_name}_translation_vote_camera"] = translation_vote

    selected_names = [str(row["region"]) for row in selected_regions]
    stable_names = [
        str(row["region"]) for row in selected_regions if bool(row["stable"])
    ]
    translation_consistent_names: list[str] = []
    stable_opposition_pairs: list[tuple[int, int, float, float]] = []
    if stable_names:
        stable_votes = np.stack([
            region_translation_votes[name] for name in stable_names
        ])
        vote_pairwise = np.linalg.norm(
            stable_votes[:, None] - stable_votes[None], axis=-1
        )
        vote_limit = args.translation_vote_cluster_mm / 1000.0
        for first in range(len(stable_names)):
            first_normal = object_normals_local[
                region_center_ids[stable_names[first]]
            ]
            for second in range(first + 1, len(stable_names)):
                second_normal = object_normals_local[
                    region_center_ids[stable_names[second]]
                ]
                normal_dot = float(first_normal @ second_normal)
                vote_difference_mm = float(
                    vote_pairwise[first, second] * 1000.0
                )
                if (
                    normal_dot <= args.auto_opposition_max_normal_dot
                    and vote_difference_mm
                    <= args.auto_opposition_max_vote_difference_mm
                ):
                    stable_opposition_pairs.append((
                        first, second, normal_dot, vote_difference_mm
                    ))

        candidate_clusters: list[np.ndarray] = []
        for bits in range(1, 1 << len(stable_names)):
            cluster = np.asarray([
                index for index in range(len(stable_names))
                if bits & (1 << index)
            ], dtype=np.int64)
            pairwise = vote_pairwise[np.ix_(cluster, cluster)]
            if np.all(pairwise <= vote_limit + 1e-12):
                candidate_clusters.append(cluster)

        def cluster_score(cluster: np.ndarray) -> tuple[int, float, int]:
            members = set(int(index) for index in cluster)
            opposition_count = sum(
                first in members and second in members
                for first, second, _, _ in stable_opposition_pairs
            )
            weight = sum(
                region_vote_weights[stable_names[int(index)]]
                for index in cluster
            )
            return opposition_count, float(weight), len(cluster)

        dominant_vote_cluster = max(candidate_clusters, key=cluster_score)
        translation_consistent_names = [
            stable_names[int(index)] for index in dominant_vote_cluster
        ]

    automatic_opposition_pairs: list[list[str]] = []
    automatic_opposition_normal_dot: list[float] = []
    automatic_opposition_vote_difference_mm: list[float] = []
    consistent_lookup = set(translation_consistent_names)
    for first, second, normal_dot, vote_difference_mm in stable_opposition_pairs:
        first_name = stable_names[first]
        second_name = stable_names[second]
        if first_name not in consistent_lookup or second_name not in consistent_lookup:
            continue
        automatic_opposition_pairs.append([first_name, second_name])
        automatic_opposition_normal_dot.append(normal_dot)
        automatic_opposition_vote_difference_mm.append(vote_difference_mm)

    output_arrays["selected_region_names"] = np.asarray(selected_names)
    output_arrays["stable_region_names"] = np.asarray(stable_names)
    output_arrays["translation_consistent_region_names"] = np.asarray(
        translation_consistent_names
    )
    output_arrays["automatic_opposition_region_pairs"] = np.asarray(
        automatic_opposition_pairs, dtype="U32"
    ).reshape(-1, 2)
    output_arrays["automatic_opposition_normal_dot"] = np.asarray(
        automatic_opposition_normal_dot, dtype=np.float32
    )
    output_arrays["automatic_opposition_vote_difference_mm"] = np.asarray(
        automatic_opposition_vote_difference_mm, dtype=np.float32
    )
    summary = {
        "method": (
            "stage1_haco_multiregion_sequence_reselection_v3"
            if args.hand_npz
            else "v14_haco_single_frame_2d_direction_fixed_patches_v1"
            if args.selection_frame
            else "v14_haco_first_contact_fixed_patches_v1"
            if args.onset_window_frames > 0
            else "v14_haco_multiregion_sequence_consensus_v2"
        ),
        "stream_id": str(np.asarray(query["stream_id"]).item()),
        "hand_source": hand_source,
        "hand_vertices_key": hand_vertices_key,
        "frames": len(ids),
        "eligible_frames": int(valid.sum()),
        "sampled_frames": int(len(eligible)),
        "successful_observations": len(observations),
        "penetration_seed_audit": penetration_seed_audit,
        "penetration_seed_source_frames": penetration_seed_source_frames,
        "selected_regions": selected_regions,
        "translation_consistent_regions": translation_consistent_names,
        "automatic_opposition_pairs": [
            {
                "regions": pair,
                "normal_dot": automatic_opposition_normal_dot[index],
                "translation_vote_difference_mm": (
                    automatic_opposition_vote_difference_mm[index]
                ),
            }
            for index, pair in enumerate(automatic_opposition_pairs)
        ],
        "low_consensus_regions": [
            str(row["region"]) for row in selected_regions if not bool(row["stable"])
        ],
        "constraints": {
            "frame_stride": args.frame_stride,
            "selection_frame": args.selection_frame,
            "minimum_phase_gate": args.minimum_phase_gate,
            "onset_window_frames": args.onset_window_frames,
            "per_region_onset_window_frames": (
                args.per_region_onset_window_frames
            ),
            "per_region_onset_from_valid_observation": (
                args.per_region_onset_from_valid_observation
            ),
            "region_onset_frames": {
                name: frame_id(ids[index])
                for name, index in region_onset_indices.items()
            },
            "pixel_radius": args.pixel_radius,
            "distance_slack_mm": args.distance_slack_mm,
            "max_contact_distance_mm": args.max_contact_distance_mm,
            "max_depth_intrusion_mm": args.max_depth_intrusion_mm,
            "penetration_seeded_candidates": (
                args.penetration_seeded_candidates
            ),
            "penetration_seed_min_vertices": (
                args.penetration_seed_min_vertices
            ),
            "penetration_seed_radius_mm": args.penetration_seed_radius_mm,
            "penetration_seed_rays": args.penetration_seed_rays,
            "depth_intrusion_sigma_mm": args.depth_intrusion_sigma_mm,
            "onset_half_life_frames": args.onset_half_life_frames,
            "min_facing_cosine": args.min_facing_cosine,
            "max_normal_dot": args.max_normal_dot,
            "normal_fallback_max_dot": args.normal_fallback_max_dot,
            "minimum_normal_compatible_candidates": (
                args.minimum_normal_compatible_candidates
            ),
            "require_normal_compatible": args.require_normal_compatible,
            "haco_topk_per_region": args.haco_topk_per_region,
            "visible_surface_only": args.visible_surface_only,
            "visibility_layers": args.visibility_layers,
            "cluster_radius_mm": args.cluster_radius_mm,
            "minimum_consensus": args.minimum_consensus,
            "minimum_dominant_observations": (
                args.minimum_dominant_observations
            ),
            "translation_vote_cluster_mm": args.translation_vote_cluster_mm,
            "auto_opposition_max_normal_dot": (
                args.auto_opposition_max_normal_dot
            ),
            "auto_opposition_max_vote_difference_mm": (
                args.auto_opposition_max_vote_difference_mm
            ),
            "weights": {
                "pixel": args.w_pixel,
                "distance": args.w_distance,
                "depth_intrusion": args.w_depth_intrusion,
                "penetration_seed": args.penetration_seed_weight,
                "facing": args.w_facing,
                "normal": args.w_normal,
            },
        },
        "observations": observations,
        "skipped": skipped,
    }
    out_npz = Path(args.out_npz).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **output_arrays)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_npz),
        "summary": str(out_json),
        "sampled_frames": len(eligible),
        "selected_regions": selected_regions,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
