#!/usr/bin/env python3
"""Select stable canonical YCB contact patches from a HACO sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--phase-npz")
    parser.add_argument("--phase-key", default="predicted_contact_gate")
    parser.add_argument("--minimum-phase-gate", type=float, default=0.999)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--dense-root")
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--minimum-contact-vertices", type=int, default=3)
    parser.add_argument("--haco-components-per-region", type=int, default=1)
    parser.add_argument("--pixel-radius", type=float, default=30.0)
    parser.add_argument("--pixel-soft-topk", type=int, default=8)
    parser.add_argument("--pixel-sigma", type=float, default=12.0)
    parser.add_argument("--candidate-topk", type=int, default=512)
    parser.add_argument("--distance-slack-mm", type=float, default=60.0)
    parser.add_argument("--max-contact-distance-mm", type=float, default=90.0)
    parser.add_argument("--max-depth-intrusion-mm", type=float, default=12.0)
    parser.add_argument("--depth-intrusion-sigma-mm", type=float, default=4.0)
    parser.add_argument("--onset-half-life-frames", type=float, default=24.0)
    parser.add_argument("--min-facing-cosine", type=float, default=0.15)
    parser.add_argument("--max-normal-dot", type=float, default=1.0)
    parser.add_argument("--visible-surface-only", action="store_true")
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
    parser.add_argument("--patch-radius-mm", type=float, default=6.0)
    parser.add_argument("--patch-normal-cosine", type=float, default=0.8)
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
    return output


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
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.depth_intrusion_sigma_mm <= 0:
        raise ValueError("--depth-intrusion-sigma-mm must be positive")
    if args.onset_half_life_frames <= 0:
        raise ValueError("--onset-half-life-frames must be positive")
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
    hand_vertices = root_relative + wrists[:, None]

    mesh = trimesh.load(Path(args.object_mesh).expanduser().resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    object_local = np.asarray(mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(mesh.faces, dtype=np.int64)
    object_normals_local = np.asarray(mesh.vertex_normals, dtype=np.float32)
    object_sparse_graph = mesh_graph(object_local, object_faces)
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    intrinsics = load_intrinsics(
        args, query, str(np.asarray(query["stream_id"]).item())
    )

    observations: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    contact_onset_index = int(np.flatnonzero(valid)[0])
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
        object_uv = project(object_vertices, intrinsics)
        hand_uv = project(hand, intrinsics)
        visible = np.ones(len(object_vertices), dtype=bool)
        if args.visible_surface_only:
            visible = visible_object_vertices(
                object_uv,
                object_vertices[:, 2],
                args.visibility_bin_px,
                args.visibility_depth_tolerance_mm / 1000.0,
            )

        selected_this_frame: list[str] = []
        for region_index, region_name in enumerate(region_names):
            raw_mask = (
                (region_ids == region_index) & (probability >= threshold)
            )
            if int(raw_mask.sum()) < args.minimum_contact_vertices:
                continue
            mask = strongest_components(
                raw_mask,
                mano_graph,
                probability,
                args.haco_components_per_region,
            )
            if int(mask.sum()) < args.minimum_contact_vertices:
                continue
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
            candidate_valid = (
                (distance_mm <= args.max_contact_distance_mm)
                & (distance_mm <= minimum_distance + args.distance_slack_mm)
                & (depth_intrusion_mm <= args.max_depth_intrusion_mm)
                & (facing >= args.min_facing_cosine)
                & (normal_dot <= args.max_normal_dot)
            )
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
            selected_offset = int(np.argmin(score))
            selected_id = int(candidates[selected_offset])
            contact_vertices = np.flatnonzero(mask)
            haco_probability = float(probability[contact_vertices].mean())
            onset_weight = 0.5 ** (
                max(int(index) - contact_onset_index, 0)
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
                "haco_vertices": int(len(contact_vertices)),
                "haco_probability": haco_probability,
                "onset_weight": float(onset_weight),
                "intrusion_weight": float(intrusion_weight),
                "vote_weight": float(quality),
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
        "observation_onset_weights": np.asarray(
            [row["onset_weight"] for row in observations], dtype=np.float32
        ),
        "observation_vote_weights": np.asarray(
            [row["vote_weight"] for row in observations], dtype=np.float32
        ),
    }

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
            "stable": bool(consensus >= args.minimum_consensus),
            "selected_vertex_id": center_id,
            "patch_vertices": int(len(patch_ids)),
        })
        output_arrays[f"{region_name}_selected_vertex_id"] = np.asarray(
            center_id, dtype=np.int64
        )
        output_arrays[f"{region_name}_patch_vertex_ids"] = patch_ids
        output_arrays[f"{region_name}_patch_vertices_canonical"] = object_local[patch_ids]
        output_arrays[f"{region_name}_patch_normals_canonical"] = object_normals_local[patch_ids]

    selected_names = [str(row["region"]) for row in selected_regions]
    output_arrays["selected_region_names"] = np.asarray(selected_names)
    summary = {
        "method": "v14_haco_multiregion_sequence_consensus_v2",
        "stream_id": str(np.asarray(query["stream_id"]).item()),
        "frames": len(ids),
        "eligible_frames": int(valid.sum()),
        "sampled_frames": int(len(eligible)),
        "successful_observations": len(observations),
        "selected_regions": selected_regions,
        "low_consensus_regions": [
            str(row["region"]) for row in selected_regions if not bool(row["stable"])
        ],
        "constraints": {
            "frame_stride": args.frame_stride,
            "minimum_phase_gate": args.minimum_phase_gate,
            "pixel_radius": args.pixel_radius,
            "distance_slack_mm": args.distance_slack_mm,
            "max_contact_distance_mm": args.max_contact_distance_mm,
            "max_depth_intrusion_mm": args.max_depth_intrusion_mm,
            "depth_intrusion_sigma_mm": args.depth_intrusion_sigma_mm,
            "onset_half_life_frames": args.onset_half_life_frames,
            "min_facing_cosine": args.min_facing_cosine,
            "max_normal_dot": args.max_normal_dot,
            "visible_surface_only": args.visible_surface_only,
            "cluster_radius_mm": args.cluster_radius_mm,
            "minimum_consensus": args.minimum_consensus,
            "weights": {
                "pixel": args.w_pixel,
                "distance": args.w_distance,
                "depth_intrusion": args.w_depth_intrusion,
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
