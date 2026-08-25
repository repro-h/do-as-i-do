#!/usr/bin/env python3
"""Interactively inspect V14/Stage1/Stage2 contact over a full sequence."""

from __future__ import annotations

import argparse
import heapq
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from refine_v14_haco_one_way_chamfer import physical_pose
from refine_v14_haco_sequence_chamfer import aligned_indices, frame_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--stage2-npz")
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--gt-hand-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--filtered-contact-topk", type=int, default=64)
    parser.add_argument("--filtered-min-weight", type=float, default=0.05)
    parser.add_argument("--object-distance-sigma-mm", type=float, default=8.0)
    parser.add_argument("--collision-geodesic-sigma-mm", type=float, default=15.0)
    parser.add_argument("--inside-low-fraction", type=float, default=0.01)
    parser.add_argument("--inside-high-fraction", type=float, default=0.02)
    parser.add_argument("--lightweight-single-frame", action="store_true")
    parser.add_argument(
        "--clearance-regions",
        nargs="*",
        default=None,
        help=(
            "Only visualize Stage2 clearance top-k points and push directions "
            "for these HACO region names."
        ),
    )
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8098)
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
    output /= np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1e-8)
    center = np.median(vertices, axis=0)
    radial_alignment = np.sum(output * (vertices - center), axis=-1)
    if float(np.median(radial_alignment)) < 0.0:
        output = -output
    return output


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh: {type(loaded).__name__}")
    return (
        np.asarray(loaded.vertices, dtype=np.float32) * float(scale),
        np.asarray(loaded.faces, dtype=np.int64),
    )


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def nearest_distances(
    query: np.ndarray, reference: np.ndarray, chunk: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty(len(query), dtype=np.float32)
    indices = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        end = min(start + chunk, len(query))
        pairwise = np.linalg.norm(
            query[start:end, None] - reference[None], axis=-1
        )
        selected = pairwise.argmin(axis=1)
        indices[start:end] = selected
        distances[start:end] = pairwise[
            np.arange(end - start), selected
        ]
    return distances, indices


def multisource_geodesic(
    vertices: np.ndarray, edges: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(len(vertices))
    ]
    for first, second in edges:
        weight = float(np.linalg.norm(vertices[first] - vertices[second]))
        adjacency[int(first)].append((int(second), weight))
        adjacency[int(second)].append((int(first), weight))
    distance = np.full(len(vertices), np.inf, dtype=np.float32)
    queue: list[tuple[float, int]] = []
    for seed in np.unique(seeds):
        distance[int(seed)] = 0.0
        heapq.heappush(queue, (0.0, int(seed)))
    while queue:
        current, vertex = heapq.heappop(queue)
        if current > float(distance[vertex]):
            continue
        for neighbor, edge_length in adjacency[vertex]:
            candidate = current + edge_length
            if candidate < float(distance[neighbor]):
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distance


def smooth_contact_gate(
    inside_fraction: float, low: float, high: float
) -> float:
    if not 0.0 <= low < high:
        raise ValueError("Inside thresholds must satisfy 0 <= low < high")
    gate = np.clip((high - inside_fraction) / (high - low), 0.0, 1.0)
    return float(gate * gate * (3.0 - 2.0 * gate))


def main() -> None:
    args = parse_args()
    if args.filtered_contact_topk <= 0:
        raise ValueError("--filtered-contact-topk must be positive")
    if args.object_distance_sigma_mm <= 0:
        raise ValueError("--object-distance-sigma-mm must be positive")
    if args.collision_geodesic_sigma_mm <= 0:
        raise ValueError("--collision-geodesic-sigma-mm must be positive")
    print("[viser] loading trajectory", flush=True)
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    print("[viser] loading WiLoR query", flush=True)
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    print("[viser] loading HACO contacts", flush=True)
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    print("[viser] loading Stage1", flush=True)
    stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
    print("[viser] loading Stage2", flush=True)
    stage2 = (
        load_npz(Path(args.stage2_npz).expanduser().resolve())
        if args.stage2_npz else None
    )
    print("[viser] loading supervision and GT hand", flush=True)
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    gt = load_npz(Path(args.gt_hand_npz).expanduser().resolve())

    ids = np.asarray(query["frame_ids"])
    count = len(ids)
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    contact_indices = aligned_indices(contact["frame_ids"], ids)
    stage1_indices = aligned_indices(stage1["frame_ids"], ids)
    stage2_indices = (
        aligned_indices(stage2["frame_ids"], ids)
        if stage2 is not None else None
    )
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    v14 = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    ) + wrist[:, None]
    stage1_vertices = np.asarray(
        stage1["refined_hand_vertices_camera"][stage1_indices], dtype=np.float32
    )
    stage2_vertices = (
        np.asarray(
            stage2["refined_hand_vertices_camera"][stage2_indices],
            dtype=np.float32,
        )
        if stage2 is not None and stage2_indices is not None else None
    )
    stage2_filtered_mask = (
        np.asarray(stage2["filtered_contact_mask"][stage2_indices]).astype(bool)
        if (
            stage2 is not None
            and stage2_indices is not None
            and "filtered_contact_mask" in stage2
        )
        else None
    )
    stage2_filtered_weight = (
        np.asarray(
            stage2["filtered_contact_weight"][stage2_indices],
            dtype=np.float32,
        )
        if (
            stage2 is not None
            and stage2_indices is not None
            and "filtered_contact_weight" in stage2
        )
        else None
    )
    stage2_clearance_weight = (
        np.asarray(
            stage2["clearance_reference_weight"][stage2_indices],
            dtype=np.float32,
        )
        if (
            stage2 is not None
            and stage2_indices is not None
            and "clearance_reference_weight" in stage2
        )
        else None
    )
    stage2_contact_targets = (
        np.asarray(
            stage2["contact_target_point_camera"][stage2_indices],
            dtype=np.float32,
        )
        if (
            stage2 is not None
            and stage2_indices is not None
            and "contact_target_point_camera" in stage2
        )
        else None
    )
    stage2_contact_normals = (
        np.asarray(
            stage2["contact_target_normal_camera"][stage2_indices],
            dtype=np.float32,
        )
        if (
            stage2 is not None
            and stage2_indices is not None
            and "contact_target_normal_camera" in stage2
        )
        else None
    )
    stage2_push_directions = (
        np.asarray(
            stage2[
                "contact_normal_pushout_direction_camera"
            ][stage2_indices],
            dtype=np.float32,
        )
        if (
            stage2 is not None
            and stage2_indices is not None
            and "contact_normal_pushout_direction_camera" in stage2
        )
        else None
    )
    stage2_push_gate = (
        np.asarray(
            stage2["contact_normal_pushout_gate"][stage2_indices],
            dtype=np.float32,
        )
        if (
            stage2 is not None
            and stage2_indices is not None
            and "contact_normal_pushout_gate" in stage2
        )
        else None
    )
    stage2_inside_region_id = (
        np.asarray(
            stage2["refined_inside_object_region_id"][stage2_indices],
            dtype=np.int64,
        )
        if (
            stage2 is not None
            and stage2_indices is not None
            and "refined_inside_object_region_id" in stage2
        )
        else None
    )
    stage2_contact_region_id = (
        np.asarray(stage2["contact_region_id"], dtype=np.int64)
        if stage2 is not None and "contact_region_id" in stage2
        else None
    )
    stage2_contact_region_names = (
        [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in np.asarray(stage2["contact_region_names"])
        ]
        if stage2 is not None and "contact_region_names" in stage2
        else []
    )
    hand_faces = np.asarray(query["mano_faces"], dtype=np.int64)
    probability = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    haco_mask = (
        np.asarray(contact["contact_mask"][contact_indices]).astype(bool)
        if "contact_mask" in contact
        else probability > threshold
    )

    side = str(query["hand_side"].item()).lower()
    gt_vertices = np.asarray(gt[f"{side}_vertices"], dtype=np.float32)[:count]
    gt_faces = np.asarray(gt[f"{side}_faces"], dtype=np.int64)
    gt_valid = np.asarray(gt[f"{side}_valid"]).astype(bool)[:count]

    print("[viser] loading YCB mesh", flush=True)
    object_local, object_faces = load_mesh(
        Path(args.object_mesh).expanduser().resolve(), args.object_scale
    )
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    object_vertices = np.empty(
        (count, len(object_local), 3), dtype=np.float32
    )
    object_frames = (
        [max(0, min(count - 1, int(args.initial_frame)))]
        if args.lightweight_single_frame else range(count)
    )
    for output_index in object_frames:
        supervision_index = supervision_indices[output_index]
        pose = physical_pose(
            supervision["gt_ycb_object_pose"][supervision_index],
            normalized_left,
        )
        object_vertices[output_index] = (
            object_local @ pose[:3, :3].T + pose[:3, 3]
        )

    stage1_inside = np.asarray(
        stage1["refined_object_vertex_inside_capped_mano"][stage1_indices]
    ).astype(bool)
    stage2_inside = (
        np.asarray(
            stage2["refined_object_vertex_inside_capped_mano"][stage2_indices]
        ).astype(bool)
        if stage2 is not None and stage2_indices is not None else None
    )
    stage1_inside_count = stage1_inside.sum(axis=1)
    stage2_inside_count = (
        stage2_inside.sum(axis=1)
        if stage2_inside is not None else np.zeros(count, dtype=np.int32)
    )
    hand_edges = mesh_edges(hand_faces)
    valid = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
    )

    print(f"[viser] starting server on port {args.port}", flush=True)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    initial = max(0, min(count - 1, int(args.initial_frame)))
    frame_slider = None
    play_button = None
    fps_slider = None
    if not args.lightweight_single_frame:
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=count - 1, step=1, initial_value=initial
        )
        play_button = server.gui.add_button("Play / Pause")
        fps_slider = server.gui.add_slider(
            "FPS", min=1, max=30, step=1, initial_value=int(args.fps)
        )
    threshold_slider = server.gui.add_slider(
        "HACO threshold", min=0.0, max=1.0, step=0.01,
        initial_value=threshold,
    )
    point_size = server.gui.add_slider(
        "Point size", min=0.001, max=0.015, step=0.001,
        initial_value=0.005,
    )
    normal_length = server.gui.add_slider(
        "Patch normal length", min=0.002, max=0.040, step=0.002,
        initial_value=0.014,
    )
    controls = {
        "object": server.gui.add_checkbox("GT YCB object", initial_value=True),
        "v14": server.gui.add_checkbox("V14 WiLoR hand", initial_value=False),
        "stage1": server.gui.add_checkbox(
            "Stage1 rigid hand", initial_value=stage2_vertices is None
        ),
        "stage2": server.gui.add_checkbox(
            "Stage2 local hand", initial_value=stage2_vertices is not None
        ),
        "gt": server.gui.add_checkbox("DexYCB GT hand", initial_value=True),
        "stage2_contact": server.gui.add_checkbox(
            "Raw HACO contacts", initial_value=False
        ),
        "filtered_contact": server.gui.add_checkbox(
            "Filtered Stage1 contacts",
            initial_value=not args.lightweight_single_frame,
        ),
        "stage2_filtered_contact": server.gui.add_checkbox(
            "Stage2 optimization contacts",
            initial_value=stage2_filtered_mask is not None,
        ),
        "stage2_contact_targets": server.gui.add_checkbox(
            "Stage2 object contact targets",
            initial_value=stage2_contact_targets is not None,
        ),
        "stage2_region_contacts": server.gui.add_checkbox(
            "Stage2 contacts by HACO region",
            initial_value=stage2_contact_region_id is not None,
        ),
        "stage2_patch_normals": server.gui.add_checkbox(
            "Stage2 object patch normals",
            initial_value=stage2_contact_normals is not None,
        ),
        "stage2_hand_normals": server.gui.add_checkbox(
            "Stage2 HACO hand normals",
            initial_value=stage2_vertices is not None,
        ),
        "stage2_push_directions": server.gui.add_checkbox(
            "Stage2 collision push directions",
            initial_value=stage2_push_directions is not None,
        ),
        "stage2_clearance_topk": server.gui.add_checkbox(
            "Stage2 clearance HACO top-k",
            initial_value=stage2_clearance_weight is not None,
        ),
        "stage2_clearance_directions": server.gui.add_checkbox(
            "Stage2 clearance push directions",
            initial_value=stage2_push_directions is not None,
        ),
        "collision_seeds": server.gui.add_checkbox(
            "Collision seed MANO vertices",
            initial_value=not args.lightweight_single_frame,
        ),
        "gt_contact": server.gui.add_checkbox(
            "HACO vertex indices on GT", initial_value=False
        ),
        "stage1_inside": server.gui.add_checkbox(
            "Stage1 contained YCB vertices",
            initial_value=stage2_inside is None,
        ),
        "stage2_inside": server.gui.add_checkbox(
            "Stage2 contained YCB vertices",
            initial_value=stage2_inside is not None,
        ),
        "stage2_inside_regions": server.gui.add_checkbox(
            "Stage2 contained YCB by region",
            initial_value=stage2_inside_region_id is not None,
        ),
    }
    handles = []
    playing = {"value": False}
    suppress = {"value": False}
    filter_geometry_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    region_colors = {
        "palm": np.asarray([245, 70, 55], dtype=np.uint8),
        "index": np.asarray([35, 190, 255], dtype=np.uint8),
        "middle": np.asarray([255, 145, 35], dtype=np.uint8),
        "pinky": np.asarray([90, 225, 75], dtype=np.uint8),
        "ring": np.asarray([145, 90, 245], dtype=np.uint8),
        "thumb": np.asarray([245, 45, 175], dtype=np.uint8),
    }

    def clear() -> None:
        while handles:
            handles.pop().remove()

    def add_hand(
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[int, int, int],
        opacity: float,
    ) -> None:
        handles.append(server.scene.add_mesh_simple(
            name, vertices=vertices, faces=faces, color=color, opacity=opacity
        ))

    def filter_geometry(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = filter_geometry_cache.get(index)
        if cached is not None:
            return cached
        object_distance, _ = nearest_distances(
            stage1_vertices[index], object_vertices[index]
        )
        contained = object_vertices[index, stage1_inside[index]]
        if len(contained):
            _, seeds = nearest_distances(contained, stage1_vertices[index])
            seeds = np.unique(seeds)
            geodesic = multisource_geodesic(
                stage1_vertices[index], hand_edges, seeds
            )
        else:
            seeds = np.empty(0, dtype=np.int64)
            geodesic = np.full(len(stage1_vertices[index]), np.inf, dtype=np.float32)
        cached = (object_distance, geodesic, seeds)
        filter_geometry_cache[index] = cached
        return cached

    def filtered_contacts(
        index: int, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        object_distance, geodesic, seeds = filter_geometry(index)
        normalized_probability = np.square(np.clip(
            (probability[index] - float(threshold_slider.value))
            / max(1.0 - float(threshold_slider.value), 1e-6),
            0.0,
            1.0,
        ))
        object_gate = np.exp(-np.square(
            object_distance / (args.object_distance_sigma_mm / 1000.0)
        ))
        inside_fraction = float(stage1_inside_count[index]) / max(
            len(object_vertices[index]), 1
        )
        contact_gate = smooth_contact_gate(
            inside_fraction,
            args.inside_low_fraction,
            args.inside_high_fraction,
        )
        collision_priority = 1.0 - contact_gate
        if len(seeds):
            collision_gate = np.exp(-np.square(
                geodesic / (args.collision_geodesic_sigma_mm / 1000.0)
            ))
        else:
            collision_gate = np.ones_like(object_gate)
        region_gate = (
            1.0 - collision_priority
            + collision_priority * (0.05 + 0.95 * collision_gate)
        )
        weight = normalized_probability * object_gate * region_gate
        selected = np.flatnonzero(
            active & (weight >= args.filtered_min_weight)
        )
        if len(selected) > args.filtered_contact_topk:
            order = np.argsort(weight[selected])[-args.filtered_contact_topk:]
            selected = selected[order]
        return selected, weight, seeds, contact_gate

    def show_frame(value: int) -> None:
        index = max(0, min(count - 1, int(value)))
        clear()
        if controls["object"].value:
            handles.append(server.scene.add_mesh_simple(
                "/gt_ycb_object",
                vertices=object_vertices[index],
                faces=object_faces,
                color=(35, 205, 220),
                opacity=0.34,
            ))
        if controls["v14"].value and valid[index]:
            add_hand("/v14_wilor_hand", v14[index], hand_faces, (65, 135, 245), 0.34)
        if controls["stage1"].value and valid[index]:
            add_hand(
                "/stage1_rigid_hand", stage1_vertices[index], hand_faces,
                (245, 155, 35), 0.40,
            )
        if (
            controls["stage2"].value
            and stage2_vertices is not None
            and valid[index]
        ):
            add_hand(
                "/stage2_local_hand", stage2_vertices[index], hand_faces,
                (220, 65, 190), 0.48,
            )
        if controls["gt"].value and gt_valid[index]:
            add_hand(
                "/dexycb_gt_hand", gt_vertices[index], gt_faces,
                (50, 215, 90), 0.38,
            )

        active = (
            haco_mask[index]
            & (probability[index] > float(threshold_slider.value))
        )
        if args.lightweight_single_frame:
            selected = np.empty(0, dtype=np.int64)
            filtered_weight = np.zeros_like(probability[index])
            collision_seeds = np.empty(0, dtype=np.int64)
            adaptive_gate = float("nan")
        else:
            selected, filtered_weight, collision_seeds, adaptive_gate = (
                filtered_contacts(index, active)
            )
        if active.any():
            strength = probability[index, active, None]
            contact_colors = np.concatenate((
                np.full_like(strength, 255.0),
                150.0 * (1.0 - strength),
                30.0 * (1.0 - strength),
            ), axis=1).clip(0, 255).astype(np.uint8)
            if controls["stage2_contact"].value:
                contact_vertices = (
                    stage2_vertices[index]
                    if stage2_vertices is not None
                    else stage1_vertices[index]
                )
                handles.append(server.scene.add_point_cloud(
                    "/refined_haco_contacts",
                    points=contact_vertices[active],
                    colors=contact_colors,
                    point_size=float(point_size.value),
                ))
            if controls["gt_contact"].value and gt_valid[index]:
                handles.append(server.scene.add_point_cloud(
                    "/gt_haco_vertex_indices",
                    points=gt_vertices[index, active],
                    colors=np.tile(
                        np.asarray([[255, 225, 35]], dtype=np.uint8),
                        (int(active.sum()), 1),
                    ),
                    point_size=float(point_size.value),
                ))
        if controls["filtered_contact"].value and len(selected):
            strength = filtered_weight[selected, None]
            filtered_colors = np.concatenate((
                70.0 * (1.0 - strength),
                np.full_like(strength, 255.0),
                210.0 * (1.0 - strength),
            ), axis=1).clip(0, 255).astype(np.uint8)
            handles.append(server.scene.add_point_cloud(
                "/filtered_stage1_contacts",
                points=stage1_vertices[index, selected],
                colors=filtered_colors,
                point_size=float(point_size.value) * 1.25,
            ))
        if (
            controls["stage2_filtered_contact"].value
            and stage2_vertices is not None
            and stage2_filtered_mask is not None
            and stage2_filtered_weight is not None
        ):
            stage2_selected = stage2_filtered_mask[index]
            if stage2_selected.any():
                strength = stage2_filtered_weight[
                    index, stage2_selected, None
                ]
                colors = np.concatenate((
                    np.full_like(strength, 255.0),
                    40.0 + 180.0 * strength,
                    np.full_like(strength, 255.0),
                ), axis=1).clip(0, 255).astype(np.uint8)
                handles.append(server.scene.add_point_cloud(
                    "/stage2_optimization_contacts",
                    points=stage2_vertices[index, stage2_selected],
                    colors=colors,
                    point_size=float(point_size.value) * 1.4,
                ))
                if (
                    controls["stage2_contact_targets"].value
                    and stage2_contact_targets is not None
                ):
                    targets = stage2_contact_targets[index, stage2_selected]
                    sources = stage2_vertices[index, stage2_selected]
                    handles.append(server.scene.add_point_cloud(
                        "/stage2_object_contact_targets",
                        points=targets,
                        colors=np.tile(
                            np.asarray([[40, 255, 80]], dtype=np.uint8),
                            (len(targets), 1),
                        ),
                        point_size=float(point_size.value) * 1.6,
                    ))
                    handles.append(server.scene.add_line_segments(
                        "/stage2_contact_correspondences",
                        points=np.stack((sources, targets), axis=1),
                        colors=np.tile(
                            np.asarray([[[255, 210, 30], [255, 210, 30]]],
                                       dtype=np.uint8),
                            (len(targets), 1, 1),
                        ),
                        line_width=2.0,
                    ))
                if (
                    controls["stage2_region_contacts"].value
                    and stage2_contact_targets is not None
                    and stage2_contact_region_id is not None
                ):
                    current_hand_normals = vertex_normals(
                        stage2_vertices[index], hand_faces
                    )
                    for region_index, region_name in enumerate(
                        stage2_contact_region_names
                    ):
                        region_selected = (
                            stage2_selected
                            & (stage2_contact_region_id == region_index)
                        )
                        if not region_selected.any():
                            continue
                        color = region_colors.get(
                            region_name,
                            np.asarray([230, 210, 40], dtype=np.uint8),
                        )
                        sources = stage2_vertices[index, region_selected]
                        targets = stage2_contact_targets[index, region_selected]
                        point_colors = np.tile(color[None], (len(sources), 1))
                        line_colors = np.tile(
                            color[None, None], (len(sources), 2, 1)
                        )
                        prefix = f"/stage2_regions/{region_name}"
                        handles.append(server.scene.add_point_cloud(
                            f"{prefix}/haco_topk_hand",
                            points=sources,
                            colors=point_colors,
                            point_size=float(point_size.value) * 1.8,
                        ))
                        handles.append(server.scene.add_point_cloud(
                            f"{prefix}/object_patch_targets",
                            points=targets,
                            colors=point_colors,
                            point_size=float(point_size.value) * 2.0,
                        ))
                        handles.append(server.scene.add_line_segments(
                            f"{prefix}/correspondences",
                            points=np.stack((sources, targets), axis=1),
                            colors=line_colors,
                            line_width=3.0,
                        ))
                        if controls["stage2_hand_normals"].value:
                            hand_normals = current_hand_normals[region_selected]
                            hand_endpoints = sources + (
                                hand_normals * float(normal_length.value)
                            )
                            handles.append(server.scene.add_line_segments(
                                f"{prefix}/hand_normals",
                                points=np.stack(
                                    (sources, hand_endpoints), axis=1
                                ),
                                colors=line_colors,
                                line_width=4.0,
                            ))
                            hand_endpoint_color = np.minimum(
                                color.astype(np.int16) + 55, 255
                            ).astype(np.uint8)
                            handles.append(server.scene.add_point_cloud(
                                f"{prefix}/hand_normal_endpoints",
                                points=hand_endpoints,
                                colors=np.tile(
                                    hand_endpoint_color[None],
                                    (len(hand_endpoints), 1),
                                ),
                                point_size=float(point_size.value) * 1.25,
                            ))
                        if (
                            controls["stage2_push_directions"].value
                            and stage2_push_directions is not None
                            and stage2_push_gate is not None
                        ):
                            push_vectors = stage2_push_directions[
                                index, region_selected
                            ]
                            push_gate = (
                                stage2_push_gate[index, region_selected] > 0
                            )
                            push_norm = np.linalg.norm(
                                push_vectors, axis=-1
                            )
                            valid_push = (
                                push_gate
                                & np.isfinite(push_vectors).all(axis=-1)
                                & (push_norm > 1e-6)
                            )
                            if valid_push.any():
                                push_vectors = (
                                    push_vectors[valid_push]
                                    / push_norm[valid_push, None]
                                )
                                push_sources = sources[valid_push]
                                push_endpoints = push_sources + (
                                    push_vectors
                                    * float(normal_length.value)
                                )
                                push_color = np.minimum(
                                    color.astype(np.int16) + 35, 255
                                ).astype(np.uint8)
                                handles.append(server.scene.add_line_segments(
                                    f"{prefix}/collision_push_direction",
                                    points=np.stack(
                                        (push_sources, push_endpoints),
                                        axis=1,
                                    ),
                                    colors=np.tile(
                                        push_color[None, None],
                                        (len(push_sources), 2, 1),
                                    ),
                                    line_width=5.0,
                                ))
                                handles.append(server.scene.add_point_cloud(
                                    f"{prefix}/collision_push_endpoints",
                                    points=push_endpoints,
                                    colors=np.tile(
                                        push_color[None],
                                        (len(push_endpoints), 1),
                                    ),
                                    point_size=float(point_size.value) * 1.5,
                                ))
                        if (
                            controls["stage2_patch_normals"].value
                            and stage2_contact_normals is not None
                        ):
                            normals = stage2_contact_normals[
                                index, region_selected
                            ]
                            normal_norm = np.linalg.norm(
                                normals, axis=-1, keepdims=True
                            )
                            normals = normals / np.maximum(normal_norm, 1e-8)
                            endpoints = targets + (
                                normals * float(normal_length.value)
                            )
                            handles.append(server.scene.add_line_segments(
                                f"{prefix}/patch_normals",
                                points=np.stack((targets, endpoints), axis=1),
                                colors=line_colors,
                                line_width=4.0,
                            ))
                            endpoint_color = np.maximum(
                                color.astype(np.int16) - 80, 0
                            ).astype(np.uint8)
                            handles.append(server.scene.add_point_cloud(
                                f"{prefix}/normal_endpoints",
                                points=endpoints,
                                colors=np.tile(
                                    endpoint_color[None], (len(endpoints), 1)
                                ),
                                point_size=float(point_size.value) * 1.25,
                            ))
        if (
            stage2_vertices is not None
            and stage2_contact_region_id is not None
            and stage2_clearance_weight is not None
            and (
                controls["stage2_clearance_topk"].value
                or controls["stage2_clearance_directions"].value
            )
        ):
            clearance_selected = stage2_clearance_weight[index] > 0
            for region_index, region_name in enumerate(
                stage2_contact_region_names
            ):
                if (
                    args.clearance_regions is not None
                    and region_name not in args.clearance_regions
                ):
                    continue
                region_selected = (
                    clearance_selected
                    & (stage2_contact_region_id == region_index)
                )
                if not region_selected.any():
                    continue
                color = region_colors.get(
                    region_name,
                    np.asarray([230, 210, 40], dtype=np.uint8),
                )
                prefix = f"/stage2_clearance/{region_name}"
                if controls["stage2_clearance_topk"].value:
                    sources = stage2_vertices[index, region_selected]
                    handles.append(server.scene.add_point_cloud(
                        f"{prefix}/haco_topk",
                        points=sources,
                        colors=np.tile(color[None], (len(sources), 1)),
                        point_size=float(point_size.value) * 1.8,
                    ))
                if (
                    controls["stage2_clearance_directions"].value
                    and stage2_push_directions is not None
                    and stage2_push_gate is not None
                ):
                    push_selected = (
                        (stage2_push_gate[index] > 0)
                        & (stage2_contact_region_id == region_index)
                    )
                    if not push_selected.any():
                        continue
                    push_sources = stage2_vertices[index, push_selected]
                    push_vectors = stage2_push_directions[
                        index, push_selected
                    ]
                    push_norm = np.linalg.norm(
                        push_vectors, axis=-1, keepdims=True
                    )
                    valid_push = (
                        np.isfinite(push_vectors).all(axis=-1)
                        & (push_norm[:, 0] > 1e-6)
                    )
                    if not valid_push.any():
                        continue
                    push_sources = push_sources[valid_push]
                    push_vectors = (
                        push_vectors[valid_push]
                        / push_norm[valid_push]
                    )
                    push_endpoints = push_sources + (
                        push_vectors * float(normal_length.value)
                    )
                    line_colors = np.tile(
                        color[None, None], (len(push_sources), 2, 1)
                    )
                    handles.append(server.scene.add_line_segments(
                        f"{prefix}/push_directions",
                        points=np.stack(
                            (push_sources, push_endpoints), axis=1
                        ),
                        colors=line_colors,
                        line_width=5.0,
                    ))
        if controls["collision_seeds"].value and len(collision_seeds):
            handles.append(server.scene.add_point_cloud(
                "/collision_seed_mano_vertices",
                points=stage1_vertices[index, collision_seeds],
                colors=np.tile(
                    np.asarray([[40, 80, 255]], dtype=np.uint8),
                    (len(collision_seeds), 1),
                ),
                point_size=float(point_size.value) * 0.8,
            ))
        if controls["stage1_inside"].value and stage1_inside[index].any():
            selected = object_vertices[index, stage1_inside[index]]
            handles.append(server.scene.add_point_cloud(
                "/stage1_contained_object_vertices",
                points=selected,
                colors=np.tile(
                    np.asarray([[255, 125, 25]], dtype=np.uint8),
                    (len(selected), 1),
                ),
                point_size=float(point_size.value) * 0.8,
            ))
        if (
            controls["stage2_inside"].value
            and stage2_inside is not None
            and stage2_inside[index].any()
        ):
            if (
                controls["stage2_inside_regions"].value
                and stage2_inside_region_id is not None
            ):
                for region_index, region_name in enumerate(
                    stage2_contact_region_names
                ):
                    selected_mask = (
                        stage2_inside[index]
                        & (stage2_inside_region_id[index] == region_index)
                    )
                    if not selected_mask.any():
                        continue
                    selected = object_vertices[index, selected_mask]
                    color = region_colors.get(
                        region_name,
                        np.asarray([230, 210, 40], dtype=np.uint8),
                    )
                    handles.append(server.scene.add_point_cloud(
                        f"/stage2_contained_regions/{region_name}",
                        points=selected,
                        colors=np.tile(
                            color[None], (len(selected), 1)
                        ),
                        point_size=float(point_size.value) * 1.0,
                    ))
            else:
                selected = object_vertices[index, stage2_inside[index]]
                handles.append(server.scene.add_point_cloud(
                    "/stage2_contained_object_vertices",
                    points=selected,
                    colors=np.tile(
                        np.asarray([[255, 25, 25]], dtype=np.uint8),
                        (len(selected), 1),
                    ),
                    point_size=float(point_size.value) * 0.9,
                ))
        print(
            f"frame={frame_id(ids[index])} index={index} "
            f"haco={int(active.sum())} "
            f"filtered={len(selected)} "
            f"stage2_filtered={int(stage2_filtered_mask[index].sum()) if stage2_filtered_mask is not None else 0} "
            f"contact_gate={adaptive_gate:.3f} "
            f"stage1_inside={int(stage1_inside_count[index])} "
            f"stage2_inside={int(stage2_inside_count[index])} "
            f"gt_valid={bool(gt_valid[index])}",
            flush=True,
        )

    if frame_slider is not None:
        @frame_slider.on_update
        def _(_) -> None:
            if not suppress["value"]:
                show_frame(int(frame_slider.value))

    if play_button is not None:
        @play_button.on_click
        def _(_) -> None:
            playing["value"] = not playing["value"]

    def current_frame() -> int:
        return int(frame_slider.value) if frame_slider is not None else initial

    for control in controls.values():
        control.on_update(lambda _: show_frame(current_frame()))
    threshold_slider.on_update(lambda _: show_frame(current_frame()))
    point_size.on_update(lambda _: show_frame(current_frame()))
    normal_length.on_update(lambda _: show_frame(current_frame()))

    def playback() -> None:
        while True:
            if playing["value"] and frame_slider is not None and fps_slider is not None:
                next_frame = (int(frame_slider.value) + 1) % count
                suppress["value"] = True
                frame_slider.value = next_frame
                suppress["value"] = False
                show_frame(next_frame)
                time.sleep(1.0 / max(float(fps_slider.value), 1.0))
            else:
                time.sleep(0.05)

    if not args.lightweight_single_frame:
        threading.Thread(target=playback, daemon=True).start()
    show_frame(initial)
    print(f"Viewer: http://localhost:{args.port}")
    print(
        "Blue=V14, orange=Stage1, magenta=Stage2, green=GT hand, "
        "cyan=GT YCB, warm=raw HACO, bright green=filtered contacts, "
        "magenta points=Stage2 optimization contacts, blue points=collision "
        "seeds, red=contained YCB vertices"
    )
    print(
        "Region colors: palm=red, index=cyan, middle=orange, "
        "pinky=green, ring=violet, thumb=magenta. Dark endpoint marks "
        "the positive object-normal direction."
    )
    print("Press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
