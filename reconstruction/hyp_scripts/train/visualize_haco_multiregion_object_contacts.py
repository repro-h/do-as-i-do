#!/usr/bin/env python3
"""Select and visualize per-region YCB contacts from V14 and HACO."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from refine_v14_haco_sequence_contact_containment import mano_contact_region_ids
from visualize_haco_choir_opposition_candidates import (
    choir_distance,
    colors,
    frame_id,
    geodesic_patch,
    index_for,
    load_intrinsics,
    load_npz,
    physical_pose,
    project,
)


PALETTE = {
    "palm": (255, 205, 40),
    "index": (40, 255, 100),
    "middle": (40, 180, 255),
    "pinky": (190, 90, 255),
    "ring": (255, 130, 40),
    "thumb": (255, 45, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--dense-root")
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=tuple(PALETTE),
        help="Only select and visualize these MANO contact regions",
    )
    parser.add_argument("--minimum-contact-vertices", type=int, default=3)
    parser.add_argument("--haco-components-per-region", type=int, default=1)
    parser.add_argument("--pixel-radius", type=float, default=45.0)
    parser.add_argument("--pixel-soft-topk", type=int, default=8)
    parser.add_argument("--pixel-sigma", type=float, default=12.0)
    parser.add_argument("--candidate-topk", type=int, default=512)
    parser.add_argument("--distance-slack-mm", type=float, default=30.0)
    parser.add_argument("--max-contact-distance-mm", type=float, default=100.0)
    parser.add_argument("--min-facing-cosine", type=float, default=0.15)
    parser.add_argument("--max-normal-dot", type=float, default=0.0)
    parser.add_argument(
        "--surface-side",
        choices=("both", "outer", "inner"),
        default="both",
        help="Optionally retain only radial outer- or inner-facing mesh vertices",
    )
    parser.add_argument("--surface-side-cosine", type=float, default=0.05)
    parser.add_argument("--visible-surface-only", action="store_true")
    parser.add_argument("--visibility-bin-px", type=float, default=4.0)
    parser.add_argument(
        "--visibility-depth-tolerance-mm", type=float, default=8.0
    )
    parser.add_argument("--w-pixel", type=float, default=1.0)
    parser.add_argument("--w-distance", type=float, default=1.0)
    parser.add_argument("--w-facing", type=float, default=20.0)
    parser.add_argument("--w-normal", type=float, default=20.0)
    parser.add_argument("--patch-radius-mm", type=float, default=8.0)
    parser.add_argument("--patch-normal-cosine", type=float, default=0.7)
    parser.add_argument("--out-npz")
    parser.add_argument("--out-json")
    parser.add_argument(
        "--candidate-surface-only",
        action="store_true",
        help="Hide the selected center and geodesic patch in Viser",
    )
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args()


def lighter(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, int(0.55 * value + 115)) for value in rgb)


def strongest_components(
    mask: np.ndarray,
    faces: np.ndarray,
    probability: np.ndarray,
    count: int,
) -> np.ndarray:
    adjacency: list[set[int]] = [set() for _ in range(len(mask))]
    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            adjacency[first].add(second)
            adjacency[second].add(first)
    remaining = set(np.flatnonzero(mask).tolist())
    components: list[np.ndarray] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        stack = [seed]
        while stack:
            vertex = stack.pop()
            neighbors = adjacency[vertex].intersection(remaining)
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
            component.extend(neighbors)
        components.append(np.asarray(component, dtype=np.int64))
    components.sort(
        key=lambda vertices: float(probability[vertices].sum()), reverse=True
    )
    selected = components[: max(1, count)]
    output = np.zeros_like(mask, dtype=bool)
    if selected:
        output[np.concatenate(selected)] = True
    return output


def visible_object_vertices(
    uv: np.ndarray,
    depth: np.ndarray,
    bin_size: float,
    tolerance: float,
) -> np.ndarray:
    valid = np.isfinite(uv).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    bins = np.zeros((len(uv), 2), dtype=np.int64)
    bins[valid] = np.floor(
        uv[valid] / max(bin_size, 1.0)
    ).astype(np.int64)
    minimum_depth: dict[tuple[int, int], float] = {}
    for index in np.flatnonzero(valid):
        key = (int(bins[index, 0]), int(bins[index, 1]))
        minimum_depth[key] = min(minimum_depth.get(key, np.inf), float(depth[index]))
    visible = np.zeros(len(uv), dtype=bool)
    for index in np.flatnonzero(valid):
        x, y = int(bins[index, 0]), int(bins[index, 1])
        local_minimum = minimum_depth.get((x, y), np.inf)
        visible[index] = depth[index] <= local_minimum + tolerance
    return visible


def main() -> None:
    args = parse_args()
    requested = frame_id(args.frame_id)
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())

    trajectory_index = index_for(trajectory["frame_ids"], requested)
    query_index = index_for(query["frame_ids"], requested)
    contact_index = index_for(contact["frame_ids"], requested)
    supervision_index = index_for(supervision["frame_ids"], requested)

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_index], dtype=np.float32
    )
    hand = (
        np.asarray(
            query["vertices_3d_root_relative_original"][query_index],
            dtype=np.float32,
        )
        + wrist[None]
    )
    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    hand_mesh = trimesh.Trimesh(vertices=hand, faces=faces, process=False)
    hand_normals = np.asarray(hand_mesh.vertex_normals, dtype=np.float32)

    probability = np.asarray(
        contact["contact_probability"][contact_index], dtype=np.float32
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )

    mesh = trimesh.load(
        Path(args.object_mesh).expanduser().resolve(), process=False
    )
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    object_vertices_local = np.asarray(mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(mesh.faces, dtype=np.int64)
    object_normals_local = np.asarray(mesh.vertex_normals, dtype=np.float32)
    object_center_local = np.median(object_vertices_local, axis=0)
    object_radial_local = object_vertices_local - object_center_local
    object_radial_local /= np.maximum(
        np.linalg.norm(object_radial_local, axis=-1, keepdims=True), 1e-12
    )
    shell_cosine = np.sum(object_normals_local * object_radial_local, axis=-1)
    pose = physical_pose(
        supervision["gt_ycb_object_pose"][supervision_index],
        bool(np.asarray(supervision.get("normalized_left", False)).item()),
    )
    object_vertices = object_vertices_local @ pose[:3, :3].T + pose[:3, 3]
    object_normals = object_normals_local @ pose[:3, :3].T
    object_normals /= np.maximum(
        np.linalg.norm(object_normals, axis=-1, keepdims=True), 1e-12
    )

    stream_id = str(np.asarray(query["stream_id"]).item())
    intrinsics = load_intrinsics(args, query, stream_id)
    object_uv = project(object_vertices, intrinsics)
    hand_uv = project(hand, intrinsics)
    object_visible = np.ones(len(object_vertices), dtype=bool)
    if args.visible_surface_only:
        object_visible = visible_object_vertices(
            object_uv,
            object_vertices[:, 2],
            args.visibility_bin_px,
            args.visibility_depth_tolerance_mm / 1000.0,
        )

    region_results: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {
        "frame_id": np.asarray(requested),
        "intrinsics": intrinsics.astype(np.float32),
    }
    visual_regions: list[dict[str, object]] = []

    for region_index, region_name in enumerate(region_names):
        if args.regions and region_name not in args.regions:
            continue
        raw_contact_mask = (
            (region_ids == region_index) & (probability >= threshold)
        )
        raw_active_count = int(raw_contact_mask.sum())
        result: dict[str, object] = {
            "region": region_name,
            "raw_contact_vertices": raw_active_count,
            "selected": False,
        }
        if raw_active_count < args.minimum_contact_vertices:
            result["skip_reason"] = "insufficient_haco_vertices"
            region_results.append(result)
            continue
        contact_mask = strongest_components(
            raw_contact_mask,
            faces,
            probability,
            args.haco_components_per_region,
        )
        active_count = int(contact_mask.sum())
        result["component_contact_vertices"] = active_count
        if active_count < args.minimum_contact_vertices:
            result["skip_reason"] = "haco_component_too_small"
            region_results.append(result)
            continue

        region_hand = hand[contact_mask]
        region_uv = hand_uv[contact_mask]

        pixel_distance = choir_distance(
            object_uv,
            region_uv,
            args.pixel_soft_topk,
            args.pixel_sigma,
        )
        pixel_eligible = np.flatnonzero(
            np.isfinite(pixel_distance)
            & (pixel_distance <= args.pixel_radius)
            & object_visible
        )
        if not len(pixel_eligible):
            result["skip_reason"] = "no_2d_candidates"
            result["minimum_pixel_distance"] = float(
                np.nanmin(pixel_distance)
            )
            region_results.append(result)
            continue
        pixel_candidates = pixel_eligible[
            np.argsort(pixel_distance[pixel_eligible])[: args.candidate_topk]
        ]

        candidate_points = object_vertices[pixel_candidates]
        candidate_uv = object_uv[pixel_candidates]
        uv_pairwise = np.linalg.norm(
            candidate_uv[:, None] - region_uv[None], axis=-1
        )
        matched_contact = np.argmin(uv_pairwise, axis=-1)
        matched_hand = region_hand[matched_contact]
        matched_hand_normal = hand_normals[contact_mask][matched_contact]
        contact_distance = np.linalg.norm(
            candidate_points - matched_hand, axis=-1
        )
        to_hand = matched_hand - candidate_points
        to_hand /= np.maximum(
            np.linalg.norm(to_hand, axis=-1, keepdims=True), 1e-12
        )
        facing_cosine = np.sum(
            object_normals[pixel_candidates] * to_hand, axis=-1
        )
        normal_dot = np.sum(
            object_normals[pixel_candidates] * matched_hand_normal, axis=-1
        )
        candidate_shell_cosine = shell_cosine[pixel_candidates]
        if args.surface_side == "outer":
            surface_side_valid = (
                candidate_shell_cosine >= args.surface_side_cosine
            )
        elif args.surface_side == "inner":
            surface_side_valid = (
                candidate_shell_cosine <= -args.surface_side_cosine
            )
        else:
            surface_side_valid = np.ones(len(pixel_candidates), dtype=bool)
        candidate_distance_mm = contact_distance * 1000.0
        minimum_distance_mm = float(candidate_distance_mm.min())
        valid = (
            (candidate_distance_mm <= args.max_contact_distance_mm)
            & (
                candidate_distance_mm
                <= minimum_distance_mm + args.distance_slack_mm
            )
            & (facing_cosine >= args.min_facing_cosine)
            & (normal_dot <= args.max_normal_dot)
            & surface_side_valid
        )

        result.update(
            {
                "pixel_candidates": int(len(pixel_candidates)),
                "minimum_pixel_distance": float(
                    pixel_distance[pixel_candidates].min()
                ),
                "minimum_contact_distance_mm": minimum_distance_mm,
                "distance_gate_candidates": int(
                    (
                        (candidate_distance_mm <= args.max_contact_distance_mm)
                        & (
                            candidate_distance_mm
                            <= minimum_distance_mm + args.distance_slack_mm
                        )
                    ).sum()
                ),
                "facing_gate_candidates": int(
                    (facing_cosine >= args.min_facing_cosine).sum()
                ),
                "normal_gate_candidates": int(
                    (normal_dot <= args.max_normal_dot).sum()
                ),
                "outer_surface_candidates": int(
                    (candidate_shell_cosine >= args.surface_side_cosine).sum()
                ),
                "inner_surface_candidates": int(
                    (candidate_shell_cosine <= -args.surface_side_cosine).sum()
                ),
                "valid_candidates": int(valid.sum()),
            }
        )
        if not valid.any():
            result["skip_reason"] = "distance_or_direction_gate"
            result["maximum_facing_cosine"] = float(facing_cosine.max())
            result["minimum_normal_dot"] = float(normal_dot.min())
            region_results.append(result)
            continue

        valid_ids = pixel_candidates[valid]
        valid_pixel = pixel_distance[valid_ids]
        valid_distance_mm = candidate_distance_mm[valid]
        valid_facing = facing_cosine[valid]
        valid_normal_dot = normal_dot[valid]
        valid_shell_cosine = candidate_shell_cosine[valid]
        score = (
            args.w_pixel * valid_pixel
            + args.w_distance * valid_distance_mm
            + args.w_facing * (1.0 - valid_facing)
            + args.w_normal * (1.0 + valid_normal_dot)
        )
        selected_offset = int(np.argmin(score))
        selected_id = int(valid_ids[selected_offset])
        patch_ids = geodesic_patch(
            object_vertices_local,
            object_faces,
            object_normals_local,
            selected_id,
            args.patch_radius_mm / 1000.0,
            args.patch_normal_cosine,
        )

        result.update(
            {
                "selected": True,
                "selected_vertex_id": selected_id,
                "selected_score": float(score[selected_offset]),
                "selected_pixel_distance": float(valid_pixel[selected_offset]),
                "selected_contact_distance_mm": float(
                    valid_distance_mm[selected_offset]
                ),
                "selected_facing_cosine": float(
                    valid_facing[selected_offset]
                ),
                "selected_normal_dot": float(
                    valid_normal_dot[selected_offset]
                ),
                "patch_vertices": int(len(patch_ids)),
            }
        )
        region_results.append(result)

        prefix = region_name
        arrays[f"{prefix}_contact_vertex_ids"] = np.flatnonzero(contact_mask)
        arrays[f"{prefix}_raw_contact_vertex_ids"] = np.flatnonzero(
            raw_contact_mask
        )
        arrays[f"{prefix}_contact_vertices_camera"] = region_hand
        arrays[f"{prefix}_candidate_vertex_ids"] = valid_ids
        arrays[f"{prefix}_candidate_vertices_camera"] = object_vertices[valid_ids]
        arrays[f"{prefix}_candidate_shell_cosine"] = valid_shell_cosine.astype(
            np.float32
        )
        arrays[f"{prefix}_selected_vertex_id"] = np.asarray(
            selected_id, dtype=np.int64
        )
        arrays[f"{prefix}_selected_vertex_camera"] = object_vertices[selected_id]
        arrays[f"{prefix}_patch_vertex_ids"] = patch_ids
        arrays[f"{prefix}_patch_vertices_canonical"] = object_vertices_local[patch_ids]
        arrays[f"{prefix}_patch_normals_canonical"] = object_normals_local[patch_ids]
        visual_regions.append(
            {
                "name": region_name,
                "raw_contact_mask": raw_contact_mask,
                "contact_mask": contact_mask,
                "candidate_ids": valid_ids,
                "candidate_shell_cosine": valid_shell_cosine,
                "selected_id": selected_id,
                "patch_ids": patch_ids,
            }
        )

    selected_names = [
        str(result["region"])
        for result in region_results
        if bool(result["selected"])
    ]
    summary = {
        "method": "v14_haco_multiregion_object_contact_v1",
        "frame_id": requested,
        "requested_regions": list(args.regions or region_names),
        "contact_threshold": threshold,
        "selected_regions": selected_names,
        "constraints": {
            "pixel_radius": args.pixel_radius,
            "distance_slack_mm": args.distance_slack_mm,
            "max_contact_distance_mm": args.max_contact_distance_mm,
            "min_facing_cosine": args.min_facing_cosine,
            "max_normal_dot": args.max_normal_dot,
            "surface_side": args.surface_side,
            "surface_side_cosine": args.surface_side_cosine,
            "visible_surface_only": args.visible_surface_only,
            "visibility_bin_px": args.visibility_bin_px,
            "visibility_depth_tolerance_mm": (
                args.visibility_depth_tolerance_mm
            ),
            "patch_radius_mm": args.patch_radius_mm,
            "patch_normal_cosine": args.patch_normal_cosine,
        },
        "regions": region_results,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out_json:
        output = Path(args.out_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not selected_names:
        raise RuntimeError("No active HACO region produced a valid object patch")
    arrays["selected_region_names"] = np.asarray(selected_names)
    if args.out_npz:
        output = Path(args.out_npz).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)

    server = viser.ViserServer(port=args.port)
    server.scene.add_mesh_simple(
        "/object",
        vertices=object_vertices,
        faces=object_faces,
        color=(170, 180, 195),
        opacity=0.65,
    )
    server.scene.add_mesh_simple(
        "/v14_hand",
        vertices=hand,
        faces=faces,
        color=(80, 175, 245),
        opacity=0.55,
    )
    for visual in visual_regions:
        name = str(visual["name"])
        color = PALETTE[name]
        raw_contact_mask = np.asarray(visual["raw_contact_mask"], dtype=bool)
        contact_mask = np.asarray(visual["contact_mask"], dtype=bool)
        candidate_ids = np.asarray(visual["candidate_ids"], dtype=np.int64)
        candidate_shell_cosine = np.asarray(
            visual["candidate_shell_cosine"], dtype=np.float32
        )
        selected_id = int(visual["selected_id"])
        patch_ids = np.asarray(visual["patch_ids"], dtype=np.int64)
        server.scene.add_point_cloud(
            f"/haco_raw/{name}",
            points=hand[raw_contact_mask],
            colors=colors(int(raw_contact_mask.sum()), lighter(color)),
            point_size=0.002,
        )
        server.scene.add_point_cloud(
            f"/haco/{name}",
            points=hand[contact_mask],
            colors=colors(int(contact_mask.sum()), color),
            point_size=0.004,
        )
        outer = candidate_shell_cosine >= args.surface_side_cosine
        inner = candidate_shell_cosine <= -args.surface_side_cosine
        ambiguous = ~(outer | inner)
        for side, selected, side_color in (
            ("outer", outer, lighter(color)),
            ("inner", inner, (255, 45, 45)),
            ("ambiguous", ambiguous, (150, 150, 150)),
        ):
            if selected.any():
                server.scene.add_point_cloud(
                    f"/candidates_{side}/{name}",
                    points=object_vertices[candidate_ids[selected]],
                    colors=colors(int(selected.sum()), side_color),
                    point_size=0.0025,
                )
        if not args.candidate_surface_only:
            server.scene.add_point_cloud(
                f"/patch/{name}",
                points=object_vertices[patch_ids],
                colors=colors(len(patch_ids), color),
                point_size=0.006,
            )
            server.scene.add_point_cloud(
                f"/selected/{name}",
                points=object_vertices[[selected_id]],
                colors=colors(1, color),
                point_size=0.011,
            )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
