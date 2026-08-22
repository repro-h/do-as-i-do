#!/usr/bin/env python3
"""Post-refine local MANO pose with capped-volume YCB containment constraints."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from audit_capped_mano_ycb_vertices import contains_points_vote
from refine_v14_haco_local_mano_pose import (
    axis_angle_to_matrix,
    geometry_summary,
    load_wilor_mano,
    mano_camera_vertices,
    nearest_geometry,
)
from refine_v14_haco_one_way_chamfer import (
    deterministic_surface_samples,
    distribution,
    load_mesh,
    load_npz,
    physical_pose,
    write_npz,
)
from refine_v14_haco_sequence_chamfer import (
    aligned_indices,
    batched_euler_matrix,
)
from visualize_capped_mano_wrist import cap_faces, directed_boundary_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--local-refinement-npz")
    parser.add_argument("--contact-sequence-npz")
    parser.add_argument("--phase-npz")
    parser.add_argument(
        "--base-mode", choices=("local", "stage1"), default="local"
    )
    parser.add_argument("--containment-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--wilor-root", required=True)
    parser.add_argument("--wilor-checkpoint", required=True)
    parser.add_argument("--wilor-config", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--object-samples", type=int, default=2048)
    parser.add_argument("--correspondence-topk", type=int, default=8)
    parser.add_argument("--collision-margin-mm", type=float, default=0.5)
    parser.add_argument("--contact-target-mm", type=float, default=6.0)
    parser.add_argument("--contact-activation-mm", type=float, default=12.0)
    parser.add_argument("--contact-probability-power", type=float, default=2.0)
    parser.add_argument("--contact-weight-floor", type=float, default=0.05)
    parser.add_argument("--region-balanced-contact", action="store_true")
    parser.add_argument("--contact-region-min-vertices", type=int, default=3)
    parser.add_argument(
        "--contact-point-selection",
        choices=("adaptive_filtered", "stage1_probability"),
        default="adaptive_filtered",
        help=(
            "Select contact vertices using the existing adaptive filter or "
            "the per-region high-probability rule used by fixed-patch Stage1."
        ),
    )
    parser.add_argument(
        "--stage1-contact-vertex-topk",
        type=int,
        default=16,
        help=(
            "With stage1_probability selection, retain this many HACO "
            "vertices per Stage1 fixed-patch region and frame."
        ),
    )
    parser.add_argument("--filter-contact-points", action="store_true")
    parser.add_argument("--filtered-contact-topk", type=int, default=48)
    parser.add_argument("--filtered-component-topk", type=int, default=8)
    parser.add_argument("--filtered-maximum-total", type=int, default=96)
    parser.add_argument("--filtered-min-weight", type=float, default=0.05)
    parser.add_argument(
        "--filtered-keeper-confidence", type=float, default=0.6
    )
    parser.add_argument(
        "--filtered-keeper-distance-mm", type=float, default=8.0
    )
    parser.add_argument("--object-distance-sigma-mm", type=float, default=8.0)
    parser.add_argument(
        "--collision-geodesic-sigma-mm", type=float, default=15.0
    )
    parser.add_argument("--collision-region-floor", type=float, default=0.05)
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-collision", type=float, default=5.0)
    parser.add_argument("--w-object-normal-pushout", type=float, default=0.0)
    parser.add_argument("--w-tangential", type=float, default=2.0)
    parser.add_argument("--w-vertex-anchor", type=float, default=1.0)
    parser.add_argument("--w-pose-anchor", type=float, default=5e-4)
    parser.add_argument("--w-pose-velocity", type=float, default=1e-3)
    parser.add_argument("--w-pose-acceleration", type=float, default=2e-3)
    parser.add_argument("--adaptive-balance", action="store_true")
    parser.add_argument("--adaptive-refresh-steps", type=int, default=10)
    parser.add_argument(
        "--adaptive-inside-low-fraction", type=float, default=0.0025
    )
    parser.add_argument(
        "--adaptive-inside-high-fraction", type=float, default=0.01
    )
    parser.add_argument("--adaptive-contact-floor", type=float, default=0.2)
    parser.add_argument(
        "--adaptive-collision-min-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--adaptive-collision-max-scale", type=float, default=4.0
    )
    parser.add_argument("--adaptive-gate-ema", type=float, default=0.8)
    parser.add_argument(
        "--adaptive-reset-optimizer-on-refresh", action="store_true"
    )
    parser.add_argument("--max-joint-delta-deg", type=float, default=4.0)
    parser.add_argument(
        "--thumb-max-joint-delta-deg",
        type=float,
        help="Optional shared delta limit for the thumb MCP/PIP/DIP joints.",
    )
    parser.add_argument("--thumb-mcp-max-joint-delta-deg", type=float)
    parser.add_argument("--thumb-pip-max-joint-delta-deg", type=float)
    parser.add_argument("--thumb-dip-max-joint-delta-deg", type=float)
    parser.add_argument("--mcp-max-joint-delta-deg", type=float)
    parser.add_argument("--pip-max-joint-delta-deg", type=float)
    parser.add_argument("--dip-max-joint-delta-deg", type=float)
    parser.add_argument(
        "--mcp-regularization-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--pip-regularization-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--dip-regularization-scale", type=float, default=1.0
    )
    parser.add_argument("--thumb-mcp-regularization-scale", type=float)
    parser.add_argument("--thumb-pip-regularization-scale", type=float)
    parser.add_argument("--thumb-dip-regularization-scale", type=float)
    parser.add_argument(
        "--contact-pivot-residual-se3",
        action="store_true",
        help=(
            "Jointly optimize a small camera-space rigid correction whose "
            "rotation pivot is the Stage1-selected HACO contact centroid."
        ),
    )
    parser.add_argument(
        "--optimization-mode",
        choices=(
            "local_pose",
            "joint_and_rigid",
            "rigid_only",
            "rotation_only",
            "alternating_camera_z_pose",
        ),
        default="local_pose",
        help=(
            "Optimize MANO local pose, local pose plus contact-pivot rigid "
            "correction, or only the contact-pivot rigid correction."
        ),
    )
    parser.add_argument(
        "--max-residual-translation-mm", type=float, default=5.0
    )
    parser.add_argument("--alternating-z-steps", type=int, default=50)
    parser.add_argument("--alternating-pose-steps", type=int, default=75)
    parser.add_argument(
        "--max-residual-rotation-deg", type=float, default=4.0
    )
    parser.add_argument(
        "--w-residual-translation-anchor", type=float, default=0.1
    )
    parser.add_argument(
        "--w-residual-rotation-anchor", type=float, default=1e-3
    )
    parser.add_argument(
        "--w-residual-translation-velocity", type=float, default=0.1
    )
    parser.add_argument(
        "--w-residual-rotation-velocity", type=float, default=1e-3
    )
    parser.add_argument(
        "--freeze-contact-correspondences",
        action="store_true",
        help="Keep the initial hand-to-object contact targets during refreshes.",
    )
    parser.add_argument("--reprojection-fx", type=float, default=600.0)
    parser.add_argument("--reprojection-fy", type=float, default=600.0)
    parser.add_argument(
        "--reprojection-tolerance-px", type=float, default=2.0
    )
    parser.add_argument("--w-reprojection", type=float, default=0.0)
    parser.add_argument(
        "--contact-correspondence-mode",
        choices=("nearest", "wrist_ray_first_surface"),
        default="nearest",
    )
    parser.add_argument("--contact-ray-radius-mm", type=float, default=15.0)
    parser.add_argument("--contact-pixel-radius", type=float, default=35.0)
    parser.add_argument("--contact-ray-depth-slack-mm", type=float, default=20.0)
    parser.add_argument("--contact-facing-min-cosine", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--frame-chunk", type=int, default=4)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--roundtrip-max-rmse-mm", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def nearest_object_correspondences(
    hand: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    frame_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = []
    points = []
    normals = []
    for start in range(0, len(hand), frame_chunk):
        end = min(start + frame_chunk, len(hand))
        pairwise = torch.cdist(hand[start:end], object_points[start:end])
        distance, nearest_index = pairwise.min(dim=-1)
        nearest_point = torch.gather(
            object_points[start:end],
            1,
            nearest_index[..., None].expand(-1, -1, 3),
        )
        nearest_normal = torch.gather(
            object_normals[start:end],
            1,
            nearest_index[..., None].expand(-1, -1, 3),
        )
        distances.append(distance)
        points.append(nearest_point)
        normals.append(nearest_normal)
    return torch.cat(distances), torch.cat(points), torch.cat(normals)


@torch.no_grad()
def wrist_ray_first_surface_correspondences(
    hand: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    wrist_origin: torch.Tensor,
    frame_chunk: int,
    fx: float,
    fy: float,
    ray_radius_mm: float,
    pixel_radius: float,
    depth_slack_mm: float,
    facing_min_cosine: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = []
    points = []
    normals = []
    ray_radius = ray_radius_mm / 1000.0
    depth_slack = depth_slack_mm / 1000.0
    for start in range(0, len(hand), frame_chunk):
        end = min(start + frame_chunk, len(hand))
        current_hand = hand[start:end]
        current_object = object_points[start:end]
        current_normals = object_normals[start:end]
        origin = wrist_origin[start:end]

        hand_ray = current_hand - origin[:, None]
        hand_ray_length = hand_ray.norm(dim=-1).clamp_min(1e-8)
        hand_ray_direction = hand_ray / hand_ray_length[..., None]
        object_from_origin = current_object - origin[:, None]
        ray_depth = torch.einsum(
            "bvc,bnc->bvn", hand_ray_direction, object_from_origin
        )
        object_radius_squared = object_from_origin.square().sum(dim=-1)
        perpendicular = torch.sqrt(torch.clamp(
            object_radius_squared[:, None] - ray_depth.square(), min=0.0
        ))

        hand_depth = current_hand[..., 2].clamp_min(1e-4)
        object_depth = current_object[..., 2].clamp_min(1e-4)
        hand_uv = torch.stack((
            fx * current_hand[..., 0] / hand_depth,
            fy * current_hand[..., 1] / hand_depth,
        ), dim=-1)
        object_uv = torch.stack((
            fx * current_object[..., 0] / object_depth,
            fy * current_object[..., 1] / object_depth,
        ), dim=-1)
        pixel_distance = torch.cdist(hand_uv, object_uv)

        toward_wrist = functional.normalize(
            origin[:, None] - current_object, dim=-1
        )
        facing = (
            current_normals * toward_wrist
        ).sum(dim=-1) >= facing_min_cosine
        valid = (
            (ray_depth > 0)
            & (ray_depth <= hand_ray_length[..., None] + depth_slack)
            & (perpendicular <= ray_radius)
            & (pixel_distance <= pixel_radius)
            & facing[:, None]
        )
        score = ray_depth + 2.0 * perpendicular
        score = score.masked_fill(~valid, torch.inf)
        selected = score.argmin(dim=-1)
        has_valid = valid.any(dim=-1)

        nearest_distance = torch.cdist(current_hand, current_object)
        nearest_selected = nearest_distance.argmin(dim=-1)
        selected = torch.where(has_valid, selected, nearest_selected)
        selected_point = torch.gather(
            current_object,
            1,
            selected[..., None].expand(-1, -1, 3),
        )
        selected_normal = torch.gather(
            current_normals,
            1,
            selected[..., None].expand(-1, -1, 3),
        )
        distances.append(torch.linalg.norm(
            current_hand - selected_point, dim=-1
        ))
        points.append(selected_point)
        normals.append(selected_normal)
    return torch.cat(distances), torch.cat(points), torch.cat(normals)


@torch.no_grad()
def closest_face_correspondences(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate closest surface point using nearest face-center candidates."""
    triangles = vertices[faces]
    centers = triangles.mean(dim=1)
    candidate_count = min(topk, len(faces))
    candidate_faces = torch.cdist(points, centers).topk(
        candidate_count, largest=False, dim=-1
    ).indices
    candidate_triangles = triangles[candidate_faces]
    vertex0 = candidate_triangles[:, :, 0]
    edge0 = candidate_triangles[:, :, 1] - vertex0
    edge1 = candidate_triangles[:, :, 2] - vertex0
    offset = points[:, None] - vertex0
    dot00 = (edge0 * edge0).sum(dim=-1)
    dot01 = (edge0 * edge1).sum(dim=-1)
    dot11 = (edge1 * edge1).sum(dim=-1)
    dot20 = (offset * edge0).sum(dim=-1)
    dot21 = (offset * edge1).sum(dim=-1)
    denominator = (dot00 * dot11 - dot01.square()).clamp_min(1e-12)
    weight1 = (dot11 * dot20 - dot01 * dot21) / denominator
    weight2 = (dot00 * dot21 - dot01 * dot20) / denominator
    barycentric = torch.stack(
        (1.0 - weight1 - weight2, weight1, weight2), dim=-1
    ).clamp_min(0.0)
    barycentric = barycentric / barycentric.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    closest = (candidate_triangles * barycentric[..., None]).sum(dim=-2)
    selected = torch.linalg.norm(
        closest - points[:, None], dim=-1
    ).argmin(dim=-1)
    row = torch.arange(len(points), device=points.device)
    return candidate_faces[row, selected], barycentric[row, selected]


def build_object_geometry(
    mesh: object,
    supervision: dict[str, np.ndarray],
    indices: np.ndarray,
    normalized_left: bool,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    canonical_vertex_normals = np.asarray(
        mesh.vertex_normals, dtype=np.float32
    )
    if canonical_vertex_normals.shape != canonical_vertices.shape:
        raise ValueError(
            "Object vertex-normal shape mismatch: "
            f"{canonical_vertex_normals.shape} != {canonical_vertices.shape}"
        )
    if not np.isfinite(canonical_vertex_normals).all():
        raise ValueError("Object vertex normals contain non-finite values")
    sampled, sampled_normals = deterministic_surface_samples(mesh, sample_count)
    object_vertices = np.empty(
        (len(indices), len(canonical_vertices), 3), dtype=np.float32
    )
    object_vertex_normals = np.empty_like(object_vertices)
    object_points = np.empty((len(indices), sample_count, 3), dtype=np.float32)
    object_normals = np.empty_like(object_points)
    for output_index, supervision_index in enumerate(indices):
        pose = physical_pose(
            supervision["gt_ycb_object_pose"][supervision_index],
            normalized_left,
        )
        object_vertices[output_index] = (
            canonical_vertices @ pose[:3, :3].T + pose[:3, 3]
        )
        object_vertex_normals[output_index] = (
            canonical_vertex_normals @ pose[:3, :3].T
        )
        object_points[output_index] = sampled @ pose[:3, :3].T + pose[:3, 3]
        object_normals[output_index] = sampled_normals @ pose[:3, :3].T
    return (
        object_vertices,
        object_vertex_normals,
        object_points,
        object_normals,
    )


@torch.no_grad()
def exact_inside_counts(
    hand_vertices: np.ndarray,
    mano_faces: np.ndarray,
    object_vertices: np.ndarray,
    boundary: np.ndarray,
    device: torch.device,
    point_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    virtual_faces = cap_faces(boundary, hand_vertices.shape[1])
    capped_faces = np.concatenate((mano_faces, virtual_faces), axis=0)
    masks = np.zeros(object_vertices.shape[:2], dtype=bool)
    counts = np.zeros(len(hand_vertices), dtype=np.int32)
    for frame_index in range(len(hand_vertices)):
        center = hand_vertices[frame_index, boundary].mean(axis=0)
        capped = np.concatenate(
            (hand_vertices[frame_index], center[None]), axis=0
        )
        lower = capped.min(axis=0) - 1e-4
        upper = capped.max(axis=0) + 1e-4
        broadphase = np.all(
            (object_vertices[frame_index] >= lower)
            & (object_vertices[frame_index] <= upper),
            axis=-1,
        )
        candidate = np.flatnonzero(broadphase)
        if len(candidate):
            selected, _ = contains_points_vote(
                object_vertices[frame_index, candidate],
                capped,
                capped_faces,
                point_chunk,
                ray_count=1,
                device=device,
            )
            masks[frame_index, candidate] = selected
        counts[frame_index] = int(masks[frame_index].sum())
    return masks, counts


def containment_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, object]:
    return {
        "initial_inside_vertices_per_frame": distribution(before),
        "refined_inside_vertices_per_frame": distribution(after),
        "initial_inside_vertices_total": int(before.sum()),
        "refined_inside_vertices_total": int(after.sum()),
        "frames_with_inside_before": int((before > 0).sum()),
        "frames_with_inside_after": int((after > 0).sum()),
        "improved_frames": int((after < before).sum()),
        "degraded_frames": int((after > before).sum()),
    }


def adaptive_contact_gate(
    inside_count: np.ndarray,
    object_vertex_count: int,
    low_fraction: float,
    high_fraction: float,
) -> np.ndarray:
    if not 0 <= low_fraction < high_fraction:
        raise ValueError(
            "Adaptive inside thresholds must satisfy "
            "0 <= low < high"
        )
    fraction = inside_count.astype(np.float32) / max(object_vertex_count, 1)
    gate = np.clip(
        (high_fraction - fraction) / (high_fraction - low_fraction),
        0.0,
        1.0,
    )
    return gate * gate * (3.0 - 2.0 * gate)


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def multisource_geodesic(
    vertices: np.ndarray,
    edges: np.ndarray,
    seeds: np.ndarray,
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


def collision_face_components(
    face_indices: np.ndarray,
    faces: np.ndarray,
) -> list[np.ndarray]:
    selected_faces = np.unique(face_indices.astype(np.int64))
    if not len(selected_faces):
        return []
    vertex_to_faces: dict[int, list[int]] = {}
    for face_index in selected_faces:
        for vertex in faces[face_index]:
            vertex_to_faces.setdefault(int(vertex), []).append(int(face_index))
    remaining = set(int(index) for index in selected_faces)
    components = []
    while remaining:
        seed = remaining.pop()
        queue = [seed]
        component = [seed]
        while queue:
            current = queue.pop()
            neighbors = set()
            for vertex in faces[current]:
                neighbors.update(vertex_to_faces[int(vertex)])
            for neighbor in neighbors & remaining:
                remaining.remove(neighbor)
                queue.append(neighbor)
                component.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def main() -> None:
    args = parse_args()
    if args.adaptive_refresh_steps <= 0:
        raise ValueError("--adaptive-refresh-steps must be positive")
    if not 0.0 <= args.adaptive_gate_ema < 1.0:
        raise ValueError("--adaptive-gate-ema must be in [0, 1)")
    if not 0.0 <= args.adaptive_contact_floor <= 1.0:
        raise ValueError("--adaptive-contact-floor must be in [0, 1]")
    if args.filtered_contact_topk <= 0:
        raise ValueError("--filtered-contact-topk must be positive")
    if args.stage1_contact_vertex_topk <= 0:
        raise ValueError("--stage1-contact-vertex-topk must be positive")
    if (
        args.contact_point_selection == "stage1_probability"
        and not args.region_balanced_contact
    ):
        raise ValueError(
            "stage1_probability selection requires --region-balanced-contact"
        )
    if args.filtered_component_topk <= 0:
        raise ValueError("--filtered-component-topk must be positive")
    if args.filtered_maximum_total < args.filtered_contact_topk:
        raise ValueError(
            "--filtered-maximum-total must be at least the global top-k"
        )
    if not 0.0 <= args.filtered_keeper_confidence <= 1.0:
        raise ValueError("--filtered-keeper-confidence must be in [0, 1]")
    if args.filtered_keeper_distance_mm <= 0:
        raise ValueError("--filtered-keeper-distance-mm must be positive")
    if args.object_distance_sigma_mm <= 0:
        raise ValueError("--object-distance-sigma-mm must be positive")
    if args.collision_geodesic_sigma_mm <= 0:
        raise ValueError("--collision-geodesic-sigma-mm must be positive")
    if not 0.0 <= args.collision_region_floor <= 1.0:
        raise ValueError("--collision-region-floor must be in [0, 1]")
    if (
        args.adaptive_collision_min_scale <= 0
        or args.adaptive_collision_max_scale
        < args.adaptive_collision_min_scale
    ):
        raise ValueError("Invalid adaptive collision scale range")
    if args.w_object_normal_pushout < 0:
        raise ValueError("--w-object-normal-pushout must be non-negative")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if args.max_residual_translation_mm <= 0:
        raise ValueError("--max-residual-translation-mm must be positive")
    if args.max_residual_rotation_deg <= 0:
        raise ValueError("--max-residual-rotation-deg must be positive")
    if args.optimization_mode in (
        "joint_and_rigid",
        "rigid_only",
        "rotation_only",
        "alternating_camera_z_pose",
    ):
        args.contact_pivot_residual_se3 = True
    elif args.contact_pivot_residual_se3:
        args.optimization_mode = "joint_and_rigid"
    residual_weights = [
        args.w_residual_translation_anchor,
        args.w_residual_rotation_anchor,
        args.w_residual_translation_velocity,
        args.w_residual_rotation_velocity,
    ]
    if any(value < 0 for value in residual_weights):
        raise ValueError("Residual SE3 regularization weights must be non-negative")
    if args.reprojection_fx <= 0 or args.reprojection_fy <= 0:
        raise ValueError("Reprojection focal lengths must be positive")
    if args.reprojection_tolerance_px < 0 or args.w_reprojection < 0:
        raise ValueError("Invalid reprojection constraint")
    if (
        args.contact_ray_radius_mm <= 0
        or args.contact_pixel_radius <= 0
        or args.contact_ray_depth_slack_mm < 0
        or not -1.0 <= args.contact_facing_min_cosine <= 1.0
    ):
        raise ValueError("Invalid wrist-ray contact correspondence settings")
    joint_limits = [
        args.mcp_max_joint_delta_deg,
        args.pip_max_joint_delta_deg,
        args.dip_max_joint_delta_deg,
    ]
    if any(value is not None and value <= 0 for value in joint_limits):
        raise ValueError("Joint-group delta limits must be positive")
    if (
        args.thumb_max_joint_delta_deg is not None
        and args.thumb_max_joint_delta_deg <= 0
    ):
        raise ValueError("Thumb joint delta limit must be positive")
    thumb_joint_limits = [
        args.thumb_mcp_max_joint_delta_deg,
        args.thumb_pip_max_joint_delta_deg,
        args.thumb_dip_max_joint_delta_deg,
    ]
    if any(value is not None and value <= 0 for value in thumb_joint_limits):
        raise ValueError("Thumb per-joint delta limits must be positive")
    regularization_scales = [
        args.mcp_regularization_scale,
        args.pip_regularization_scale,
        args.dip_regularization_scale,
    ]
    if any(value <= 0 for value in regularization_scales):
        raise ValueError("Joint-group regularization scales must be positive")
    thumb_regularization_scales = [
        args.thumb_mcp_regularization_scale,
        args.thumb_pip_regularization_scale,
        args.thumb_dip_regularization_scale,
    ]
    if any(
        value is not None and value <= 0
        for value in thumb_regularization_scales
    ):
        raise ValueError("Thumb per-joint regularization scales must be positive")
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
    containment = load_npz(Path(args.containment_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    local = None
    contact_source = None
    phase = None
    if args.base_mode == "local":
        if not args.local_refinement_npz:
            raise ValueError("--base-mode local requires --local-refinement-npz")
        local = load_npz(
            Path(args.local_refinement_npz).expanduser().resolve()
        )
    else:
        if not args.contact_sequence_npz or not args.phase_npz:
            raise ValueError(
                "--base-mode stage1 requires --contact-sequence-npz and "
                "--phase-npz"
            )
        contact_source = load_npz(
            Path(args.contact_sequence_npz).expanduser().resolve()
        )
        phase = load_npz(Path(args.phase_npz).expanduser().resolve())
    ids = np.asarray(query["frame_ids"])
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    stage1_indices = aligned_indices(stage1["frame_ids"], ids)
    containment_indices = aligned_indices(containment["frame_ids"], ids)
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)
    local_indices = None
    contact_indices = None
    phase_indices = None
    if local is not None:
        local_indices = aligned_indices(local["frame_ids"], ids)
    else:
        assert contact_source is not None and phase is not None
        contact_indices = aligned_indices(contact_source["frame_ids"], ids)
        phase_indices = aligned_indices(phase["frame_ids"], ids)
    frame_count = len(ids)

    wrist_np = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    stage1_translation_np = np.asarray(
        stage1["translation_camera"][stage1_indices], dtype=np.float32
    )
    stage1_angles_np = np.asarray(
        stage1["rotation_euler_xyz"][stage1_indices], dtype=np.float32
    )
    global_orient_np = np.asarray(
        query["mano_global_orient_canonical_right"], dtype=np.float32
    )
    betas_np = np.asarray(query["mano_betas"], dtype=np.float32)
    mano_faces_np = np.asarray(query["mano_faces"], dtype=np.int64)
    if local is not None:
        assert local_indices is not None
        base_vertices_np = np.asarray(
            local["refined_hand_vertices_camera"][local_indices],
            dtype=np.float32,
        )
        base_pose_np = np.asarray(
            local["refined_hand_pose_canonical_right"][local_indices],
            dtype=np.float32,
        )
        contact_mask_np = np.asarray(
            local["contact_mask"][local_indices]
        ).astype(bool)
        probability_np = np.asarray(
            local["contact_probability"][local_indices], dtype=np.float32
        )
        contact_gate_np = np.asarray(
            local["contact_gate"][local_indices], dtype=np.float32
        )
        contact_threshold = float(np.asarray(
            local.get("contact_threshold", np.asarray(0.5))
        ).item())
        contact_valid_np = (
            np.asarray(local["contact_valid"][local_indices]).astype(bool)
            if "contact_valid" in local
            else np.ones(frame_count, dtype=bool)
        )
    else:
        assert contact_source is not None and contact_indices is not None
        assert phase is not None and phase_indices is not None
        base_vertices_np = np.asarray(
            stage1["refined_hand_vertices_camera"][stage1_indices],
            dtype=np.float32,
        )
        base_pose_np = np.asarray(
            query["mano_hand_pose_canonical_right"], dtype=np.float32
        )
        contact_mask_np = np.asarray(
            contact_source["contact_mask"][contact_indices]
        ).astype(bool)
        probability_np = np.asarray(
            contact_source["contact_probability"][contact_indices],
            dtype=np.float32,
        )
        contact_gate_np = np.asarray(
            phase["predicted_contact_gate"][phase_indices], dtype=np.float32
        )
        contact_threshold = float(np.asarray(
            contact_source["contact_threshold"]
        ).item())
        contact_valid_np = (
            np.asarray(
                contact_source["contact_valid"][contact_indices]
            ).astype(bool)
            if "contact_valid" in contact_source
            else np.ones(frame_count, dtype=bool)
        )

    finite_probability_np = np.isfinite(probability_np)
    invalid_probability_count = int((~finite_probability_np).sum())
    if invalid_probability_count:
        print(
            "Ignoring "
            f"{invalid_probability_count} non-finite HACO contact probabilities"
        )
    contact_mask_np &= finite_probability_np
    contact_mask_np &= contact_valid_np[:, None]
    probability_np = np.nan_to_num(
        probability_np, nan=0.0, posinf=1.0, neginf=0.0
    )
    finite_contact_gate_np = np.isfinite(contact_gate_np)
    contact_valid_np &= finite_contact_gate_np
    contact_gate_np = np.nan_to_num(
        contact_gate_np, nan=0.0, posinf=0.0, neginf=0.0
    )
    contact_gate_np[~contact_valid_np] = 0.0
    containment_key = next(
        (
            key for key in (
                "object_vertex_inside_capped_mano",
                "refined_object_vertex_inside_capped_mano",
                "initial_object_vertex_inside_capped_mano",
            )
            if key in containment
        ),
        None,
    )
    if containment_key is None:
        raise KeyError(
            "Containment archive lacks an object-inside-MANO mask"
        )
    inside_mask_np = np.asarray(
        containment[containment_key][containment_indices]
    ).astype(bool)
    valid_np = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
    )
    inside_mask_np &= valid_np[:, None]
    inside_count_np = inside_mask_np.sum(axis=1).astype(np.int32)
    optimization_gate_np = valid_np & (
        (contact_gate_np > 0) | (inside_count_np > 0)
    )
    active_indices_np = np.flatnonzero(optimization_gate_np)
    if not len(active_indices_np):
        raise RuntimeError("No contact or containment constraints are active")

    mesh = load_mesh(Path(args.object_mesh).expanduser().resolve(), args.object_scale)
    normalized_left = bool(np.asarray(
        supervision.get("normalized_left", False)
    ).item())
    (
        object_vertices_np,
        object_vertex_normals_np,
        object_points_np,
        object_normals_np,
    ) = build_object_geometry(
        mesh,
        supervision,
        supervision_indices,
        normalized_left,
        args.object_samples,
    )
    if inside_mask_np.shape[1] != object_vertices_np.shape[1]:
        raise ValueError(
            "Containment/object vertex mismatch: "
            f"{inside_mask_np.shape[1]} != {object_vertices_np.shape[1]}"
        )

    device = torch.device(args.device)
    mano = load_wilor_mano(
        Path(args.wilor_root).expanduser().resolve(),
        Path(args.wilor_checkpoint).expanduser().resolve(),
        Path(args.wilor_config).expanduser().resolve(),
        Path(args.mano_data_dir).expanduser().resolve(),
        device,
    )
    global_orient = torch.from_numpy(global_orient_np).to(device)
    base_pose = torch.from_numpy(base_pose_np).to(device)
    betas = torch.from_numpy(betas_np).to(device)
    wrist = torch.from_numpy(wrist_np).to(device)
    stage1_translation = torch.from_numpy(stage1_translation_np).to(device)
    stage1_rotation = batched_euler_matrix(
        torch.from_numpy(stage1_angles_np).to(device)
    )
    object_points = torch.from_numpy(object_points_np).to(device)
    object_normals = torch.from_numpy(object_normals_np).to(device)
    mano_faces = torch.from_numpy(mano_faces_np).to(device)
    contact_gate = torch.from_numpy(contact_gate_np).to(device)
    contact_mask = torch.from_numpy(contact_mask_np).to(device)
    probability = torch.from_numpy(probability_np).to(device)
    optimization_gate = torch.from_numpy(optimization_gate_np).to(device)
    mirror_left = str(query["hand_side"].item()).lower() == "left"

    with torch.no_grad():
        reconstructed_parts = []
        for start in range(0, frame_count, args.frame_chunk):
            end = min(start + args.frame_chunk, frame_count)
            reconstructed_parts.append(mano_camera_vertices(
                mano,
                global_orient[start:end],
                base_pose[start:end],
                betas[start:end],
                wrist[start:end],
                stage1_translation[start:end],
                stage1_rotation[start:end],
                mirror_left,
            ))
        reconstructed = torch.cat(reconstructed_parts)
        roundtrip_rmse = torch.sqrt(
            ((reconstructed - torch.from_numpy(base_vertices_np).to(device)) * 1000.0)
            .square()
            .sum(dim=-1)
            .mean(dim=-1)
        )
    if float(roundtrip_rmse.max().cpu()) > args.roundtrip_max_rmse_mm:
        raise RuntimeError(
            f"{args.base_mode} MANO roundtrip exceeded threshold: "
            f"{float(roundtrip_rmse.max().cpu()):.6f} mm"
        )

    wrist_origin = wrist + stage1_translation

    def contact_correspondences(
        hand_vertices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if args.contact_correspondence_mode == "wrist_ray_first_surface":
            return wrist_ray_first_surface_correspondences(
                hand_vertices,
                object_points,
                object_normals,
                wrist_origin,
                args.frame_chunk,
                args.reprojection_fx,
                args.reprojection_fy,
                args.contact_ray_radius_mm,
                args.contact_pixel_radius,
                args.contact_ray_depth_slack_mm,
                args.contact_facing_min_cosine,
            )
        return nearest_object_correspondences(
            hand_vertices, object_points, object_normals, args.frame_chunk
        )

    initial_distance, fixed_contact_point, fixed_contact_normal = (
        contact_correspondences(reconstructed)
    )
    initial_normal_inside = (
        (fixed_contact_point - reconstructed) * fixed_contact_normal
    ).sum(dim=-1)
    normalized_confidence = torch.clamp(
        (probability - contact_threshold)
        / max(1.0 - contact_threshold, 1e-6),
        min=0.0,
        max=1.0,
    )
    confidence = normalized_confidence.pow(args.contact_probability_power)
    probability_contact_weight = (
        args.contact_weight_floor
        + (1.0 - args.contact_weight_floor) * confidence
    ) * contact_mask * (contact_gate[:, None] > 0)
    initial_plausible_contact = (
        contact_mask
        & (initial_distance <= args.contact_activation_mm / 1000.0)
        & (contact_gate[:, None] > 0)
    )
    unfiltered_contact_weight = (
        args.contact_weight_floor
        + (1.0 - args.contact_weight_floor) * confidence
    ) * initial_plausible_contact
    total_active_vertices = max(
        1, int(optimization_gate_np.sum()) * reconstructed.shape[1]
    )

    boundary = directed_boundary_loop(mano_faces_np)

    def build_collision_correspondences(
        hand: torch.Tensor,
        inside_mask: np.ndarray,
    ) -> tuple[
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
    ]:
        points_by_frame: list[torch.Tensor | None] = [None] * frame_count
        normals_by_frame: list[torch.Tensor | None] = [None] * frame_count
        faces_by_frame: list[torch.Tensor | None] = [None] * frame_count
        barycentric_by_frame: list[torch.Tensor | None] = [None] * frame_count
        for index in np.flatnonzero(inside_mask.sum(axis=1) > 0):
            points = torch.from_numpy(
                object_vertices_np[index, inside_mask[index]]
            ).to(device)
            normals = torch.from_numpy(
                object_vertex_normals_np[index, inside_mask[index]]
            ).to(device)
            face_index, barycentric = closest_face_correspondences(
                points,
                hand[index],
                mano_faces,
                args.correspondence_topk,
            )
            points_by_frame[index] = points
            normals_by_frame[index] = normals
            faces_by_frame[index] = face_index
            barycentric_by_frame[index] = barycentric
        return (
            points_by_frame,
            normals_by_frame,
            faces_by_frame,
            barycentric_by_frame,
        )

    with torch.no_grad():
        (
            correspondence_points,
            correspondence_object_normals,
            correspondence_faces,
            correspondence_barycentric,
        ) = build_collision_correspondences(reconstructed, inside_mask_np)
    current_inside_mask_np = inside_mask_np.copy()
    current_inside_count_np = inside_count_np.copy()
    total_collision_points = max(1, int(current_inside_count_np.sum()))

    if args.adaptive_balance:
        adaptive_gate_np = adaptive_contact_gate(
            current_inside_count_np,
            object_vertices_np.shape[1],
            args.adaptive_inside_low_fraction,
            args.adaptive_inside_high_fraction,
        )
    else:
        adaptive_gate_np = np.ones(frame_count, dtype=np.float32)
    adaptive_gate = torch.from_numpy(adaptive_gate_np).to(device)

    def adaptive_scales() -> tuple[torch.Tensor, torch.Tensor]:
        contact_scale = args.adaptive_contact_floor + (
            1.0 - args.adaptive_contact_floor
        ) * adaptive_gate
        collision_scale = args.adaptive_collision_min_scale + (
            args.adaptive_collision_max_scale
            - args.adaptive_collision_min_scale
        ) * (1.0 - adaptive_gate)
        return contact_scale, collision_scale

    frame_contact_scale, frame_collision_scale = adaptive_scales()
    hand_edges_np = mesh_edges(mano_faces_np)

    contact_region_names: list[str] = []
    contact_region_ids_np = np.full(reconstructed.shape[1], -1, dtype=np.int64)
    contact_region_mask = None
    stage1_contact_region_indices: list[int] = []
    stage1_contact_region_names: list[str] = []
    if args.region_balanced_contact:
        if "contact_region_id" not in stage1 or "contact_region_names" not in stage1:
            raise KeyError(
                "--region-balanced-contact requires a region-balanced Stage1 archive"
            )
        contact_region_ids_np = np.asarray(
            stage1["contact_region_id"], dtype=np.int64
        )
        contact_region_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in np.asarray(stage1["contact_region_names"])
        ]
        if contact_region_ids_np.shape != (reconstructed.shape[1],):
            raise ValueError(
                "Stage1 contact_region_id does not match MANO vertex count"
            )
        contact_region_mask = torch.from_numpy(np.stack([
            contact_region_ids_np == index
            for index in range(len(contact_region_names))
        ])).to(device)
        if args.contact_point_selection == "stage1_probability":
            if "fixed_patch_region_names" not in stage1:
                raise KeyError(
                    "stage1_probability selection requires Stage1 "
                    "fixed_patch_region_names"
                )
            fixed_names = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in np.asarray(stage1["fixed_patch_region_names"])
            ]
            unknown = sorted(set(fixed_names).difference(contact_region_names))
            if unknown:
                raise KeyError(f"Unknown Stage1 fixed-patch regions: {unknown}")
            stage1_contact_region_indices = [
                contact_region_names.index(name) for name in fixed_names
            ]
            stage1_contact_region_names = fixed_names

    def build_contact_weights(
        hand: torch.Tensor,
        distance: torch.Tensor,
        collision_faces: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, int, float]:
        if args.contact_point_selection == "stage1_probability":
            assert contact_region_mask is not None
            weights = torch.zeros_like(probability_contact_weight)
            for frame_index in range(frame_count):
                for region_index in stage1_contact_region_indices:
                    candidates = torch.nonzero(
                        contact_mask[frame_index]
                        & contact_region_mask[region_index]
                        & (contact_gate[frame_index] > 0),
                        as_tuple=False,
                    ).flatten()
                    if not len(candidates):
                        continue
                    count = min(args.stage1_contact_vertex_topk, len(candidates))
                    chosen = candidates[torch.topk(
                        probability_contact_weight[frame_index, candidates],
                        count,
                    ).indices]
                    weights[frame_index, chosen] = (
                        probability_contact_weight[frame_index, chosen]
                    )
        elif not args.filter_contact_points:
            weights = unfiltered_contact_weight
        else:
            geodesic_gate_np = np.ones(
                (frame_count, hand.shape[1]), dtype=np.float32
            )
            component_geodesics: list[list[np.ndarray]] = [
                [] for _ in range(frame_count)
            ]
            hand_np = hand.cpu().numpy().astype(np.float32)
            sigma = args.collision_geodesic_sigma_mm / 1000.0
            for frame_index, face_index in enumerate(collision_faces):
                if face_index is None or not len(face_index):
                    continue
                components = collision_face_components(
                    face_index.cpu().numpy(), mano_faces_np
                )
                for component in components:
                    seed_vertices = np.unique(
                        mano_faces_np[component].reshape(-1)
                    )
                    component_geodesics[frame_index].append(
                        multisource_geodesic(
                            hand_np[frame_index],
                            hand_edges_np,
                            seed_vertices,
                        )
                    )
                if component_geodesics[frame_index]:
                    nearest_component = np.min(
                        np.stack(component_geodesics[frame_index]), axis=0
                    )
                    geodesic_gate_np[frame_index] = np.exp(
                        -np.square(nearest_component / sigma)
                    )
            geodesic_gate = torch.from_numpy(geodesic_gate_np).to(device)
            collision_priority = 1.0 - adaptive_gate[:, None]
            region_gate = (
                1.0 - collision_priority
                + collision_priority
                * (
                    args.collision_region_floor
                    + (1.0 - args.collision_region_floor)
                    * geodesic_gate
                )
            )
            object_gate = torch.exp(-torch.square(
                distance / (args.object_distance_sigma_mm / 1000.0)
            ))
            dynamic_plausible = (
                contact_mask
                & (distance <= args.contact_activation_mm / 1000.0)
                & (contact_gate[:, None] > 0)
            )
            score = confidence * object_gate * region_gate * dynamic_plausible
            weights = torch.zeros_like(score)
            for frame_index in range(frame_count):
                selected = torch.zeros(
                    hand.shape[1], dtype=torch.bool, device=device
                )
                selection_score = score[frame_index].clone()

                for geodesic_np in component_geodesics[frame_index]:
                    component_gate = torch.from_numpy(
                        np.exp(-np.square(geodesic_np / sigma))
                    ).to(device)
                    component_score = (
                        confidence[frame_index]
                        * object_gate[frame_index]
                        * (
                            args.collision_region_floor
                            + (1.0 - args.collision_region_floor)
                            * component_gate
                        )
                        * dynamic_plausible[frame_index]
                    )
                    candidates = torch.nonzero(
                        component_score >= args.filtered_min_weight,
                        as_tuple=False,
                    ).flatten()
                    if len(candidates) > args.filtered_component_topk:
                        keep = torch.topk(
                            component_score[candidates],
                            args.filtered_component_topk,
                        ).indices
                        candidates = candidates[keep]
                    selected[candidates] = True
                    selection_score[candidates] = torch.maximum(
                        selection_score[candidates],
                        component_score[candidates],
                    )

                remaining = args.filtered_maximum_total - int(selected.sum())
                keeper_score = confidence[frame_index] * object_gate[frame_index]
                keepers = torch.nonzero(
                    dynamic_plausible[frame_index]
                    & (
                        normalized_confidence[frame_index]
                        >= args.filtered_keeper_confidence
                    )
                    & (
                        distance[frame_index]
                        <= args.filtered_keeper_distance_mm / 1000.0
                    )
                    & ~selected,
                    as_tuple=False,
                ).flatten()
                if remaining > 0 and len(keepers):
                    if len(keepers) > remaining:
                        keep = torch.topk(
                            keeper_score[keepers], remaining
                        ).indices
                        keepers = keepers[keep]
                    selected[keepers] = True
                    selection_score[keepers] = torch.maximum(
                        selection_score[keepers], keeper_score[keepers]
                    )

                remaining = args.filtered_maximum_total - int(selected.sum())
                candidates = torch.nonzero(
                    score[frame_index] >= args.filtered_min_weight,
                    as_tuple=False,
                ).flatten()
                candidates = candidates[~selected[candidates]]
                global_count = min(
                    args.filtered_contact_topk, max(remaining, 0)
                )
                if global_count <= 0:
                    candidates = candidates[:0]
                elif len(candidates) > global_count:
                    keep = torch.topk(
                        score[frame_index, candidates],
                        global_count,
                    ).indices
                    candidates = candidates[keep]
                selected[candidates] = True
                weights[frame_index, selected] = selection_score[selected]
        selected_count = int((weights > 0).sum().cpu())
        weight_sum = weights.sum()
        effective_count = float(
            (weight_sum.square() / weights.square().sum().clamp_min(1e-12))
            .cpu()
        )
        return weights, selected_count, effective_count

    current_contact_distance = initial_distance
    contact_weight, selected_contact_count, contact_effective_count = (
        build_contact_weights(
            reconstructed,
            current_contact_distance,
            correspondence_faces,
        )
    )
    total_contact_weight = contact_weight.sum().clamp_min(1e-6)
    optimize_local_pose = args.optimization_mode not in (
        "rigid_only", "rotation_only"
    )
    optimize_residual_se3 = args.optimization_mode != "local_pose"
    optimize_residual_translation = args.optimization_mode in (
        "joint_and_rigid", "rigid_only", "alternating_camera_z_pose"
    )
    optimize_residual_rotation = args.optimization_mode in (
        "joint_and_rigid", "rigid_only", "rotation_only"
    )
    delta = torch.zeros(
        (frame_count, 15, 3),
        device=device,
        requires_grad=optimize_local_pose,
    )
    pivot_weight = contact_weight.detach()
    pivot_denominator = pivot_weight.sum(dim=-1, keepdim=True)
    fallback_pivot = wrist + stage1_translation
    contact_pivot = torch.where(
        pivot_denominator > 0,
        (
            reconstructed * pivot_weight[..., None]
        ).sum(dim=1) / pivot_denominator.clamp_min(1e-6),
        fallback_pivot,
    ).detach()
    residual_translation = torch.zeros(
        (frame_count, 3),
        device=device,
        requires_grad=optimize_residual_translation,
    )
    residual_rotation = torch.zeros(
        (frame_count, 3),
        device=device,
        requires_grad=optimize_residual_rotation,
    )
    optimized_parameters = [delta] if optimize_local_pose else []
    if optimize_residual_translation:
        optimized_parameters.append(residual_translation)
    if optimize_residual_rotation:
        optimized_parameters.append(residual_rotation)
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.lr)
    alternating_mode = args.optimization_mode == "alternating_camera_z_pose"
    if alternating_mode:
        if args.contact_point_selection != "stage1_probability":
            raise ValueError(
                "alternating_camera_z_pose requires "
                "--contact-point-selection stage1_probability"
            )
        if args.alternating_z_steps <= 0 or args.alternating_pose_steps <= 0:
            raise ValueError("Alternating Z/Pose step counts must be positive")
        if not stage1_contact_region_names:
            raise RuntimeError("No Stage1 fixed contact regions for pose optimization")
    pose_region_joint_slices = {
        "index": slice(0, 3),
        "middle": slice(3, 6),
        "pinky": slice(6, 9),
        "ring": slice(9, 12),
        "thumb": slice(12, 15),
    }
    optimized_joint_mask = torch.ones((1, 15, 1), device=device)
    if alternating_mode:
        optimized_joint_mask.zero_()
        for region_name in stage1_contact_region_names:
            joint_slice = pose_region_joint_slices.get(region_name)
            if joint_slice is not None:
                optimized_joint_mask[:, joint_slice] = 1.0
        if not optimized_joint_mask.any():
            raise RuntimeError(
                "Stage1 contact regions contain no articulated finger joints"
            )
    active_indices = torch.from_numpy(active_indices_np).to(device)
    contact_target = args.contact_target_mm / 1000.0
    collision_margin = args.collision_margin_mm / 1000.0
    group_limit_deg = [
        args.max_joint_delta_deg if value is None else value
        for value in joint_limits
    ]
    max_delta = torch.tensor(
        group_limit_deg * 5, device=device, dtype=delta.dtype
    ).view(1, 15, 1) * (math.pi / 180.0)
    if args.thumb_max_joint_delta_deg is not None:
        max_delta[:, 12:15] = (
            args.thumb_max_joint_delta_deg * math.pi / 180.0
        )
    for offset, value in enumerate(thumb_joint_limits):
        if value is not None:
            max_delta[:, 12 + offset] = value * math.pi / 180.0
    joint_regularization_scale = torch.tensor(
        regularization_scales * 5, device=device, dtype=delta.dtype
    ).view(1, 15, 1)
    for offset, value in enumerate(thumb_regularization_scales):
        if value is not None:
            joint_regularization_scale[:, 12 + offset] = value

    def weighted_joint_mean(value: torch.Tensor) -> torch.Tensor:
        if not value.numel():
            return torch.zeros((), device=device, dtype=delta.dtype)
        weights = joint_regularization_scale.expand(value.shape[0], -1, 3)
        return (value.square() * weights).sum() / weights.sum()
    best_total = float("inf")
    best_delta = torch.zeros_like(delta)
    best_residual_translation = torch.zeros_like(residual_translation)
    best_residual_rotation = torch.zeros_like(residual_rotation)
    history = []

    def apply_residual_se3(
        vertices: torch.Tensor,
        indices: torch.Tensor | slice,
    ) -> torch.Tensor:
        if not optimize_residual_se3:
            return vertices
        gate = optimization_gate[indices, None]
        translation = residual_translation[indices] * gate
        rotation_vector = residual_rotation[indices] * gate
        rotation = axis_angle_to_matrix(rotation_vector)
        pivot = contact_pivot[indices]
        return (
            torch.bmm(
                vertices - pivot[:, None],
                rotation.transpose(1, 2),
            )
            + pivot[:, None]
            + translation[:, None]
        )

    for step in range(1, args.steps + 1):
        alternating_phase = None
        if alternating_mode:
            cycle_steps = args.alternating_z_steps + args.alternating_pose_steps
            alternating_phase = (
                "camera_z"
                if (step - 1) % cycle_steps < args.alternating_z_steps
                else "local_pose"
            )
        if (
            args.adaptive_balance
            and step > 1
            and (step - 1) % args.adaptive_refresh_steps == 0
        ):
            with torch.no_grad():
                current_parts = []
                for frame_start in range(0, frame_count, args.frame_chunk):
                    frame_end = min(
                        frame_start + args.frame_chunk, frame_count
                    )
                    current_delta = delta[frame_start:frame_end] * (
                        optimization_gate[
                            frame_start:frame_end, None, None
                        ]
                    )
                    current_pose = (
                        axis_angle_to_matrix(current_delta)
                        @ base_pose[frame_start:frame_end]
                    )
                    current_vertices = mano_camera_vertices(
                        mano,
                        global_orient[frame_start:frame_end],
                        current_pose,
                        betas[frame_start:frame_end],
                        wrist[frame_start:frame_end],
                        stage1_translation[frame_start:frame_end],
                        stage1_rotation[frame_start:frame_end],
                        mirror_left,
                    )
                    current_parts.append(apply_residual_se3(
                        current_vertices, slice(frame_start, frame_end)
                    ))
                current_hand = torch.cat(current_parts)
                current_inside_mask_np, current_inside_count_np = (
                    exact_inside_counts(
                        current_hand.cpu().numpy().astype(np.float32),
                        mano_faces_np,
                        object_vertices_np,
                        boundary,
                        device,
                        args.point_chunk,
                    )
                )
                (
                    correspondence_points,
                    correspondence_object_normals,
                    correspondence_faces,
                    correspondence_barycentric,
                ) = build_collision_correspondences(
                    current_hand, current_inside_mask_np
                )
                total_collision_points = max(
                    1, int(current_inside_count_np.sum())
                )
                refreshed_gate = torch.from_numpy(adaptive_contact_gate(
                    current_inside_count_np,
                    object_vertices_np.shape[1],
                    args.adaptive_inside_low_fraction,
                    args.adaptive_inside_high_fraction,
                )).to(device)
                adaptive_gate = (
                    args.adaptive_gate_ema * adaptive_gate
                    + (1.0 - args.adaptive_gate_ema) * refreshed_gate
                )
                frame_contact_scale, frame_collision_scale = (
                    adaptive_scales()
                )
                if args.freeze_contact_correspondences:
                    current_contact_distance = torch.linalg.norm(
                        current_hand - fixed_contact_point, dim=-1
                    )
                else:
                    (
                        current_contact_distance,
                        fixed_contact_point,
                        fixed_contact_normal,
                    ) = contact_correspondences(current_hand)
                (
                    contact_weight,
                    selected_contact_count,
                    contact_effective_count,
                ) = build_contact_weights(
                    current_hand,
                    current_contact_distance,
                    correspondence_faces,
                )
                total_contact_weight = contact_weight.sum().clamp_min(1e-6)
                if args.adaptive_reset_optimizer_on_refresh:
                    optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        contact_region_active = None
        total_contact_region_scale = None
        if args.region_balanced_contact:
            contact_region_count = torch.stack([
                ((contact_weight > 0) & contact_region_mask[index][None]).sum(dim=-1)
                for index in range(len(contact_region_names))
            ], dim=-1)
            contact_region_active = (
                (contact_region_count >= args.contact_region_min_vertices)
                & optimization_gate[:, None]
            )
            total_contact_region_scale = (
                contact_region_active * frame_contact_scale[:, None]
            ).sum().clamp_min(1e-6)
        contact_value = 0.0
        collision_value = 0.0
        object_normal_pushout_value = 0.0
        tangential_value = 0.0
        vertex_anchor_value = 0.0
        reprojection_value = 0.0
        for start in range(0, len(active_indices_np), args.frame_chunk):
            indices = active_indices[start:start + args.frame_chunk]
            effective_delta = delta[indices] * optimization_gate[
                indices, None, None
            ]
            refined_pose = axis_angle_to_matrix(effective_delta) @ base_pose[indices]
            refined = mano_camera_vertices(
                mano,
                global_orient[indices],
                refined_pose,
                betas[indices],
                wrist[indices],
                stage1_translation[indices],
                stage1_rotation[indices],
                mirror_left,
            )
            refined = apply_residual_se3(refined, indices)
            fixed_distance = torch.linalg.norm(
                refined - fixed_contact_point[indices], dim=-1
            )
            contact_error = torch.clamp(
                fixed_distance - contact_target, min=0.0
            ).square()
            if args.region_balanced_contact:
                region_weight = (
                    contact_weight[indices, None]
                    * contact_region_mask[None]
                )
                region_denominator = region_weight.sum(dim=-1).clamp_min(1e-6)
                region_error = (
                    contact_error[:, None] * region_weight
                ).sum(dim=-1) / region_denominator
                region_scale = (
                    contact_region_active[indices]
                    * frame_contact_scale[indices, None]
                )
                chunk_contact = (
                    region_error * region_scale
                ).sum() / total_contact_region_scale
            else:
                chunk_contact = (
                    contact_error
                    * contact_weight[indices]
                    * frame_contact_scale[indices, None]
                ).sum() / total_contact_weight
            displacement = refined - reconstructed[indices]
            normal_component = (
                displacement * fixed_contact_normal[indices]
            ).sum(dim=-1, keepdim=True) * fixed_contact_normal[indices]
            tangent = displacement - normal_component
            chunk_tangential = (
                tangent.square().sum(dim=-1)
                * contact_weight[indices]
                * frame_contact_scale[indices, None]
            ).sum() / total_contact_weight
            chunk_vertex_anchor = displacement.square().sum() / total_active_vertices
            initial_depth = reconstructed[indices, :, 2].clamp_min(1e-4)
            refined_depth = refined[:, :, 2].clamp_min(1e-4)
            initial_projection = torch.stack((
                args.reprojection_fx
                * reconstructed[indices, :, 0] / initial_depth,
                args.reprojection_fy
                * reconstructed[indices, :, 1] / initial_depth,
            ), dim=-1)
            refined_projection = torch.stack((
                args.reprojection_fx * refined[:, :, 0] / refined_depth,
                args.reprojection_fy * refined[:, :, 1] / refined_depth,
            ), dim=-1)
            reprojection_error = torch.linalg.norm(
                refined_projection - initial_projection, dim=-1
            )
            chunk_reprojection = torch.clamp(
                reprojection_error - args.reprojection_tolerance_px,
                min=0.0,
            ).square().mean()

            chunk_collision_sum = torch.zeros((), device=device)
            chunk_object_normal_pushout_sum = torch.zeros((), device=device)
            for local_index, global_index_tensor in enumerate(indices):
                global_index = int(global_index_tensor.item())
                points = correspondence_points[global_index]
                point_normals = correspondence_object_normals[global_index]
                face_index = correspondence_faces[global_index]
                barycentric = correspondence_barycentric[global_index]
                if (
                    points is None
                    or point_normals is None
                    or face_index is None
                    or barycentric is None
                ):
                    continue
                selected_faces = mano_faces[face_index]
                triangles = refined[local_index, selected_faces]
                surface = (triangles * barycentric[..., None]).sum(dim=-2)
                normal = functional.normalize(
                    torch.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                        dim=-1,
                    ),
                    dim=-1,
                )
                if mirror_left:
                    normal = -normal
                signed_clearance = ((points - surface) * normal).sum(dim=-1)
                chunk_collision_sum = chunk_collision_sum + torch.clamp(
                    collision_margin - signed_clearance, min=0.0
                ).square().sum() * frame_collision_scale[global_index]
                object_clearance = (
                    (surface - points) * point_normals
                ).sum(dim=-1)
                chunk_object_normal_pushout_sum = (
                    chunk_object_normal_pushout_sum
                    + torch.clamp(
                        collision_margin - object_clearance, min=0.0
                    ).square().sum() * frame_collision_scale[global_index]
                )
            chunk_collision = chunk_collision_sum / total_collision_points
            chunk_object_normal_pushout = (
                chunk_object_normal_pushout_sum / total_collision_points
            )
            chunk_loss = (
                args.w_contact * chunk_contact
                + args.w_collision * chunk_collision
                + args.w_object_normal_pushout
                * chunk_object_normal_pushout
                + args.w_tangential * chunk_tangential
                + args.w_vertex_anchor * chunk_vertex_anchor
                + args.w_reprojection * chunk_reprojection
            )
            chunk_loss.backward()
            contact_value += float(chunk_contact.detach())
            collision_value += float(chunk_collision.detach())
            object_normal_pushout_value += float(
                chunk_object_normal_pushout.detach()
            )
            tangential_value += float(chunk_tangential.detach())
            vertex_anchor_value += float(chunk_vertex_anchor.detach())
            reprojection_value += float(chunk_reprojection.detach())

        effective_delta = delta * optimization_gate[:, None, None]
        active = optimization_gate
        pose_anchor = weighted_joint_mean(effective_delta[active])
        velocity = effective_delta[1:] - effective_delta[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        regularization = (
            args.w_pose_anchor * pose_anchor
            + args.w_pose_velocity * weighted_joint_mean(velocity)
            + args.w_pose_acceleration * weighted_joint_mean(acceleration)
        )
        if optimize_residual_se3:
            residual_translation_effective = (
                residual_translation * optimization_gate[:, None]
            )
            residual_rotation_effective = (
                residual_rotation * optimization_gate[:, None]
            )
            translation_velocity = (
                residual_translation_effective[1:]
                - residual_translation_effective[:-1]
            )
            rotation_velocity = (
                residual_rotation_effective[1:]
                - residual_rotation_effective[:-1]
            )
            regularization = regularization + (
                args.w_residual_translation_anchor
                * residual_translation_effective[active].square().mean()
                + args.w_residual_rotation_anchor
                * residual_rotation_effective[active].square().mean()
                + args.w_residual_translation_velocity
                * translation_velocity.square().mean()
                + args.w_residual_rotation_velocity
                * rotation_velocity.square().mean()
            )
        regularization.backward()
        total_value = (
            args.w_contact * contact_value
            + args.w_collision * collision_value
            + args.w_object_normal_pushout * object_normal_pushout_value
            + args.w_tangential * tangential_value
            + args.w_vertex_anchor * vertex_anchor_value
            + args.w_reprojection * reprojection_value
            + float(regularization.detach())
        )
        if not math.isfinite(total_value):
            raise FloatingPointError(
                f"Non-finite Stage2 loss at optimization step {step}"
            )
        gradients = [
            parameter.grad
            for parameter in optimized_parameters
            if parameter.grad is not None
        ]
        if len(gradients) != len(optimized_parameters) or any(
            not torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise FloatingPointError(
                f"Non-finite Stage2 gradient at optimization step {step}"
            )
        if alternating_mode:
            delta.grad.mul_(optimized_joint_mask)
            residual_translation.grad[:, :2].zero_()
            if alternating_phase == "camera_z":
                delta.grad.zero_()
            else:
                residual_translation.grad.zero_()
        delta_before_step = delta.detach().clone() if alternating_mode else None
        translation_before_step = (
            residual_translation.detach().clone() if alternating_mode else None
        )
        gradient_norm = torch.sqrt(sum(
            gradient.square().sum() for gradient in gradients
        ))
        if gradient_norm > args.max_grad_norm:
            scale = args.max_grad_norm / gradient_norm
            for gradient in gradients:
                gradient.mul_(scale)
        if total_value < best_total:
            best_total = total_value
            best_delta = delta.detach().clone()
            best_residual_translation = residual_translation.detach().clone()
            best_residual_rotation = residual_rotation.detach().clone()
        optimizer.step()
        with torch.no_grad():
            if alternating_mode:
                if alternating_phase == "camera_z":
                    delta.copy_(delta_before_step)
                else:
                    residual_translation.copy_(translation_before_step)
            if not torch.isfinite(delta).all():
                raise FloatingPointError(
                    f"Non-finite Stage2 parameters after optimization step {step}"
                )
            if optimize_local_pose:
                norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                delta.mul_(torch.clamp(max_delta / norm, max=1.0))
                delta.mul_(optimized_joint_mask)
                delta[~optimization_gate] = 0
            translation_limit = args.max_residual_translation_mm / 1000.0
            if alternating_mode:
                residual_translation[:, :2].zero_()
                residual_translation[:, 2].clamp_(
                    -translation_limit, translation_limit
                )
            else:
                translation_norm = residual_translation.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                residual_translation.mul_(torch.clamp(
                    translation_limit / translation_norm, max=1.0
                ))
            rotation_limit = math.radians(args.max_residual_rotation_deg)
            rotation_norm = residual_rotation.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            residual_rotation.mul_(torch.clamp(
                rotation_limit / rotation_norm, max=1.0
            ))
            residual_translation[~optimization_gate] = 0
            residual_rotation[~optimization_gate] = 0
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "alternating_phase": alternating_phase,
                "total": total_value,
                "contact": contact_value,
                "collision": collision_value,
                "object_normal_pushout": object_normal_pushout_value,
                "tangential": tangential_value,
                "vertex_anchor": vertex_anchor_value,
                "reprojection": reprojection_value,
                "regularization": float(regularization.detach()),
                "inside_vertices": int(current_inside_count_np.sum()),
                "adaptive_contact_gate_median": float(
                    adaptive_gate[active].median().cpu()
                ),
                "contact_scale_median": float(
                    frame_contact_scale[active].median().cpu()
                ),
                "collision_scale_median": float(
                    frame_collision_scale[active].median().cpu()
                ),
                "selected_contact_vertices": selected_contact_count,
                "contact_effective_vertices": contact_effective_count,
                "joint_delta_median_deg": float(
                    delta.detach()[active].norm(dim=-1).median().cpu()
                    * 180.0 / math.pi
                ),
                "joint_delta_max_deg": float(
                    delta.detach()[active].norm(dim=-1).max().cpu()
                    * 180.0 / math.pi
                ),
                "residual_translation_median_mm": float(
                    residual_translation.detach()[active]
                    .norm(dim=-1).median().cpu() * 1000.0
                ),
                "residual_rotation_median_deg": float(
                    residual_rotation.detach()[active]
                    .norm(dim=-1).median().cpu() * 180.0 / math.pi
                ),
            }
            history.append(row)
            print(row)

    if args.adaptive_balance:
        best_delta = delta.detach().clone()
        best_residual_translation = residual_translation.detach().clone()
        best_residual_rotation = residual_rotation.detach().clone()

    refined_parts = []
    refined_pose_parts = []
    with torch.no_grad():
        for start in range(0, frame_count, args.frame_chunk):
            end = min(start + args.frame_chunk, frame_count)
            effective_delta = best_delta[start:end] * optimization_gate[
                start:end, None, None
            ]
            refined_pose = axis_angle_to_matrix(effective_delta) @ base_pose[start:end]
            refined_pose_parts.append(refined_pose)
            refined_vertices = mano_camera_vertices(
                mano,
                global_orient[start:end],
                refined_pose,
                betas[start:end],
                wrist[start:end],
                stage1_translation[start:end],
                stage1_rotation[start:end],
                mirror_left,
            )
            if optimize_residual_se3:
                rotation = axis_angle_to_matrix(
                    best_residual_rotation[start:end]
                    * optimization_gate[start:end, None]
                )
                pivot = contact_pivot[start:end]
                refined_vertices = (
                    torch.bmm(
                        refined_vertices - pivot[:, None],
                        rotation.transpose(1, 2),
                    )
                    + pivot[:, None]
                    + best_residual_translation[start:end, None]
                    * optimization_gate[start:end, None, None]
                )
            refined_parts.append(refined_vertices)
        refined = torch.cat(refined_parts)
        refined_pose = torch.cat(refined_pose_parts)
    refined_np = refined.cpu().numpy().astype(np.float32)
    initial_depth_np = np.maximum(base_vertices_np[..., 2], 1e-4)
    refined_depth_np = np.maximum(refined_np[..., 2], 1e-4)
    initial_projection_np = np.stack((
        args.reprojection_fx * base_vertices_np[..., 0] / initial_depth_np,
        args.reprojection_fy * base_vertices_np[..., 1] / initial_depth_np,
    ), axis=-1)
    refined_projection_np = np.stack((
        args.reprojection_fx * refined_np[..., 0] / refined_depth_np,
        args.reprojection_fy * refined_np[..., 1] / refined_depth_np,
    ), axis=-1)
    reprojection_error_px = np.linalg.norm(
        refined_projection_np - initial_projection_np, axis=-1
    ).astype(np.float32)
    refined_distance, refined_normal_inside = nearest_geometry(
        refined, object_points, object_normals, args.frame_chunk
    )
    refined_inside_mask, refined_inside_count = exact_inside_counts(
        refined_np,
        mano_faces_np,
        object_vertices_np,
        boundary,
        device,
        args.point_chunk,
    )
    if args.filter_contact_points:
        with torch.no_grad():
            (
                _,
                _,
                final_collision_faces,
                _,
            ) = build_collision_correspondences(
                refined, refined_inside_mask
            )
            if args.adaptive_balance:
                adaptive_gate = torch.from_numpy(adaptive_contact_gate(
                    refined_inside_count,
                    object_vertices_np.shape[1],
                    args.adaptive_inside_low_fraction,
                    args.adaptive_inside_high_fraction,
                )).to(device)
            (
                contact_weight,
                selected_contact_count,
                contact_effective_count,
            ) = build_contact_weights(
                refined,
                refined_distance,
                final_collision_faces,
            )
    filtered_contact_mask_np = (
        contact_weight > 0
    ).cpu().numpy()
    filtered_contact_weight_np = contact_weight.cpu().numpy().astype(np.float32)
    selected_initial_mm = (
        initial_distance.cpu().numpy()[filtered_contact_mask_np] * 1000.0
    )
    selected_refined_mm = (
        refined_distance.cpu().numpy()[filtered_contact_mask_np] * 1000.0
    )
    filtered_contact_metrics = {
        "selected_vertices": int(filtered_contact_mask_np.sum()),
        "effective_vertices": contact_effective_count,
        "initial_distance_mm": distribution(selected_initial_mm),
        "refined_distance_mm": distribution(selected_refined_mm),
        "refined_within_2mm_fraction": float(
            np.mean(selected_refined_mm <= 2.0)
        ) if len(selected_refined_mm) else None,
        "refined_within_5mm_fraction": float(
            np.mean(selected_refined_mm <= 5.0)
        ) if len(selected_refined_mm) else None,
    }
    contact_region_count_np = np.zeros(
        (frame_count, len(contact_region_names)), dtype=np.int32
    )
    initial_contact_region_distance_np = np.full(
        (frame_count, len(contact_region_names)), np.nan, dtype=np.float32
    )
    refined_contact_region_distance_np = np.full_like(
        initial_contact_region_distance_np, np.nan
    )
    contact_region_summary: dict[str, object] = {}
    if args.region_balanced_contact:
        initial_distance_np = initial_distance.cpu().numpy() * 1000.0
        refined_distance_np = refined_distance.cpu().numpy() * 1000.0
        for region, name in enumerate(contact_region_names):
            region_vertices = contact_region_ids_np == region
            selected = filtered_contact_mask_np & region_vertices[None]
            contact_region_count_np[:, region] = selected.sum(axis=1)
            for frame in range(frame_count):
                if selected[frame].any():
                    initial_contact_region_distance_np[frame, region] = np.median(
                        initial_distance_np[frame, selected[frame]]
                    )
                    refined_contact_region_distance_np[frame, region] = np.median(
                        refined_distance_np[frame, selected[frame]]
                    )
            evaluated = (
                contact_region_count_np[:, region]
                >= args.contact_region_min_vertices
            )
            contact_region_summary[name] = {
                "active_frames": int(evaluated.sum()),
                "initial_distance_mm": distribution(
                    initial_contact_region_distance_np[evaluated, region]
                ),
                "refined_distance_mm": distribution(
                    refined_contact_region_distance_np[evaluated, region]
                ),
            }

    initial_geometry, _ = geometry_summary(
        initial_distance.cpu().numpy(),
        initial_normal_inside.cpu().numpy(),
        contact_mask_np,
        contact_gate_np,
        1.5,
        20.0,
    )
    refined_geometry, _ = geometry_summary(
        refined_distance.cpu().numpy(),
        refined_normal_inside.cpu().numpy(),
        contact_mask_np,
        contact_gate_np,
        1.5,
        20.0,
    )
    collision_metrics = containment_metrics(inside_count_np, refined_inside_count)

    gt_audit = None
    initial_gt_frame = np.full(frame_count, np.nan, dtype=np.float32)
    refined_gt_frame = np.full(frame_count, np.nan, dtype=np.float32)
    if args.gt_hand_npz:
        gt = load_npz(Path(args.gt_hand_npz).expanduser().resolve())
        side = str(query["hand_side"].item()).lower()
        gt_vertices = np.asarray(gt[f"{side}_vertices"], dtype=np.float32)
        gt_valid = np.asarray(gt[f"{side}_valid"]).astype(bool)
        evaluated = valid_np & gt_valid[:frame_count]
        initial_error = np.linalg.norm(
            base_vertices_np[evaluated] - gt_vertices[:frame_count][evaluated],
            axis=-1,
        ) * 1000.0
        refined_error = np.linalg.norm(
            refined_np[evaluated] - gt_vertices[:frame_count][evaluated],
            axis=-1,
        ) * 1000.0
        initial_gt_frame[evaluated] = np.median(initial_error, axis=-1)
        refined_gt_frame[evaluated] = np.median(refined_error, axis=-1)
        gt_audit = {
            "initial_vertex_error_mm": distribution(initial_error),
            "refined_vertex_error_mm": distribution(refined_error),
            "initial_frame_median_mm": distribution(initial_gt_frame[evaluated]),
            "refined_frame_median_mm": distribution(refined_gt_frame[evaluated]),
            "improved_frames": int(
                (refined_gt_frame[evaluated] < initial_gt_frame[evaluated]).sum()
            ),
            "degraded_over_1mm": int(
                (
                    refined_gt_frame[evaluated]
                    - initial_gt_frame[evaluated]
                    > 1.0
                ).sum()
            ),
        }

    effective_delta = best_delta * optimization_gate[:, None, None]
    delta_deg = effective_delta.norm(dim=-1).cpu().numpy() * 180.0 / math.pi
    summary = {
        "method": (
            "object_normal_adaptive_collision_contact_local_mano_pushout_v1"
            if args.adaptive_balance and args.w_object_normal_pushout > 0
            else "filtered_adaptive_collision_contact_local_mano_pushout_v2"
            if args.adaptive_balance and args.filter_contact_points
            else "adaptive_collision_contact_local_mano_pushout_v1"
            if args.adaptive_balance
            else (
                "stage1_constrained_local_mano_containment_pushout_v1"
                if args.base_mode == "stage1"
                else "local_mano_fixed_correspondence_containment_pushout_v1"
            )
        ),
        "base_mode": args.base_mode,
        "optimization_mode": args.optimization_mode,
        "alternating_optimization": {
            "enabled": alternating_mode,
            "camera_z_steps": args.alternating_z_steps,
            "local_pose_steps": args.alternating_pose_steps,
            "pose_regions": stage1_contact_region_names,
            "camera_xy_frozen": alternating_mode,
            "rigid_rotation_frozen": alternating_mode,
        },
        "stream_id": str(query["stream_id"].item()),
        "hand_side": str(query["hand_side"].item()),
        "frames": frame_count,
        "active_frames": int(optimization_gate_np.sum()),
        "collision_points": int(inside_count_np.sum()),
        "containment_key": containment_key,
        "collision_margin_mm": args.collision_margin_mm,
        "weights": {
            "contact": args.w_contact,
            "collision": args.w_collision,
            "object_normal_pushout": args.w_object_normal_pushout,
            "tangential": args.w_tangential,
            "vertex_anchor": args.w_vertex_anchor,
            "pose_anchor": args.w_pose_anchor,
            "pose_velocity": args.w_pose_velocity,
            "pose_acceleration": args.w_pose_acceleration,
            "residual_translation_anchor": (
                args.w_residual_translation_anchor
            ),
            "residual_rotation_anchor": args.w_residual_rotation_anchor,
            "residual_translation_velocity": (
                args.w_residual_translation_velocity
            ),
            "residual_rotation_velocity": (
                args.w_residual_rotation_velocity
            ),
            "reprojection": args.w_reprojection,
        },
        "contact_pivot_residual_se3": {
            "enabled": optimize_residual_se3,
            "max_translation_mm": args.max_residual_translation_mm,
            "max_rotation_deg": args.max_residual_rotation_deg,
            "translation_norm_mm": distribution(
                np.linalg.norm(
                    best_residual_translation.cpu().numpy()[optimization_gate_np],
                    axis=-1,
                ) * 1000.0
            ),
            "rotation_norm_deg": distribution(
                np.linalg.norm(
                    best_residual_rotation.cpu().numpy()[optimization_gate_np],
                    axis=-1,
                ) * 180.0 / math.pi
            ),
            "translation_frozen": not optimize_residual_translation,
            "contact_correspondences_frozen": (
                args.freeze_contact_correspondences
            ),
            "reprojection_fx": args.reprojection_fx,
            "reprojection_fy": args.reprojection_fy,
            "reprojection_tolerance_px": args.reprojection_tolerance_px,
            "reprojection_error_px": distribution(
                reprojection_error_px[optimization_gate_np]
            ),
            "reprojection_frame_median_px": distribution(
                np.median(reprojection_error_px, axis=-1)[optimization_gate_np]
            ),
        },
        "max_joint_delta_deg": args.max_joint_delta_deg,
        "thumb_max_joint_delta_deg": args.thumb_max_joint_delta_deg,
        "thumb_joint_constraints": {
            "max_delta_deg": {
                "mcp": float(max_delta[0, 12, 0].cpu()) * 180.0 / math.pi,
                "pip": float(max_delta[0, 13, 0].cpu()) * 180.0 / math.pi,
                "dip": float(max_delta[0, 14, 0].cpu()) * 180.0 / math.pi,
            },
            "regularization_scale": {
                "mcp": float(joint_regularization_scale[0, 12, 0].cpu()),
                "pip": float(joint_regularization_scale[0, 13, 0].cpu()),
                "dip": float(joint_regularization_scale[0, 14, 0].cpu()),
            },
        },
        "joint_group_constraints": {
            "order": ["mcp", "pip", "dip"],
            "max_delta_deg": {
                "mcp": group_limit_deg[0],
                "pip": group_limit_deg[1],
                "dip": group_limit_deg[2],
            },
            "regularization_scale": {
                "mcp": args.mcp_regularization_scale,
                "pip": args.pip_regularization_scale,
                "dip": args.dip_regularization_scale,
            },
        },
        "contact_filter": {
            "point_selection": args.contact_point_selection,
            "stage1_region_topk": args.stage1_contact_vertex_topk,
            "stage1_regions": [
                contact_region_names[index]
                for index in stage1_contact_region_indices
            ],
            "enabled": args.filter_contact_points,
            "global_topk": args.filtered_contact_topk,
            "component_topk": args.filtered_component_topk,
            "maximum_total": args.filtered_maximum_total,
            "minimum_weight": args.filtered_min_weight,
            "keeper_confidence": args.filtered_keeper_confidence,
            "keeper_distance_mm": args.filtered_keeper_distance_mm,
            "object_distance_sigma_mm": args.object_distance_sigma_mm,
            "collision_geodesic_sigma_mm": (
                args.collision_geodesic_sigma_mm
            ),
            "collision_region_floor": args.collision_region_floor,
            "metrics": filtered_contact_metrics,
        },
        "contact_correspondence": {
            "mode": args.contact_correspondence_mode,
            "frozen": args.freeze_contact_correspondences,
            "ray_radius_mm": args.contact_ray_radius_mm,
            "pixel_radius": args.contact_pixel_radius,
            "ray_depth_slack_mm": args.contact_ray_depth_slack_mm,
            "facing_min_cosine": args.contact_facing_min_cosine,
        },
        "contact_aggregation": (
            "mano_region_balanced"
            if args.region_balanced_contact else "global_vertex"
        ),
        "contact_regions": {
            "names": contact_region_names,
            "minimum_vertices": args.contact_region_min_vertices,
            "metrics": contact_region_summary,
        },
        "adaptive_balance": {
            "enabled": args.adaptive_balance,
            "refresh_steps": args.adaptive_refresh_steps,
            "inside_low_fraction": args.adaptive_inside_low_fraction,
            "inside_high_fraction": args.adaptive_inside_high_fraction,
            "contact_floor": args.adaptive_contact_floor,
            "collision_min_scale": args.adaptive_collision_min_scale,
            "collision_max_scale": args.adaptive_collision_max_scale,
            "gate_ema": args.adaptive_gate_ema,
            "reset_optimizer_on_refresh": (
                args.adaptive_reset_optimizer_on_refresh
            ),
            "final_contact_gate": distribution(
                adaptive_gate.cpu().numpy()[optimization_gate_np]
            ),
        },
        "input_roundtrip_frame_rmse_mm": distribution(
            roundtrip_rmse.cpu().numpy()
        ),
        "initial_geometry": initial_geometry,
        "refined_geometry": refined_geometry,
        "containment": collision_metrics,
        "joint_delta_deg": distribution(delta_deg[optimization_gate_np]),
        "gt_audit": gt_audit,
        "history": history,
        "warning": (
            "Containment and closest-face correspondences are refreshed "
            f"every {args.adaptive_refresh_steps} steps."
            if args.adaptive_balance
            else (
                "Containment active set and closest-face correspondences are "
                f"fixed from the input {args.base_mode} hand for this test."
            )
        ),
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    write_npz(output_path, {
        "frame_ids": ids,
        "initial_hand_vertices_camera": base_vertices_np,
        "refined_hand_vertices_camera": refined_np,
        "mano_faces": mano_faces_np,
        "initial_hand_pose_canonical_right": base_pose_np,
        "refined_hand_pose_canonical_right": refined_pose.cpu().numpy().astype(np.float32),
        "joint_rotation_delta_rotvec": effective_delta.cpu().numpy().astype(np.float32),
        "contact_pivot_camera": contact_pivot.cpu().numpy().astype(np.float32),
        "residual_translation_camera": (
            best_residual_translation.cpu().numpy().astype(np.float32)
        ),
        "residual_rotation_rotvec": (
            best_residual_rotation.cpu().numpy().astype(np.float32)
        ),
        "reprojection_error_px": reprojection_error_px,
        "initial_object_vertex_inside_capped_mano": inside_mask_np,
        "refined_object_vertex_inside_capped_mano": refined_inside_mask,
        "initial_inside_object_vertices": inside_count_np,
        "refined_inside_object_vertices": refined_inside_count,
        "contact_mask": contact_mask_np,
        "contact_probability": probability_np.astype(np.float16),
        "contact_gate": contact_gate_np,
        "adaptive_contact_gate": adaptive_gate.cpu().numpy().astype(np.float32),
        "filtered_contact_mask": filtered_contact_mask_np,
        "filtered_contact_weight": filtered_contact_weight_np.astype(np.float16),
        "contact_target_point_camera": (
            fixed_contact_point.cpu().numpy().astype(np.float32)
        ),
        "contact_target_normal_camera": (
            fixed_contact_normal.cpu().numpy().astype(np.float32)
        ),
        "contact_region_names": np.asarray(contact_region_names),
        "contact_region_id": contact_region_ids_np.astype(np.int16),
        "contact_region_vertex_count": contact_region_count_np,
        "initial_contact_region_distance_median_mm": (
            initial_contact_region_distance_np
        ),
        "refined_contact_region_distance_median_mm": (
            refined_contact_region_distance_np
        ),
        "initial_gt_vertex_error_median_mm": initial_gt_frame,
        "refined_gt_vertex_error_median_mm": refined_gt_frame,
        "stream_id": np.asarray(str(query["stream_id"].item())),
        "method": np.asarray(summary["method"]),
    })
    summary_path = Path(
        args.out_json or output_path.with_suffix(".json")
    ).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
