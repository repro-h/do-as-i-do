#!/usr/bin/env python3
"""Post-refine local MANO pose with capped-volume YCB containment constraints."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--local-refinement-npz", required=True)
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
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-collision", type=float, default=5.0)
    parser.add_argument("--w-pose-anchor", type=float, default=5e-4)
    parser.add_argument("--w-pose-velocity", type=float, default=1e-3)
    parser.add_argument("--w-pose-acceleration", type=float, default=2e-3)
    parser.add_argument("--max-joint-delta-deg", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--frame-chunk", type=int, default=4)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--roundtrip-max-rmse-mm", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    sampled, sampled_normals = deterministic_surface_samples(mesh, sample_count)
    object_vertices = np.empty(
        (len(indices), len(canonical_vertices), 3), dtype=np.float32
    )
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
        object_points[output_index] = sampled @ pose[:3, :3].T + pose[:3, 3]
        object_normals[output_index] = sampled_normals @ pose[:3, :3].T
    return object_vertices, object_points, object_normals


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


def main() -> None:
    args = parse_args()
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
    local = load_npz(Path(args.local_refinement_npz).expanduser().resolve())
    containment = load_npz(Path(args.containment_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    ids = np.asarray(query["frame_ids"])
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    stage1_indices = aligned_indices(stage1["frame_ids"], ids)
    local_indices = aligned_indices(local["frame_ids"], ids)
    containment_indices = aligned_indices(containment["frame_ids"], ids)
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)
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
    base_vertices_np = np.asarray(
        local["refined_hand_vertices_camera"][local_indices], dtype=np.float32
    )
    base_pose_np = np.asarray(
        local["refined_hand_pose_canonical_right"][local_indices], dtype=np.float32
    )
    global_orient_np = np.asarray(
        query["mano_global_orient_canonical_right"], dtype=np.float32
    )
    betas_np = np.asarray(query["mano_betas"], dtype=np.float32)
    mano_faces_np = np.asarray(query["mano_faces"], dtype=np.int64)
    contact_mask_np = np.asarray(local["contact_mask"][local_indices]).astype(bool)
    probability_np = np.asarray(
        local["contact_probability"][local_indices], dtype=np.float32
    )
    contact_gate_np = np.asarray(
        local["contact_gate"][local_indices], dtype=np.float32
    )
    inside_mask_np = np.asarray(
        containment["object_vertex_inside_capped_mano"][containment_indices]
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
    object_vertices_np, object_points_np, object_normals_np = build_object_geometry(
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
            "Local MANO roundtrip exceeded threshold: "
            f"{float(roundtrip_rmse.max().cpu()):.6f} mm"
        )

    initial_distance, initial_normal_inside = nearest_geometry(
        reconstructed, object_points, object_normals, args.frame_chunk
    )
    threshold = float(np.asarray(
        local.get("contact_threshold", np.asarray(0.5))
    ).item())
    confidence = torch.clamp(
        (probability - threshold) / max(1.0 - threshold, 1e-6),
        min=0.0,
        max=1.0,
    ).pow(args.contact_probability_power)
    plausible_contact = (
        contact_mask
        & (initial_distance <= args.contact_activation_mm / 1000.0)
        & (contact_gate[:, None] > 0)
    )
    contact_weight = (
        args.contact_weight_floor
        + (1.0 - args.contact_weight_floor) * confidence
    ) * plausible_contact
    total_contact_weight = contact_weight.sum().clamp_min(1e-6)

    correspondence_points: list[torch.Tensor | None] = [None] * frame_count
    correspondence_faces: list[torch.Tensor | None] = [None] * frame_count
    correspondence_barycentric: list[torch.Tensor | None] = [None] * frame_count
    with torch.no_grad():
        for index in np.flatnonzero(inside_count_np > 0):
            points = torch.from_numpy(
                object_vertices_np[index, inside_mask_np[index]]
            ).to(device)
            face_index, barycentric = closest_face_correspondences(
                points,
                reconstructed[index],
                mano_faces,
                args.correspondence_topk,
            )
            correspondence_points[index] = points
            correspondence_faces[index] = face_index
            correspondence_barycentric[index] = barycentric
    total_collision_points = max(1, int(inside_count_np.sum()))

    delta = torch.zeros(
        (frame_count, 15, 3), device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    active_indices = torch.from_numpy(active_indices_np).to(device)
    contact_target = args.contact_target_mm / 1000.0
    collision_margin = args.collision_margin_mm / 1000.0
    max_delta = math.radians(args.max_joint_delta_deg)
    best_total = float("inf")
    best_delta = torch.zeros_like(delta)
    history = []

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        contact_value = 0.0
        collision_value = 0.0
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
            pairwise = torch.cdist(refined, object_points[indices])
            nearest_distance = pairwise.min(dim=-1).values
            contact_error = torch.clamp(
                nearest_distance - contact_target, min=0.0
            ).square()
            chunk_contact = (
                contact_error * contact_weight[indices]
            ).sum() / total_contact_weight

            chunk_collision_sum = torch.zeros((), device=device)
            for local_index, global_index_tensor in enumerate(indices):
                global_index = int(global_index_tensor.item())
                points = correspondence_points[global_index]
                face_index = correspondence_faces[global_index]
                barycentric = correspondence_barycentric[global_index]
                if points is None or face_index is None or barycentric is None:
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
                ).square().sum()
            chunk_collision = chunk_collision_sum / total_collision_points
            chunk_loss = (
                args.w_contact * chunk_contact
                + args.w_collision * chunk_collision
            )
            chunk_loss.backward()
            contact_value += float(chunk_contact.detach())
            collision_value += float(chunk_collision.detach())

        effective_delta = delta * optimization_gate[:, None, None]
        active = optimization_gate
        pose_anchor = effective_delta[active].square().mean()
        velocity = effective_delta[1:] - effective_delta[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        regularization = (
            args.w_pose_anchor * pose_anchor
            + args.w_pose_velocity * velocity.square().mean()
            + args.w_pose_acceleration * acceleration.square().mean()
        )
        regularization.backward()
        total_value = (
            args.w_contact * contact_value
            + args.w_collision * collision_value
            + float(regularization.detach())
        )
        if total_value < best_total:
            best_total = total_value
            best_delta = delta.detach().clone()
        optimizer.step()
        with torch.no_grad():
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta.mul_(torch.clamp(max_delta / norm, max=1.0))
            delta[~optimization_gate] = 0
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": total_value,
                "contact": contact_value,
                "collision": collision_value,
                "regularization": float(regularization.detach()),
                "joint_delta_median_deg": float(
                    best_delta[active].norm(dim=-1).median().cpu()
                    * 180.0 / math.pi
                ),
                "joint_delta_max_deg": float(
                    best_delta[active].norm(dim=-1).max().cpu()
                    * 180.0 / math.pi
                ),
            }
            history.append(row)
            print(row)

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
            refined_parts.append(mano_camera_vertices(
                mano,
                global_orient[start:end],
                refined_pose,
                betas[start:end],
                wrist[start:end],
                stage1_translation[start:end],
                stage1_rotation[start:end],
                mirror_left,
            ))
        refined = torch.cat(refined_parts)
        refined_pose = torch.cat(refined_pose_parts)
    refined_np = refined.cpu().numpy().astype(np.float32)
    refined_distance, refined_normal_inside = nearest_geometry(
        refined, object_points, object_normals, args.frame_chunk
    )
    boundary = directed_boundary_loop(mano_faces_np)
    refined_inside_mask, refined_inside_count = exact_inside_counts(
        refined_np,
        mano_faces_np,
        object_vertices_np,
        boundary,
        device,
        args.point_chunk,
    )

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
        "method": "local_mano_fixed_correspondence_containment_pushout_v1",
        "stream_id": str(query["stream_id"].item()),
        "hand_side": str(query["hand_side"].item()),
        "frames": frame_count,
        "active_frames": int(optimization_gate_np.sum()),
        "collision_points": int(inside_count_np.sum()),
        "collision_margin_mm": args.collision_margin_mm,
        "initial_geometry": initial_geometry,
        "refined_geometry": refined_geometry,
        "containment": collision_metrics,
        "joint_delta_deg": distribution(delta_deg[optimization_gate_np]),
        "gt_audit": gt_audit,
        "history": history,
        "warning": (
            "Containment active set and closest-face correspondences are fixed "
            "from the input local refinement for this first test."
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
        "initial_object_vertex_inside_capped_mano": inside_mask_np,
        "refined_object_vertex_inside_capped_mano": refined_inside_mask,
        "initial_inside_object_vertices": inside_count_np,
        "refined_inside_object_vertices": refined_inside_count,
        "contact_mask": contact_mask_np,
        "contact_probability": probability_np.astype(np.float16),
        "contact_gate": contact_gate_np,
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
