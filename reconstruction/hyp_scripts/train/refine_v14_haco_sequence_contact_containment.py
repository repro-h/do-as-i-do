#!/usr/bin/env python3
"""Joint rigid Stage1 refinement using HACO contact and MANO containment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from refine_v14_haco_containment_pushout import (
    build_object_geometry,
    closest_face_correspondences,
    containment_metrics,
    exact_inside_counts,
)
from refine_v14_haco_one_way_chamfer import (
    distribution,
    load_mesh,
    load_npz,
    write_npz,
)
from refine_v14_haco_sequence_chamfer import (
    aligned_indices,
    audit_geometry,
    correction_gate_from_contact_distance,
    transform_batch,
)
from visualize_capped_mano_wrist import directed_boundary_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--phase-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--object-samples", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--softmax-sigma-mm", type=float, default=10.0)
    parser.add_argument("--contact-probability-power", type=float, default=2.0)
    parser.add_argument("--contact-weight-floor", type=float, default=0.05)
    parser.add_argument("--contact-target-mm", type=float, default=6.0)
    parser.add_argument("--correction-stop-mm", type=float, default=10.0)
    parser.add_argument("--correction-full-mm", type=float, default=18.0)
    parser.add_argument("--collision-margin-mm", type=float, default=0.5)
    parser.add_argument("--collision-stop-count", type=int, default=10)
    parser.add_argument("--correspondence-topk", type=int, default=8)
    parser.add_argument("--containment-refresh", type=int, default=25)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--frame-chunk", type=int, default=4)
    parser.add_argument("--max-translation-mm", type=float, default=20.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-collision", type=float, default=5.0)
    parser.add_argument("--w-translation-anchor", type=float, default=0.1)
    parser.add_argument("--w-rotation-anchor", type=float, default=0.01)
    parser.add_argument("--w-translation-velocity", type=float, default=1.0)
    parser.add_argument("--w-translation-acceleration", type=float, default=2.0)
    parser.add_argument("--w-rotation-velocity", type=float, default=0.01)
    parser.add_argument("--w-rotation-acceleration", type=float, default=0.02)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--penetration-trust-mm", type=float, default=20.0)
    parser.add_argument("--use-oracle-gate", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def collision_correspondences(
    hand: torch.Tensor,
    object_vertices: np.ndarray,
    inside_mask: np.ndarray,
    faces: torch.Tensor,
    topk: int,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], list[torch.Tensor | None]]:
    points: list[torch.Tensor | None] = [None] * len(hand)
    face_indices: list[torch.Tensor | None] = [None] * len(hand)
    barycentric: list[torch.Tensor | None] = [None] * len(hand)
    for index in np.flatnonzero(inside_mask.any(axis=1)):
        selected = torch.from_numpy(object_vertices[index, inside_mask[index]]).to(
            hand.device
        )
        selected_faces, selected_barycentric = closest_face_correspondences(
            selected, hand[index], faces, topk
        )
        points[index] = selected
        face_indices[index] = selected_faces
        barycentric[index] = selected_barycentric
    return points, face_indices, barycentric


def gt_audit(
    path: str | None,
    query: dict[str, np.ndarray],
    valid: np.ndarray,
    initial: np.ndarray,
    refined: np.ndarray,
) -> tuple[dict[str, object] | None, np.ndarray, np.ndarray]:
    initial_frame = np.full(len(initial), np.nan, dtype=np.float32)
    refined_frame = np.full(len(initial), np.nan, dtype=np.float32)
    if not path:
        return None, initial_frame, refined_frame
    gt = load_npz(Path(path).expanduser().resolve())
    side = str(query["hand_side"].item()).lower()
    vertices = np.asarray(gt[f"{side}_vertices"], dtype=np.float32)[:len(initial)]
    evaluated = valid & np.asarray(gt[f"{side}_valid"]).astype(bool)[:len(initial)]
    initial_error = np.linalg.norm(initial[evaluated] - vertices[evaluated], axis=-1) * 1000.0
    refined_error = np.linalg.norm(refined[evaluated] - vertices[evaluated], axis=-1) * 1000.0
    initial_frame[evaluated] = np.median(initial_error, axis=-1)
    refined_frame[evaluated] = np.median(refined_error, axis=-1)
    return ({
        "initial_vertex_error_mm": distribution(initial_error),
        "refined_vertex_error_mm": distribution(refined_error),
        "initial_frame_median_mm": distribution(initial_frame[evaluated]),
        "refined_frame_median_mm": distribution(refined_frame[evaluated]),
        "improved_frames": int((refined_frame[evaluated] < initial_frame[evaluated]).sum()),
        "degraded_over_1mm": int(
            ((refined_frame[evaluated] - initial_frame[evaluated]) > 1.0).sum()
        ),
    }, initial_frame, refined_frame)


def main() -> None:
    args = parse_args()
    if args.containment_refresh <= 0:
        raise ValueError("--containment-refresh must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    phase = load_npz(Path(args.phase_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    ids = np.asarray(query["frame_ids"])
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    contact_indices = aligned_indices(contact["frame_ids"], ids)
    phase_indices = aligned_indices(phase["frame_ids"], ids)
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)
    frame_count = len(ids)

    wrist_np = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    vertices_np = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    ) + wrist_np[:, None]
    faces_np = np.asarray(query["mano_faces"], dtype=np.int64)
    probability_np = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    contact_mask_np = np.asarray(contact["contact_mask"][contact_indices]).astype(bool)
    gate_key = "oracle_contact_gate" if args.use_oracle_gate else "predicted_contact_gate"
    phase_gate_np = np.asarray(phase[gate_key][phase_indices], dtype=np.float32)
    valid_np = (
        np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
    )
    phase_gate_np *= valid_np.astype(np.float32)

    mesh = load_mesh(Path(args.object_mesh).expanduser().resolve(), args.object_scale)
    normalized_left = bool(np.asarray(supervision.get("normalized_left", False)).item())
    object_vertices_np, object_points_np, object_normals_np = build_object_geometry(
        mesh, supervision, supervision_indices, normalized_left, args.object_samples
    )
    boundary = directed_boundary_loop(faces_np)

    vertices = torch.from_numpy(vertices_np).to(device)
    wrists = torch.from_numpy(wrist_np).to(device)
    object_points = torch.from_numpy(object_points_np).to(device)
    object_normals = torch.from_numpy(object_normals_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    contact_mask = torch.from_numpy(contact_mask_np).to(device)
    probability = torch.from_numpy(probability_np).to(device)
    phase_gate = torch.from_numpy(phase_gate_np).to(device)

    initial_metrics, initial_per_frame = audit_geometry(
        vertices, contact_mask, object_points, object_normals, phase_gate,
        args.penetration_tolerance_mm, args.penetration_trust_mm, args.frame_chunk,
    )
    contact_gate_np = correction_gate_from_contact_distance(
        initial_per_frame["contact_distance_median_mm"], phase_gate_np,
        args.correction_stop_mm, args.correction_full_mm,
    )
    contact_gate = torch.from_numpy(contact_gate_np).to(device)
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    confidence = torch.clamp(
        (probability - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    ).pow(args.contact_probability_power)
    # Keep every phase-active HACO contact as a one-sided safety tether.  The
    # correction gate decides whether contact alone may move a frame, but it
    # must not disable contact preservation when collision push-out is active.
    contact_weight = (
        args.contact_weight_floor + (1.0 - args.contact_weight_floor) * confidence
    ) * contact_mask * phase_gate[:, None]

    translation = torch.zeros((frame_count, 3), device=device, requires_grad=True)
    angles = torch.zeros((frame_count, 3), device=device, requires_grad=True)
    optimizer = torch.optim.Adam((translation, angles), lr=args.lr)
    max_translation = args.max_translation_mm / 1000.0
    max_angle = math.radians(args.max_rotation_deg)
    contact_target = args.contact_target_mm / 1000.0
    collision_margin = args.collision_margin_mm / 1000.0
    sigma = args.softmax_sigma_mm / 1000.0
    topk = min(max(args.topk, 1), args.object_samples)
    mirror_left = str(query["hand_side"].item()).lower() == "left"
    history: list[dict[str, object]] = []
    refresh_history: list[dict[str, object]] = []
    inside_mask = np.zeros((frame_count, len(object_vertices_np[0])), dtype=bool)
    inside_count = np.zeros(frame_count, dtype=np.int32)
    collision_points: list[torch.Tensor | None] = [None] * frame_count
    collision_faces: list[torch.Tensor | None] = [None] * frame_count
    collision_barycentric: list[torch.Tensor | None] = [None] * frame_count

    for step in range(1, args.steps + 1):
        if (step - 1) % args.containment_refresh == 0:
            with torch.no_grad():
                current = transform_batch(vertices, wrists, translation, angles)
            current_np = current.cpu().numpy().astype(np.float32)
            inside_mask, inside_count = exact_inside_counts(
                current_np, faces_np, object_vertices_np, boundary,
                device, args.point_chunk,
            )
            inside_mask &= valid_np[:, None]
            inside_count = inside_mask.sum(axis=1).astype(np.int32)
            collision_points, collision_faces, collision_barycentric = (
                collision_correspondences(
                    current, object_vertices_np, inside_mask, faces,
                    args.correspondence_topk,
                )
            )
            refresh = {
                "step": step,
                "inside_total": int(inside_count.sum()),
                "frames_with_inside": int((inside_count > 0).sum()),
                "collision_active_frames": int(
                    (inside_count > args.collision_stop_count).sum()
                ),
            }
            refresh_history.append(refresh)
            print({"containment_refresh": refresh}, flush=True)

        collision_active_np = valid_np & (inside_count > args.collision_stop_count)
        optimization_active_np = (contact_gate_np > 0) | collision_active_np
        if not optimization_active_np.any():
            raise RuntimeError("Joint contact/containment gate selected no frames")
        active_indices_np = np.flatnonzero(optimization_active_np)
        active_indices = torch.from_numpy(active_indices_np).to(device)
        optimization_active = torch.from_numpy(optimization_active_np).to(device)
        total_contact_weight = contact_weight[optimization_active].sum().clamp_min(1e-6)
        total_collision_points = max(
            1, int(inside_count[collision_active_np].sum())
        )

        optimizer.zero_grad(set_to_none=True)
        contact_value = 0.0
        collision_value = 0.0
        for start in range(0, len(active_indices_np), args.frame_chunk):
            indices = active_indices[start:start + args.frame_chunk]
            refined = transform_batch(
                vertices[indices], wrists[indices], translation[indices], angles[indices]
            )
            pairwise = torch.cdist(refined, object_points[indices])
            nearest = torch.topk(pairwise, topk, dim=-1, largest=False).values
            soft_weight = torch.softmax(
                -nearest.square() / (2.0 * sigma * sigma), dim=-1
            )
            effective_distance = torch.sqrt(
                (soft_weight * nearest.square()).sum(dim=-1) + 1e-12
            )
            contact_error = torch.clamp(
                effective_distance - contact_target, min=0.0
            ).square()
            chunk_contact = (
                contact_error * contact_weight[indices]
            ).sum() / total_contact_weight

            collision_sum = torch.zeros((), device=device)
            for local_index, global_tensor in enumerate(indices):
                global_index = int(global_tensor.item())
                if not collision_active_np[global_index]:
                    continue
                points = collision_points[global_index]
                face_index = collision_faces[global_index]
                barycentric = collision_barycentric[global_index]
                if points is None or face_index is None or barycentric is None:
                    continue
                triangles = refined[local_index, faces[face_index]]
                surface = (triangles * barycentric[..., None]).sum(dim=-2)
                normal = functional.normalize(
                    torch.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0], dim=-1,
                    ), dim=-1,
                )
                if mirror_left:
                    normal = -normal
                signed_clearance = ((points - surface) * normal).sum(dim=-1)
                collision_sum = collision_sum + torch.clamp(
                    collision_margin - signed_clearance, min=0.0
                ).square().sum()
            chunk_collision = collision_sum / total_collision_points
            chunk_loss = (
                args.w_contact * chunk_contact + args.w_collision * chunk_collision
            )
            chunk_loss.backward()
            contact_value += float(chunk_contact.detach())
            collision_value += float(chunk_collision.detach())

        active = optimization_active
        translation_anchor = translation[active].square().sum(dim=-1).mean()
        rotation_anchor = angles[active].square().sum(dim=-1).mean()
        translation_velocity = translation[1:] - translation[:-1]
        rotation_velocity = angles[1:] - angles[:-1]
        translation_acceleration = translation_velocity[1:] - translation_velocity[:-1]
        rotation_acceleration = rotation_velocity[1:] - rotation_velocity[:-1]
        regularization = (
            args.w_translation_anchor * translation_anchor
            + args.w_rotation_anchor * rotation_anchor
            + args.w_translation_velocity * translation_velocity.square().sum(dim=-1).mean()
            + args.w_translation_acceleration
            * translation_acceleration.square().sum(dim=-1).mean()
            + args.w_rotation_velocity * rotation_velocity.square().sum(dim=-1).mean()
            + args.w_rotation_acceleration
            * rotation_acceleration.square().sum(dim=-1).mean()
        )
        regularization.backward()
        optimizer.step()
        with torch.no_grad():
            translation_norm = translation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            angle_norm = angles.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            translation.mul_(torch.clamp(max_translation / translation_norm, max=1.0))
            angles.mul_(torch.clamp(max_angle / angle_norm, max=1.0))
            translation[~torch.from_numpy(valid_np).to(device)] = 0
            angles[~torch.from_numpy(valid_np).to(device)] = 0

        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": (
                    args.w_contact * contact_value
                    + args.w_collision * collision_value
                    + float(regularization.detach())
                ),
                "contact": contact_value,
                "collision": collision_value,
                "regularization": float(regularization.detach()),
                "active_frames": int(active.sum().item()),
                "inside_total_at_refresh": int(inside_count.sum()),
                "translation_median_mm": float(
                    (translation[active].norm(dim=-1) * 1000.0).median().detach()
                ),
                "rotation_median_deg": float(
                    (angles[active].norm(dim=-1) * 180.0 / math.pi).median().detach()
                ),
            }
            history.append(row)
            print(row, flush=True)

    with torch.no_grad():
        refined = transform_batch(vertices, wrists, translation, angles)
    refined_np = refined.cpu().numpy().astype(np.float32)
    final_inside_mask, final_inside_count = exact_inside_counts(
        refined_np, faces_np, object_vertices_np, boundary, device, args.point_chunk
    )
    final_inside_mask &= valid_np[:, None]
    final_inside_count = final_inside_mask.sum(axis=1).astype(np.int32)
    initial_inside_mask, initial_inside_count = exact_inside_counts(
        vertices_np, faces_np, object_vertices_np, boundary, device, args.point_chunk
    )
    initial_inside_mask &= valid_np[:, None]
    initial_inside_count = initial_inside_mask.sum(axis=1).astype(np.int32)
    refined_metrics, refined_per_frame = audit_geometry(
        refined, contact_mask, object_points, object_normals, phase_gate,
        args.penetration_tolerance_mm, args.penetration_trust_mm, args.frame_chunk,
    )
    collision_summary = containment_metrics(initial_inside_count, final_inside_count)
    gt_summary, initial_gt_frame, refined_gt_frame = gt_audit(
        args.gt_hand_npz, query, valid_np, vertices_np, refined_np
    )

    summary = {
        "method": "joint_haco_contact_capped_mano_containment_rigid_stage1_v3",
        "stream_id": str(query["stream_id"].item()),
        "frames": frame_count,
        "gate_source": gate_key,
        "contact_active_frames": int((contact_gate_np > 0).sum()),
        "contact_preservation_frames": int((phase_gate_np > 0).sum()),
        "initial_collision_active_frames": int(
            (initial_inside_count > args.collision_stop_count).sum()
        ),
        "contact": {"initial": initial_metrics, "refined": refined_metrics},
        "containment": collision_summary,
        "translation_norm_mm": distribution(
            np.linalg.norm(translation.detach().cpu().numpy(), axis=-1) * 1000.0
        ),
        "rotation_norm_deg": distribution(
            np.linalg.norm(angles.detach().cpu().numpy(), axis=-1) * 180.0 / math.pi
        ),
        "weights": {
            "contact": args.w_contact,
            "collision": args.w_collision,
            "translation_anchor": args.w_translation_anchor,
            "rotation_anchor": args.w_rotation_anchor,
        },
        "containment_refresh": args.containment_refresh,
        "collision_stop_count": args.collision_stop_count,
        "gt_audit": gt_summary,
        "refresh_history": refresh_history,
        "history": history,
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    write_npz(output_path, {
        "frame_ids": ids,
        "initial_hand_vertices_camera": vertices_np,
        "refined_hand_vertices_camera": refined_np,
        "mano_faces": faces_np,
        "contact_mask": contact_mask_np,
        "contact_probability": probability_np.astype(np.float16),
        "contact_gate": phase_gate_np,
        "contact_correction_gate": contact_gate_np,
        "translation_camera": translation.detach().cpu().numpy().astype(np.float32),
        "rotation_euler_xyz": angles.detach().cpu().numpy().astype(np.float32),
        "initial_contact_distance_median_mm": initial_per_frame["contact_distance_median_mm"],
        "refined_contact_distance_median_mm": refined_per_frame["contact_distance_median_mm"],
        "initial_object_vertex_inside_capped_mano": initial_inside_mask,
        "refined_object_vertex_inside_capped_mano": final_inside_mask,
        "initial_inside_object_vertices": initial_inside_count,
        "refined_inside_object_vertices": final_inside_count,
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
