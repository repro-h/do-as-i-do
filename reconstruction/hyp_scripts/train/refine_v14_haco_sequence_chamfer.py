#!/usr/bin/env python3
"""Phase-gated rigid sequence refinement with all HACO contact vertices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from refine_v14_haco_one_way_chamfer import (
    deterministic_surface_samples,
    distribution,
    load_mesh,
    load_npz,
    physical_pose,
    write_npz,
)


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
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--frame-chunk", type=int, default=4)
    parser.add_argument("--max-translation-mm", type=float, default=20.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
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


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def aligned_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {frame_id(value): index for index, value in enumerate(source)}
    return np.asarray([lookup[frame_id(value)] for value in target], dtype=np.int64)


def batched_euler_matrix(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles.unbind(dim=-1)
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    rx = torch.stack((
        one, zero, zero,
        zero, cx, -sx,
        zero, sx, cx,
    ), dim=-1).reshape(-1, 3, 3)
    ry = torch.stack((
        cy, zero, sy,
        zero, one, zero,
        -sy, zero, cy,
    ), dim=-1).reshape(-1, 3, 3)
    rz = torch.stack((
        cz, -sz, zero,
        sz, cz, zero,
        zero, zero, one,
    ), dim=-1).reshape(-1, 3, 3)
    return rz @ ry @ rx


def transform_batch(
    vertices: torch.Tensor,
    wrists: torch.Tensor,
    translation: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    rotation = batched_euler_matrix(angles)
    centered = vertices - wrists[:, None]
    return torch.bmm(centered, rotation.transpose(1, 2)) + wrists[:, None] + translation[:, None]


@torch.no_grad()
def audit_geometry(
    hand: torch.Tensor,
    contact_mask: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    gate: torch.Tensor,
    tolerance_mm: float,
    trust_mm: float,
    frame_chunk: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    contact_distances = []
    penetration_depths = []
    penetrating_count = np.zeros(len(hand), dtype=np.int32)
    contact_median = np.full(len(hand), np.nan, dtype=np.float32)
    contact_p90 = np.full(len(hand), np.nan, dtype=np.float32)
    for start in range(0, len(hand), frame_chunk):
        end = min(start + frame_chunk, len(hand))
        pairwise = torch.cdist(hand[start:end], object_points[start:end])
        nearest_distance, nearest_index = pairwise.min(dim=-1)
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
        inside = ((nearest_point - hand[start:end]) * nearest_normal).sum(dim=-1)
        trusted = nearest_distance <= trust_mm / 1000.0
        penetrating = (inside > tolerance_mm / 1000.0) & trusted
        depth = torch.clamp(inside - tolerance_mm / 1000.0, min=0.0)
        for local in range(end - start):
            index = start + local
            penetrating_count[index] = int(penetrating[local].sum().item())
            selected_depth = depth[local, penetrating[local]].cpu().numpy() * 1000.0
            if len(selected_depth):
                penetration_depths.append(selected_depth)
            selected = contact_mask[index] & (gate[index] > 0)
            selected_distance = nearest_distance[local, selected].cpu().numpy() * 1000.0
            if len(selected_distance):
                contact_distances.append(selected_distance)
                contact_median[index] = float(np.median(selected_distance))
                contact_p90[index] = float(np.percentile(selected_distance, 90))
    all_contact = np.concatenate(contact_distances) if contact_distances else np.empty(0)
    all_penetration = (
        np.concatenate(penetration_depths) if penetration_depths else np.empty(0)
    )
    return (
        {
            "contact_distance_mm": distribution(all_contact),
            "penetrating_vertices_total": int(penetrating_count.sum()),
            "penetrating_vertices_per_frame": distribution(penetrating_count),
            "penetration_depth_mm": distribution(all_penetration),
        },
        {
            "contact_distance_median_mm": contact_median,
            "contact_distance_p90_mm": contact_p90,
            "penetrating_vertices": penetrating_count,
        },
    )


def main() -> None:
    args = parse_args()
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
    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    probability_np = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    contact_mask_np = np.asarray(
        contact["contact_mask"][contact_indices]
    ).astype(bool)
    gate_key = "oracle_contact_gate" if args.use_oracle_gate else "predicted_contact_gate"
    gate_np = np.asarray(phase[gate_key][phase_indices], dtype=np.float32)
    valid_np = (
        np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
    )
    gate_np *= valid_np.astype(np.float32)
    active_indices_np = np.flatnonzero(gate_np > 0)
    if len(active_indices_np) == 0:
        raise RuntimeError(f"{gate_key} selected no frames")

    mesh = load_mesh(Path(args.object_mesh).expanduser().resolve(), args.object_scale)
    local_points_np, local_normals_np = deterministic_surface_samples(
        mesh, args.object_samples
    )
    normalized_left = bool(np.asarray(
        supervision.get("normalized_left", False)
    ).item())
    object_points_np = np.empty(
        (frame_count, args.object_samples, 3), dtype=np.float32
    )
    object_normals_np = np.empty_like(object_points_np)
    for output_index, supervision_index in enumerate(supervision_indices):
        pose = physical_pose(
            supervision["gt_ycb_object_pose"][supervision_index], normalized_left
        )
        object_points_np[output_index] = (
            local_points_np @ pose[:3, :3].T + pose[:3, 3]
        )
        object_normals_np[output_index] = local_normals_np @ pose[:3, :3].T

    device = torch.device(args.device)
    vertices = torch.from_numpy(vertices_np).to(device)
    wrists = torch.from_numpy(wrist_np).to(device)
    object_points = torch.from_numpy(object_points_np).to(device)
    object_normals = torch.from_numpy(object_normals_np).to(device)
    contact_mask = torch.from_numpy(contact_mask_np).to(device)
    probability = torch.from_numpy(probability_np).to(device)
    gate = torch.from_numpy(gate_np).to(device)
    active_indices = torch.from_numpy(active_indices_np).to(device)
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    confidence = torch.clamp(
        (probability - threshold) / max(1.0 - threshold, 1e-6),
        min=0.0,
        max=1.0,
    ).pow(args.contact_probability_power)
    contact_weight = (
        args.contact_weight_floor
        + (1.0 - args.contact_weight_floor) * confidence
    ) * contact_mask
    contact_weight = contact_weight * gate[:, None]
    total_contact_weight = contact_weight.sum().clamp_min(1e-6)

    translation = torch.zeros(
        (frame_count, 3), device=device, requires_grad=True
    )
    angles = torch.zeros((frame_count, 3), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([translation, angles], lr=args.lr)
    max_translation = args.max_translation_mm / 1000.0
    max_angle = math.radians(args.max_rotation_deg)
    sigma = args.softmax_sigma_mm / 1000.0
    topk = min(max(1, args.topk), args.object_samples)
    best_total = float("inf")
    best_translation = torch.zeros_like(translation)
    best_angles = torch.zeros_like(angles)
    history = []

    initial_metrics, initial_per_frame = audit_geometry(
        vertices, contact_mask, object_points, object_normals, gate,
        args.penetration_tolerance_mm, args.penetration_trust_mm,
        args.frame_chunk,
    )
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        contact_value = 0.0
        for start in range(0, len(active_indices_np), args.frame_chunk):
            frame_indices = active_indices[start:start + args.frame_chunk]
            effective_translation = translation[frame_indices] * gate[frame_indices, None]
            effective_angles = angles[frame_indices] * gate[frame_indices, None]
            refined = transform_batch(
                vertices[frame_indices], wrists[frame_indices],
                effective_translation, effective_angles,
            )
            pairwise = torch.cdist(refined, object_points[frame_indices])
            nearest, _ = torch.topk(pairwise, topk, dim=-1, largest=False)
            soft_weight = torch.softmax(
                -nearest.square() / (2.0 * sigma * sigma), dim=-1
            )
            error = (soft_weight * nearest.square()).sum(dim=-1)
            chunk_loss = (
                error * contact_weight[frame_indices]
            ).sum() / total_contact_weight
            chunk_loss.backward()
            contact_value += float(chunk_loss.detach())

        effective_translation = translation * gate[:, None]
        effective_angles = angles * gate[:, None]
        active = gate > 0
        translation_anchor = effective_translation[active].square().sum(dim=-1).mean()
        rotation_anchor = effective_angles[active].square().sum(dim=-1).mean()
        translation_velocity = (
            effective_translation[1:] - effective_translation[:-1]
        )
        rotation_velocity = effective_angles[1:] - effective_angles[:-1]
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
        total_value = contact_value + float(regularization.detach())
        if total_value < best_total:
            best_total = total_value
            best_translation = translation.detach().clone()
            best_angles = angles.detach().clone()
        optimizer.step()
        with torch.no_grad():
            translation_norm = translation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rotation_norm = angles.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            translation.mul_(torch.clamp(max_translation / translation_norm, max=1.0))
            angles.mul_(torch.clamp(max_angle / rotation_norm, max=1.0))
            translation[gate == 0] = 0
            angles[gate == 0] = 0
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": total_value,
                "contact": contact_value,
                "regularization": float(regularization.detach()),
                "translation_median_mm": float(
                    (translation[active].norm(dim=-1) * 1000.0).median().detach()
                ),
                "translation_max_mm": float(
                    (translation[active].norm(dim=-1) * 1000.0).max().detach()
                ),
                "rotation_median_deg": float(
                    (angles[active].norm(dim=-1) * 180.0 / math.pi).median().detach()
                ),
            }
            history.append(row)
            print(row)

    effective_translation = best_translation * gate[:, None]
    effective_angles = best_angles * gate[:, None]
    refined = transform_batch(
        vertices, wrists, effective_translation, effective_angles
    ).detach()
    refined_metrics, refined_per_frame = audit_geometry(
        refined, contact_mask, object_points, object_normals, gate,
        args.penetration_tolerance_mm, args.penetration_trust_mm,
        args.frame_chunk,
    )

    gt_audit = None
    initial_gt_error = np.full(frame_count, np.nan, dtype=np.float32)
    refined_gt_error = np.full(frame_count, np.nan, dtype=np.float32)
    if args.gt_hand_npz:
        gt_data = load_npz(Path(args.gt_hand_npz).expanduser().resolve())
        side = str(query["hand_side"].item()).lower()
        gt_vertices = np.asarray(gt_data[f"{side}_vertices"], dtype=np.float32)
        gt_valid = np.asarray(gt_data[f"{side}_valid"]).astype(bool)
        evaluated = valid_np & gt_valid[:frame_count]
        initial_error = np.linalg.norm(
            vertices_np[evaluated] - gt_vertices[:frame_count][evaluated], axis=-1
        ) * 1000.0
        refined_error = np.linalg.norm(
            refined.cpu().numpy()[evaluated] - gt_vertices[:frame_count][evaluated],
            axis=-1,
        ) * 1000.0
        initial_gt_error[evaluated] = np.median(initial_error, axis=-1)
        refined_gt_error[evaluated] = np.median(refined_error, axis=-1)
        gt_audit = {
            "initial_vertex_error_mm": distribution(initial_error),
            "refined_vertex_error_mm": distribution(refined_error),
            "initial_frame_median_mm": distribution(initial_gt_error[evaluated]),
            "refined_frame_median_mm": distribution(refined_gt_error[evaluated]),
        }

    summary = {
        "method": "phase_gated_all_haco_one_way_soft_topk_chamfer_sequence_v1",
        "stream_id": str(query["stream_id"].item()),
        "frames": frame_count,
        "gate_source": gate_key,
        "active_frames": int((gate_np > 0).sum()),
        "initial": initial_metrics,
        "refined": refined_metrics,
        "translation_norm_mm": distribution(
            np.linalg.norm(
                effective_translation.cpu().numpy()[gate_np > 0], axis=-1
            ) * 1000.0
        ),
        "rotation_norm_deg": distribution(
            np.linalg.norm(
                effective_angles.cpu().numpy()[gate_np > 0], axis=-1
            )
            * 180.0 / math.pi
        ),
        "gt_audit": gt_audit,
        "history": history,
        "warning": "No penetration loss or collision safety gate is applied.",
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    payload = {
        "frame_ids": ids,
        "initial_hand_vertices_camera": vertices_np,
        "refined_hand_vertices_camera": refined.cpu().numpy().astype(np.float32),
        "mano_faces": faces,
        "contact_mask": contact_mask_np,
        "contact_probability": probability_np.astype(np.float16),
        "contact_gate": gate_np,
        "translation_camera": effective_translation.cpu().numpy().astype(np.float32),
        "rotation_euler_xyz": effective_angles.cpu().numpy().astype(np.float32),
        "initial_contact_distance_median_mm": initial_per_frame["contact_distance_median_mm"],
        "refined_contact_distance_median_mm": refined_per_frame["contact_distance_median_mm"],
        "initial_contact_distance_p90_mm": initial_per_frame["contact_distance_p90_mm"],
        "refined_contact_distance_p90_mm": refined_per_frame["contact_distance_p90_mm"],
        "initial_penetrating_vertices": initial_per_frame["penetrating_vertices"],
        "refined_penetrating_vertices": refined_per_frame["penetrating_vertices"],
        "initial_gt_vertex_error_median_mm": initial_gt_error,
        "refined_gt_vertex_error_median_mm": refined_gt_error,
        "stream_id": np.asarray(str(query["stream_id"].item())),
        "method": np.asarray(summary["method"]),
    }
    write_npz(output_path, payload)
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
