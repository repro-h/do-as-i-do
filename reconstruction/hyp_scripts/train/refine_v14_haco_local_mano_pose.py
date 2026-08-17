#!/usr/bin/env python3
"""Refine local MANO pose after freezing V14 and Stage-1 rigid placement."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
from refine_v14_haco_sequence_chamfer import (
    aligned_indices,
    batched_euler_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--phase-npz", required=True)
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
    parser.add_argument("--contact-target-mm", type=float, default=6.0)
    parser.add_argument("--contact-activation-mm", type=float, default=12.0)
    parser.add_argument("--contact-probability-power", type=float, default=2.0)
    parser.add_argument("--contact-weight-floor", type=float, default=0.05)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--penetration-trust-mm", type=float, default=20.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=8.0)
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-penetration", type=float, default=2.0)
    parser.add_argument("--w-pose-anchor", type=float, default=1e-4)
    parser.add_argument("--w-pose-velocity", type=float, default=1e-3)
    parser.add_argument("--w-pose-acceleration", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--frame-chunk", type=int, default=4)
    parser.add_argument("--roundtrip-max-rmse-mm", type=float, default=0.1)
    parser.add_argument("--use-oracle-gate", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ), dim=-1).reshape(*rotvec.shape[:-1], 3, 3)
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    sine_factor = torch.sinc(theta / math.pi)[..., None]
    cosine_factor = (
        0.5 * torch.sinc(theta / (2.0 * math.pi)).square()
    )[..., None]
    identity = torch.eye(
        3, device=rotvec.device, dtype=rotvec.dtype
    ).expand(*rotvec.shape[:-1], 3, 3)
    return identity + sine_factor * skew + cosine_factor * (skew @ skew)


def load_wilor_mano(
    wilor_root: Path,
    checkpoint: Path,
    config: Path,
    mano_data_dir: Path,
    device: torch.device,
) -> torch.nn.Module:
    sys.path.insert(0, str(wilor_root))
    from wilor.models import load_wilor

    previous = Path.cwd()
    try:
        os.chdir(wilor_root)
        model, _ = load_wilor(
            checkpoint_path=str(checkpoint),
            cfg_path=str(config),
            init_renderer=False,
            mano_data_dir=str(mano_data_dir),
        )
    finally:
        os.chdir(previous)
    mano = model.mano.to(device).eval().requires_grad_(False)
    del model
    return mano


def transform_local_to_camera(
    local: torch.Tensor,
    wrist: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.bmm(local, rotation.transpose(1, 2))
        + wrist[:, None]
        + translation[:, None]
    )


def mano_camera_vertices(
    mano: torch.nn.Module,
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
    wrist: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    mirror_left: bool,
) -> torch.Tensor:
    output = mano(
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        pose2rot=False,
    )
    local = output.vertices - output.joints[:, :1]
    if mirror_left:
        mirror = torch.as_tensor(
            [-1.0, 1.0, 1.0], device=local.device, dtype=local.dtype
        )
        local = local * mirror
    return transform_local_to_camera(
        local, wrist, translation, rotation
    )


@torch.no_grad()
def nearest_geometry(
    hand: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    frame_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance_parts = []
    inside_parts = []
    for start in range(0, len(hand), frame_chunk):
        end = min(start + frame_chunk, len(hand))
        pairwise = torch.cdist(hand[start:end], object_points[start:end])
        distance, nearest_index = pairwise.min(dim=-1)
        nearest_point = torch.gather(
            object_points[start:end], 1,
            nearest_index[..., None].expand(-1, -1, 3),
        )
        nearest_normal = torch.gather(
            object_normals[start:end], 1,
            nearest_index[..., None].expand(-1, -1, 3),
        )
        inside = ((nearest_point - hand[start:end]) * nearest_normal).sum(dim=-1)
        distance_parts.append(distance)
        inside_parts.append(inside)
    return torch.cat(distance_parts), torch.cat(inside_parts)


def geometry_summary(
    distance: np.ndarray,
    inside: np.ndarray,
    contact_mask: np.ndarray,
    phase_gate: np.ndarray,
    tolerance_mm: float,
    trust_mm: float,
) -> tuple[dict[str, object], np.ndarray]:
    phase = phase_gate > 0
    selected_contact = contact_mask & phase[:, None]
    contact_mm = distance[selected_contact] * 1000.0
    penetrating = (
        (inside > tolerance_mm / 1000.0)
        & (distance <= trust_mm / 1000.0)
        & phase[:, None]
    )
    penetration_mm = np.maximum(
        inside[penetrating] * 1000.0 - tolerance_mm, 0.0
    )
    return ({
        "contact_distance_mm": distribution(contact_mm),
        "penetrating_vertices_total": int(penetrating.sum()),
        "penetration_depth_mm": distribution(penetration_mm),
    }, penetrating.sum(axis=1).astype(np.int32))


def main() -> None:
    args = parse_args()
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    phase = load_npz(Path(args.phase_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    required_query = {
        "mano_global_orient_canonical_right",
        "mano_hand_pose_canonical_right",
        "mano_betas",
        "model_valid",
        "hand_side",
        "mano_faces",
    }
    missing = sorted(required_query - set(query))
    if missing:
        raise KeyError(f"WiLoR query cache lacks {missing}")

    ids = np.asarray(query["frame_ids"])
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    stage1_indices = aligned_indices(stage1["frame_ids"], ids)
    contact_indices = aligned_indices(contact["frame_ids"], ids)
    phase_indices = aligned_indices(phase["frame_ids"], ids)
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)
    frame_count = len(ids)
    gate_key = "oracle_contact_gate" if args.use_oracle_gate else "predicted_contact_gate"

    wrist_np = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    stage1_vertices_np = np.asarray(
        stage1["refined_hand_vertices_camera"][stage1_indices], dtype=np.float32
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
    base_pose_np = np.asarray(
        query["mano_hand_pose_canonical_right"], dtype=np.float32
    )
    betas_np = np.asarray(query["mano_betas"], dtype=np.float32)
    probability_np = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    contact_mask_np = np.asarray(
        contact["contact_mask"][contact_indices]
    ).astype(bool)
    phase_gate_np = np.asarray(phase[gate_key][phase_indices], dtype=np.float32)
    valid_np = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
    )
    phase_gate_np *= valid_np.astype(np.float32)
    active_indices_np = np.flatnonzero(phase_gate_np > 0)
    if not len(active_indices_np):
        raise RuntimeError(f"{gate_key} selected no valid frames")

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
            supervision["gt_ycb_object_pose"][supervision_index],
            normalized_left,
        )
        object_points_np[output_index] = (
            local_points_np @ pose[:3, :3].T + pose[:3, 3]
        )
        object_normals_np[output_index] = local_normals_np @ pose[:3, :3].T

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
    stage1_angles = torch.from_numpy(stage1_angles_np).to(device)
    stage1_rotation = batched_euler_matrix(stage1_angles)
    object_points = torch.from_numpy(object_points_np).to(device)
    object_normals = torch.from_numpy(object_normals_np).to(device)
    phase_gate = torch.from_numpy(phase_gate_np).to(device)
    probability = torch.from_numpy(probability_np).to(device)
    contact_mask = torch.from_numpy(contact_mask_np).to(device)
    mirror_left = str(query["hand_side"].item()).lower() == "left"

    with torch.no_grad():
        reconstructed_parts = []
        for start in range(0, frame_count, args.frame_chunk):
            end = min(start + args.frame_chunk, frame_count)
            reconstructed_parts.append(mano_camera_vertices(
                mano,
                global_orient[start:end], base_pose[start:end], betas[start:end],
                wrist[start:end], stage1_translation[start:end],
                stage1_rotation[start:end], mirror_left,
            ))
        reconstructed_stage1 = torch.cat(reconstructed_parts)
        roundtrip_error = torch.linalg.norm(
            reconstructed_stage1
            - torch.from_numpy(stage1_vertices_np).to(device), dim=-1
        ) * 1000.0
        roundtrip_rmse = torch.sqrt(roundtrip_error.square().mean(dim=-1))
        roundtrip_max = float(roundtrip_rmse.max().cpu())
    if roundtrip_max > args.roundtrip_max_rmse_mm:
        raise RuntimeError(
            "Stage-1 MANO roundtrip exceeded threshold: "
            f"{roundtrip_max:.6f} > {args.roundtrip_max_rmse_mm:.6f} mm"
        )

    initial_distance, initial_inside = nearest_geometry(
        reconstructed_stage1, object_points, object_normals, args.frame_chunk
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    confidence = torch.clamp(
        (probability - threshold) / max(1.0 - threshold, 1e-6),
        min=0.0, max=1.0,
    ).pow(args.contact_probability_power)
    plausible_contact = (
        contact_mask
        & (initial_distance <= args.contact_activation_mm / 1000.0)
        & (phase_gate[:, None] > 0)
    )
    contact_weight = (
        args.contact_weight_floor
        + (1.0 - args.contact_weight_floor) * confidence
    ) * plausible_contact
    total_contact_weight = contact_weight.sum().clamp_min(1e-6)
    total_active_vertices = max(1, len(active_indices_np) * contact_mask.shape[1])

    delta = torch.zeros(
        (frame_count, 15, 3), device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    active_indices = torch.from_numpy(active_indices_np).to(device)
    contact_target = args.contact_target_mm / 1000.0
    penetration_tolerance = args.penetration_tolerance_mm / 1000.0
    penetration_trust = args.penetration_trust_mm / 1000.0
    max_delta = math.radians(args.max_joint_delta_deg)
    best_total = float("inf")
    best_delta = torch.zeros_like(delta)
    history = []

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        contact_value = 0.0
        penetration_value = 0.0
        for start in range(0, len(active_indices_np), args.frame_chunk):
            indices = active_indices[start:start + args.frame_chunk]
            effective_delta = delta[indices] * phase_gate[indices, None, None]
            refined_pose = axis_angle_to_matrix(effective_delta) @ base_pose[indices]
            refined = mano_camera_vertices(
                mano,
                global_orient[indices], refined_pose, betas[indices],
                wrist[indices], stage1_translation[indices],
                stage1_rotation[indices], mirror_left,
            )
            pairwise = torch.cdist(refined, object_points[indices])
            nearest_distance, nearest_index = pairwise.min(dim=-1)
            nearest_point = torch.gather(
                object_points[indices], 1,
                nearest_index[..., None].expand(-1, -1, 3),
            )
            nearest_normal = torch.gather(
                object_normals[indices], 1,
                nearest_index[..., None].expand(-1, -1, 3),
            )
            contact_error = torch.clamp(
                nearest_distance - contact_target, min=0.0
            ).square()
            chunk_contact = (
                contact_error * contact_weight[indices]
            ).sum() / total_contact_weight
            inside = ((nearest_point - refined) * nearest_normal).sum(dim=-1)
            trusted = nearest_distance <= penetration_trust
            penetration_error = torch.clamp(
                inside - penetration_tolerance, min=0.0
            ).square() * trusted
            chunk_penetration = penetration_error.sum() / total_active_vertices
            chunk_loss = (
                args.w_contact * chunk_contact
                + args.w_penetration * chunk_penetration
            )
            chunk_loss.backward()
            contact_value += float(chunk_contact.detach())
            penetration_value += float(chunk_penetration.detach())

        effective_delta = delta * phase_gate[:, None, None]
        active = phase_gate > 0
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
            + args.w_penetration * penetration_value
            + float(regularization.detach())
        )
        if total_value < best_total:
            best_total = total_value
            best_delta = delta.detach().clone()
        optimizer.step()
        with torch.no_grad():
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta.mul_(torch.clamp(max_delta / norm, max=1.0))
            delta[phase_gate == 0] = 0
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": total_value,
                "contact": contact_value,
                "penetration": penetration_value,
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
            effective_delta = (
                best_delta[start:end] * phase_gate[start:end, None, None]
            )
            refined_pose = (
                axis_angle_to_matrix(effective_delta) @ base_pose[start:end]
            )
            refined_pose_parts.append(refined_pose)
            refined_parts.append(mano_camera_vertices(
                mano,
                global_orient[start:end], refined_pose, betas[start:end],
                wrist[start:end], stage1_translation[start:end],
                stage1_rotation[start:end], mirror_left,
            ))
        refined = torch.cat(refined_parts)
        refined_pose = torch.cat(refined_pose_parts)
    refined_distance, refined_inside = nearest_geometry(
        refined, object_points, object_normals, args.frame_chunk
    )

    initial_geometry, initial_penetrating = geometry_summary(
        initial_distance.cpu().numpy(), initial_inside.cpu().numpy(),
        contact_mask_np, phase_gate_np,
        args.penetration_tolerance_mm, args.penetration_trust_mm,
    )
    refined_geometry, refined_penetrating = geometry_summary(
        refined_distance.cpu().numpy(), refined_inside.cpu().numpy(),
        contact_mask_np, phase_gate_np,
        args.penetration_tolerance_mm, args.penetration_trust_mm,
    )

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
            stage1_vertices_np[evaluated] - gt_vertices[:frame_count][evaluated],
            axis=-1,
        ) * 1000.0
        refined_np = refined.cpu().numpy().astype(np.float32)
        refined_error = np.linalg.norm(
            refined_np[evaluated] - gt_vertices[:frame_count][evaluated], axis=-1
        ) * 1000.0
        initial_gt_frame[evaluated] = np.median(initial_error, axis=-1)
        refined_gt_frame[evaluated] = np.median(refined_error, axis=-1)
        gt_audit = {
            "stage1_vertex_error_mm": distribution(initial_error),
            "local_refined_vertex_error_mm": distribution(refined_error),
            "stage1_frame_median_mm": distribution(initial_gt_frame[evaluated]),
            "local_refined_frame_median_mm": distribution(refined_gt_frame[evaluated]),
        }
    else:
        refined_np = refined.cpu().numpy().astype(np.float32)

    effective_best_delta = best_delta * phase_gate[:, None, None]
    delta_deg = effective_best_delta.norm(dim=-1).cpu().numpy() * 180.0 / math.pi
    summary = {
        "method": "stage1_frozen_local_mano_contact_collision_v1",
        "stream_id": str(query["stream_id"].item()),
        "hand_side": str(query["hand_side"].item()),
        "frames": frame_count,
        "active_frames": int((phase_gate_np > 0).sum()),
        "gate_source": gate_key,
        "stage1_roundtrip_frame_rmse_mm": distribution(
            roundtrip_rmse.cpu().numpy()
        ),
        "plausible_contact_vertices": int(plausible_contact.sum().cpu()),
        "initial": initial_geometry,
        "refined": refined_geometry,
        "joint_delta_deg": distribution(delta_deg[phase_gate_np > 0]),
        "gt_audit": gt_audit,
        "history": history,
        "warning": (
            "Collision uses nearest sampled surface normals, not an exact SDF."
        ),
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    write_npz(output_path, {
        "frame_ids": ids,
        "stage1_hand_vertices_camera": stage1_vertices_np,
        "refined_hand_vertices_camera": refined_np,
        "mano_faces": np.asarray(query["mano_faces"], dtype=np.int64),
        "base_hand_pose_canonical_right": base_pose_np,
        "refined_hand_pose_canonical_right": refined_pose.cpu().numpy().astype(np.float32),
        "joint_rotation_delta_rotvec": effective_best_delta.cpu().numpy().astype(np.float32),
        "contact_mask": contact_mask_np,
        "contact_probability": probability_np.astype(np.float16),
        "contact_gate": phase_gate_np,
        "initial_penetrating_vertices": initial_penetrating,
        "refined_penetrating_vertices": refined_penetrating,
        "stage1_gt_vertex_error_median_mm": initial_gt_frame,
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
