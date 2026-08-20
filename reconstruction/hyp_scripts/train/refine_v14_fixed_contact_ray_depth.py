#!/usr/bin/env python3
"""Refine V14 wrist depth with fixed object patches and collision feedback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from refine_v14_haco_containment_pushout import exact_inside_counts
from refine_v14_haco_one_way_chamfer import distribution, load_npz, write_npz
from refine_v14_haco_sequence_chamfer import aligned_indices, frame_id
from refine_v14_haco_sequence_contact_containment import (
    mano_contact_region_ids,
    physical_pose,
)
from visualize_capped_mano_wrist import directed_boundary_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--fixed-patch-npz", required=True)
    parser.add_argument("--phase-npz", required=True)
    parser.add_argument("--phase-key", default="predicted_contact_gate")
    parser.add_argument("--minimum-phase-gate", type=float, default=0.25)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--probe-mm", type=float, default=2.0)
    parser.add_argument("--max-step-mm", type=float, default=3.0)
    parser.add_argument("--max-correction-mm", type=float, default=60.0)
    parser.add_argument("--contact-target-mm", type=float, default=6.0)
    parser.add_argument("--inside-low-fraction", type=float, default=0.002)
    parser.add_argument("--inside-high-fraction", type=float, default=0.01)
    parser.add_argument("--collision-object-samples", type=int, default=2048)
    parser.add_argument("--temporal-step-weight", type=float, default=0.2)
    parser.add_argument("--contact-step-scale", type=float, default=0.5)
    parser.add_argument("--minimum-contact-vertices", type=int, default=3)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def transform_object(
    vertices: np.ndarray, poses: np.ndarray
) -> np.ndarray:
    return (
        np.einsum("vi,fji->fvj", vertices, poses[:, :3, :3])
        + poses[:, None, :3, 3]
    )


def shifted_hand(
    hand: np.ndarray, rays: np.ndarray, offset_m: np.ndarray
) -> np.ndarray:
    return hand + offset_m[:, None, None] * rays[:, None]


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def region_contact_feedback(
    hand: np.ndarray,
    contact_mask: np.ndarray,
    region_ids: np.ndarray,
    fixed_regions: dict[str, np.ndarray],
    region_names: list[str],
    target_mm: float,
    minimum_vertices: int,
    ray: np.ndarray,
) -> tuple[float, float, int]:
    gaps: list[float] = []
    projected: list[float] = []
    active_regions = 0
    for name, patch in fixed_regions.items():
        region_index = region_names.index(name)
        selected = contact_mask & (region_ids == region_index)
        if int(selected.sum()) < minimum_vertices or not len(patch):
            continue
        points = hand[selected]
        pairwise = np.linalg.norm(
            points[:, None] - patch[None], axis=-1
        )
        nearest_index = pairwise.argmin(axis=1)
        nearest = patch[nearest_index]
        distance_mm = pairwise[np.arange(len(points)), nearest_index] * 1000.0
        gap = float(np.median(distance_mm))
        gaps.append(gap)
        active_regions += 1
        if gap > target_mm:
            ray_delta_mm = (nearest - points) @ ray * 1000.0
            projected.append(float(np.median(ray_delta_mm)))
    if not gaps:
        return float("nan"), 0.0, 0
    contact_delta = float(np.median(projected)) if projected else 0.0
    return float(np.median(gaps)), contact_delta, active_regions


def feedback_all_frames(
    hand: np.ndarray,
    contact_mask: np.ndarray,
    region_ids: np.ndarray,
    fixed_region_camera: dict[str, np.ndarray],
    region_names: list[str],
    rays: np.ndarray,
    target_mm: float,
    minimum_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaps = np.full(len(hand), np.nan, dtype=np.float32)
    deltas = np.zeros(len(hand), dtype=np.float32)
    active = np.zeros(len(hand), dtype=np.int32)
    for index in range(len(hand)):
        patches = {
            name: values[index] for name, values in fixed_region_camera.items()
        }
        gaps[index], deltas[index], active[index] = region_contact_feedback(
            hand[index],
            contact_mask[index],
            region_ids,
            patches,
            region_names,
            target_mm,
            minimum_vertices,
            rays[index],
        )
    return gaps, deltas, active


def sampled_inside_counts(
    hand: np.ndarray,
    faces: np.ndarray,
    object_vertices: np.ndarray,
    boundary: np.ndarray,
    device: torch.device,
    point_chunk: int,
) -> np.ndarray:
    _, counts = exact_inside_counts(
        hand, faces, object_vertices, boundary, device, point_chunk
    )
    return counts.astype(np.int32)


def main() -> None:
    args = parse_args()
    if not 0 <= args.inside_low_fraction < args.inside_high_fraction:
        raise ValueError("Inside thresholds must satisfy 0 <= low < high")
    if args.iterations <= 0 or args.probe_mm <= 0 or args.max_step_mm <= 0:
        raise ValueError("Iteration and step settings must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    fixed = load_npz(Path(args.fixed_patch_npz).expanduser().resolve())
    phase = load_npz(Path(args.phase_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    ids = np.asarray(query["frame_ids"])
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    contact_indices = aligned_indices(contact["frame_ids"], ids)
    phase_indices = aligned_indices(phase["frame_ids"], ids)
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    hand = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    ) + wrist[:, None]
    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    gate = np.asarray(phase[args.phase_key][phase_indices], dtype=np.float32)
    contact_mask = np.asarray(
        contact["contact_mask"][contact_indices]
    ).astype(bool)
    valid = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
        & np.isfinite(hand).all(axis=(1, 2))
    )
    active = valid & (gate >= args.minimum_phase_gate)
    if not active.any():
        raise RuntimeError("No phase-active valid frame")

    stable_names = (
        [str(value) for value in fixed["stable_region_names"]]
        if "stable_region_names" in fixed else []
    )
    if not stable_names:
        raise RuntimeError("Fixed patch archive has no stable regions")
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )
    unknown = sorted(set(stable_names).difference(region_names))
    if unknown:
        raise KeyError(f"Unknown fixed patch regions: {unknown}")

    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    poses = np.stack([
        physical_pose(
            supervision["gt_ycb_object_pose"][index], normalized_left
        )
        for index in supervision_indices
    ]).astype(np.float32)
    fixed_region_camera = {
        name: transform_object(
            np.asarray(fixed[f"{name}_patch_vertices_canonical"], dtype=np.float32),
            poses,
        )
        for name in stable_names
    }

    mesh = trimesh.load(Path(args.object_mesh).expanduser().resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    object_local = np.asarray(mesh.vertices, dtype=np.float32)
    sample_count = min(args.collision_object_samples, len(object_local))
    sample_indices = np.linspace(
        0, len(object_local) - 1, sample_count, dtype=np.int64
    )
    object_sample_camera = transform_object(object_local[sample_indices], poses)
    object_full_camera = transform_object(object_local, poses)
    boundary = directed_boundary_loop(faces)

    rays = wrist / np.maximum(
        np.linalg.norm(wrist, axis=-1, keepdims=True), 1e-8
    )
    offsets_mm = np.zeros(len(ids), dtype=np.float32)
    probe_m = args.probe_mm / 1000.0
    history: list[dict[str, object]] = []

    initial_gaps, _, active_regions = feedback_all_frames(
        hand,
        contact_mask,
        region_ids,
        fixed_region_camera,
        region_names,
        rays,
        args.contact_target_mm,
        args.minimum_contact_vertices,
    )
    initial_sampled_inside = sampled_inside_counts(
        hand, faces, object_sample_camera, boundary, device, args.point_chunk
    )

    for iteration in range(1, args.iterations + 1):
        current = shifted_hand(hand, rays, offsets_mm / 1000.0)
        current_gap, contact_delta, active_regions = feedback_all_frames(
            current,
            contact_mask,
            region_ids,
            fixed_region_camera,
            region_names,
            rays,
            args.contact_target_mm,
            args.minimum_contact_vertices,
        )
        current_inside = sampled_inside_counts(
            current, faces, object_sample_camera, boundary, device, args.point_chunk
        )
        plus = shifted_hand(
            hand, rays, (offsets_mm / 1000.0) + probe_m
        )
        minus = shifted_hand(
            hand, rays, (offsets_mm / 1000.0) - probe_m
        )
        plus_inside = sampled_inside_counts(
            plus, faces, object_sample_camera, boundary, device, args.point_chunk
        )
        minus_inside = sampled_inside_counts(
            minus, faces, object_sample_camera, boundary, device, args.point_chunk
        )

        collision_direction = np.zeros(len(ids), dtype=np.float32)
        plus_better = (plus_inside < current_inside) & (
            plus_inside < minus_inside
        )
        minus_better = (minus_inside < current_inside) & (
            minus_inside < plus_inside
        )
        equal_improvement = (
            (plus_inside == minus_inside) & (plus_inside < current_inside)
        )
        collision_direction[plus_better] = args.probe_mm
        collision_direction[minus_better] = -args.probe_mm
        collision_direction[equal_improvement] = (
            np.where(contact_delta[equal_improvement] >= 0.0, 1.0, -1.0)
            * args.probe_mm
        )
        inside_fraction = current_inside.astype(np.float32) / float(sample_count)
        collision_gate = smoothstep(
            args.inside_low_fraction,
            args.inside_high_fraction,
            inside_fraction,
        )
        contact_step = np.clip(
            contact_delta * args.contact_step_scale,
            -args.max_step_mm,
            args.max_step_mm,
        )
        collision_step = np.clip(
            collision_direction, -args.max_step_mm, args.max_step_mm
        )
        delta = collision_gate * collision_step + (1.0 - collision_gate) * contact_step

        neighbor = offsets_mm.copy()
        if len(offsets_mm) > 2:
            neighbor[1:-1] = 0.5 * (offsets_mm[:-2] + offsets_mm[2:])
        delta += args.temporal_step_weight * (neighbor - offsets_mm)
        delta = np.clip(delta, -args.max_step_mm, args.max_step_mm)
        delta[~active] = 0.0
        delta[(active_regions == 0) & (current_inside == 0)] = 0.0
        offsets_mm = np.clip(
            offsets_mm + delta,
            -args.max_correction_mm,
            args.max_correction_mm,
        )

        row = {
            "iteration": iteration,
            "active_frames": int(active.sum()),
            "sampled_inside_total": int(current_inside[active].sum()),
            "sampled_inside_median": float(np.median(current_inside[active])),
            "contact_gap_median_mm": (
                float(np.nanmedian(current_gap[active]))
                if np.isfinite(current_gap[active]).any() else None
            ),
            "offset_median_mm": float(np.median(np.abs(offsets_mm[active]))),
            "offset_max_mm": float(np.max(np.abs(offsets_mm[active]))),
            "collision_dominant_frames": int(
                (active & (collision_gate >= 0.5)).sum()
            ),
        }
        history.append(row)
        if iteration == 1 or iteration % 5 == 0 or iteration == args.iterations:
            print(row, flush=True)

    refined = shifted_hand(hand, rays, offsets_mm / 1000.0).astype(np.float32)
    refined_wrist = (
        wrist + offsets_mm[:, None] / 1000.0 * rays
    ).astype(np.float32)
    refined_gaps, _, refined_active_regions = feedback_all_frames(
        refined,
        contact_mask,
        region_ids,
        fixed_region_camera,
        region_names,
        rays,
        args.contact_target_mm,
        args.minimum_contact_vertices,
    )
    initial_inside_mask, initial_inside = exact_inside_counts(
        hand, faces, object_full_camera, boundary, device, args.point_chunk
    )
    refined_inside_mask, refined_inside = exact_inside_counts(
        refined, faces, object_full_camera, boundary, device, args.point_chunk
    )
    initial_inside_mask &= valid[:, None]
    refined_inside_mask &= valid[:, None]
    initial_inside = initial_inside_mask.sum(axis=1).astype(np.int32)
    refined_inside = refined_inside_mask.sum(axis=1).astype(np.int32)

    gt_summary = None
    if args.gt_hand_npz:
        gt = load_npz(Path(args.gt_hand_npz).expanduser().resolve())
        side = str(query["hand_side"].item()).lower()
        gt_vertices = np.asarray(gt[f"{side}_vertices"], dtype=np.float32)[:len(ids)]
        gt_valid = valid & np.asarray(gt[f"{side}_valid"]).astype(bool)[:len(ids)]
        initial_error = np.linalg.norm(hand[gt_valid] - gt_vertices[gt_valid], axis=-1)
        refined_error = np.linalg.norm(refined[gt_valid] - gt_vertices[gt_valid], axis=-1)
        gt_summary = {
            "initial_vertex_error_mm": distribution(initial_error * 1000.0),
            "refined_vertex_error_mm": distribution(refined_error * 1000.0),
        }

    evaluated = active & np.isfinite(initial_gaps) & np.isfinite(refined_gaps)
    summary = {
        "method": "fixed_contact_threshold_ray_depth_feedback_v1",
        "stream_id": str(query["stream_id"].item()),
        "fixed_patch_source": str(
            Path(args.fixed_patch_npz).expanduser().resolve()
        ),
        "fixed_regions": stable_names,
        "active_frames": int(active.sum()),
        "contact_evaluated_frames": int(evaluated.sum()),
        "ray_offset_mm": distribution(offsets_mm[active]),
        "absolute_ray_offset_mm": distribution(np.abs(offsets_mm[active])),
        "contact_gap_mm": {
            "initial": distribution(initial_gaps[evaluated]),
            "refined": distribution(refined_gaps[evaluated]),
        },
        "containment": {
            "initial_total": int(initial_inside.sum()),
            "refined_total": int(refined_inside.sum()),
            "improved_frames": int((refined_inside < initial_inside).sum()),
            "degraded_frames": int((refined_inside > initial_inside).sum()),
            "initial_per_frame": distribution(initial_inside[valid]),
            "refined_per_frame": distribution(refined_inside[valid]),
        },
        "gt_audit": gt_summary,
        "settings": {
            "probe_mm": args.probe_mm,
            "max_step_mm": args.max_step_mm,
            "max_correction_mm": args.max_correction_mm,
            "inside_low_fraction": args.inside_low_fraction,
            "inside_high_fraction": args.inside_high_fraction,
            "contact_target_mm": args.contact_target_mm,
            "collision_object_samples": sample_count,
        },
        "history": history,
        "warning": (
            "No object SDF is used. Collision direction is selected from local "
            "plus/minus wrist-ray probes using capped-MANO containment."
        ),
    }
    output = Path(args.out_npz).expanduser().resolve()
    summary_path = Path(args.out_json).expanduser().resolve()
    write_npz(output, {
        "frame_ids": ids,
        "initial_hand_vertices_camera": hand,
        "refined_hand_vertices_camera": refined,
        "mano_faces": faces,
        "initial_wrist_camera": wrist,
        "refined_wrist_camera": refined_wrist,
        "translation_camera": (refined_wrist - wrist).astype(np.float32),
        "wrist_ray_camera": rays.astype(np.float32),
        "ray_offset_mm": offsets_mm,
        "contact_gate": gate,
        "prediction_valid": valid,
        "initial_fixed_contact_gap_mm": initial_gaps,
        "refined_fixed_contact_gap_mm": refined_gaps,
        "fixed_patch_region_names": np.asarray(stable_names),
        "initial_object_vertex_inside_capped_mano": initial_inside_mask,
        "refined_object_vertex_inside_capped_mano": refined_inside_mask,
        "initial_inside_object_vertices": initial_inside,
        "refined_inside_object_vertices": refined_inside,
        "stream_id": np.asarray(str(query["stream_id"].item())),
        "method": np.asarray(summary["method"]),
    })
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Output: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
