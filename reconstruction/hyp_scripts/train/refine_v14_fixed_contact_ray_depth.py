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
    parser.add_argument(
        "--initial-hand-npz",
        help="Optional prior refinement whose camera-space hand initializes this pass",
    )
    parser.add_argument(
        "--initial-hand-vertices-key",
        default="refined_hand_vertices_camera",
    )
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--fixed-patch-npz", required=True)
    parser.add_argument(
        "--contact-target-source",
        choices=("fixed_patch", "candidate_surface"),
        default="fixed_patch",
        help=(
            "Use the selected geodesic patch or the full HACO-aligned object "
            "candidate surface as the contact target"
        ),
    )
    parser.add_argument(
        "--fixed-regions",
        nargs="+",
        help=(
            "Explicit patch regions to use; by default use stable_region_names"
        ),
    )
    parser.add_argument("--phase-npz", required=True)
    parser.add_argument(
        "--frame-id",
        help="Optional single frame to optimize instead of the full sequence",
    )
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
    parser.add_argument("--contact-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--contact-probability-power", type=float, default=2.0)
    parser.add_argument("--region-support-power", type=float, default=0.5)
    parser.add_argument("--minimum-region-weight", type=float, default=0.25)
    parser.add_argument(
        "--region-reduction",
        choices=("median", "max"),
        default="max",
        help=(
            "Aggregate active HACO regions per frame. 'max' makes the most "
            "distant region drive wrist-ray correction."
        ),
    )
    parser.add_argument("--inside-low-fraction", type=float, default=0.002)
    parser.add_argument("--inside-high-fraction", type=float, default=0.01)
    parser.add_argument("--collision-object-samples", type=int, default=2048)
    parser.add_argument(
        "--inside-allowance-count",
        type=int,
        default=8,
        help=(
            "Per-frame sampled containment budget. States within max(initial, "
            "allowance) may improve contact; above it collision dominates."
        ),
    )
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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def region_contact_feedback(
    hand: np.ndarray,
    contact_mask: np.ndarray,
    contact_probability: np.ndarray,
    region_ids: np.ndarray,
    fixed_regions: dict[str, np.ndarray],
    region_names: list[str],
    target_mm: float,
    minimum_vertices: int,
    ray: np.ndarray,
    probability_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaps = np.full(len(region_names), np.nan, dtype=np.float32)
    projected = np.full(len(region_names), np.nan, dtype=np.float32)
    support = np.zeros(len(region_names), dtype=np.float32)
    for name, patch in fixed_regions.items():
        region_index = region_names.index(name)
        selected = contact_mask & (region_ids == region_index)
        if int(selected.sum()) < minimum_vertices or not len(patch):
            continue
        points = hand[selected]
        probability = np.clip(contact_probability[selected], 1e-6, 1.0)
        weights = probability ** probability_power
        pairwise = np.linalg.norm(
            points[:, None] - patch[None], axis=-1
        )
        nearest_index = pairwise.argmin(axis=1)
        nearest = patch[nearest_index]
        distance_mm = pairwise[np.arange(len(points)), nearest_index] * 1000.0
        gap = weighted_median(distance_mm, weights)
        gaps[region_index] = gap
        ray_delta_mm = (nearest - points) @ ray * 1000.0
        projected[region_index] = weighted_median(ray_delta_mm, weights)
        support[region_index] = float(weights.sum())
    return gaps, projected, support


def feedback_all_frames(
    hand: np.ndarray,
    contact_mask: np.ndarray,
    contact_probability: np.ndarray,
    region_ids: np.ndarray,
    fixed_region_camera: dict[str, np.ndarray],
    region_names: list[str],
    rays: np.ndarray,
    target_mm: float,
    minimum_vertices: int,
    probability_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaps = np.full((len(hand), len(region_names)), np.nan, dtype=np.float32)
    projected = np.full_like(gaps, np.nan)
    support = np.zeros_like(gaps)
    for index in range(len(hand)):
        patches = {
            name: values[index] for name, values in fixed_region_camera.items()
        }
        gaps[index], projected[index], support[index] = region_contact_feedback(
            hand[index],
            contact_mask[index],
            contact_probability[index],
            region_ids,
            patches,
            region_names,
            target_mm,
            minimum_vertices,
            rays[index],
            probability_power,
        )
    return gaps, projected, support


def reduce_region_feedback(
    gaps: np.ndarray,
    projected: np.ndarray,
    support: np.ndarray,
    target_mm: float,
    reduction: str,
    support_power: float,
    minimum_region_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(gaps)
    active = valid.sum(axis=1).astype(np.int32)
    aggregate = np.full(len(gaps), np.nan, dtype=np.float32)
    delta = np.zeros(len(gaps), dtype=np.float32)
    for index in np.flatnonzero(active):
        values = gaps[index, valid[index]]
        if reduction == "max":
            strengths = support[index, valid[index]]
            strengths /= max(float(strengths.max()), 1e-12)
            strengths = np.maximum(
                strengths ** support_power, minimum_region_weight
            )
            violation = np.maximum(values - target_mm, 0.0) * strengths
            local = int(np.argmax(violation))
            region_index = np.flatnonzero(valid[index])[local]
            aggregate[index] = values[local]
            if values[local] > target_mm:
                delta[index] = projected[index, region_index]
        else:
            aggregate[index] = float(np.median(values))
            pulling = valid[index] & (gaps[index] > target_mm)
            if pulling.any():
                delta[index] = float(np.nanmedian(projected[index, pulling]))
    return aggregate, delta, active


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
    all_ids = np.asarray(query["frame_ids"])
    if args.frame_id is not None:
        requested = frame_id(args.frame_id)
        query_frame_ids = [frame_id(value) for value in all_ids]
        if requested not in query_frame_ids:
            raise KeyError(f"Frame {requested} not found in query archive")
        selected_query_indices = np.asarray(
            [query_frame_ids.index(requested)], dtype=np.int64
        )
        ids = all_ids[selected_query_indices]
    else:
        selected_query_indices = np.arange(len(all_ids), dtype=np.int64)
        ids = all_ids
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    contact_indices = aligned_indices(contact["frame_ids"], ids)
    phase_indices = aligned_indices(phase["frame_ids"], ids)
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    hand = np.asarray(
        query["vertices_3d_root_relative_original"][selected_query_indices],
        dtype=np.float32,
    ) + wrist[:, None]
    initial_hand_source = None
    if args.initial_hand_npz:
        initial_hand_source = str(
            Path(args.initial_hand_npz).expanduser().resolve()
        )
        initial_hand = load_npz(Path(initial_hand_source))
        initial_indices = aligned_indices(initial_hand["frame_ids"], ids)
        if args.initial_hand_vertices_key not in initial_hand:
            raise KeyError(
                f"Initial hand archive lacks {args.initial_hand_vertices_key!r}"
            )
        hand = np.asarray(
            initial_hand[args.initial_hand_vertices_key][initial_indices],
            dtype=np.float32,
        )
        if "refined_wrist_camera" in initial_hand:
            wrist = np.asarray(
                initial_hand["refined_wrist_camera"][initial_indices],
                dtype=np.float32,
            )
    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    gate = np.asarray(phase[args.phase_key][phase_indices], dtype=np.float32)
    contact_mask = np.asarray(
        contact["contact_mask"][contact_indices]
    ).astype(bool)
    contact_probability = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    valid = (
        np.asarray(query["model_valid"])[selected_query_indices].astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
        & np.isfinite(hand).all(axis=(1, 2))
    )
    active = valid & (gate >= args.minimum_phase_gate)
    if not active.any():
        raise RuntimeError("No phase-active valid frame")

    default_name_key = (
        "selected_region_names"
        if args.contact_target_source == "candidate_surface"
        else "stable_region_names"
    )
    default_names = (
        [str(value) for value in fixed[default_name_key]]
        if default_name_key in fixed else []
    )
    fixed_names = list(args.fixed_regions or default_names)
    if not fixed_names:
        raise RuntimeError(
            f"Contact archive has no {default_name_key}; inspect the candidates "
            "and pass --fixed-regions REGION [REGION ...]"
        )
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )
    unknown = sorted(set(fixed_names).difference(region_names))
    if unknown:
        raise KeyError(f"Unknown fixed patch regions: {unknown}")
    target_key = (
        "candidate_vertex_ids"
        if args.contact_target_source == "candidate_surface"
        else "patch_vertices_canonical"
    )
    missing = sorted(
        name for name in fixed_names if f"{name}_{target_key}" not in fixed
    )
    if missing:
        raise KeyError(
            f"Contact archive lacks {args.contact_target_source} for: {missing}"
        )

    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    poses = np.stack([
        physical_pose(
            supervision["gt_ycb_object_pose"][index], normalized_left
        )
        for index in supervision_indices
    ]).astype(np.float32)
    mesh = trimesh.load(Path(args.object_mesh).expanduser().resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    object_local = np.asarray(mesh.vertices, dtype=np.float32)
    if args.contact_target_source == "candidate_surface":
        fixed_region_local = {
            name: object_local[
                np.asarray(fixed[f"{name}_candidate_vertex_ids"], dtype=np.int64)
            ]
            for name in fixed_names
        }
    else:
        fixed_region_local = {
            name: np.asarray(
                fixed[f"{name}_patch_vertices_canonical"], dtype=np.float32
            )
            for name in fixed_names
        }
    fixed_region_camera = {
        name: transform_object(vertices, poses)
        for name, vertices in fixed_region_local.items()
    }
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

    initial_region_gaps, initial_region_projected, initial_region_support = feedback_all_frames(
        hand,
        contact_mask,
        contact_probability,
        region_ids,
        fixed_region_camera,
        region_names,
        rays,
        args.contact_target_mm,
        args.minimum_contact_vertices,
        args.contact_probability_power,
    )
    initial_gaps, _, active_regions = reduce_region_feedback(
        initial_region_gaps,
        initial_region_projected,
        initial_region_support,
        args.contact_target_mm,
        args.region_reduction,
        args.region_support_power,
        args.minimum_region_weight,
    )
    initial_sampled_inside = sampled_inside_counts(
        hand, faces, object_sample_camera, boundary, device, args.point_chunk
    )
    inside_budget = np.maximum(
        initial_sampled_inside,
        np.full(len(ids), args.inside_allowance_count, dtype=np.int32),
    )
    best_offsets_mm = offsets_mm.copy()
    best_gap = initial_gaps.copy()
    best_inside = initial_sampled_inside.copy()

    for iteration in range(1, args.iterations + 1):
        current = shifted_hand(hand, rays, offsets_mm / 1000.0)
        current_region_gap, current_region_projected, current_region_support = feedback_all_frames(
            current,
            contact_mask,
            contact_probability,
            region_ids,
            fixed_region_camera,
            region_names,
            rays,
            args.contact_target_mm,
            args.minimum_contact_vertices,
            args.contact_probability_power,
        )
        current_gap, contact_delta, active_regions = reduce_region_feedback(
            current_region_gap,
            current_region_projected,
            current_region_support,
            args.contact_target_mm,
            args.region_reduction,
            args.region_support_power,
            args.minimum_region_weight,
        )
        current_inside = sampled_inside_counts(
            current, faces, object_sample_camera, boundary, device, args.point_chunk
        )
        feasible = active & (current_inside <= inside_budget)
        better_contact = feasible & np.isfinite(current_gap) & (
            ~np.isfinite(best_gap) | (current_gap < best_gap - 1e-4)
        )
        equal_contact = (
            feasible
            & np.isfinite(current_gap)
            & np.isfinite(best_gap)
            & (np.abs(current_gap - best_gap) <= 1e-4)
            & (current_inside < best_inside)
        )
        update_best = better_contact | equal_contact
        best_offsets_mm[update_best] = offsets_mm[update_best]
        best_gap[update_best] = current_gap[update_best]
        best_inside[update_best] = current_inside[update_best]
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
        plus_region_gap, plus_region_projected, plus_region_support = feedback_all_frames(
            plus,
            contact_mask,
            contact_probability,
            region_ids,
            fixed_region_camera,
            region_names,
            rays,
            args.contact_target_mm,
            args.minimum_contact_vertices,
            args.contact_probability_power,
        )
        minus_region_gap, minus_region_projected, minus_region_support = feedback_all_frames(
            minus,
            contact_mask,
            contact_probability,
            region_ids,
            fixed_region_camera,
            region_names,
            rays,
            args.contact_target_mm,
            args.minimum_contact_vertices,
            args.contact_probability_power,
        )
        plus_gap, _, _ = reduce_region_feedback(
            plus_region_gap,
            plus_region_projected,
            plus_region_support,
            args.contact_target_mm,
            args.region_reduction,
            args.region_support_power,
            args.minimum_region_weight,
        )
        minus_gap, _, _ = reduce_region_feedback(
            minus_region_gap,
            minus_region_projected,
            minus_region_support,
            args.contact_target_mm,
            args.region_reduction,
            args.region_support_power,
            args.minimum_region_weight,
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
        plus_contact_better = np.isfinite(plus_gap) & (
            (~np.isfinite(minus_gap)) | (plus_gap < minus_gap)
        )
        minus_contact_better = np.isfinite(minus_gap) & (
            (~np.isfinite(plus_gap)) | (minus_gap < plus_gap)
        )
        contact_step[plus_contact_better] = args.probe_mm
        contact_step[minus_contact_better] = -args.probe_mm
        contact_satisfied = (
            np.isfinite(current_gap)
            & (current_gap <= args.contact_target_mm + args.contact_tolerance_mm)
        )
        contact_step[contact_satisfied] = 0.0
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
            "budget_feasible_frames": int(feasible.sum()),
        }
        history.append(row)
        if iteration == 1 or iteration % 5 == 0 or iteration == args.iterations:
            print(row, flush=True)

    final = shifted_hand(hand, rays, offsets_mm / 1000.0)
    final_region_gap, final_region_projected, final_region_support = (
        feedback_all_frames(
            final,
            contact_mask,
            contact_probability,
            region_ids,
            fixed_region_camera,
            region_names,
            rays,
            args.contact_target_mm,
            args.minimum_contact_vertices,
            args.contact_probability_power,
        )
    )
    final_gap, _, _ = reduce_region_feedback(
        final_region_gap,
        final_region_projected,
        final_region_support,
        args.contact_target_mm,
        args.region_reduction,
        args.region_support_power,
        args.minimum_region_weight,
    )
    final_inside = sampled_inside_counts(
        final, faces, object_sample_camera, boundary, device, args.point_chunk
    )
    final_feasible = active & (final_inside <= inside_budget)
    final_better = final_feasible & np.isfinite(final_gap) & (
        ~np.isfinite(best_gap) | (final_gap < best_gap - 1e-4)
    )
    best_offsets_mm[final_better] = offsets_mm[final_better]
    best_gap[final_better] = final_gap[final_better]
    best_inside[final_better] = final_inside[final_better]
    offsets_mm = best_offsets_mm

    refined = shifted_hand(hand, rays, offsets_mm / 1000.0).astype(np.float32)
    refined_wrist = (
        wrist + offsets_mm[:, None] / 1000.0 * rays
    ).astype(np.float32)
    refined_region_gaps, refined_region_projected, refined_region_support = feedback_all_frames(
        refined,
        contact_mask,
        contact_probability,
        region_ids,
        fixed_region_camera,
        region_names,
        rays,
        args.contact_target_mm,
        args.minimum_contact_vertices,
        args.contact_probability_power,
    )
    refined_gaps, _, refined_active_regions = reduce_region_feedback(
        refined_region_gaps,
        refined_region_projected,
        refined_region_support,
        args.contact_target_mm,
        args.region_reduction,
        args.region_support_power,
        args.minimum_region_weight,
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
        gt_indices = aligned_indices(gt["frame_ids"], ids)
        gt_vertices = np.asarray(gt[f"{side}_vertices"], dtype=np.float32)[gt_indices]
        gt_valid = valid & np.asarray(gt[f"{side}_valid"]).astype(bool)[gt_indices]
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
        "initial_hand_source": initial_hand_source,
        "contact_target_source_npz": str(
            Path(args.fixed_patch_npz).expanduser().resolve()
        ),
        "contact_target_source": args.contact_target_source,
        "fixed_regions": fixed_names,
        "active_frames": int(active.sum()),
        "contact_evaluated_frames": int(evaluated.sum()),
        "ray_offset_mm": distribution(offsets_mm[active]),
        "absolute_ray_offset_mm": distribution(np.abs(offsets_mm[active])),
        "contact_gap_mm": {
            "initial": distribution(initial_gaps[evaluated]),
            "refined": distribution(refined_gaps[evaluated]),
        },
        "per_region_contact_gap_mm": {
            name: {
                "initial": distribution(
                    initial_region_gaps[
                        active & np.isfinite(initial_region_gaps[:, region_index]),
                        region_index,
                    ]
                ),
                "refined": distribution(
                    refined_region_gaps[
                        active & np.isfinite(refined_region_gaps[:, region_index]),
                        region_index,
                    ]
                ),
            }
            for region_index, name in enumerate(region_names)
            if name in fixed_names
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
            "contact_tolerance_mm": args.contact_tolerance_mm,
            "region_reduction": args.region_reduction,
            "collision_object_samples": sample_count,
            "inside_allowance_count": args.inside_allowance_count,
            "contact_probability_power": args.contact_probability_power,
            "region_support_power": args.region_support_power,
            "minimum_region_weight": args.minimum_region_weight,
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
        "contact_region_names": np.asarray(region_names),
        "initial_fixed_contact_gap_by_region_mm": initial_region_gaps,
        "refined_fixed_contact_gap_by_region_mm": refined_region_gaps,
        "fixed_patch_region_names": np.asarray(fixed_names),
        "contact_target_source": np.asarray(args.contact_target_source),
        "initial_object_vertex_inside_capped_mano": initial_inside_mask,
        "refined_object_vertex_inside_capped_mano": refined_inside_mask,
        "initial_inside_object_vertices": initial_inside,
        "refined_inside_object_vertices": refined_inside,
        "stream_id": np.asarray(str(query["stream_id"].item())),
        "initial_hand_source": np.asarray(initial_hand_source or "v14"),
        "method": np.asarray(summary["method"]),
    })
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Output: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
