#!/usr/bin/env python3
"""Joint rigid Stage1 refinement using HACO contact and MANO containment."""

from __future__ import annotations

import argparse
import json
import math
import pickle
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


MANO_CONTACT_REGIONS = {
    "palm": (0,),
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
}


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def physical_pose(pose: np.ndarray, normalized_left: bool) -> np.ndarray:
    """Convert a canonical-right object pose back to physical camera axes."""
    result = np.asarray(pose, dtype=np.float32).copy()
    if normalized_left:
        result[:3, :3] = MIRROR_X @ result[:3, :3] @ MIRROR_X
        result[:3, 3] = MIRROR_X @ result[:3, 3]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--initial-hand-npz")
    parser.add_argument("--initial-hand-vertices-key")
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
    parser.add_argument(
        "--fixed-contact-vertex-topk",
        type=int,
        default=16,
        help="Per region, retain this many probability-aware MANO contact vertices.",
    )
    parser.add_argument(
        "--fixed-contact-selection-sigma-mm",
        type=float,
        default=6.0,
        help="Confidence bonus used when selecting fixed-patch contact vertices.",
    )
    parser.add_argument(
        "--opposition-frame-loss",
        action="store_true",
        help=(
            "For fixed thumb/index patches, align their midpoint and axis "
            "instead of forcing the current rigid hand to close both contacts."
        ),
    )
    parser.add_argument(
        "--auto-opposition-loss",
        action="store_true",
        help=(
            "Use the first valid automatic opposition pair stored in the "
            "fixed-patch archive; fall back to ordinary multiregion contact "
            "when no pair is available."
        ),
    )
    parser.add_argument(
        "--opposition-auxiliary-contact-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--opposition-axis-scale-mm",
        type=float,
        default=20.0,
        help="Metric scale assigned to opposition-axis angular error.",
    )
    parser.add_argument(
        "--opposition-vertex-topk",
        type=int,
        default=0,
        help=(
            "Per opposition region, use only this many highest-probability "
            "HACO vertices to form the hand contact center; 0 uses all."
        ),
    )
    parser.add_argument("--mano-data-dir")
    parser.add_argument("--fixed-patch-npz")
    parser.add_argument(
        "--fixed-region-selection",
        choices=("auto", "translation_consistent", "stable", "selected"),
        default="auto",
        help=(
            "Choose which region-name set to consume from the fixed-patch "
            "archive. Auto prefers translation-consistent regions."
        ),
    )
    parser.add_argument(
        "--fixed-contact-source",
        choices=("patch", "candidate_surface"),
        default="patch",
        help=(
            "Use geodesic patch vertices or the full HACO-aligned object "
            "candidate surface from --fixed-patch-npz"
        ),
    )
    parser.add_argument("--region-balanced-contact", action="store_true")
    parser.add_argument("--contact-region-min-vertices", type=int, default=3)
    parser.add_argument("--contact-target-mm", type=float, default=6.0)
    parser.add_argument("--correction-stop-mm", type=float, default=10.0)
    parser.add_argument("--correction-full-mm", type=float, default=18.0)
    parser.add_argument("--collision-margin-mm", type=float, default=0.5)
    parser.add_argument("--collision-stop-count", type=int, default=10)
    parser.add_argument(
        "--phase-average-inside-limit",
        type=float,
        help=(
            "Select the best whole-sequence contact state whose mean number "
            "of inside object vertices over phase-active frames stays within "
            "this budget. Collision loss may remain disabled."
        ),
    )
    parser.add_argument("--correspondence-topk", type=int, default=8)
    parser.add_argument("--containment-refresh", type=int, default=25)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--frame-chunk", type=int, default=4)
    parser.add_argument("--max-translation-mm", type=float, default=20.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--translation-only",
        action="store_true",
        help="Optimize camera-space translation while keeping rotation exactly zero.",
    )
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-collision", type=float, default=5.0)
    parser.add_argument("--w-object-normal-pushout", type=float, default=0.0)
    parser.add_argument("--w-translation-anchor", type=float, default=0.1)
    parser.add_argument("--w-rotation-anchor", type=float, default=0.01)
    parser.add_argument("--w-translation-velocity", type=float, default=1.0)
    parser.add_argument("--w-translation-acceleration", type=float, default=2.0)
    parser.add_argument("--w-rotation-velocity", type=float, default=0.01)
    parser.add_argument("--w-rotation-acceleration", type=float, default=0.02)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--penetration-trust-mm", type=float, default=20.0)
    parser.add_argument("--containment-best-state", action="store_true")
    parser.add_argument(
        "--fixed-region-reduction",
        choices=("mean", "max"),
        default="mean",
        help="Combine per-frame fixed-region contact losses by mean or bottleneck max",
    )
    parser.add_argument(
        "--best-state-selection",
        choices=("inside", "contact_feasible"),
        default="inside",
    )
    parser.add_argument("--best-state-inside-allowance", type=int, default=0)
    parser.add_argument("--use-oracle-gate", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def mano_contact_region_ids(
    mano_data_dir: str, hand_side: str
) -> tuple[np.ndarray, list[str]]:
    model_name = "MANO_LEFT.pkl" if hand_side == "left" else "MANO_RIGHT.pkl"
    model_path = Path(mano_data_dir).expanduser().resolve() / model_name
    with model_path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    weights = raw.get("weights", raw.get("lbs_weights"))
    if weights is None:
        raise KeyError(f"No MANO skinning weights in {model_path}")
    if hasattr(weights, "toarray"):
        weights = weights.toarray()
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape[0] != 778 or weights.shape[1] < 16:
        raise ValueError(f"Unexpected MANO weights shape: {weights.shape}")

    names = list(MANO_CONTACT_REGIONS)
    scores = np.stack(
        [weights[:, joints].sum(axis=1) for joints in MANO_CONTACT_REGIONS.values()],
        axis=-1,
    )
    return np.argmax(scores, axis=-1).astype(np.int64), names


def initial_hand_vertices_key(
    data: dict[str, np.ndarray], requested: str | None
) -> str:
    if requested:
        if requested not in data:
            raise KeyError(f"Initial hand archive lacks {requested!r}")
        return requested
    for key in (
        "refined_hand_vertices_camera",
        "stage1_hand_vertices_camera",
        "initial_hand_vertices_camera",
    ):
        if key in data:
            return key
    raise KeyError("Could not find camera-space vertices in initial hand archive")


@torch.no_grad()
def contact_region_distance_median_mm(
    hand: torch.Tensor,
    object_points: torch.Tensor,
    contact_mask: np.ndarray,
    region_ids: np.ndarray,
    region_count: int,
    frame_chunk: int,
) -> np.ndarray:
    output = np.full((len(hand), region_count), np.nan, dtype=np.float32)
    for start in range(0, len(hand), frame_chunk):
        stop = min(start + frame_chunk, len(hand))
        distance = torch.cdist(hand[start:stop], object_points[start:stop]).min(
            dim=-1
        ).values.cpu().numpy()
        for local, frame in enumerate(range(start, stop)):
            for region in range(region_count):
                selected = contact_mask[frame] & (region_ids == region)
                if selected.any():
                    output[frame, region] = np.median(distance[local, selected]) * 1000.0
    return output


@torch.no_grad()
def fixed_region_bottleneck_distance_mm(
    hand: np.ndarray,
    contact_mask: np.ndarray,
    contact_probability: np.ndarray,
    region_ids: np.ndarray,
    region_names: list[str],
    fixed_regions: dict[str, np.ndarray],
    minimum_vertices: int,
    patch_topk: int,
    vertex_topk: int,
    probability_threshold: float,
    probability_power: float,
    weight_floor: float,
    selection_sigma_mm: float,
) -> np.ndarray:
    """Maximum probability-aware soft-top-k region gap for every frame."""
    output = np.full(len(hand), np.nan, dtype=np.float32)
    for frame in range(len(hand)):
        gaps: list[float] = []
        for name, targets in fixed_regions.items():
            region = region_names.index(name)
            selected = contact_mask[frame] & (region_ids == region)
            if int(selected.sum()) < minimum_vertices or not len(targets[frame]):
                continue
            points = hand[frame, selected]
            pairwise = np.linalg.norm(
                points[:, None] - targets[frame][None], axis=-1
            )
            patch_count = min(max(1, patch_topk), pairwise.shape[-1])
            nearest = np.partition(
                pairwise, patch_count - 1, axis=-1
            )[:, :patch_count]
            distance_mm = np.sqrt(np.mean(nearest ** 2, axis=-1)) * 1000.0
            normalized = np.clip(
                (contact_probability[frame, selected] - probability_threshold)
                / max(1.0 - probability_threshold, 1e-6),
                0.0,
                1.0,
            )
            weights = weight_floor + (1.0 - weight_floor) * (
                normalized ** probability_power
            )
            selection_score = distance_mm - selection_sigma_mm * np.log(
                np.maximum(weights, 1e-6)
            )
            count = min(max(1, vertex_topk), len(distance_mm))
            chosen = np.argpartition(selection_score, count - 1)[:count]
            gap = np.sqrt(np.average(
                distance_mm[chosen] ** 2,
                weights=weights[chosen],
            ))
            gaps.append(float(gap))
        if gaps:
            output[frame] = max(gaps)
    return output


@torch.no_grad()
def opposition_frame_error_mm(
    hand: np.ndarray,
    contact_mask: np.ndarray,
    contact_probability: np.ndarray,
    region_ids: np.ndarray,
    region_names: list[str],
    fixed_regions: dict[str, np.ndarray],
    minimum_vertices: int,
    probability_threshold: float,
    probability_power: float,
    weight_floor: float,
    axis_scale_mm: float,
    opposition_pair: tuple[str, str] = ("thumb", "index"),
    vertex_topk: int = 0,
) -> np.ndarray:
    """Translation-compatible contact-pair midpoint and axis error."""
    output = np.full(len(hand), np.nan, dtype=np.float32)
    first_name, second_name = opposition_pair
    if first_name not in fixed_regions or second_name not in fixed_regions:
        return output
    first_region = region_names.index(first_name)
    second_region = region_names.index(second_name)
    for frame in range(len(hand)):
        centers: dict[str, np.ndarray] = {}
        valid_frame = True
        for name, region in (
            (first_name, first_region),
            (second_name, second_region),
        ):
            selected = contact_mask[frame] & (region_ids == region)
            if int(selected.sum()) < minimum_vertices:
                valid_frame = False
                break
            normalized = np.clip(
                (contact_probability[frame, selected] - probability_threshold)
                / max(1.0 - probability_threshold, 1e-6),
                0.0,
                1.0,
            )
            weights = weight_floor + (1.0 - weight_floor) * (
                normalized ** probability_power
            )
            points = hand[frame, selected]
            if vertex_topk > 0 and len(weights) > vertex_topk:
                chosen = np.argpartition(weights, -vertex_topk)[-vertex_topk:]
                points = points[chosen]
                weights = weights[chosen]
            centers[name] = np.average(
                points, axis=0, weights=weights
            )
        if not valid_frame:
            continue
        hand_axis = centers[second_name] - centers[first_name]
        target_first = fixed_regions[first_name][frame].mean(axis=0)
        target_second = fixed_regions[second_name][frame].mean(axis=0)
        target_axis = target_second - target_first
        hand_norm = float(np.linalg.norm(hand_axis))
        target_norm = float(np.linalg.norm(target_axis))
        if hand_norm <= 1e-8 or target_norm <= 1e-8:
            continue
        cosine = float(
            np.dot(hand_axis, target_axis) / (hand_norm * target_norm)
        )
        midpoint = 0.5 * (centers[first_name] + centers[second_name])
        target_midpoint = 0.5 * (target_first + target_second)
        midpoint_mm = float(np.linalg.norm(midpoint - target_midpoint) * 1000.0)
        axis_error = max(1.0 - np.clip(cosine, -1.0, 1.0), 0.0)
        output[frame] = np.sqrt(
            midpoint_mm ** 2 + axis_scale_mm ** 2 * axis_error
        )
    return output


def opposition_frame_loss(
    hand: torch.Tensor,
    contact_weight: torch.Tensor,
    contact_region_mask: torch.Tensor,
    contact_region_names: list[str],
    fixed_regions: dict[str, torch.Tensor],
    frame_indices: torch.Tensor,
    minimum_vertices: int,
    axis_scale_mm: float,
    opposition_pair: tuple[str, str] = ("thumb", "index"),
    vertex_topk: int = 0,
) -> torch.Tensor:
    centers: dict[str, torch.Tensor] = {}
    active: dict[str, torch.Tensor] = {}
    first_name, second_name = opposition_pair
    for name in opposition_pair:
        region = contact_region_names.index(name)
        selected = contact_region_mask[region]
        weights = contact_weight[:, selected]
        points = hand[:, selected]
        if vertex_topk > 0 and weights.shape[-1] > vertex_topk:
            chosen = torch.topk(
                weights, vertex_topk, dim=-1, largest=True
            ).indices
            weights = torch.gather(weights, -1, chosen)
            points = torch.gather(
                points,
                1,
                chosen[..., None].expand(-1, -1, points.shape[-1]),
            )
        centers[name] = (
            points * weights[..., None]
        ).sum(dim=1) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        active[name] = (
            (weights > 0).sum(dim=-1) >= minimum_vertices
        )
    valid = active[first_name] & active[second_name]
    if not valid.any():
        return torch.zeros((), device=hand.device, dtype=hand.dtype)
    target_first = fixed_regions[first_name][frame_indices].mean(dim=1)
    target_second = fixed_regions[second_name][frame_indices].mean(dim=1)
    hand_midpoint = 0.5 * (centers[first_name] + centers[second_name])
    target_midpoint = 0.5 * (target_first + target_second)
    midpoint_error = (hand_midpoint - target_midpoint).square().sum(dim=-1)
    hand_axis = functional.normalize(
        centers[second_name] - centers[first_name], dim=-1
    )
    target_axis = functional.normalize(target_second - target_first, dim=-1)
    axis_error = torch.clamp(
        1.0 - (hand_axis * target_axis).sum(dim=-1), min=0.0
    )
    axis_scale = axis_scale_mm / 1000.0
    return (
        midpoint_error[valid] + axis_scale * axis_scale * axis_error[valid]
    ).mean()


def fixed_region_contact_loss(
    hand: torch.Tensor,
    contact_weight: torch.Tensor,
    contact_region_mask: torch.Tensor,
    contact_region_names: list[str],
    fixed_regions: dict[str, torch.Tensor],
    frame_indices: torch.Tensor,
    minimum_vertices: int,
    patch_topk: int,
    vertex_topk: int,
    contact_target: float,
    selection_sigma: float,
    reduction: str,
    excluded_regions: set[str] | None = None,
) -> torch.Tensor:
    frame_losses = []
    frame_active = []
    excluded = excluded_regions or set()
    for region_name, region_patch in fixed_regions.items():
        if region_name in excluded:
            continue
        region_index = contact_region_names.index(region_name)
        selected = contact_region_mask[region_index]
        if not selected.any():
            continue
        distances = torch.cdist(hand[:, selected], region_patch[frame_indices])
        nearest_count = min(patch_topk, distances.shape[-1])
        nearest = torch.topk(
            distances, nearest_count, dim=-1, largest=False
        ).values
        effective_distance = torch.sqrt(
            nearest.square().mean(dim=-1) + 1e-12
        )
        error = torch.clamp(
            effective_distance - contact_target, min=0.0
        ).square()
        weights = contact_weight[:, selected]
        selection_score = (
            effective_distance
            - selection_sigma * torch.log(weights.clamp_min(1e-6))
        ).masked_fill(weights <= 0, torch.inf)
        selected_count = min(max(1, vertex_topk), selection_score.shape[-1])
        selected_vertices = torch.topk(
            selection_score, selected_count, dim=-1, largest=False
        ).indices
        selected_error = torch.gather(error, -1, selected_vertices)
        selected_weights = torch.gather(weights, -1, selected_vertices)
        region_weights = selected_weights.sum(dim=-1)
        frame_losses.append(
            (selected_error * selected_weights).sum(dim=-1)
            / region_weights.clamp_min(1e-6)
        )
        frame_active.append(
            (selected_weights > 0).sum(dim=-1) >= minimum_vertices
        )
    if not frame_losses:
        return torch.zeros((), device=hand.device, dtype=hand.dtype)
    losses = torch.stack(frame_losses, dim=-1)
    active = torch.stack(frame_active, dim=-1)
    if reduction == "max":
        reduced = losses.masked_fill(~active, -torch.inf).max(dim=-1).values
    else:
        reduced = (losses * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1)
    valid = active.any(dim=-1)
    return (
        reduced[valid].mean()
        if valid.any()
        else torch.zeros((), device=hand.device, dtype=hand.dtype)
    )


@torch.no_grad()
def collision_correspondences(
    hand: torch.Tensor,
    object_vertices: np.ndarray,
    object_normals: np.ndarray,
    inside_mask: np.ndarray,
    faces: torch.Tensor,
    topk: int,
) -> tuple[
    list[torch.Tensor | None],
    list[torch.Tensor | None],
    list[torch.Tensor | None],
    list[torch.Tensor | None],
]:
    points: list[torch.Tensor | None] = [None] * len(hand)
    normals: list[torch.Tensor | None] = [None] * len(hand)
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
        normals[index] = torch.from_numpy(
            object_normals[index, inside_mask[index]]
        ).to(device=hand.device, dtype=hand.dtype)
        face_indices[index] = selected_faces
        barycentric[index] = selected_barycentric
    return points, normals, face_indices, barycentric


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
    initial_hand_source = "v14_trajectory"
    initial_vertices_key = "query_root_relative_plus_predicted_wrist"
    if args.initial_hand_npz:
        initial_hand = load_npz(
            Path(args.initial_hand_npz).expanduser().resolve()
        )
        initial_indices = aligned_indices(initial_hand["frame_ids"], ids)
        initial_vertices_key = initial_hand_vertices_key(
            initial_hand, args.initial_hand_vertices_key
        )
        vertices_np = np.asarray(
            initial_hand[initial_vertices_key][initial_indices],
            dtype=np.float32,
        )
        if vertices_np.shape != (frame_count, 778, 3):
            raise ValueError(
                f"Initial hand shape mismatch: {vertices_np.shape}"
            )
        if "refined_wrist_camera" in initial_hand:
            wrist_np = np.asarray(
                initial_hand["refined_wrist_camera"][initial_indices],
                dtype=np.float32,
            )
        elif "translation_camera" in initial_hand:
            wrist_np = wrist_np + np.asarray(
                initial_hand["translation_camera"][initial_indices],
                dtype=np.float32,
            )
        initial_hand_source = str(
            Path(args.initial_hand_npz).expanduser().resolve()
        )
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
        & np.isfinite(vertices_np).all(axis=(1, 2))
    )
    phase_gate_np *= valid_np.astype(np.float32)

    mesh = load_mesh(Path(args.object_mesh).expanduser().resolve(), args.object_scale)
    normalized_left = bool(np.asarray(supervision.get("normalized_left", False)).item())
    (
        object_vertices_np,
        object_vertex_normals_np,
        object_points_np,
        object_normals_np,
    ) = build_object_geometry(
        mesh, supervision, supervision_indices, normalized_left, args.object_samples
    )
    fixed_region_np: dict[str, np.ndarray] = {}
    fixed_patch_source = None
    fixed_region_selection_key = None
    automatic_opposition_pair: tuple[str, str] | None = None
    fixed_patch: dict[str, np.ndarray] = {}
    if args.fixed_patch_npz:
        fixed_patch = load_npz(
            Path(args.fixed_patch_npz).expanduser().resolve()
        )
        if args.fixed_region_selection == "translation_consistent":
            if (
                "translation_consistent_region_names" not in fixed_patch
                or not len(fixed_patch["translation_consistent_region_names"])
            ):
                raise RuntimeError(
                    "Fixed patch archive has no translation-consistent regions"
                )
            fixed_region_selection_key = "translation_consistent_region_names"
            fixed_region_names = [
                str(value)
                for value in fixed_patch["translation_consistent_region_names"]
            ]
        elif args.fixed_region_selection == "stable":
            if (
                "stable_region_names" not in fixed_patch
                or not len(fixed_patch["stable_region_names"])
            ):
                raise RuntimeError("Fixed patch archive has no stable regions")
            fixed_region_selection_key = "stable_region_names"
            fixed_region_names = [
                str(value) for value in fixed_patch["stable_region_names"]
            ]
        elif args.fixed_region_selection == "selected":
            if (
                "selected_region_names" not in fixed_patch
                or not len(fixed_patch["selected_region_names"])
            ):
                raise RuntimeError("Fixed patch archive has no selected regions")
            fixed_region_selection_key = "selected_region_names"
            fixed_region_names = [
                str(value) for value in fixed_patch["selected_region_names"]
            ]
        elif (
            "translation_consistent_region_names" in fixed_patch
            and len(fixed_patch["translation_consistent_region_names"])
        ):
            fixed_region_selection_key = "translation_consistent_region_names"
            fixed_region_names = [
                str(value)
                for value in fixed_patch["translation_consistent_region_names"]
            ]
        elif (
            args.fixed_contact_source == "candidate_surface"
            and "selected_region_names" in fixed_patch
        ):
            fixed_region_selection_key = "selected_region_names"
            fixed_region_names = [
                str(value) for value in fixed_patch["selected_region_names"]
            ]
        elif "stable_region_names" in fixed_patch:
            fixed_region_selection_key = "stable_region_names"
            fixed_region_names = [
                str(value) for value in fixed_patch["stable_region_names"]
            ]
            if not fixed_region_names:
                raise RuntimeError(
                    "Fixed patch archive has no consensus-stable regions"
                )
        elif "selected_region_names" in fixed_patch:
            fixed_region_selection_key = "selected_region_names"
            fixed_region_names = [
                str(value) for value in fixed_patch["selected_region_names"]
            ]
        else:
            fixed_region_selection_key = "implicit_thumb_index"
            fixed_region_names = [
                name for name in ("thumb", "index")
                if f"{name}_patch_vertices_canonical" in fixed_patch
            ]
        if not fixed_region_names:
            raise KeyError(
                "Fixed patch archive has no selected region patches"
            )
        for region_name in fixed_region_names:
            key = (
                f"{region_name}_candidate_vertex_ids"
                if args.fixed_contact_source == "candidate_surface"
                else f"{region_name}_patch_vertices_canonical"
            )
            if key not in fixed_patch:
                raise KeyError(f"Fixed patch archive lacks {key!r}")
            if args.fixed_contact_source == "candidate_surface":
                candidate_ids = np.asarray(fixed_patch[key], dtype=np.int64)
                canonical = np.asarray(mesh.vertices, dtype=np.float32)[
                    candidate_ids
                ]
            else:
                canonical = np.asarray(fixed_patch[key], dtype=np.float32)
            camera = np.empty(
                (frame_count, len(canonical), 3), dtype=np.float32
            )
            for output_index, supervision_index in enumerate(supervision_indices):
                pose = physical_pose(
                    supervision["gt_ycb_object_pose"][supervision_index],
                    normalized_left,
                )
                camera[output_index] = (
                    canonical @ pose[:3, :3].T + pose[:3, 3]
                )
            fixed_region_np[region_name] = camera
        fixed_patch_source = str(
            Path(args.fixed_patch_npz).expanduser().resolve()
        )
        if args.auto_opposition_loss:
            for pair in fixed_patch.get(
                "automatic_opposition_region_pairs", np.empty((0, 2))
            ):
                candidate = (str(pair[0]), str(pair[1]))
                if set(candidate).issubset(fixed_region_np):
                    automatic_opposition_pair = candidate
                    break
    if args.opposition_frame_loss and not {
        "thumb", "index"
    }.issubset(fixed_region_np):
        raise ValueError(
            "--opposition-frame-loss requires fixed thumb and index patches"
        )
    opposition_pair = (
        automatic_opposition_pair
        if automatic_opposition_pair is not None
        else ("thumb", "index")
    )
    use_opposition_loss = bool(
        args.opposition_frame_loss or automatic_opposition_pair is not None
    )
    boundary = directed_boundary_loop(faces_np)

    vertices = torch.from_numpy(vertices_np).to(device)
    wrists = torch.from_numpy(wrist_np).to(device)
    object_points = torch.from_numpy(object_points_np).to(device)
    object_normals = torch.from_numpy(object_normals_np).to(device)
    fixed_regions = {
        name: torch.from_numpy(values).to(device)
        for name, values in fixed_region_np.items()
    }
    faces = torch.from_numpy(faces_np).to(device)
    contact_mask = torch.from_numpy(contact_mask_np).to(device)
    probability = torch.from_numpy(probability_np).to(device)
    phase_gate = torch.from_numpy(phase_gate_np).to(device)

    initial_metrics, initial_per_frame = audit_geometry(
        vertices, contact_mask, object_points, object_normals, phase_gate,
        args.penetration_tolerance_mm, args.penetration_trust_mm, args.frame_chunk,
    )

    contact_region_names: list[str] = []
    contact_region_ids_np = np.full(vertices_np.shape[1], -1, dtype=np.int64)
    contact_region_count_np = np.zeros((frame_count, 0), dtype=np.int32)
    contact_region_active_np = np.zeros((frame_count, 0), dtype=bool)
    initial_region_distance = np.empty((frame_count, 0), dtype=np.float32)
    correction_distance_np = initial_per_frame["contact_distance_median_mm"].copy()
    contact_region_mask_np = None
    if args.region_balanced_contact:
        if not args.mano_data_dir:
            raise ValueError("--region-balanced-contact requires --mano-data-dir")
        contact_region_ids_np, contact_region_names = mano_contact_region_ids(
            args.mano_data_dir, str(query["hand_side"].item()).lower()
        )
        contact_region_mask_np = np.stack(
            [contact_region_ids_np == index for index in range(len(contact_region_names))]
        )
        contact_region_count_np = np.stack(
            [
                (contact_mask_np & contact_region_mask_np[index][None]).sum(axis=1)
                for index in range(len(contact_region_names))
            ],
            axis=-1,
        ).astype(np.int32)
        contact_region_active_np = (
            (contact_region_count_np >= args.contact_region_min_vertices)
            & (phase_gate_np[:, None] > 0)
        )
        if not contact_region_active_np.any():
            raise RuntimeError("No HACO contact region passed the minimum vertex count")
        initial_region_distance = contact_region_distance_median_mm(
            vertices, object_points, contact_mask_np, contact_region_ids_np,
            len(contact_region_names), args.frame_chunk,
        )
        # A well-aligned palm must not hide a missed thumb or index pinch.
        # Contact-only correction is gated by the worst HACO-active region.
        for frame in range(frame_count):
            selected = initial_region_distance[frame, contact_region_active_np[frame]]
            selected = selected[np.isfinite(selected)]
            if len(selected):
                correction_distance_np[frame] = selected.max()

    contact_gate_np = correction_gate_from_contact_distance(
        correction_distance_np, phase_gate_np,
        args.correction_stop_mm, args.correction_full_mm,
    )
    if args.fixed_patch_npz:
        if not args.region_balanced_contact:
            raise ValueError("--fixed-patch-npz requires --region-balanced-contact")
        unknown_regions = sorted(
            set(fixed_regions).difference(contact_region_names)
        )
        if unknown_regions:
            raise KeyError(
                f"Fixed patch regions are unknown to MANO: {unknown_regions}"
            )
        # The patch was selected independently of the old global-distance
        # gate. Every phase-active frame must be allowed to reach it.
        contact_gate_np = phase_gate_np.copy()
    contact_gate = torch.from_numpy(contact_gate_np).to(device)
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    confidence = torch.clamp(
        (probability - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    ).pow(args.contact_probability_power)
    # Keep every phase-active HACO contact as a one-sided safety tether.  The
    # correction gate decides whether contact alone may move a frame, but it
    # must not disable contact preservation when collision push-out is active.
    contact_base_weight = (
        args.contact_weight_floor + (1.0 - args.contact_weight_floor) * confidence
    ) * contact_mask
    if args.fixed_patch_npz:
        fixed_region_indices = [
            contact_region_names.index(name) for name in fixed_regions
        ]
        patch_region = torch.from_numpy(np.isin(
            contact_region_ids_np,
            fixed_region_indices,
        )).to(device=device, dtype=contact_base_weight.dtype)
        contact_base_weight = contact_base_weight * patch_region[None]
    contact_weight = contact_base_weight * phase_gate[:, None]

    contact_region_mask = None
    contact_region_active = None
    total_contact_region_scale = None
    if args.region_balanced_contact:
        contact_region_mask = torch.from_numpy(contact_region_mask_np).to(device)
        contact_region_active = torch.from_numpy(contact_region_active_np).to(device)

    translation = torch.zeros((frame_count, 3), device=device, requires_grad=True)
    angles = torch.zeros(
        (frame_count, 3),
        device=device,
        requires_grad=not args.translation_only,
    )
    optimized_parameters = [translation]
    if not args.translation_only:
        optimized_parameters.append(angles)
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.lr)
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
    collision_object_normals: list[torch.Tensor | None] = [None] * frame_count
    collision_faces: list[torch.Tensor | None] = [None] * frame_count
    collision_barycentric: list[torch.Tensor | None] = [None] * frame_count
    correction_support_np = np.zeros(frame_count, dtype=bool)
    best_inside_count = np.full(frame_count, np.iinfo(np.int32).max, dtype=np.int32)
    baseline_inside_count: np.ndarray | None = None
    best_contact_gap = np.full(frame_count, np.inf, dtype=np.float32)
    best_translation = torch.zeros((frame_count, 3), device=device)
    best_angles = torch.zeros((frame_count, 3), device=device)
    phase_budget_mask = valid_np & (phase_gate_np > 0)
    if args.phase_average_inside_limit is not None and not phase_budget_mask.any():
        raise RuntimeError(
            "--phase-average-inside-limit requires phase-active valid frames"
        )
    if args.phase_average_inside_limit is not None and not fixed_region_np:
        raise RuntimeError(
            "--phase-average-inside-limit requires --fixed-patch-npz"
        )
    best_phase_budget_score = math.inf
    best_phase_budget_average = math.inf
    best_phase_budget_translation: torch.Tensor | None = None
    best_phase_budget_angles: torch.Tensor | None = None

    def state_contact_gap(hand_np: np.ndarray) -> np.ndarray:
        if use_opposition_loss:
            gap = opposition_frame_error_mm(
                hand_np,
                contact_mask_np,
                probability_np,
                contact_region_ids_np,
                contact_region_names,
                fixed_region_np,
                args.contact_region_min_vertices,
                threshold,
                args.contact_probability_power,
                args.contact_weight_floor,
                args.opposition_axis_scale_mm,
                opposition_pair,
                args.opposition_vertex_topk,
            )
            if args.opposition_auxiliary_contact_weight > 0:
                auxiliary = fixed_region_bottleneck_distance_mm(
                    hand_np,
                    contact_mask_np,
                    probability_np,
                    contact_region_ids_np,
                    contact_region_names,
                    fixed_region_np,
                    args.contact_region_min_vertices,
                    args.topk,
                    args.fixed_contact_vertex_topk,
                    threshold,
                    args.contact_probability_power,
                    args.contact_weight_floor,
                    args.fixed_contact_selection_sigma_mm,
                )
                valid = np.isfinite(gap) & np.isfinite(auxiliary)
                gap[valid] += (
                    args.opposition_auxiliary_contact_weight * auxiliary[valid]
                )
            return gap
        return fixed_region_bottleneck_distance_mm(
            hand_np,
            contact_mask_np,
            probability_np,
            contact_region_ids_np,
            contact_region_names,
            fixed_region_np,
            args.contact_region_min_vertices,
            args.topk,
            args.fixed_contact_vertex_topk,
            threshold,
            args.contact_probability_power,
            args.contact_weight_floor,
            args.fixed_contact_selection_sigma_mm,
        )

    def update_phase_budget_state(
        hand_np: np.ndarray,
        current_inside_count: np.ndarray,
    ) -> None:
        nonlocal best_phase_budget_score
        nonlocal best_phase_budget_average
        nonlocal best_phase_budget_translation
        nonlocal best_phase_budget_angles
        if args.phase_average_inside_limit is None:
            return
        average_inside = float(current_inside_count[phase_budget_mask].mean())
        if average_inside > args.phase_average_inside_limit:
            return
        current_gap = state_contact_gap(hand_np)
        evaluated = phase_budget_mask & np.isfinite(current_gap)
        if not evaluated.any():
            return
        score = float(current_gap[evaluated].mean())
        if score >= best_phase_budget_score:
            return
        best_phase_budget_score = score
        best_phase_budget_average = average_inside
        best_phase_budget_translation = translation.detach().clone()
        best_phase_budget_angles = angles.detach().clone()

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
            if baseline_inside_count is None:
                baseline_inside_count = inside_count.copy()
            update_phase_budget_state(current_np, inside_count)
            if args.best_state_selection == "contact_feasible":
                if not fixed_region_np:
                    raise ValueError(
                        "contact_feasible best-state selection requires "
                        "--fixed-patch-npz"
                    )
                current_contact_gap = state_contact_gap(current_np)
                feasible = inside_count <= (
                    baseline_inside_count + args.best_state_inside_allowance
                )
                improved_state = (
                    feasible
                    & np.isfinite(current_contact_gap)
                    & (current_contact_gap < best_contact_gap)
                )
                best_contact_gap[improved_state] = current_contact_gap[
                    improved_state
                ]
            else:
                improved_state = inside_count <= best_inside_count
            if improved_state.any():
                improved_tensor = torch.from_numpy(improved_state).to(device)
                best_inside_count[improved_state] = inside_count[improved_state]
                best_translation[improved_tensor] = translation.detach()[improved_tensor]
                best_angles[improved_tensor] = angles.detach()[improved_tensor]
            (
                collision_points,
                collision_object_normals,
                collision_faces,
                collision_barycentric,
            ) = (
                collision_correspondences(
                    current,
                    object_vertices_np,
                    object_vertex_normals_np,
                    inside_mask,
                    faces,
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
                "phase_average_inside": (
                    float(inside_count[phase_budget_mask].mean())
                    if phase_budget_mask.any()
                    else None
                ),
            }
            refresh_history.append(refresh)
            print({"containment_refresh": refresh}, flush=True)

        collision_active_np = valid_np & (inside_count > args.collision_stop_count)
        optimization_active_np = (contact_gate_np > 0) | collision_active_np
        if not optimization_active_np.any():
            raise RuntimeError("Joint contact/containment gate selected no frames")
        # Once a constrained frame has received a correction, retain it in the
        # support even after its collision is resolved. Frames that never had
        # contact or collision must remain exact no-ops instead of being moved
        # indirectly by temporal regularization.
        correction_support_np |= optimization_active_np
        active_indices_np = np.flatnonzero(optimization_active_np)
        active_indices = torch.from_numpy(active_indices_np).to(device)
        optimization_active = torch.from_numpy(optimization_active_np).to(device)
        correction_support = torch.from_numpy(correction_support_np).to(device)
        total_contact_weight = contact_weight[optimization_active].sum().clamp_min(1e-6)
        if args.region_balanced_contact:
            total_contact_region_scale = (
                contact_region_active[optimization_active]
                * phase_gate[optimization_active, None]
            ).sum().clamp_min(1e-6)
        total_collision_points = max(
            1, int(inside_count[collision_active_np].sum())
        )

        optimizer.zero_grad(set_to_none=True)
        contact_value = 0.0
        collision_value = 0.0
        object_normal_pushout_value = 0.0
        for start in range(0, len(active_indices_np), args.frame_chunk):
            indices = active_indices[start:start + args.frame_chunk]
            refined = transform_batch(
                vertices[indices], wrists[indices], translation[indices], angles[indices]
            )
            if args.fixed_patch_npz and use_opposition_loss:
                chunk_contact = opposition_frame_loss(
                    refined,
                    contact_weight[indices],
                    contact_region_mask,
                    contact_region_names,
                    fixed_regions,
                    indices,
                    args.contact_region_min_vertices,
                    args.opposition_axis_scale_mm,
                    opposition_pair,
                    args.opposition_vertex_topk,
                )
                if args.opposition_auxiliary_contact_weight > 0:
                    auxiliary_contact = fixed_region_contact_loss(
                        refined,
                        contact_weight[indices],
                        contact_region_mask,
                        contact_region_names,
                        fixed_regions,
                        indices,
                        args.contact_region_min_vertices,
                        topk,
                        args.fixed_contact_vertex_topk,
                        contact_target,
                        args.fixed_contact_selection_sigma_mm / 1000.0,
                        args.fixed_region_reduction,
                        set(opposition_pair),
                    )
                    chunk_contact = (
                        chunk_contact
                        + args.opposition_auxiliary_contact_weight
                        * auxiliary_contact
                    )
            elif args.fixed_patch_npz:
                region_frame_losses = []
                region_frame_active = []
                for region_name, region_patch in fixed_regions.items():
                    region_index = contact_region_names.index(region_name)
                    selected = contact_region_mask[region_index]
                    if not selected.any():
                        continue
                    distances = torch.cdist(
                        refined[:, selected], region_patch[indices]
                    )
                    patch_topk = min(topk, distances.shape[-1])
                    nearest = torch.topk(
                        distances, patch_topk, dim=-1, largest=False
                    ).values
                    effective_distance = torch.sqrt(
                        nearest.square().mean(dim=-1) + 1e-12
                    )
                    error = torch.clamp(
                        effective_distance - contact_target, min=0.0
                    ).square()
                    weights = contact_weight[indices][:, selected]
                    selection_sigma = (
                        args.fixed_contact_selection_sigma_mm / 1000.0
                    )
                    selection_score = (
                        effective_distance
                        - selection_sigma * torch.log(weights.clamp_min(1e-6))
                    ).masked_fill(weights <= 0, torch.inf)
                    vertex_topk = min(
                        max(1, args.fixed_contact_vertex_topk),
                        selection_score.shape[-1],
                    )
                    selected_vertices = torch.topk(
                        selection_score,
                        vertex_topk,
                        dim=-1,
                        largest=False,
                    ).indices
                    selected_error = torch.gather(
                        error, -1, selected_vertices
                    )
                    selected_weights = torch.gather(
                        weights, -1, selected_vertices
                    )
                    region_weights = selected_weights.sum(dim=-1)
                    region_frame_losses.append(
                        (selected_error * selected_weights).sum(dim=-1)
                        / region_weights.clamp_min(1e-6)
                    )
                    region_frame_active.append(
                        (selected_weights > 0).sum(dim=-1)
                        >= args.contact_region_min_vertices
                    )
                if region_frame_losses:
                    frame_losses = torch.stack(region_frame_losses, dim=-1)
                    frame_active = torch.stack(region_frame_active, dim=-1)
                    if args.fixed_region_reduction == "max":
                        reduced = frame_losses.masked_fill(
                            ~frame_active, -torch.inf
                        ).max(dim=-1).values
                    else:
                        reduced = (
                            frame_losses * frame_active
                        ).sum(dim=-1) / frame_active.sum(dim=-1).clamp_min(1)
                    valid_frame = frame_active.any(dim=-1)
                    chunk_contact = (
                        reduced[valid_frame].mean()
                        if valid_frame.any()
                        else torch.zeros((), device=device)
                    )
                else:
                    chunk_contact = torch.zeros((), device=device)
            else:
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
            if args.fixed_patch_npz:
                pass
            elif args.region_balanced_contact:
                region_weight = (
                    contact_base_weight[indices, None]
                    * contact_region_mask[None]
                )
                region_denominator = region_weight.sum(dim=-1).clamp_min(1e-6)
                region_error = (
                    contact_error[:, None] * region_weight
                ).sum(dim=-1) / region_denominator
                region_scale = (
                    contact_region_active[indices] * phase_gate[indices, None]
                )
                chunk_contact = (
                    region_error * region_scale
                ).sum() / total_contact_region_scale
            else:
                chunk_contact = (
                    contact_error * contact_weight[indices]
                ).sum() / total_contact_weight

            collision_sum = torch.zeros((), device=device)
            object_normal_pushout_sum = torch.zeros((), device=device)
            for local_index, global_tensor in enumerate(indices):
                global_index = int(global_tensor.item())
                if not collision_active_np[global_index]:
                    continue
                points = collision_points[global_index]
                point_normals = collision_object_normals[global_index]
                face_index = collision_faces[global_index]
                barycentric = collision_barycentric[global_index]
                if (
                    points is None
                    or point_normals is None
                    or face_index is None
                    or barycentric is None
                ):
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
                object_clearance = (
                    (surface - points) * point_normals
                ).sum(dim=-1)
                object_normal_pushout_sum = (
                    object_normal_pushout_sum
                    + torch.clamp(
                        collision_margin - object_clearance, min=0.0
                    ).square().sum()
                )
            chunk_collision = collision_sum / total_collision_points
            chunk_object_normal_pushout = (
                object_normal_pushout_sum / total_collision_points
            )
            chunk_loss = (
                args.w_contact * chunk_contact
                + args.w_collision * chunk_collision
                + args.w_object_normal_pushout * chunk_object_normal_pushout
            )
            chunk_loss.backward()
            contact_value += float(chunk_contact.detach())
            collision_value += float(chunk_collision.detach())
            object_normal_pushout_value += float(
                chunk_object_normal_pushout.detach()
            )

        active = correction_support
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
            translation.mul_(torch.clamp(max_translation / translation_norm, max=1.0))
            if args.translation_only:
                angles.zero_()
            else:
                angle_norm = angles.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                angles.mul_(torch.clamp(max_angle / angle_norm, max=1.0))
            translation[~correction_support] = 0
            angles[~correction_support] = 0

        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": (
                    args.w_contact * contact_value
                    + args.w_collision * collision_value
                    + args.w_object_normal_pushout
                    * object_normal_pushout_value
                    + float(regularization.detach())
                ),
                "contact": contact_value,
                "collision": collision_value,
                "object_normal_pushout": object_normal_pushout_value,
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
        final_candidate = transform_batch(vertices, wrists, translation, angles)
    final_candidate_np = final_candidate.cpu().numpy().astype(np.float32)
    candidate_inside_mask, candidate_inside_count = exact_inside_counts(
        final_candidate_np,
        faces_np,
        object_vertices_np,
        boundary,
        device,
        args.point_chunk,
    )
    candidate_inside_mask &= valid_np[:, None]
    candidate_inside_count = candidate_inside_mask.sum(axis=1).astype(np.int32)
    update_phase_budget_state(final_candidate_np, candidate_inside_count)
    if args.best_state_selection == "contact_feasible":
        if baseline_inside_count is None:
            baseline_inside_count = candidate_inside_count.copy()
        candidate_contact_gap = state_contact_gap(final_candidate_np)
        improved_state = (
            (candidate_inside_count <= (
                baseline_inside_count + args.best_state_inside_allowance
            ))
            & np.isfinite(candidate_contact_gap)
            & (candidate_contact_gap < best_contact_gap)
        )
        best_contact_gap[improved_state] = candidate_contact_gap[improved_state]
    else:
        improved_state = candidate_inside_count <= best_inside_count
    if improved_state.any():
        improved_tensor = torch.from_numpy(improved_state).to(device)
        best_inside_count[improved_state] = candidate_inside_count[improved_state]
        best_translation[improved_tensor] = translation.detach()[improved_tensor]
        best_angles[improved_tensor] = angles.detach()[improved_tensor]
    initial_inside_mask, initial_inside_count = exact_inside_counts(
        vertices_np, faces_np, object_vertices_np, boundary, device, args.point_chunk
    )
    initial_inside_mask &= valid_np[:, None]
    initial_inside_count = initial_inside_mask.sum(axis=1).astype(np.int32)
    selected_translation = translation.detach().clone()
    selected_angles = angles.detach().clone()
    if args.phase_average_inside_limit is not None:
        if (
            best_phase_budget_translation is None
            or best_phase_budget_angles is None
        ):
            raise RuntimeError(
                "No optimization state satisfied --phase-average-inside-limit"
            )
        selected_translation = best_phase_budget_translation
        selected_angles = best_phase_budget_angles
    elif args.containment_best_state:
        protected = correction_support_np & valid_np
        protected_tensor = torch.from_numpy(protected).to(device)
        selected_translation[protected_tensor] = best_translation[
            protected_tensor
        ]
        selected_angles[protected_tensor] = best_angles[
            protected_tensor
        ]
    with torch.no_grad():
        refined = transform_batch(
            vertices, wrists, selected_translation, selected_angles
        )
    refined_np = refined.cpu().numpy().astype(np.float32)
    final_inside_mask, final_inside_count = exact_inside_counts(
        refined_np, faces_np, object_vertices_np, boundary, device, args.point_chunk
    )
    final_inside_mask &= valid_np[:, None]
    final_inside_count = final_inside_mask.sum(axis=1).astype(np.int32)
    refined_metrics, refined_per_frame = audit_geometry(
        refined, contact_mask, object_points, object_normals, phase_gate,
        args.penetration_tolerance_mm, args.penetration_trust_mm, args.frame_chunk,
    )
    initial_opposition_error = np.full(frame_count, np.nan, dtype=np.float32)
    refined_opposition_error = np.full(frame_count, np.nan, dtype=np.float32)
    opposition_summary = None
    if use_opposition_loss:
        initial_opposition_error = opposition_frame_error_mm(
            vertices_np, contact_mask_np, probability_np,
            contact_region_ids_np, contact_region_names, fixed_region_np,
            args.contact_region_min_vertices, threshold,
            args.contact_probability_power, args.contact_weight_floor,
            args.opposition_axis_scale_mm,
            opposition_pair,
        )
        refined_opposition_error = opposition_frame_error_mm(
            refined_np, contact_mask_np, probability_np,
            contact_region_ids_np, contact_region_names, fixed_region_np,
            args.contact_region_min_vertices, threshold,
            args.contact_probability_power, args.contact_weight_floor,
            args.opposition_axis_scale_mm,
            opposition_pair,
        )
        evaluated_opposition = (
            np.isfinite(initial_opposition_error)
            & np.isfinite(refined_opposition_error)
        )
        opposition_summary = {
            "initial_error_mm": distribution(
                initial_opposition_error[evaluated_opposition]
            ),
            "refined_error_mm": distribution(
                refined_opposition_error[evaluated_opposition]
            ),
            "improved_frames": int((
                refined_opposition_error[evaluated_opposition]
                < initial_opposition_error[evaluated_opposition]
            ).sum()),
        }
    collision_summary = containment_metrics(initial_inside_count, final_inside_count)
    gt_summary, initial_gt_frame, refined_gt_frame = gt_audit(
        args.gt_hand_npz, query, valid_np, vertices_np, refined_np
    )
    refined_region_distance = np.empty((frame_count, 0), dtype=np.float32)
    region_distance_summary: dict[str, object] = {}
    if args.region_balanced_contact:
        refined_region_distance = contact_region_distance_median_mm(
            refined, object_points, contact_mask_np, contact_region_ids_np,
            len(contact_region_names), args.frame_chunk,
        )
        for index, name in enumerate(contact_region_names):
            evaluated = contact_region_active_np[:, index]
            region_distance_summary[name] = {
                "initial": distribution(initial_region_distance[evaluated, index]),
                "refined": distribution(refined_region_distance[evaluated, index]),
            }

    summary = {
        "method": (
            "v14_opposition_midpoint_axis_rigid_stage1_v1"
            if use_opposition_loss
            else "iterative_multiregion_containment_first_rigid_stage1_v5"
            if args.initial_hand_npz
            and args.fixed_patch_npz
            and args.containment_best_state
            else
            "v14_fixed_multiregion_containment_first_rigid_stage1_v4"
            if args.fixed_patch_npz and args.containment_best_state
            else
            "v14_fixed_multiregion_object_normal_rigid_stage1_v3"
            if args.fixed_patch_npz and args.w_object_normal_pushout > 0
            else "v14_fixed_multiregion_canonical_contact_patch_rigid_stage1_v2"
            if args.fixed_patch_npz
            else "region_balanced_haco_contact_capped_mano_containment_rigid_stage1_v4"
            if args.region_balanced_contact
            else "joint_haco_contact_capped_mano_containment_rigid_stage1_v3"
        ),
        "stream_id": str(query["stream_id"].item()),
        "degrees_of_freedom": (
            "camera_translation_only"
            if args.translation_only
            else "camera_translation_and_rotation"
        ),
        "initial_hand_source": initial_hand_source,
        "initial_hand_vertices_key": initial_vertices_key,
        "frames": frame_count,
        "gate_source": gate_key,
        "contact_active_frames": int((contact_gate_np > 0).sum()),
        "contact_preservation_frames": int((phase_gate_np > 0).sum()),
        "initial_collision_active_frames": int(
            (initial_inside_count > args.collision_stop_count).sum()
        ),
        "correction_support_frames": int(correction_support_np.sum()),
        "contact": {"initial": initial_metrics, "refined": refined_metrics},
        "containment": collision_summary,
        "translation_norm_mm": distribution(
            np.linalg.norm(selected_translation.cpu().numpy(), axis=-1) * 1000.0
        ),
        "rotation_norm_deg": distribution(
            np.linalg.norm(selected_angles.cpu().numpy(), axis=-1)
            * 180.0 / math.pi
        ),
        "weights": {
            "contact": args.w_contact,
            "collision": args.w_collision,
            "object_normal_pushout": args.w_object_normal_pushout,
            "translation_anchor": args.w_translation_anchor,
            "rotation_anchor": args.w_rotation_anchor,
        },
        "contact_aggregation": (
            "opposition_midpoint_axis"
            if use_opposition_loss
            else "fixed_multiregion_canonical_patches"
            if args.fixed_patch_npz
            else "mano_region_balanced" if args.region_balanced_contact else "global_vertex"
        ),
        "fixed_patch_source": fixed_patch_source,
        "fixed_region_selection_key": fixed_region_selection_key,
        "fixed_contact_source": args.fixed_contact_source,
        "fixed_patch_regions": list(fixed_regions),
        "contact_regions": {
            "names": contact_region_names,
            "minimum_vertices": args.contact_region_min_vertices,
            "active_frame_regions": int(contact_region_active_np.sum()),
            "active_frames_by_region": {
                name: int(contact_region_active_np[:, index].sum())
                for index, name in enumerate(contact_region_names)
            },
            "distance_mm": region_distance_summary,
        },
        "containment_refresh": args.containment_refresh,
        "containment_best_state": args.containment_best_state,
        "fixed_region_reduction": args.fixed_region_reduction,
        "fixed_contact_vertex_selection": {
            "vertex_topk": args.fixed_contact_vertex_topk,
            "patch_topk": args.topk,
            "selection_sigma_mm": args.fixed_contact_selection_sigma_mm,
            "probability_power": args.contact_probability_power,
            "weight_floor": args.contact_weight_floor,
        },
        "opposition_frame": {
            "enabled": use_opposition_loss,
            "source": (
                "automatic" if automatic_opposition_pair is not None else "manual"
                if args.opposition_frame_loss else None
            ),
            "pair": list(opposition_pair) if use_opposition_loss else None,
            "auxiliary_contact_weight": args.opposition_auxiliary_contact_weight,
            "axis_scale_mm": args.opposition_axis_scale_mm,
            "vertex_topk": args.opposition_vertex_topk,
            "audit": opposition_summary,
        },
        "best_state_selection": args.best_state_selection,
        "best_state_inside_allowance": args.best_state_inside_allowance,
        "phase_average_inside_budget": {
            "enabled": args.phase_average_inside_limit is not None,
            "limit": args.phase_average_inside_limit,
            "selected_average": (
                float(final_inside_count[phase_budget_mask].mean())
                if phase_budget_mask.any()
                else None
            ),
            "selected_contact_score_mm": (
                best_phase_budget_score
                if math.isfinite(best_phase_budget_score)
                else None
            ),
            "selected_phase_frames": int(phase_budget_mask.sum()),
        },
        "best_fixed_region_bottleneck_gap_mm": distribution(
            best_contact_gap[np.isfinite(best_contact_gap)]
        ),
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
        "contact_region_names": np.asarray(contact_region_names),
        "contact_region_id": contact_region_ids_np.astype(np.int16),
        "contact_region_vertex_count": contact_region_count_np,
        "contact_region_active": contact_region_active_np,
        "initial_contact_region_distance_median_mm": initial_region_distance,
        "refined_contact_region_distance_median_mm": refined_region_distance,
        "contact_gate": phase_gate_np,
        "contact_correction_gate": contact_gate_np,
        "contact_correction_distance_mm": correction_distance_np,
        "translation_camera": selected_translation.cpu().numpy().astype(np.float32),
        "rotation_euler_xyz": selected_angles.cpu().numpy().astype(np.float32),
        "refined_wrist_camera": (
            wrist_np + selected_translation.cpu().numpy()
        ).astype(np.float32),
        "initial_contact_distance_median_mm": initial_per_frame["contact_distance_median_mm"],
        "refined_contact_distance_median_mm": refined_per_frame["contact_distance_median_mm"],
        "initial_opposition_frame_error_mm": initial_opposition_error,
        "refined_opposition_frame_error_mm": refined_opposition_error,
        "initial_object_vertex_inside_capped_mano": initial_inside_mask,
        "refined_object_vertex_inside_capped_mano": final_inside_mask,
        "initial_inside_object_vertices": initial_inside_count,
        "refined_inside_object_vertices": final_inside_count,
        "initial_gt_vertex_error_median_mm": initial_gt_frame,
        "refined_gt_vertex_error_median_mm": refined_gt_frame,
        "stream_id": np.asarray(str(query["stream_id"].item())),
        "initial_hand_source": np.asarray(initial_hand_source),
        "initial_hand_vertices_key": np.asarray(initial_vertices_key),
        **({
            **{
                f"fixed_{name}_patch_vertices_camera": values
                for name, values in fixed_region_np.items()
            },
            "fixed_patch_region_names": np.asarray(list(fixed_region_np)),
            "fixed_patch_source": np.asarray(fixed_patch_source),
        } if args.fixed_patch_npz else {}),
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
