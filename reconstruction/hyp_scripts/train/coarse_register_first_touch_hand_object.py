#!/usr/bin/env python3
"""Coarsely register a hand at first touch without an object or hand SDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

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
    parser.add_argument("--phase-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--gt-hand-npz")
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
    )
    parser.add_argument("--phase-key", default="predicted_contact_gate")
    parser.add_argument("--onset-minimum-gate", type=float, default=0.25)
    parser.add_argument("--onset-pre-frames", type=int, default=2)
    parser.add_argument("--onset-post-frames", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=4096)
    parser.add_argument("--candidate-chunk", type=int, default=128)
    parser.add_argument("--translation-range-mm", type=float, default=80.0)
    parser.add_argument("--rotation-range-deg", type=float, default=12.0)
    parser.add_argument("--contact-topk", type=int, default=8)
    parser.add_argument("--minimum-contact-vertices", type=int, default=3)
    parser.add_argument("--anchor-vertices", type=int, default=64)
    parser.add_argument("--w-reprojection-mm-per-px", type=float, default=0.75)
    parser.add_argument("--w-translation-prior", type=float, default=0.02)
    parser.add_argument("--w-rotation-prior", type=float, default=0.1)
    parser.add_argument("--exact-candidates", type=int, default=48)
    parser.add_argument("--collision-object-samples", type=int, default=2048)
    parser.add_argument("--contact-slack-mm", type=float, default=10.0)
    parser.add_argument("--maximum-reprojection-px", type=float, default=20.0)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = points[..., 2]
    output = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1) & (depth > 1e-6)
    output[..., 0][valid] = (
        intrinsics[0, 0] * points[..., 0][valid] / depth[valid]
        + intrinsics[0, 2]
    )
    output[..., 1][valid] = (
        intrinsics[1, 1] * points[..., 1][valid] / depth[valid]
        + intrinsics[1, 2]
    )
    return output


def load_intrinsics(
    query: dict[str, np.ndarray], values: list[float] | None
) -> np.ndarray:
    if values is not None:
        fx, fy, cx, cy = values
        return np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    for key in ("intrinsics", "camera_intrinsics"):
        if key in query:
            value = np.asarray(query[key], dtype=np.float32)
            if value.shape[-2:] == (3, 3):
                return value.reshape(-1, 3, 3)[0]
    raise KeyError(
        "Query archive lacks a 3x3 intrinsics matrix; pass "
        "--intrinsics FX FY CX CY"
    )


def to_object(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum(
        "fvi,fij->fvj", points - poses[:, None, :3, 3], poses[:, :3, :3]
    )


def points_to_object(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum(
        "fi,fij->fj", points - poses[:, :3, 3], poses[:, :3, :3]
    )


def to_camera(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return (
        np.einsum("...fvi,fji->...fvj", points, poses[:, :3, :3])
        + poses[:, None, :3, 3]
    )


def candidate_hands_object(
    hand_object: np.ndarray,
    wrist_object: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    centered = hand_object - wrist_object[:, None]
    return (
        np.einsum("fvi,cki->cfvk", centered, rotations)
        + wrist_object[None, :, None]
        + translations[:, None, None]
    )


def transform_hands_object_per_frame(
    hand_object: np.ndarray,
    wrist_object: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    centered = hand_object - wrist_object[:, None]
    return (
        np.einsum("fvi,fki->fvk", centered, rotations)
        + wrist_object[:, None]
        + translations[:, None]
    )


def generate_candidates(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    translations = rng.uniform(
        -args.translation_range_mm / 1000.0,
        args.translation_range_mm / 1000.0,
        size=(args.num_candidates, 3),
    ).astype(np.float32)
    rotation_vectors = np.deg2rad(rng.uniform(
        -args.rotation_range_deg,
        args.rotation_range_deg,
        size=(args.num_candidates, 3),
    )).astype(np.float32)
    translations[0] = 0.0
    rotation_vectors[0] = 0.0
    return translations, rotation_vectors


def score_candidates(
    hand_object: np.ndarray,
    wrist_object: np.ndarray,
    poses: np.ndarray,
    base_uv: np.ndarray,
    contact_masks: np.ndarray,
    region_ids: np.ndarray,
    region_count: int,
    object_tree: cKDTree,
    translations: np.ndarray,
    rotation_vectors: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_count = len(translations)
    contact_score = np.full(candidate_count, np.inf, dtype=np.float32)
    reprojection = np.full(candidate_count, np.inf, dtype=np.float32)
    anchor_indices = np.linspace(
        0, hand_object.shape[1] - 1,
        min(args.anchor_vertices, hand_object.shape[1]),
        dtype=np.int64,
    )
    for start in range(0, candidate_count, args.candidate_chunk):
        stop = min(start + args.candidate_chunk, candidate_count)
        rotation = Rotation.from_rotvec(
            rotation_vectors[start:stop]
        ).as_matrix().astype(np.float32)
        candidate_object = candidate_hands_object(
            hand_object, wrist_object, rotation, translations[start:stop]
        )
        region_values: list[np.ndarray] = []
        for frame in range(len(hand_object)):
            for region in range(region_count):
                selected = contact_masks[frame] & (region_ids == region)
                if int(selected.sum()) < args.minimum_contact_vertices:
                    continue
                points = candidate_object[:, frame, selected]
                distance = object_tree.query(
                    points.reshape(-1, 3), workers=-1
                )[0].reshape(stop - start, -1)
                topk = min(args.contact_topk, distance.shape[1])
                region_values.append(
                    np.partition(distance, topk - 1, axis=1)[:, :topk].mean(axis=1)
                    * 1000.0
                )
        if region_values:
            contact_score[start:stop] = np.stack(region_values, axis=1).mean(axis=1)

        candidate_camera = to_camera(
            candidate_object[:, :, anchor_indices], poses
        )
        candidate_uv = project(candidate_camera, intrinsics)
        difference = np.linalg.norm(
            candidate_uv - base_uv[None, :, anchor_indices], axis=-1
        )
        reprojection[start:stop] = np.nanmedian(difference, axis=(1, 2))

    translation_norm = np.linalg.norm(translations, axis=-1) * 1000.0
    rotation_norm = np.linalg.norm(rotation_vectors, axis=-1) * 180.0 / np.pi
    objective = (
        contact_score
        + args.w_reprojection_mm_per_px * reprojection
        + args.w_translation_prior * translation_norm
        + args.w_rotation_prior * rotation_norm
    )
    return objective, contact_score, reprojection


def transform_object_vertices(
    vertices: np.ndarray, poses: np.ndarray
) -> np.ndarray:
    return (
        np.einsum("vi,fji->fvj", vertices, poses[:, :3, :3])
        + poses[:, None, :3, 3]
    )


def contact_distribution(
    hand_object: np.ndarray,
    contact_masks: np.ndarray,
    region_ids: np.ndarray,
    region_count: int,
    tree: cKDTree,
) -> dict[str, object]:
    values: list[np.ndarray] = []
    for frame in range(len(hand_object)):
        for region in range(region_count):
            selected = contact_masks[frame] & (region_ids == region)
            if selected.any():
                values.append(tree.query(hand_object[frame, selected], workers=-1)[0])
    return distribution(np.concatenate(values) * 1000.0 if values else np.empty(0))


def main() -> None:
    args = parse_args()
    if args.num_candidates < 2 or args.exact_candidates < 1:
        raise ValueError("Candidate counts must be positive")
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

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    hand = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    ) + wrist[:, None]
    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    probability = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    contact_masks = probability >= threshold
    gate = np.asarray(phase[args.phase_key][phase_indices], dtype=np.float32)
    valid = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
    )
    onset_candidates = np.flatnonzero(valid & (gate >= args.onset_minimum_gate))
    if not len(onset_candidates):
        raise RuntimeError("No first-touch onset frame")
    onset = int(onset_candidates[0])
    window = np.arange(
        max(0, onset - args.onset_pre_frames),
        min(len(ids), onset + args.onset_post_frames + 1),
        dtype=np.int64,
    )
    window = window[valid[window]]
    if not len(window):
        raise RuntimeError("First-touch window has no valid frames")

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
    object_tree = cKDTree(object_local)
    intrinsics = load_intrinsics(query, args.intrinsics)
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )

    hand_object = to_object(hand, poses)
    wrist_object = points_to_object(wrist, poses)
    base_uv = project(hand, intrinsics)
    translations, rotation_vectors = generate_candidates(args)
    objective, contact_score, reprojection = score_candidates(
        hand_object[window],
        wrist_object[window],
        poses[window],
        base_uv[window],
        contact_masks[window],
        region_ids,
        len(region_names),
        object_tree,
        translations,
        rotation_vectors,
        intrinsics,
        args,
    )

    finite = np.isfinite(objective)
    ranked = np.flatnonzero(finite)[np.argsort(objective[finite])]
    if not len(ranked):
        raise RuntimeError("No finite coarse registration candidate")
    exact_ids = ranked[: args.exact_candidates]
    if 0 not in exact_ids:
        exact_ids = np.concatenate((np.asarray([0]), exact_ids[:-1]))
    exact_rotation = Rotation.from_rotvec(
        rotation_vectors[exact_ids]
    ).as_matrix().astype(np.float32)
    exact_object = candidate_hands_object(
        hand_object[window],
        wrist_object[window],
        exact_rotation,
        translations[exact_ids],
    )
    exact_camera = to_camera(exact_object, poses[window])

    sample_count = min(args.collision_object_samples, len(object_local))
    sample_ids = np.linspace(
        0, len(object_local) - 1, sample_count, dtype=np.int64
    )
    object_sample_camera = transform_object_vertices(
        object_local[sample_ids], poses[window]
    )
    flattened_hand = exact_camera.reshape(-1, hand.shape[1], 3)
    flattened_object = np.broadcast_to(
        object_sample_camera[None],
        (len(exact_ids),) + object_sample_camera.shape,
    ).reshape(-1, sample_count, 3).copy()
    boundary = directed_boundary_loop(faces)
    _, sampled_inside = exact_inside_counts(
        flattened_hand,
        faces,
        flattened_object,
        boundary,
        device,
        args.point_chunk,
    )
    sampled_inside = sampled_inside.reshape(len(exact_ids), len(window)).sum(axis=1)

    exact_contact = contact_score[exact_ids]
    exact_reprojection = reprojection[exact_ids]
    contact_limit = float(np.nanmin(exact_contact) + args.contact_slack_mm)
    feasible = (
        np.isfinite(exact_contact)
        & (exact_contact <= contact_limit)
        & (exact_reprojection <= args.maximum_reprojection_px)
    )
    candidate_pool = np.flatnonzero(feasible)
    if not len(candidate_pool):
        candidate_pool = np.arange(len(exact_ids))
    order = np.lexsort((
        objective[exact_ids[candidate_pool]],
        sampled_inside[candidate_pool],
    ))
    selected_local = int(candidate_pool[order[0]])
    selected_id = int(exact_ids[selected_local])

    selected_translation = translations[selected_id]
    selected_rotvec = rotation_vectors[selected_id]
    scaled_rotvec = gate[:, None] * selected_rotvec[None]
    scaled_rotation = Rotation.from_rotvec(scaled_rotvec).as_matrix().astype(np.float32)
    refined_object = transform_hands_object_per_frame(
        hand_object,
        wrist_object,
        scaled_rotation,
        gate[:, None] * selected_translation[None],
    )
    refined = to_camera(refined_object[None], poses)[0]
    refined_wrist_object = wrist_object + gate[:, None] * selected_translation[None]
    refined_wrist = (
        np.einsum("fi,fji->fj", refined_wrist_object, poses[:, :3, :3])
        + poses[:, :3, 3]
    )

    object_camera = transform_object_vertices(object_local, poses)
    initial_inside_mask, initial_inside = exact_inside_counts(
        hand, faces, object_camera, boundary, device, args.point_chunk
    )
    refined_inside_mask, refined_inside = exact_inside_counts(
        refined, faces, object_camera, boundary, device, args.point_chunk
    )
    initial_inside_mask &= valid[:, None]
    refined_inside_mask &= valid[:, None]
    initial_inside = initial_inside_mask.sum(axis=1).astype(np.int32)
    refined_inside = refined_inside_mask.sum(axis=1).astype(np.int32)

    evaluated_contact = contact_masks & valid[:, None] & (gate[:, None] > 0)
    initial_contact = contact_distribution(
        hand_object, evaluated_contact, region_ids, len(region_names), object_tree
    )
    refined_contact = contact_distribution(
        refined_object, evaluated_contact, region_ids, len(region_names), object_tree
    )
    refined_uv = project(refined, intrinsics)
    reprojection_all = np.linalg.norm(refined_uv - base_uv, axis=-1)

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

    summary = {
        "method": "first_touch_shared_object_se3_coarse_registration_v1",
        "stream_id": str(query["stream_id"].item()),
        "onset_frame": frame_id(ids[onset]),
        "window_frames": [frame_id(ids[index]) for index in window],
        "num_candidates": args.num_candidates,
        "exact_candidates": int(len(exact_ids)),
        "selected_candidate": selected_id,
        "selected_translation_object_mm": (
            selected_translation * 1000.0
        ).tolist(),
        "selected_translation_norm_mm": float(
            np.linalg.norm(selected_translation) * 1000.0
        ),
        "selected_rotation_object_deg": np.rad2deg(selected_rotvec).tolist(),
        "selected_rotation_norm_deg": float(
            np.linalg.norm(selected_rotvec) * 180.0 / np.pi
        ),
        "selected_contact_score_mm": float(contact_score[selected_id]),
        "selected_reprojection_px": float(reprojection[selected_id]),
        "sampled_inside_window": int(sampled_inside[selected_local]),
        "containment": {
            "initial_total": int(initial_inside.sum()),
            "refined_total": int(refined_inside.sum()),
            "improved_frames": int((refined_inside < initial_inside).sum()),
            "degraded_frames": int((refined_inside > initial_inside).sum()),
            "initial_per_frame": distribution(initial_inside[valid]),
            "refined_per_frame": distribution(refined_inside[valid]),
        },
        "contact": {"initial": initial_contact, "refined": refined_contact},
        "reprojection_px": distribution(reprojection_all[valid]),
        "gt_audit": gt_summary,
        "warning": (
            "No SDF is used. Capped-MANO ray-parity containment is evaluated "
            "only as a discrete candidate-ranking and audit metric."
        ),
    }
    output = Path(args.out_npz).expanduser().resolve()
    summary_path = Path(args.out_json).expanduser().resolve()
    write_npz(output, {
        "frame_ids": ids,
        "initial_hand_vertices_camera": hand,
        "refined_hand_vertices_camera": refined.astype(np.float32),
        "mano_faces": faces,
        "initial_wrist_camera": wrist,
        "refined_wrist_camera": refined_wrist.astype(np.float32),
        "translation_camera": (refined_wrist - wrist).astype(np.float32),
        "shared_translation_object": selected_translation.astype(np.float32),
        "shared_rotation_object_rotvec": selected_rotvec.astype(np.float32),
        "correction_gate": gate.astype(np.float32),
        "prediction_valid": valid,
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
