#!/usr/bin/env python3
"""Contact-aware rigid hand translation fitting for one sequence."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


DISTAL_MANO_JOINTS = {
    "thumb": 15,
    "index": 3,
    "middle": 6,
    "ring": 12,
    "pinky": 9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mesh-scale", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hand-samples", type=int, default=192)
    parser.add_argument("--object-samples", type=int, default=1024)
    parser.add_argument("--contact-enter-mm", type=float, default=8.0)
    parser.add_argument("--contact-target-mm", type=float, default=2.0)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.5)
    parser.add_argument(
        "--penetration-max-distance-mm",
        type=float,
        default=20.0,
        help=(
            "Only trust nearest-normal penetration signs within this surface "
            "distance. This avoids classifying distant points by a local "
            "surface tangent half-space."
        ),
    )
    parser.add_argument("--min-contact-points", type=int, default=3)
    parser.add_argument("--contact-topk", type=int, default=12)
    parser.add_argument(
        "--contact-vertex-scope",
        choices=("sampled", "mano_semantic"),
        default="sampled",
        help=(
            "Choose contact vertices from every sampled hand vertex or from "
            "MANO distal-finger and palm regions derived from LBS weights."
        ),
    )
    parser.add_argument("--mano-data-dir", default=None)
    parser.add_argument("--contact-per-finger-vertices", type=int, default=32)
    parser.add_argument("--contact-palm-vertices", type=int, default=64)
    parser.add_argument("--enter-patience", type=int, default=3)
    parser.add_argument("--exit-patience", type=int, default=5)
    parser.add_argument("--contact-update-frames", type=int, default=8)
    parser.add_argument(
        "--contact-redetect-steps",
        type=int,
        default=50,
        help="Re-detect active contacts from the current hand every N steps; 0 disables.",
    )
    parser.add_argument(
        "--contact-start-frame",
        type=int,
        default=-1,
        help="Ignore contact candidates before this output frame.",
    )
    parser.add_argument(
        "--contact-end-frame",
        type=int,
        default=-1,
        help="Ignore contact candidates after this output frame.",
    )
    parser.add_argument("--normal-dot-max", type=float, default=-0.1)
    parser.add_argument(
        "--translation-mode",
        choices=("xyz", "ray_depth"),
        default="xyz",
        help=(
            "Optimize free camera-space XYZ translation or one signed scalar "
            "along each frame's initial wrist camera ray."
        ),
    )
    parser.add_argument("--max-translation-mm", type=float, default=20.0)
    parser.add_argument(
        "--ray-probe-mm",
        type=float,
        default=2.0,
        help="Audit data losses at best ray depth minus/plus this offset.",
    )
    parser.add_argument("--w-contact", type=float, default=2.0)
    parser.add_argument("--w-penetration", type=float, default=4.0)
    parser.add_argument("--w-projection", type=float, default=0.25)
    parser.add_argument(
        "--projection-target",
        choices=("initial", "gt"),
        default="initial",
    )
    parser.add_argument("--w-anchor", type=float, default=0.5)
    parser.add_argument("--w-velocity", type=float, default=1.0)
    parser.add_argument("--w-acceleration", type=float, default=2.0)
    parser.add_argument("--w-final-acceleration", type=float, default=0.5)
    parser.add_argument("--w-relative-acceleration", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"Empty mesh: {path}")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64) * scale,
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def deterministic_surface_samples(
    mesh: trimesh.Trimesh, count: int
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    area = np.linalg.norm(cross, axis=-1)
    cdf = np.cumsum(np.maximum(area, 1e-12))
    targets = (np.arange(count, dtype=np.float64) + 0.5) / count * cdf[-1]
    selected = np.searchsorted(cdf, targets).clip(0, len(faces) - 1)
    sequence = np.arange(count, dtype=np.float64)
    u = np.mod((sequence + 0.5) * 0.7548776662466927, 1.0)
    v = np.mod((sequence + 0.5) * 0.5698402909980532, 1.0)
    sqrt_u = np.sqrt(np.maximum(u, 1e-8))
    barycentric = np.stack(
        [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v], axis=-1
    )
    points = (triangles[selected] * barycentric[:, :, None]).sum(axis=1)
    normals = cross[selected] / np.maximum(
        np.linalg.norm(cross[selected], axis=-1, keepdims=True), 1e-8
    )
    return points.astype(np.float32), normals.astype(np.float32)


def sampled_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    output = np.empty((len(vertices), len(indices), 3), dtype=np.float32)
    for frame, value in enumerate(vertices):
        triangles = value[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normals = np.zeros_like(value, dtype=np.float64)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        normals /= np.maximum(
            np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8
        )
        output[frame] = normals[indices]
    return output


def mano_semantic_contact_indices(
    mano_data_dir: Path,
    hand_side: str,
    per_finger: int,
    palm_count: int,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    model_name = "MANO_LEFT.pkl" if hand_side == "left" else "MANO_RIGHT.pkl"
    model_path = mano_data_dir / model_name
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

    groups: dict[str, list[int]] = {}
    for name, joint_index in DISTAL_MANO_JOINTS.items():
        count = min(max(1, per_finger), len(weights))
        indices = np.argpartition(weights[:, joint_index], -count)[-count:]
        indices = indices[np.argsort(weights[indices, joint_index])[::-1]]
        groups[name] = indices.astype(int).tolist()
    count = min(max(0, palm_count), len(weights))
    if count:
        palm = np.argpartition(weights[:, 0], -count)[-count:]
        palm = palm[np.argsort(weights[palm, 0])[::-1]]
    else:
        palm = np.empty(0, dtype=np.int64)
    groups["palm"] = palm.astype(int).tolist()
    combined = np.unique(
        np.concatenate(
            [np.asarray(value, dtype=np.int64) for value in groups.values()]
        )
    )
    return combined, groups


def transform_surface(
    points: torch.Tensor,
    normals: torch.Tensor,
    poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = poses[:, :3, :3]
    translation = poses[:, :3, 3]
    world_points = torch.einsum("tij,aj->tai", rotation, points)
    world_points = world_points + translation[:, None]
    world_normals = torch.einsum("tij,aj->tai", rotation, normals)
    return world_points, F.normalize(world_normals, dim=-1)


def nearest_surface(
    hand: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = torch.cdist(hand, object_points)
    nearest_distance, nearest_index = distance.min(dim=-1)
    nearest_point = torch.gather(
        object_points, 1, nearest_index[..., None].expand(-1, -1, 3)
    )
    nearest_normal = torch.gather(
        object_normals, 1, nearest_index[..., None].expand(-1, -1, 3)
    )
    signed_inside = ((nearest_point - hand) * nearest_normal).sum(dim=-1)
    return nearest_distance, nearest_point, nearest_normal, signed_inside


def contact_states(
    candidates: np.ndarray,
    distances: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    frames, hand_count = candidates.shape
    enough = candidates.sum(axis=1) >= args.min_contact_points
    active = np.zeros(frames, dtype=bool)
    entered = 0
    missed = 0
    state = False
    for frame in range(frames):
        if not state:
            entered = entered + 1 if enough[frame] else 0
            if entered >= args.enter_patience:
                state = True
                active[frame - args.enter_patience + 1 : frame + 1] = True
                missed = 0
        else:
            active[frame] = True
            missed = missed + 1 if not enough[frame] else 0
            if missed >= args.exit_patience:
                active[frame - args.exit_patience + 1 : frame + 1] = False
                state = False
                entered = 0

    selected = np.zeros_like(candidates)
    updates = []
    frame = 0
    while frame < frames:
        if not active[frame]:
            frame += 1
            continue
        end = frame
        while end < frames and active[end]:
            end += 1
        cursor = frame
        while cursor < end:
            chunk_end = min(cursor + args.contact_update_frames, end)
            score = candidates[cursor:chunk_end].sum(axis=0)
            chunk_candidates = candidates[cursor:chunk_end]
            distance_sum = np.where(
                chunk_candidates, distances[cursor:chunk_end], 0.0
            ).sum(axis=0)
            mean_distance = np.full(hand_count, np.inf, dtype=np.float64)
            observed = score > 0
            mean_distance[observed] = distance_sum[observed] / score[observed]
            valid = np.flatnonzero(score > 0)
            if len(valid):
                order = np.lexsort((mean_distance[valid], -score[valid]))
                chosen = valid[order[: args.contact_topk]]
                selected[cursor:chunk_end, chosen] = True
            else:
                chosen = np.empty(0, dtype=np.int64)
            updates.append(
                {
                    "frames": [int(cursor), int(chunk_end - 1)],
                    "sample_ids": chosen.astype(int).tolist(),
                }
            )
            cursor = chunk_end
        frame = end
    return selected, updates


def contact_candidates(
    distance: torch.Tensor,
    inside: torch.Tensor,
    hand_normals: torch.Tensor,
    object_normals: torch.Tensor,
    valid: torch.Tensor,
    semantic_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    normal_dot = (hand_normals * object_normals).sum(dim=-1)
    candidates = (
        (distance <= args.contact_enter_mm / 1000.0)
        & (inside <= args.penetration_tolerance_mm / 1000.0)
        & (normal_dot <= args.normal_dot_max)
        & valid[:, None]
        & semantic_mask[None]
    )
    if args.contact_start_frame >= 0:
        candidates[: args.contact_start_frame] = False
    if args.contact_end_frame >= 0:
        candidates[args.contact_end_frame + 1 :] = False
    return candidates


def project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp_min(1e-4)
    u = intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2]
    return torch.stack([u, v], dim=-1)


def stats(values: np.ndarray, scale: float = 1.0) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def signed_stats(values: np.ndarray, scale: float = 1.0) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    hand_path = Path(args.hand_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()
    mesh_path = Path(args.object_mesh).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(hand_path, allow_pickle=False) as raw:
        hand_payload = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(supervision_path, allow_pickle=False) as raw:
        supervision = {key: np.asarray(raw[key]) for key in raw.files}

    vertices = np.asarray(hand_payload["verts_cam"], dtype=np.float32)
    faces = np.asarray(hand_payload["faces"], dtype=np.int64)
    poses = np.asarray(supervision["object_pose"], dtype=np.float32)
    intrinsics = np.asarray(supervision["intrinsics"], dtype=np.float32)
    pred_joints = np.asarray(
        supervision["pred_joints_3d"], dtype=np.float32
    )
    gt_joints_2d = np.asarray(
        supervision["gt_joints_2d"], dtype=np.float32
    )
    valid = (
        np.asarray(hand_payload["pred_valid"]).astype(bool)
        & np.asarray(supervision["object_valid"]).astype(bool)
    )
    count = min(len(vertices), len(poses), len(valid))
    vertices, poses, valid = vertices[:count], poses[:count], valid[:count]
    pred_joints = pred_joints[:count]
    gt_joints_2d = gt_joints_2d[:count]
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    source_translation = np.asarray(
        hand_payload.get(
            "stage1_translation_normalized",
            np.zeros((count, 3), dtype=np.float32),
        ),
        dtype=np.float32,
    )[:count]
    if source_translation.shape != (count, 3):
        raise ValueError(
            f"Invalid source translation shape: {source_translation.shape}"
        )
    if source_translation.any():
        pred_joints = pred_joints + source_translation[:, None]
    if args.projection_target == "gt" and normalized_left:
        raise ValueError(
            "GT 2D projection mode currently expects a non-mirrored/right-hand "
            "supervision stream"
        )
    if args.projection_target == "gt":
        valid &= (
            np.asarray(supervision["supervision_valid"][:count]).astype(bool)
            & np.isfinite(gt_joints_2d).all(axis=(1, 2))
        )

    semantic_groups: dict[str, list[int]] = {}
    if args.contact_vertex_scope == "mano_semantic":
        if not args.mano_data_dir:
            raise ValueError(
                "--mano-data-dir is required with "
                "--contact-vertex-scope mano_semantic"
            )
        hand_indices = np.arange(vertices.shape[1], dtype=np.int64)
        hand_side = str(
            np.asarray(
                hand_payload.get(
                    "hand_side",
                    supervision.get("hand_side", "right"),
                )
            ).item()
        ).lower()
        semantic_indices, semantic_groups = mano_semantic_contact_indices(
            Path(args.mano_data_dir).expanduser().resolve(),
            hand_side,
            args.contact_per_finger_vertices,
            args.contact_palm_vertices,
        )
        semantic_mask_np = np.zeros(len(hand_indices), dtype=bool)
        semantic_mask_np[semantic_indices] = True
    else:
        hand_indices = np.linspace(
            0,
            vertices.shape[1] - 1,
            min(args.hand_samples, vertices.shape[1]),
            dtype=np.int64,
        )
        semantic_indices = hand_indices.copy()
        semantic_mask_np = np.ones(len(hand_indices), dtype=bool)
    sampled_hand = vertices[:, hand_indices]
    hand_normals = sampled_vertex_normals(vertices, faces, hand_indices)
    mesh = load_mesh(mesh_path, args.mesh_scale)
    local_points, local_normals = deterministic_surface_samples(
        mesh, args.object_samples
    )

    hand_tensor = torch.from_numpy(sampled_hand).to(device)
    hand_normal_tensor = torch.from_numpy(hand_normals).to(device)
    pose_tensor = torch.from_numpy(poses).to(device)
    object_points, object_normals = transform_surface(
        torch.from_numpy(local_points).to(device),
        torch.from_numpy(local_normals).to(device),
        pose_tensor,
    )
    valid_tensor = torch.from_numpy(valid).to(device)
    semantic_mask = torch.from_numpy(semantic_mask_np).to(device)
    with torch.no_grad():
        initial = nearest_surface(hand_tensor, object_points, object_normals)
        initial_distance, initial_point, initial_normal, initial_inside = initial
        candidates = contact_candidates(
            initial_distance,
            initial_inside,
            hand_normal_tensor,
            initial_normal,
            valid_tensor,
            semantic_mask,
            args,
        )

    contact_mask_np, updates = contact_states(
        candidates.cpu().numpy(),
        initial_distance.cpu().numpy(),
        args,
    )
    updates = [{"optimization_step": 0, **row} for row in updates]
    contact_mask = torch.from_numpy(contact_mask_np).to(device)
    intrinsics_tensor = torch.from_numpy(intrinsics).to(device)
    joint_tensor = torch.from_numpy(pred_joints).to(device)
    object_center = pose_tensor[:, :3, 3]
    projection_joint_ids = torch.tensor(
        [0, 5, 9, 13, 17], dtype=torch.long, device=device
    )
    if args.projection_target == "gt":
        projection_points = joint_tensor[:, projection_joint_ids]
        projection_target = torch.from_numpy(
            gt_joints_2d[:, [0, 5, 9, 13, 17]]
        ).to(device)
    else:
        projection_points = hand_tensor
        projection_target = project(projection_points, intrinsics_tensor)

    wrist_ray = F.normalize(joint_tensor[:, 0], dim=-1, eps=1e-8)
    translation_parameter_shape = (
        (count, 1) if args.translation_mode == "ray_depth" else (count, 3)
    )
    raw_translation = torch.zeros(
        translation_parameter_shape,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw_translation], lr=args.lr)
    max_translation = args.max_translation_mm / 1000.0
    history = []
    best_total = float("inf")
    best_step = 0
    best_translation = torch.zeros(
        (count, 3), dtype=torch.float32, device=device
    )
    best_contact_mask_np = contact_mask_np.copy()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        bounded_translation = torch.tanh(raw_translation) * max_translation
        if args.translation_mode == "ray_depth":
            translation = bounded_translation * wrist_ray
        else:
            translation = bounded_translation
        corrected = hand_tensor + translation[:, None]
        distance, _, nearest_normal, inside = nearest_surface(
            corrected, object_points, object_normals
        )
        if (
            args.contact_redetect_steps > 0
            and step > 1
            and (step - 1) % args.contact_redetect_steps == 0
        ):
            with torch.no_grad():
                refreshed_candidates = contact_candidates(
                    distance.detach(),
                    inside.detach(),
                    hand_normal_tensor,
                    nearest_normal.detach(),
                    valid_tensor,
                    semantic_mask,
                    args,
                )
                refreshed_mask_np, refreshed_updates = contact_states(
                    refreshed_candidates.cpu().numpy(),
                    distance.detach().cpu().numpy(),
                    args,
                )
                contact_mask_np = refreshed_mask_np
                contact_mask = torch.from_numpy(contact_mask_np).to(device)
                updates.extend(
                    [
                        {"optimization_step": step, **row}
                        for row in refreshed_updates
                    ]
                )

        penetration_trusted = (
            distance <= args.penetration_max_distance_mm / 1000.0
        )
        penetration = (
            F.relu(inside - args.penetration_tolerance_mm / 1000.0)
            * penetration_trusted
        )
        penetration_loss = (
            penetration.square() * valid_tensor[:, None]
        ).sum() / valid_tensor.sum().clamp_min(1)

        # signed_inside is positive inside and negative outside. Keep contact
        # vertices on the exterior side instead of minimizing unsigned distance.
        contact_error = F.smooth_l1_loss(
            inside,
            torch.full_like(inside, -args.contact_target_mm / 1000.0),
            reduction="none",
            beta=0.002,
        )
        contact_loss = (
            contact_error * contact_mask
        ).sum() / contact_mask.sum().clamp_min(1)

        projection_geometry = (
            projection_points + translation[:, None]
        )
        projection = project(projection_geometry, intrinsics_tensor)
        projection_loss = (
            (projection - projection_target).square().sum(dim=-1)
            * valid_tensor[:, None]
        ).sum() / (
            valid_tensor.sum().clamp_min(1) * projection.shape[1]
        )
        projection_loss = projection_loss / (100.0 ** 2)

        anchor_loss = (
            translation.square().sum(dim=-1) * valid_tensor
        ).sum() / valid_tensor.sum().clamp_min(1)
        velocity = translation[1:] - translation[:-1]
        pair_valid = valid_tensor[1:] & valid_tensor[:-1]
        velocity_loss = (
            velocity.square().sum(dim=-1) * pair_valid
        ).sum() / pair_valid.sum().clamp_min(1)
        acceleration = velocity[1:] - velocity[:-1]
        triple_valid = pair_valid[1:] & pair_valid[:-1]
        acceleration_loss = (
            acceleration.square().sum(dim=-1) * triple_valid
        ).sum() / triple_valid.sum().clamp_min(1)
        corrected_wrist = joint_tensor[:, 0] + translation
        final_velocity = corrected_wrist[1:] - corrected_wrist[:-1]
        final_acceleration = final_velocity[1:] - final_velocity[:-1]
        final_acceleration_loss = (
            final_acceleration.square().sum(dim=-1) * triple_valid
        ).sum() / triple_valid.sum().clamp_min(1)
        relative_wrist = corrected_wrist - object_center
        relative_velocity = relative_wrist[1:] - relative_wrist[:-1]
        relative_acceleration = relative_velocity[1:] - relative_velocity[:-1]
        relative_acceleration_loss = (
            relative_acceleration.square().sum(dim=-1) * triple_valid
        ).sum() / triple_valid.sum().clamp_min(1)

        total = (
            args.w_contact * contact_loss
            + args.w_penetration * penetration_loss
            + args.w_projection * projection_loss
            + args.w_anchor * anchor_loss
            + args.w_velocity * velocity_loss
            + args.w_acceleration * acceleration_loss
            + args.w_final_acceleration * final_acceleration_loss
            + args.w_relative_acceleration * relative_acceleration_loss
        )
        total_value = float(total.detach())
        if total_value < best_total:
            best_total = total_value
            best_step = step
            best_translation = translation.detach().clone()
            best_contact_mask_np = contact_mask_np.copy()
        total.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": float(total.detach()),
                "contact": float(contact_loss.detach()),
                "penetration": float(penetration_loss.detach()),
                "projection": float(projection_loss.detach()),
                "anchor": float(anchor_loss.detach()),
                "velocity": float(velocity_loss.detach()),
                "acceleration": float(acceleration_loss.detach()),
                "final_acceleration": float(
                    final_acceleration_loss.detach()
                ),
                "relative_acceleration": float(
                    relative_acceleration_loss.detach()
                ),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    with torch.no_grad():
        translation = best_translation
        corrected = hand_tensor + translation[:, None]
        final_distance, _, _, final_inside = nearest_surface(
            corrected, object_points, object_normals
        )
        ray_probe = []
        if args.translation_mode == "ray_depth" and args.ray_probe_mm > 0:
            probe_m = args.ray_probe_mm / 1000.0
            best_contact_mask = torch.from_numpy(best_contact_mask_np).to(device)
            for label, offset in (
                ("minus", -probe_m),
                ("center", 0.0),
                ("plus", probe_m),
            ):
                probe_translation = translation + offset * wrist_ray
                probe_hand = hand_tensor + probe_translation[:, None]
                probe_distance, _, _, probe_inside = nearest_surface(
                    probe_hand, object_points, object_normals
                )
                probe_penetration = (
                    F.relu(
                        probe_inside
                        - args.penetration_tolerance_mm / 1000.0
                    )
                    * (
                        probe_distance
                        <= args.penetration_max_distance_mm / 1000.0
                    )
                )
                probe_contact = F.smooth_l1_loss(
                    probe_inside,
                    torch.full_like(
                        probe_inside,
                        -args.contact_target_mm / 1000.0,
                    ),
                    reduction="none",
                    beta=0.002,
                )
                probe_projection = project(
                    projection_points + probe_translation[:, None],
                    intrinsics_tensor,
                )
                projection_per_frame = (
                    (probe_projection - projection_target)
                    .square()
                    .sum(dim=-1)
                    .mean(dim=-1)
                    / (100.0 ** 2)
                )
                penetration_per_frame = probe_penetration.square().mean(dim=-1)
                contact_count = best_contact_mask.sum(dim=-1)
                contact_per_frame = (
                    (probe_contact * best_contact_mask).sum(dim=-1)
                    / contact_count.clamp_min(1)
                )
                data_total = (
                    args.w_penetration * penetration_per_frame
                    + args.w_contact * contact_per_frame
                    + args.w_projection * projection_per_frame
                )
                ray_probe.append(
                    {
                        "label": label,
                        "offset_mm": offset * 1000.0,
                        "penetration": penetration_per_frame.cpu().numpy(),
                        "contact": contact_per_frame.cpu().numpy(),
                        "projection": projection_per_frame.cpu().numpy(),
                        "weighted_data_total": data_total.cpu().numpy(),
                    }
                )
    translation_np = translation.cpu().numpy()
    contact_mask_np = best_contact_mask_np
    corrected_vertices = vertices + translation_np[:, None]
    output = dict(hand_payload)
    output["verts_cam"] = corrected_vertices.astype(np.float32)
    output["optim_seq_translation"] = translation_np.astype(np.float32)
    output["optim_seq_translation_mode"] = np.asarray(args.translation_mode)
    if args.translation_mode == "ray_depth":
        ray_np = wrist_ray.cpu().numpy()
        ray_depth_np = (translation * wrist_ray).sum(dim=-1).cpu().numpy()
        output["optim_seq_camera_ray"] = ray_np.astype(np.float32)
        output["optim_seq_ray_depth"] = ray_depth_np.astype(np.float32)
    output["optim_seq_contact_sample_indices"] = hand_indices.astype(np.int64)
    output["optim_seq_semantic_contact_indices"] = semantic_indices.astype(
        np.int64
    )
    output["optim_seq_contact_mask"] = contact_mask_np
    output["optim_seq_source_hand"] = np.asarray(str(hand_path))
    output_path = out_dir / "hand_contact_optimized.npz"
    np.savez_compressed(output_path, **output)

    initial_penetrating = (
        (
            initial_inside.cpu().numpy()
            > args.penetration_tolerance_mm / 1000.0
        )
        & (
            initial_distance.cpu().numpy()
            <= args.penetration_max_distance_mm / 1000.0
        )
    ) & valid[:, None]
    final_penetrating = (
        (
            final_inside.cpu().numpy()
            > args.penetration_tolerance_mm / 1000.0
        )
        & (
            final_distance.cpu().numpy()
            <= args.penetration_max_distance_mm / 1000.0
        )
    ) & valid[:, None]
    contact_values_initial = initial_distance.cpu().numpy()[contact_mask_np]
    contact_values_final = final_distance.cpu().numpy()[contact_mask_np]
    initial_penetration_depth = np.maximum(
        initial_inside.cpu().numpy()
        - args.penetration_tolerance_mm / 1000.0,
        0.0,
    )
    initial_penetration_depth[~initial_penetrating] = 0.0
    initial_penetration_depth = initial_penetration_depth[valid]
    final_penetration_depth = np.maximum(
        final_inside.cpu().numpy()
        - args.penetration_tolerance_mm / 1000.0,
        0.0,
    )
    final_penetration_depth[~final_penetrating] = 0.0
    final_penetration_depth = final_penetration_depth[valid]
    audit = {
        "hand_npz": str(hand_path),
        "supervision_npz": str(supervision_path),
        "object_mesh": str(mesh_path),
        "output_npz": str(output_path),
        "settings": vars(args),
        "num_frames": count,
        "num_valid_frames": int(valid.sum()),
        "num_contact_frame_vertices": int(contact_mask_np.sum()),
        "num_semantic_contact_vertices": int(len(semantic_indices)),
        "semantic_contact_groups": semantic_groups,
        "contact_updates": updates,
        "best_step": best_step,
        "best_total": best_total,
        "translation_mm": stats(
            np.linalg.norm(translation_np, axis=-1), 1000.0
        ),
        "signed_ray_depth_mm": (
            signed_stats(
                (translation * wrist_ray).sum(dim=-1).cpu().numpy(),
                1000.0,
            )
            if args.translation_mode == "ray_depth"
            else None
        ),
        "contact_distance_mm": {
            "initial": stats(contact_values_initial, 1000.0),
            "final": stats(contact_values_final, 1000.0),
        },
        "penetrating_sample_vertices": {
            "initial": int(initial_penetrating.sum()),
            "final": int(final_penetrating.sum()),
        },
        "penetration_depth_mm": {
            "initial": stats(
                initial_penetration_depth[initial_penetration_depth > 0],
                1000.0,
            ),
            "final": stats(
                final_penetration_depth[final_penetration_depth > 0],
                1000.0,
            ),
        },
        "ray_direction_probe": (
            [
                {
                    "frame": frame,
                    "best_offset": min(
                        ray_probe,
                        key=lambda row: row["weighted_data_total"][frame],
                    )["label"],
                    "values": {
                        row["label"]: {
                            "offset_mm": row["offset_mm"],
                            "penetration": float(row["penetration"][frame]),
                            "contact": float(row["contact"][frame]),
                            "projection": float(row["projection"][frame]),
                            "weighted_data_total": float(
                                row["weighted_data_total"][frame]
                            ),
                        }
                        for row in ray_probe
                    },
                }
                for frame in range(count)
            ]
            if ray_probe
            else []
        ),
        "history": history,
    }
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
