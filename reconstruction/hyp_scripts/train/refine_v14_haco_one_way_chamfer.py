#!/usr/bin/env python3
"""Rigidly refine one V14 hand with all HACO contacts and one-way Chamfer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import trimesh


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--frame-id")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--object-samples", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--softmax-sigma-mm", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-translation-mm", type=float, default=20.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--w-translation-anchor", type=float, default=0.1)
    parser.add_argument("--w-rotation-anchor", type=float, default=0.01)
    parser.add_argument(
        "--contact-weighting",
        choices=("uniform", "probability"),
        default="probability",
    )
    parser.add_argument("--contact-probability-power", type=float, default=2.0)
    parser.add_argument("--contact-weight-floor", type=float, default=0.05)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--penetration-trust-mm", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def normalized_frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def index_for(values: np.ndarray, target: str) -> int:
    normalized = [normalized_frame_id(value) for value in values]
    if target not in normalized:
        raise KeyError(f"Frame {target} not found")
    return normalized.index(target)


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64) * scale,
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def physical_pose(pose: np.ndarray, normalized_left: bool) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    if normalized_left:
        result[:3, :3] = MIRROR_X @ result[:3, :3] @ MIRROR_X
        result[:3, 3] = MIRROR_X @ result[:3, 3]
    return result


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
    areas = np.linalg.norm(cross, axis=-1)
    cdf = np.cumsum(np.maximum(areas, 1e-12))
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


def euler_matrix(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles.unbind()
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    rx = torch.stack((
        one, zero, zero,
        zero, cx, -sx,
        zero, sx, cx,
    )).reshape(3, 3)
    ry = torch.stack((
        cy, zero, sy,
        zero, one, zero,
        -sy, zero, cy,
    )).reshape(3, 3)
    rz = torch.stack((
        cz, -sz, zero,
        sz, cz, zero,
        zero, zero, one,
    )).reshape(3, 3)
    return rz @ ry @ rx


def transform_hand(
    vertices: torch.Tensor,
    wrist: torch.Tensor,
    translation: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    rotation = euler_matrix(angles)
    return (vertices - wrist) @ rotation.T + wrist + translation


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


@torch.no_grad()
def geometry_metrics(
    hand: torch.Tensor,
    contact_mask: torch.Tensor,
    object_points: torch.Tensor,
    object_normals: torch.Tensor,
    tolerance_mm: float,
    trust_mm: float,
) -> dict[str, object]:
    distance = torch.cdist(hand[None], object_points[None])[0]
    nearest_distance, nearest_index = distance.min(dim=-1)
    nearest_point = object_points[nearest_index]
    nearest_normal = object_normals[nearest_index]
    inside = ((nearest_point - hand) * nearest_normal).sum(dim=-1)
    trusted = nearest_distance <= trust_mm / 1000.0
    penetrating = (inside > tolerance_mm / 1000.0) & trusted
    depth = torch.clamp(inside - tolerance_mm / 1000.0, min=0.0)
    depth = depth[penetrating].cpu().numpy() * 1000.0
    contact_distance = nearest_distance[contact_mask].cpu().numpy() * 1000.0
    return {
        "contact_distance_mm": distribution(contact_distance),
        "penetrating_vertices": int(penetrating.sum().item()),
        "penetration_depth_mm": distribution(depth),
    }


def gt_vertices(
    path: Path, hand_side: str, frame_index: int
) -> np.ndarray | None:
    data = load_npz(path)
    valid = data[f"{hand_side}_valid"]
    if frame_index >= len(valid) or not bool(valid[frame_index]):
        return None
    vertices = np.asarray(
        data[f"{hand_side}_vertices"][frame_index], dtype=np.float32
    )
    return vertices if np.isfinite(vertices).all() else None


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    trajectory_path = Path(args.trajectory_npz).expanduser().resolve()
    query_path = Path(args.query_npz).expanduser().resolve()
    contact_path = Path(args.contact_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()
    trajectory = load_npz(trajectory_path)
    query = load_npz(query_path)
    contact = load_npz(contact_path)
    supervision = load_npz(supervision_path)

    requested = (
        normalized_frame_id(args.frame_id)
        if args.frame_id is not None
        else normalized_frame_id(query["frame_ids"][args.frame_index])
    )
    trajectory_index = index_for(trajectory["frame_ids"], requested)
    query_index = index_for(query["frame_ids"], requested)
    supervision_index = index_for(supervision["frame_ids"], requested)
    if normalized_frame_id(contact["frame_id"].item()) != requested:
        raise ValueError("Contact frame does not match requested frame")

    wrist_np = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_index], dtype=np.float32
    )
    vertices_np = np.asarray(
        query["vertices_3d_root_relative_original"][query_index],
        dtype=np.float32,
    ) + wrist_np[None]
    faces = np.asarray(query["mano_faces"], dtype=np.int64)
    probability_np = np.asarray(
        contact["contact_probability"], dtype=np.float32
    ).reshape(-1)
    contact_mask_np = np.asarray(contact["contact_mask"]).astype(bool).reshape(-1)
    if not contact_mask_np.any():
        raise RuntimeError("HACO selected no contact vertices")

    object_mesh = load_mesh(
        Path(args.object_mesh).expanduser().resolve(), args.object_scale
    )
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    object_pose = physical_pose(
        supervision["gt_ycb_object_pose"][supervision_index], normalized_left
    )
    object_mesh.apply_transform(object_pose)
    object_points_np, object_normals_np = deterministic_surface_samples(
        object_mesh, args.object_samples
    )

    device = torch.device(args.device)
    vertices = torch.from_numpy(vertices_np).to(device)
    wrist = torch.from_numpy(wrist_np).to(device)
    object_points = torch.from_numpy(object_points_np).to(device)
    object_normals = torch.from_numpy(object_normals_np).to(device)
    contact_mask = torch.from_numpy(contact_mask_np).to(device)
    probability = torch.from_numpy(probability_np).to(device)
    contact_vertices = vertices[contact_mask]
    if args.contact_weighting == "probability":
        threshold = float(np.asarray(contact["contact_threshold"]).item())
        confidence = torch.clamp(
            (probability[contact_mask] - threshold) / max(1.0 - threshold, 1e-6),
            min=0.0,
            max=1.0,
        ).pow(args.contact_probability_power)
        contact_weight = (
            args.contact_weight_floor
            + (1.0 - args.contact_weight_floor) * confidence
        )
    else:
        contact_weight = torch.ones_like(probability[contact_mask])
    contact_weight = contact_weight / contact_weight.mean().clamp_min(1e-6)

    translation = torch.zeros(3, device=device, requires_grad=True)
    angles = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([translation, angles], lr=args.lr)
    max_translation = args.max_translation_mm / 1000.0
    max_angle = math.radians(args.max_rotation_deg)
    sigma = args.softmax_sigma_mm / 1000.0
    topk = min(max(1, args.topk), len(object_points))
    best_total = float("inf")
    best_translation = torch.zeros_like(translation)
    best_angles = torch.zeros_like(angles)
    history = []

    initial_metrics = geometry_metrics(
        vertices, contact_mask, object_points, object_normals,
        args.penetration_tolerance_mm, args.penetration_trust_mm,
    )
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        refined_contact = transform_hand(
            contact_vertices, wrist, translation, angles
        )
        pairwise = torch.cdist(refined_contact[None], object_points[None])[0]
        nearest, _ = torch.topk(pairwise, topk, dim=-1, largest=False)
        soft_weight = torch.softmax(
            -(nearest.square()) / (2.0 * sigma * sigma), dim=-1
        )
        contact_error = (soft_weight * nearest.square()).sum(dim=-1)
        contact_loss = (contact_error * contact_weight).mean()
        translation_anchor = translation.square().sum()
        rotation_anchor = angles.square().sum()
        total = (
            contact_loss
            + args.w_translation_anchor * translation_anchor
            + args.w_rotation_anchor * rotation_anchor
        )
        total_value = float(total.detach())
        if total_value < best_total:
            best_total = total_value
            best_translation = translation.detach().clone()
            best_angles = angles.detach().clone()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            translation_scale = torch.clamp(
                max_translation / translation.norm().clamp_min(1e-12),
                max=1.0,
            )
            rotation_scale = torch.clamp(
                max_angle / angles.norm().clamp_min(1e-12),
                max=1.0,
            )
            translation.mul_(translation_scale)
            angles.mul_(rotation_scale)
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": total_value,
                "contact": float(contact_loss.detach()),
                "translation_mm": float(translation.detach().norm() * 1000.0),
                "rotation_deg": float(
                    angles.detach().norm() * 180.0 / math.pi
                ),
            }
            history.append(row)
            print(row)

    refined = transform_hand(
        vertices, wrist, best_translation, best_angles
    ).detach()
    final_metrics = geometry_metrics(
        refined, contact_mask, object_points, object_normals,
        args.penetration_tolerance_mm, args.penetration_trust_mm,
    )
    hand_side = str(query["hand_side"].item()).lower()
    gt = None
    gt_audit = None
    if args.gt_hand_npz:
        gt = gt_vertices(
            Path(args.gt_hand_npz).expanduser().resolve(),
            hand_side,
            query_index,
        )
    if gt is not None and len(gt) == len(vertices_np):
        initial_error = np.linalg.norm(vertices_np - gt, axis=-1) * 1000.0
        final_error = np.linalg.norm(
            refined.cpu().numpy() - gt, axis=-1
        ) * 1000.0
        gt_audit = {
            "initial_vertex_error_mm": distribution(initial_error),
            "refined_vertex_error_mm": distribution(final_error),
        }

    summary = {
        "method": "all_haco_one_way_soft_topk_chamfer_rigid_v1",
        "frame": requested,
        "stream_id": str(query["stream_id"].item()),
        "contact_vertices": int(contact_mask_np.sum()),
        "contact_weighting": args.contact_weighting,
        "contact_probability_power": args.contact_probability_power,
        "contact_weight_floor": args.contact_weight_floor,
        "topk": topk,
        "initial": initial_metrics,
        "refined": final_metrics,
        "translation_mm": (best_translation.cpu().numpy() * 1000.0).tolist(),
        "translation_norm_mm": float(best_translation.norm() * 1000.0),
        "rotation_euler_deg": (
            best_angles.cpu().numpy() * 180.0 / math.pi
        ).tolist(),
        "rotation_norm_deg": float(best_angles.norm() * 180.0 / math.pi),
        "gt_audit": gt_audit,
        "history": history,
        "warning": "No penetration loss or collision safety gate is applied.",
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    write_npz(output_path, {
        "frame_id": np.asarray(requested),
        "stream_id": np.asarray(str(query["stream_id"].item())),
        "initial_hand_vertices_camera": vertices_np.astype(np.float32),
        "refined_hand_vertices_camera": refined.cpu().numpy().astype(np.float32),
        "mano_faces": faces,
        "contact_probability": probability_np,
        "contact_mask": contact_mask_np,
        "translation_camera": best_translation.cpu().numpy().astype(np.float32),
        "rotation_euler_xyz": best_angles.cpu().numpy().astype(np.float32),
        "object_surface_points_camera": object_points_np,
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
