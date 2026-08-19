#!/usr/bin/env python3
"""Single-frame rigid refinement from V14 to fixed object contact patches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from refine_v14_haco_sequence_chamfer import transform_batch
from refine_v14_haco_sequence_contact_containment import (
    collision_correspondences,
    mano_contact_region_ids,
)
from refine_v14_haco_containment_pushout import (
    build_object_geometry,
    exact_inside_counts,
)
from refine_v14_haco_one_way_chamfer import load_mesh, load_npz
from visualize_capped_mano_wrist import directed_boundary_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--patch-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--max-translation-mm", type=float, default=50.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--contact-target-mm", type=float, default=2.0)
    parser.add_argument("--collision-margin-mm", type=float, default=0.5)
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-collision", type=float, default=10.0)
    parser.add_argument("--w-anchor", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--refresh-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def index_for(values: np.ndarray, target: str) -> int:
    normalized = [frame_id(value) for value in values]
    if target not in normalized:
        raise KeyError(f"Frame {target} not found")
    return normalized.index(target)


def physical_pose(pose: np.ndarray, normalized_left: bool) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    if normalized_left:
        mirror = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        result[:3, :3] = mirror @ result[:3, :3] @ mirror
        result[:3, 3] = mirror @ result[:3, 3]
    return result


def main() -> None:
    args = parse_args()
    requested = frame_id(args.frame_id)
    stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    patch = load_npz(Path(args.patch_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())

    si = index_for(stage1["frame_ids"], requested)
    qi = index_for(query["frame_ids"], requested)
    ci = index_for(contact["frame_ids"], requested)
    oi = index_for(supervision["frame_ids"], requested)
    if frame_id(patch["frame_id"].item()) != requested:
        raise ValueError("Patch frame does not match requested frame")

    initial_np = np.asarray(
        stage1["initial_hand_vertices_camera"][si], dtype=np.float32
    )
    faces_np = np.asarray(query["mano_faces"], dtype=np.int64)
    probability = np.asarray(contact["contact_probability"][ci], dtype=np.float32)
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    contact_mask = (
        np.asarray(contact["contact_mask"][ci]).astype(bool)
        & (probability >= threshold)
    )
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )
    thumb_mask = contact_mask & (region_ids == region_names.index("thumb"))
    index_mask = contact_mask & (region_ids == region_names.index("index"))
    if thumb_mask.sum() < 3 or index_mask.sum() < 3:
        raise RuntimeError("Insufficient thumb/index HACO contacts")

    mesh = load_mesh(Path(args.object_mesh).expanduser().resolve(), 1.0)
    normalized_left = bool(np.asarray(
        supervision.get("normalized_left", False)
    ).item())
    pose = physical_pose(
        supervision["gt_ycb_object_pose"][oi], normalized_left
    )
    object_vertices_local = np.asarray(mesh.vertices, dtype=np.float32)
    object_vertices_np = (
        object_vertices_local @ pose[:3, :3].T + pose[:3, 3]
    )[None]
    object_points_np = object_vertices_np.copy()
    object_normals_np = np.asarray(mesh.vertex_normals, dtype=np.float32)
    object_normals_np = (
        object_normals_np @ pose[:3, :3].T
    )[None]
    thumb_patch_np = np.asarray(patch["thumb_patch_vertices_canonical"], dtype=np.float32)
    index_patch_np = np.asarray(patch["index_patch_vertices_canonical"], dtype=np.float32)
    thumb_patch_np = thumb_patch_np @ pose[:3, :3].T + pose[:3, 3]
    index_patch_np = index_patch_np @ pose[:3, :3].T + pose[:3, 3]

    boundary = directed_boundary_loop(faces_np)
    device = torch.device(args.device)
    vertices = torch.from_numpy(initial_np[None]).to(device)
    wrists = vertices.mean(dim=1)
    thumb_patch = torch.from_numpy(thumb_patch_np[None]).to(device)
    index_patch = torch.from_numpy(index_patch_np[None]).to(device)
    object_vertices = torch.from_numpy(object_vertices_np).to(device)
    object_points = torch.from_numpy(object_points_np).to(device)
    object_normals = torch.from_numpy(object_normals_np).to(device)
    mano_faces = torch.from_numpy(faces_np).to(device)
    translation = torch.zeros((1, 3), device=device, requires_grad=True)
    angles = torch.zeros((1, 3), device=device, requires_grad=True)
    optimizer = torch.optim.Adam((translation, angles), lr=args.lr)
    max_translation = args.max_translation_mm / 1000.0
    max_angle = math.radians(args.max_rotation_deg)
    contact_target = args.contact_target_mm / 1000.0
    collision_margin = args.collision_margin_mm / 1000.0
    collision_points = collision_faces = collision_barycentric = None
    inside_count = np.zeros(1, dtype=np.int32)
    history = []

    for step in range(1, args.steps + 1):
        with torch.no_grad():
            current = transform_batch(vertices, wrists, translation, angles)
        if step == 1 or (step - 1) % args.refresh_steps == 0:
            current_np = current.detach().cpu().numpy().astype(np.float32)
            inside_mask, inside_count = exact_inside_counts(
                current_np, faces_np, object_vertices_np,
                boundary, device, 1024,
            )
            collision_points, collision_faces, collision_barycentric = (
                collision_correspondences(
                    current, object_vertices_np, inside_mask, mano_faces, 8
                )
            )

        optimizer.zero_grad(set_to_none=True)
        refined = transform_batch(vertices, wrists, translation, angles)
        thumb_distance = torch.cdist(refined[:, thumb_mask], thumb_patch)
        index_distance = torch.cdist(refined[:, index_mask], index_patch)
        thumb_error = torch.topk(thumb_distance, min(8, thumb_distance.shape[-1]), dim=-1, largest=False).values
        index_error = torch.topk(index_distance, min(8, index_distance.shape[-1]), dim=-1, largest=False).values
        contact_loss = (
            torch.clamp(thumb_error - contact_target, min=0).square().mean()
            + torch.clamp(index_error - contact_target, min=0).square().mean()
        )
        collision_loss = torch.zeros((), device=device)
        if collision_points is not None and collision_points[0] is not None:
            points = collision_points[0]
            face_index = collision_faces[0]
            barycentric = collision_barycentric[0]
            triangles = refined[0, mano_faces[face_index]]
            surface = (triangles * barycentric[..., None]).sum(dim=-2)
            normal = torch.nn.functional.normalize(
                torch.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0], dim=-1,
                ), dim=-1,
            )
            signed_clearance = ((points - surface) * normal).sum(dim=-1)
            collision_loss = torch.clamp(
                collision_margin - signed_clearance, min=0
            ).square().mean()
        loss = (
            args.w_contact * contact_loss
            + args.w_collision * collision_loss
            + args.w_anchor * (
                translation.square().sum() + angles.square().sum()
            )
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            translation.mul_(torch.clamp(max_translation / translation.norm().clamp_min(1e-12), max=1.0))
            angles.mul_(torch.clamp(max_angle / angles.norm().clamp_min(1e-12), max=1.0))
        if step == 1 or step % 25 == 0 or step == args.steps:
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "contact": float(contact_loss.detach()),
                "collision": float(collision_loss.detach()),
                "inside_vertices": int(inside_count[0]),
                "translation_mm": float(translation.norm().detach() * 1000),
                "rotation_deg": float(angles.norm().detach() * 180 / math.pi),
            })

    with torch.no_grad():
        refined_np = transform_batch(vertices, wrists, translation, angles).cpu().numpy()[0]
    final_mask, final_count = exact_inside_counts(
        refined_np[None], faces_np, object_vertices_np, boundary, device, 1024
    )
    output = Path(args.out_npz).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_ids=np.asarray([requested]),
        initial_hand_vertices_camera=initial_np[None],
        refined_hand_vertices_camera=refined_np[None].astype(np.float32),
        mano_faces=faces_np,
        translation_camera=translation.detach().cpu().numpy(),
        rotation_euler_xyz=angles.detach().cpu().numpy(),
        initial_inside_object_vertices=inside_count,
        refined_inside_object_vertices=final_count,
        fixed_thumb_patch_vertices_camera=thumb_patch_np[None],
        fixed_index_patch_vertices_camera=index_patch_np[None],
        method=np.asarray("v14_fixed_contact_patch_rigid_stage1_v1"),
    )
    summary = {
        "method": "v14_fixed_contact_patch_rigid_stage1_v1",
        "frame_id": requested,
        "pair_hand_source": "v14",
        "translation_mm": float(np.linalg.norm(translation.detach().cpu().numpy()) * 1000),
        "rotation_deg": float(np.linalg.norm(angles.detach().cpu().numpy()) * 180 / math.pi),
        "initial_inside_vertices": int(inside_count[0]),
        "refined_inside_vertices": int(final_count[0]),
        "history": history,
        "output": str(output),
    }
    summary_path = Path(args.out_json or output.with_suffix(".json")).expanduser().resolve()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
