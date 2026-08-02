#!/usr/bin/env python3
"""Refine one shared SAM-to-YCB canonical residual from a DexYCB sequence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import so3_exp_map

TRAIN_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "train"
if str(TRAIN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPT_DIR))

from fit_isolated_object_sequence import (
    camera_to_pytorch3d,
    load_mesh,
    load_pose_rows,
    load_segmentation,
    normalize_frame_id,
    resolve_depth_path,
    sample_pixels,
    target_object_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-faces", type=int, default=30000)
    parser.add_argument("--sample-vertices", type=int, default=2048)
    parser.add_argument("--sample-target-pixels", type=int, default=512)
    parser.add_argument("--hand-dilation-px", type=int, default=7)
    parser.add_argument("--max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-translation-mm", type=float, default=15.0)
    parser.add_argument("--w-rep", type=float, default=2.0)
    parser.add_argument("--w-attr", type=float, default=1.0)
    parser.add_argument("--w-depth", type=float, default=5.0)
    parser.add_argument("--w-prior", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    base_path = Path(args.base_json).expanduser().resolve()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    mesh_path = Path(args.mesh).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    output_payload = copy.deepcopy(base_payload)
    _, base_rows = load_pose_rows(base_payload)
    _, output_rows = load_pose_rows(output_payload)
    frame_map = json.loads(frame_map_path.read_text(encoding="utf-8"))
    frame_rows = frame_map["frames"]
    stream_dir = Path(
        frame_map.get("stream_dir")
        or Path(frame_rows[0]["label_path"]).parent
    )
    object_id = target_object_id(stream_dir)

    frame_ids = []
    raw_keys = []
    selected_rows = []
    base_poses = []
    for row in frame_rows:
        frame_id = normalize_frame_id(row["original_frame"])
        raw_key = frame_id if frame_id in base_rows else str(int(frame_id))
        pose_row = base_rows.get(raw_key)
        if pose_row is None or pose_row.get("object_in_camera") is None:
            continue
        pose = np.asarray(pose_row["object_in_camera"], dtype=np.float32)
        if pose.size != 16 or not np.isfinite(pose).all():
            continue
        frame_ids.append(frame_id)
        raw_keys.append(raw_key)
        selected_rows.append(row)
        base_poses.append(pose.reshape(4, 4))
    if len(base_poses) < 3:
        raise RuntimeError("Need at least three valid base poses")

    base_poses_np = np.stack(base_poses)
    base_rotation = torch.as_tensor(
        base_poses_np[:, :3, :3], dtype=torch.float32, device=device
    )
    base_translation = torch.as_tensor(
        base_poses_np[:, :3, 3], dtype=torch.float32, device=device
    )
    intrinsics = np.asarray(
        base_payload["intrinsics"], dtype=np.float32
    ).reshape(3, 3)
    source_scale = float(base_payload.get("source_mesh_scale", 1.0))
    mesh_vertices_np, mesh_faces_np = load_mesh(mesh_path, args.max_faces)
    mesh_vertices = torch.as_tensor(
        mesh_vertices_np * source_scale,
        dtype=torch.float32,
        device=device,
    )
    mesh_faces = torch.as_tensor(
        mesh_faces_np, dtype=torch.int64, device=device
    )
    sample_indices = np.linspace(
        0,
        len(mesh_vertices_np) - 1,
        min(args.sample_vertices, len(mesh_vertices_np)),
        dtype=np.int64,
    )
    sampled_vertices = mesh_vertices[
        torch.as_tensor(sample_indices, device=device)
    ]

    first_segmentation = load_segmentation(
        Path(selected_rows[0]["label_path"])
    )
    source_height, source_width = first_segmentation.shape
    width = max(32, int(round(source_width * args.render_scale)))
    height = max(32, int(round(source_height * args.render_scale)))
    focal = torch.tensor(
        [[
            intrinsics[0, 0] * width / source_width,
            intrinsics[1, 1] * height / source_height,
        ]],
        dtype=torch.float32,
        device=device,
    )
    principal = torch.tensor(
        [[
            intrinsics[0, 2] * width / source_width,
            intrinsics[1, 2] * height / source_height,
        ]],
        dtype=torch.float32,
        device=device,
    )
    cameras = PerspectiveCameras(
        focal_length=focal,
        principal_point=principal,
        image_size=((height, width),),
        in_ndc=False,
        device=device,
    )
    sigma = 1e-4
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=math.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=8,
            bin_size=None,
            max_faces_per_bin=max(200000, len(mesh_faces)),
        ),
    )
    silhouette_shader = SoftSilhouetteShader(
        blend_params=BlendParams(sigma=sigma, gamma=1e-4)
    )

    object_masks = []
    valid_backgrounds = []
    depths = []
    target_pixels = []
    dilation = max(1, int(round(args.hand_dilation_px * args.render_scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    for index, (frame_id, row) in enumerate(
        zip(frame_ids, selected_rows)
    ):
        segmentation = load_segmentation(Path(row["label_path"]))
        object_mask = cv2.resize(
            (segmentation == object_id).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        hand_mask = cv2.resize(
            (segmentation == 255).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        hand_dilated = cv2.dilate(
            hand_mask, kernel, iterations=1
        ).astype(bool)
        depth_raw = cv2.imread(
            str(resolve_depth_path(stream_dir, frame_id)),
            cv2.IMREAD_UNCHANGED,
        )
        if depth_raw is None:
            raise FileNotFoundError(f"Cannot read depth for {frame_id}")
        depth = cv2.resize(
            depth_raw.astype(np.float32) / 1000.0,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        object_masks.append(object_mask)
        valid_backgrounds.append(~hand_dilated)
        depths.append(depth)
        target_pixels.append(
            sample_pixels(
                object_mask,
                args.sample_target_pixels,
                args.seed + index,
            )
        )

    object_masks_t = torch.as_tensor(
        np.stack(object_masks), dtype=torch.float32, device=device
    )
    valid_backgrounds_t = torch.as_tensor(
        np.stack(valid_backgrounds), dtype=torch.bool, device=device
    )
    depths_t = torch.as_tensor(
        np.stack(depths), dtype=torch.float32, device=device
    )

    rotation_parameter = torch.nn.Parameter(
        torch.zeros(3, dtype=torch.float32, device=device)
    )
    translation_parameter = torch.nn.Parameter(
        torch.zeros(3, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        [rotation_parameter, translation_parameter], lr=args.lr
    )
    max_rotation = math.radians(args.max_rotation_deg)
    max_translation = args.max_translation_mm / 1000.0
    history = []
    best = None

    for step in range(args.steps):
        rotation_delta_vector = (
            torch.tanh(rotation_parameter) * max_rotation
        )
        rotation_delta = so3_exp_map(rotation_delta_vector[None])[0]
        translation_delta = (
            torch.tanh(translation_parameter) * max_translation
        )
        refined_rotation = base_rotation @ rotation_delta
        refined_translation = base_translation + torch.einsum(
            "fij,j->fi", base_rotation, translation_delta
        )

        begin = (step * args.batch_size) % len(frame_ids)
        batch_indices = [
            (begin + offset) % len(frame_ids)
            for offset in range(min(args.batch_size, len(frame_ids)))
        ]
        indices = torch.as_tensor(
            batch_indices, dtype=torch.int64, device=device
        )
        vertices_camera = (
            mesh_vertices[None]
            @ refined_rotation[indices].transpose(1, 2)
            + refined_translation[indices, None, :]
        )
        meshes = Meshes(
            verts=[camera_to_pytorch3d(value) for value in vertices_camera],
            faces=[mesh_faces for _ in batch_indices],
        )
        fragments = rasterizer(meshes)
        alpha = silhouette_shader(fragments, meshes)[..., 3]
        target_mask = object_masks_t[indices]
        valid_background = valid_backgrounds_t[indices]
        rep = (
            F.relu(alpha - target_mask).square()
            * valid_background.float()
        ).sum() / valid_background.float().sum().clamp_min(1.0)

        projected_vertices = (
            sampled_vertices[None]
            @ refined_rotation[indices].transpose(1, 2)
            + refined_translation[indices, None, :]
        )
        projected_x = (
            focal[0, 0]
            * projected_vertices[..., 0]
            / projected_vertices[..., 2].clamp_min(1e-4)
            + principal[0, 0]
        )
        projected_y = (
            focal[0, 1]
            * projected_vertices[..., 1]
            / projected_vertices[..., 2].clamp_min(1e-4)
            + principal[0, 1]
        )
        projected = torch.stack([projected_x, projected_y], dim=-1)
        attr_values = []
        for local_index, frame_index in enumerate(batch_indices):
            pixels = target_pixels[frame_index]
            if not len(pixels):
                continue
            pixels_t = torch.as_tensor(
                pixels, dtype=torch.float32, device=device
            )
            distances = torch.cdist(
                pixels_t / max(width, height),
                projected[local_index] / max(width, height),
            )
            attr_values.append(distances.min(dim=1).values.square().mean())
        attr = (
            torch.stack(attr_values).mean()
            if attr_values
            else torch.zeros((), device=device)
        )

        rendered_depth = fragments.zbuf[..., 0]
        depth_valid = (
            (fragments.pix_to_face[..., 0] >= 0)
            & target_mask.bool()
            & valid_background
            & (depths_t[indices] > 0.05)
        )
        depth = (
            F.smooth_l1_loss(
                rendered_depth[depth_valid],
                depths_t[indices][depth_valid],
                beta=0.01,
                reduction="mean",
            )
            if depth_valid.any()
            else torch.zeros((), device=device)
        )
        prior = (
            (rotation_delta_vector / max(max_rotation, 1e-8)).square().mean()
            + (translation_delta / max(max_translation, 1e-8)).square().mean()
        )
        total = (
            args.w_rep * rep
            + args.w_attr * attr
            + args.w_depth * depth
            + args.w_prior * prior
        )
        row = {
            "step": step + 1,
            "total": float(total.detach()),
            "rep": float(rep.detach()),
            "attr": float(attr.detach()),
            "depth": float(depth.detach()),
            "prior": float(prior.detach()),
        }
        history.append(row)
        if best is None or row["total"] < best["loss"]:
            best = {
                "loss": row["total"],
                "step": step + 1,
                "rotation_parameter": rotation_parameter.detach().clone(),
                "translation_parameter": translation_parameter.detach().clone(),
            }
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [rotation_parameter, translation_parameter], 1.0
        )
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(json.dumps(row), flush=True)

    with torch.no_grad():
        rotation_delta_vector = (
            torch.tanh(best["rotation_parameter"]) * max_rotation
        )
        rotation_delta = so3_exp_map(rotation_delta_vector[None])[0]
        translation_delta = (
            torch.tanh(best["translation_parameter"]) * max_translation
        )
        refined_rotation = base_rotation @ rotation_delta
        refined_translation = base_translation + torch.einsum(
            "fij,j->fi", base_rotation, translation_delta
        )
    refined_rotation_np = refined_rotation.cpu().numpy()
    refined_translation_np = refined_translation.cpu().numpy()
    for index, raw_key in enumerate(raw_keys):
        pose = base_poses_np[index].copy()
        pose[:3, :3] = refined_rotation_np[index]
        pose[:3, 3] = refined_translation_np[index]
        output_rows[raw_key]["object_in_camera"] = pose.astype(float).tolist()

    residual_matrix = np.eye(4, dtype=np.float64)
    residual_matrix[:3, :3] = rotation_delta.cpu().numpy()
    residual_matrix[:3, 3] = translation_delta.cpu().numpy()
    audit = {
        "source_base_json": str(base_path),
        "frame_map_json": str(frame_map_path),
        "mesh": str(mesh_path),
        "uses_gt_object_pose": True,
        "residual_frame": "sam_object_local",
        "shared_residual": residual_matrix.tolist(),
        "rotation_residual_deg": float(
            np.degrees(np.linalg.norm(rotation_delta_vector.cpu().numpy()))
        ),
        "translation_residual_mm": (
            translation_delta.cpu().numpy() * 1000.0
        ).tolist(),
        "translation_residual_norm_mm": float(
            torch.linalg.norm(translation_delta).cpu() * 1000.0
        ),
        "best_step": int(best["step"]),
        "best_loss": float(best["loss"]),
        "settings": vars(args),
        "history_final": history[-1],
    }
    output_payload["shared_canonical_refinement"] = audit
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    out_path.with_name(f"{out_path.stem}_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    out_path.with_name(f"{out_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
