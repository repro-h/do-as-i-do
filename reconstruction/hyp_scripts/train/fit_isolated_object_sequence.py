#!/usr/bin/env python3
"""CHOIR-style isolated 6DoF object fitting for one DexYCB stream."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import yaml
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import so3_exp_map, so3_log_map
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-faces", type=int, default=30000)
    parser.add_argument("--sample-vertices", type=int, default=2048)
    parser.add_argument("--sample-target-pixels", type=int, default=384)
    parser.add_argument("--hand-dilation-px", type=int, default=7)
    parser.add_argument("--max-rotation-deg", type=float, default=12.0)
    parser.add_argument("--max-xy-mm", type=float, default=20.0)
    parser.add_argument("--max-z-mm", type=float, default=30.0)
    parser.add_argument("--xy-only-steps", type=int, default=30)
    parser.add_argument("--unlock-z-step", type=int, default=70)
    parser.add_argument("--w-rep", type=float, default=2.0)
    parser.add_argument("--w-attr", type=float, default=1.0)
    parser.add_argument("--w-depth", type=float, default=2.0)
    parser.add_argument("--w-translation-temp", type=float, default=0.5)
    parser.add_argument("--w-rotation-temp", type=float, default=0.5)
    parser.add_argument("--w-static", type=float, default=0.25)
    parser.add_argument("--w-prior", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_frame_id(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    if value.startswith("color_"):
        value = value.split("_")[-1]
    return value.zfill(6)


def load_mesh(path: Path, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
        used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[used]
        faces = inverse.reshape(-1, 3)
    return vertices, faces


def load_pose_rows(payload: dict) -> tuple[str, dict]:
    for key in ("by_frame", "frames"):
        if isinstance(payload.get(key), dict):
            return key, payload[key]
    raise KeyError("FoundationPose JSON has no by_frame/frames dictionary")


def resolve_depth_path(stream_dir: Path, frame_id: str) -> Path:
    names = (
        f"aligned_depth_to_color_{frame_id}.png",
        f"depth_{frame_id}.png",
        f"aligned_depth_{frame_id}.png",
    )
    for name in names:
        path = stream_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No aligned depth for frame {frame_id} in {stream_dir}"
    )


def load_segmentation(path: Path) -> np.ndarray:
    with np.load(path) as data:
        segmentation = np.asarray(data["seg"])
    return np.squeeze(segmentation)


def target_object_id(stream_dir: Path) -> int:
    meta = yaml.safe_load(
        (stream_dir.parent / "meta.yml").read_text(encoding="utf-8")
    ) or {}
    ycb_ids = list(meta.get("ycb_ids", []) or [])
    grasp_index = int(meta.get("ycb_grasp_ind", 0))
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(f"Invalid ycb_grasp_ind={grasp_index}")
    return int(ycb_ids[grasp_index])


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def sample_pixels(mask: np.ndarray, count: int, seed: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)
    if len(xs) > count:
        generator = np.random.default_rng(seed)
        selected = generator.choice(len(xs), count, replace=False)
        xs, ys = xs[selected], ys[selected]
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def camera_to_pytorch3d(vertices: torch.Tensor) -> torch.Tensor:
    result = vertices.clone()
    result[..., 0] *= -1.0
    result[..., 1] *= -1.0
    return result


def rotation_metrics(matrices: np.ndarray) -> dict:
    rotations = Rotation.from_matrix(matrices)
    if len(rotations) < 2:
        return {}
    velocity = np.stack(
        [
            (rotations[index].inv() * rotations[index + 1]).as_rotvec()
            for index in range(len(rotations) - 1)
        ]
    )
    speed = np.degrees(np.linalg.norm(velocity, axis=-1))
    acceleration = (
        np.degrees(np.linalg.norm(velocity[1:] - velocity[:-1], axis=-1))
        if len(velocity) > 1
        else np.empty(0)
    )
    return {
        "speed_deg_per_frame": distribution(speed),
        "acceleration_deg_per_frame2": distribution(acceleration),
    }


def translation_metrics(values: np.ndarray) -> dict:
    speed = np.linalg.norm(values[1:] - values[:-1], axis=-1) * 1000.0
    acceleration = (
        np.linalg.norm(values[2:] - 2 * values[1:-1] + values[:-2], axis=-1)
        * 1000.0
    )
    return {
        "speed_mm_per_frame": distribution(speed),
        "acceleration_mm_per_frame2": distribution(acceleration),
    }


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    pose_path = Path(args.foundationpose_json).expanduser().resolve()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    mesh_path = Path(args.mesh).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()

    pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
    output_payload = copy.deepcopy(pose_payload)
    _, pose_rows = load_pose_rows(pose_payload)
    _, output_rows = load_pose_rows(output_payload)
    frame_map = json.loads(frame_map_path.read_text(encoding="utf-8"))
    frame_rows = frame_map["frames"]
    stream_dir = Path(frame_map.get("stream_dir") or Path(frame_rows[0]["label_path"]).parent)
    object_id = target_object_id(stream_dir)

    frame_ids = []
    raw_keys = []
    base_poses = []
    selected_rows = []
    for frame_row in frame_rows:
        frame_id = normalize_frame_id(frame_row["original_frame"])
        raw_key = frame_id if frame_id in pose_rows else str(int(frame_id))
        pose_row = pose_rows.get(raw_key)
        if pose_row is None or pose_row.get("object_in_camera") is None:
            continue
        pose = np.asarray(pose_row["object_in_camera"], dtype=np.float32)
        if pose.size != 16 or not np.isfinite(pose).all():
            continue
        frame_ids.append(frame_id)
        raw_keys.append(raw_key)
        base_poses.append(pose.reshape(4, 4))
        selected_rows.append(frame_row)
    if len(base_poses) < 3:
        raise RuntimeError("Need at least three valid FoundationPose frames")

    base_poses_np = np.stack(base_poses)
    base_rotation = torch.as_tensor(
        base_poses_np[:, :3, :3], dtype=torch.float32, device=device
    )
    base_translation = torch.as_tensor(
        base_poses_np[:, :3, 3], dtype=torch.float32, device=device
    )
    intrinsics = np.asarray(pose_payload["intrinsics"], dtype=np.float32).reshape(3, 3)
    source_scale = float(pose_payload.get("source_mesh_scale", 1.0))
    mesh_vertices_np, mesh_faces_np = load_mesh(mesh_path, args.max_faces)
    mesh_vertices = torch.as_tensor(
        mesh_vertices_np * source_scale, dtype=torch.float32, device=device
    )
    mesh_faces = torch.as_tensor(mesh_faces_np, dtype=torch.int64, device=device)
    sample_indices = np.linspace(
        0,
        len(mesh_vertices_np) - 1,
        min(args.sample_vertices, len(mesh_vertices_np)),
        dtype=np.int64,
    )
    sampled_vertices = mesh_vertices[torch.as_tensor(sample_indices, device=device)]

    first_seg = load_segmentation(Path(selected_rows[0]["label_path"]))
    source_height, source_width = first_seg.shape
    width = max(32, int(round(source_width * args.render_scale)))
    height = max(32, int(round(source_height * args.render_scale)))
    scale_x = width / source_width
    scale_y = height / source_height
    focal = torch.tensor(
        [[intrinsics[0, 0] * scale_x, intrinsics[1, 1] * scale_y]],
        dtype=torch.float32,
        device=device,
    )
    principal = torch.tensor(
        [[intrinsics[0, 2] * scale_x, intrinsics[1, 2] * scale_y]],
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
    hand_masks = []
    valid_backgrounds = []
    depths = []
    target_pixels = []
    dilation = max(1, int(round(args.hand_dilation_px * args.render_scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    for index, (frame_id, row) in enumerate(zip(frame_ids, selected_rows)):
        segmentation = load_segmentation(Path(row["label_path"]))
        object_mask = resize_mask(segmentation == object_id, (width, height))
        hand_mask = resize_mask(segmentation == 255, (width, height))
        hand_dilated = cv2.dilate(
            hand_mask.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
        depth_raw = cv2.imread(
            str(resolve_depth_path(stream_dir, frame_id)), cv2.IMREAD_UNCHANGED
        )
        if depth_raw is None:
            raise FileNotFoundError(f"Cannot read depth for {frame_id}")
        depth = cv2.resize(
            depth_raw.astype(np.float32) / 1000.0,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        object_masks.append(object_mask)
        hand_masks.append(hand_mask)
        valid_backgrounds.append(~hand_dilated)
        depths.append(depth)
        target_pixels.append(
            sample_pixels(object_mask, args.sample_target_pixels, args.seed + index)
        )

    object_masks_t = torch.as_tensor(
        np.stack(object_masks), dtype=torch.float32, device=device
    )
    hand_masks_t = torch.as_tensor(
        np.stack(hand_masks), dtype=torch.bool, device=device
    )
    valid_backgrounds_t = torch.as_tensor(
        np.stack(valid_backgrounds), dtype=torch.bool, device=device
    )
    depths_t = torch.as_tensor(np.stack(depths), dtype=torch.float32, device=device)

    rotation_parameter = torch.nn.Parameter(
        torch.zeros((len(frame_ids), 3), dtype=torch.float32, device=device)
    )
    translation_parameter = torch.nn.Parameter(
        torch.zeros((len(frame_ids), 3), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        [rotation_parameter, translation_parameter], lr=args.lr
    )
    max_rotation = math.radians(args.max_rotation_deg)
    max_translation = torch.tensor(
        [args.max_xy_mm, args.max_xy_mm, args.max_z_mm],
        dtype=torch.float32,
        device=device,
    ) / 1000.0
    base_relative_rotation = so3_log_map(
        base_rotation[:-1].transpose(1, 2) @ base_rotation[1:]
    )
    base_translation_velocity = base_translation[1:] - base_translation[:-1]
    static = (
        (torch.linalg.norm(base_relative_rotation, dim=-1) < math.radians(2.0))
        & (torch.linalg.norm(base_translation_velocity, dim=-1) < 0.002)
    )
    history = []

    for step in range(args.steps):
        rotation_enabled = step >= args.xy_only_steps
        z_enabled = step >= args.unlock_z_step
        rotation_delta = torch.tanh(rotation_parameter) * max_rotation
        if not rotation_enabled:
            rotation_delta = rotation_delta * 0.0
        translation_delta = torch.tanh(translation_parameter) * max_translation
        if not z_enabled:
            translation_delta = translation_delta * torch.tensor(
                [1.0, 1.0, 0.0], device=device
            )
        rotation = so3_exp_map(rotation_delta) @ base_rotation
        translation = base_translation + translation_delta

        begin = (step * args.batch_size) % len(frame_ids)
        batch_indices = [
            (begin + offset) % len(frame_ids)
            for offset in range(min(args.batch_size, len(frame_ids)))
        ]
        indices = torch.as_tensor(batch_indices, dtype=torch.int64, device=device)
        vertices_camera = (
            mesh_vertices[None] @ rotation[indices].transpose(1, 2)
            + translation[indices, None, :]
        )
        vertices_p3d = camera_to_pytorch3d(vertices_camera)
        meshes = Meshes(
            verts=[value for value in vertices_p3d],
            faces=[mesh_faces for _ in batch_indices],
        )
        fragments = rasterizer(meshes)
        alpha = silhouette_shader(fragments, meshes)[..., 3]
        target_mask = object_masks_t[indices]
        valid_background = valid_backgrounds_t[indices]
        rep = (
            F.relu(alpha - target_mask).square() * valid_background.float()
        ).sum() / valid_background.float().sum().clamp_min(1.0)

        attr_values = []
        projected_vertices = (
            sampled_vertices[None] @ rotation[indices].transpose(1, 2)
            + translation[indices, None, :]
        )
        projected_x = (
            focal[0, 0] * projected_vertices[..., 0]
            / projected_vertices[..., 2].clamp_min(1e-4)
            + principal[0, 0]
        )
        projected_y = (
            focal[0, 1] * projected_vertices[..., 1]
            / projected_vertices[..., 2].clamp_min(1e-4)
            + principal[0, 1]
        )
        projected = torch.stack([projected_x, projected_y], dim=-1)
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
            & ~hand_masks_t[indices]
            & (depths_t[indices] > 0.05)
        )
        depth = F.smooth_l1_loss(
            rendered_depth[depth_valid],
            depths_t[indices][depth_valid],
            beta=0.01,
            reduction="mean",
        ) if depth_valid.any() else torch.zeros((), device=device)

        relative_rotation = so3_log_map(
            rotation[:-1].transpose(1, 2) @ rotation[1:]
        )
        translation_velocity = translation[1:] - translation[:-1]
        rotation_acceleration = relative_rotation[1:] - relative_rotation[:-1]
        translation_acceleration = (
            translation_velocity[1:] - translation_velocity[:-1]
        )
        rotation_temp = (
            F.smooth_l1_loss(
                relative_rotation,
                base_relative_rotation,
                beta=math.radians(2.0),
            )
            + rotation_acceleration.square().mean()
        )
        translation_temp = (
            F.smooth_l1_loss(
                translation_velocity,
                base_translation_velocity,
                beta=0.002,
            )
            + translation_acceleration.square().mean()
        )
        if static.any():
            static_loss = (
                relative_rotation[static].square().mean()
                + translation_velocity[static].square().mean()
            )
        else:
            static_loss = torch.zeros((), device=device)
        prior = rotation_delta.square().mean() + translation_delta.square().mean()
        total = (
            args.w_rep * rep
            + args.w_attr * attr
            + args.w_depth * depth
            + args.w_translation_temp * translation_temp
            + args.w_rotation_temp * rotation_temp
            + args.w_static * static_loss
            + args.w_prior * prior
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [rotation_parameter, translation_parameter], 1.0
        )
        optimizer.step()
        row = {
            "step": step + 1,
            "total": float(total.detach()),
            "rep": float(rep.detach()),
            "attr": float(attr.detach()),
            "depth": float(depth.detach()),
            "translation_temp": float(translation_temp.detach()),
            "rotation_temp": float(rotation_temp.detach()),
            "static": float(static_loss.detach()),
            "prior": float(prior.detach()),
        }
        history.append(row)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(json.dumps(row), flush=True)

    with torch.no_grad():
        rotation_delta = torch.tanh(rotation_parameter) * max_rotation
        translation_delta = torch.tanh(translation_parameter) * max_translation
        fitted_rotation = so3_exp_map(rotation_delta) @ base_rotation
        fitted_translation = base_translation + translation_delta
    fitted_rotation_np = fitted_rotation.cpu().numpy()
    fitted_translation_np = fitted_translation.cpu().numpy()
    for index, raw_key in enumerate(raw_keys):
        pose = base_poses_np[index].copy()
        pose[:3, :3] = fitted_rotation_np[index]
        pose[:3, 3] = fitted_translation_np[index]
        output_rows[raw_key]["object_in_camera"] = pose.astype(float).tolist()

    audit = {
        "source_foundationpose_json": str(pose_path),
        "frame_map_json": str(frame_map_path),
        "mesh": str(mesh_path),
        "uses_gt_object_pose": False,
        "num_fitted_frames": len(frame_ids),
        "settings": vars(args),
        "translation_residual_norm_mm": distribution(
            np.linalg.norm(
                fitted_translation_np - base_poses_np[:, :3, 3], axis=-1
            )
            * 1000.0
        ),
        "rotation_residual_deg": distribution(
            np.degrees(np.linalg.norm(rotation_delta.cpu().numpy(), axis=-1))
        ),
        "translation_temporal": {
            "before": translation_metrics(base_poses_np[:, :3, 3]),
            "after": translation_metrics(fitted_translation_np),
        },
        "rotation_temporal": {
            "before": rotation_metrics(base_poses_np[:, :3, :3]),
            "after": rotation_metrics(fitted_rotation_np),
        },
        "final_losses": history[-1],
    }
    output_payload["isolated_object_fitting"] = audit
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    audit_path = out_path.with_name(f"{out_path.stem}_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    history_path = out_path.with_name(f"{out_path.stem}_history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {audit_path}")
    print(f"Wrote: {history_path}")


if __name__ == "__main__":
    main()
