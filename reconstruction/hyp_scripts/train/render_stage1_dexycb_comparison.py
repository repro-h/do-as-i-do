#!/usr/bin/env python3
"""Render Stage-1 before/after hand-object overlays to fixed-timeline MP4s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--original-layout", required=True)
    parser.add_argument("--original-hand-meshes", required=True)
    parser.add_argument("--corrected-layout", required=True)
    parser.add_argument("--corrected-hand-meshes", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--view",
        choices=("camera", "side", "orbit"),
        default="camera",
        help="RGB camera overlay, fixed 3D side view, or rotating 3D view.",
    )
    parser.add_argument("--object-color", default="0.95,0.20,0.55")
    parser.add_argument("--hand-color", default="0.10,0.65,0.95")
    return parser.parse_args()


def parse_color(value: str) -> tuple[float, float, float]:
    color = tuple(float(item) for item in value.split(","))
    if len(color) != 3 or any(item < 0.0 or item > 1.0 for item in color):
        raise ValueError(f"Expected RGB values in [0,1], got {value}")
    return color


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(loaded).__name__}")
    return (
        np.asarray(loaded.vertices, dtype=np.float32),
        np.asarray(loaded.faces, dtype=np.int64),
    )


def load_layout(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload.get("objects", []):
        frame_index = row.get("frame_idx", row.get("frame_index"))
        if frame_index is not None:
            result[int(frame_index)] = row["local_to_scene"]
    return result


def load_hand(path: Path, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        vertices = np.asarray(data[f"{side}_vertices"], dtype=np.float32)
        faces = np.asarray(data[f"{side}_faces"], dtype=np.int64)
        valid = np.asarray(
            data[f"{side}_valid"]
            if f"{side}_valid" in data
            else np.ones(len(vertices)),
            dtype=bool,
        )
    return vertices, faces, valid


def numeric_frames(path: Path) -> list[Path]:
    paths = []
    for candidate in path.iterdir():
        if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            frame_index = int(candidate.stem)
        except ValueError:
            continue
        paths.append((frame_index, candidate))
    paths.sort()
    if not paths:
        raise RuntimeError(f"No numeric RGB frames found in {path}")
    expected = list(range(paths[-1][0] + 1))
    actual = [index for index, _ in paths]
    if actual != expected:
        raise RuntimeError("RGB frame directory is not a contiguous 0-based timeline")
    return [candidate for _, candidate in paths]


def make_renderer(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    max_faces_per_bin: int,
) -> MeshRenderer:
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=((height, width),),
        in_ndc=False,
        device=device,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=max_faces_per_bin,
        ),
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    return MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )


def camera_to_pytorch3d(vertices: torch.Tensor) -> torch.Tensor:
    converted = vertices.clone()
    converted[:, 0] *= -1.0
    converted[:, 1] *= -1.0
    return converted


def object_vertices_camera(
    vertices: np.ndarray, transform: dict, device: torch.device
) -> torch.Tensor:
    quaternion = transform["quat_wxyz_camera_frame"]
    xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    rotation = Rotation.from_quat(xyzw).as_matrix().astype(np.float32)
    translation = np.asarray(
        transform["translation_camera_frame"], dtype=np.float32
    )
    scale = np.asarray(transform.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    transformed = vertices * scale[None, :]
    transformed = transformed @ rotation.T + translation[None, :]
    return torch.as_tensor(transformed, dtype=torch.float32, device=device)


def transform_to_view(
    vertices: torch.Tensor,
    center: np.ndarray,
    forward: np.ndarray,
    distance: float,
) -> torch.Tensor:
    """Map camera-frame points to a synthetic right/down/forward camera."""
    down = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    forward = np.asarray(forward, dtype=np.float32)
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    relative = vertices - torch.as_tensor(center, dtype=torch.float32, device=vertices.device)
    basis = torch.as_tensor(
        np.stack([right, down, forward], axis=1),
        dtype=torch.float32,
        device=vertices.device,
    )
    transformed = relative @ basis
    transformed[:, 2] += float(distance)
    return transformed


def render_overlay(
    image_bgr: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    object_transform: Optional[dict],
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    hand_valid: bool,
    renderer: MeshRenderer,
    device: torch.device,
    object_color: tuple[float, float, float],
    hand_color: tuple[float, float, float],
    alpha: float,
    view_spec: Optional[tuple[np.ndarray, np.ndarray, float]] = None,
) -> np.ndarray:
    vertices_list = []
    faces_list = []
    colors_list = []
    vertex_offset = 0

    if object_transform is not None:
        vertices = object_vertices_camera(object_vertices, object_transform, device)
        if view_spec is not None:
            vertices = transform_to_view(vertices, *view_spec)
        faces = torch.as_tensor(object_faces, dtype=torch.int64, device=device)
        vertices_list.append(camera_to_pytorch3d(vertices))
        faces_list.append(faces + vertex_offset)
        colors_list.append(
            torch.tensor(object_color, dtype=torch.float32, device=device)
            .expand(len(vertices), 3)
        )
        vertex_offset += len(vertices)

    if hand_valid and np.isfinite(hand_vertices).all():
        vertices = torch.as_tensor(hand_vertices, dtype=torch.float32, device=device)
        if view_spec is not None:
            vertices = transform_to_view(vertices, *view_spec)
        faces = torch.as_tensor(hand_faces, dtype=torch.int64, device=device)
        vertices_list.append(camera_to_pytorch3d(vertices))
        faces_list.append(faces + vertex_offset)
        colors_list.append(
            torch.tensor(hand_color, dtype=torch.float32, device=device)
            .expand(len(vertices), 3)
        )

    if not vertices_list:
        return image_bgr.copy()

    mesh = Meshes(
        verts=[torch.cat(vertices_list, dim=0)],
        faces=[torch.cat(faces_list, dim=0)],
        textures=TexturesVertex(
            verts_features=torch.cat(colors_list, dim=0).unsqueeze(0)
        ),
    )
    with torch.no_grad():
        rendered = renderer(mesh)[0].detach().cpu().numpy()
    render_bgr = (rendered[..., :3][..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
    mask = rendered[..., 3] > 0.01
    result = image_bgr.copy()
    result[mask] = cv2.addWeighted(
        image_bgr[mask], 1.0 - alpha, render_bgr[mask], alpha, 0.0
    )
    return result


def scene_bounds(
    object_vertices: np.ndarray,
    layouts: tuple[dict[int, dict], dict[int, dict]],
    hands: tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
) -> tuple[np.ndarray, float]:
    samples = []
    vertex_stride = max(1, len(object_vertices) // 2000)
    for layout in layouts:
        for transform in layout.values():
            quaternion = transform["quat_wxyz_camera_frame"]
            xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
            rotation = Rotation.from_quat(xyzw).as_matrix().astype(np.float32)
            translation = np.asarray(
                transform["translation_camera_frame"], dtype=np.float32
            )
            scale = np.asarray(
                transform.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32
            )
            vertices = object_vertices[::vertex_stride] * scale[None, :]
            samples.append(vertices @ rotation.T + translation[None, :])
    for vertices, _, valid in hands:
        valid_vertices = vertices[valid]
        if len(valid_vertices):
            samples.append(valid_vertices[:, ::20, :].reshape(-1, 3))
    if not samples:
        raise RuntimeError("Cannot estimate 3D scene bounds")
    points = np.concatenate(samples, axis=0)
    lower = np.quantile(points, 0.01, axis=0)
    upper = np.quantile(points, 0.99, axis=0)
    center = (lower + upper) * 0.5
    extent = float(np.max(upper - lower))
    return center.astype(np.float32), max(extent, 0.05)


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = numeric_frames(Path(args.frames_dir).expanduser().resolve())
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read {frames[0]}")
    height, width = first.shape[:2]

    object_vertices, object_faces = load_mesh(Path(args.mesh).expanduser().resolve())
    original_layout = load_layout(Path(args.original_layout).expanduser().resolve())
    corrected_layout = load_layout(Path(args.corrected_layout).expanduser().resolve())
    original_hand = load_hand(
        Path(args.original_hand_meshes).expanduser().resolve(), args.hand_side
    )
    corrected_hand = load_hand(
        Path(args.corrected_hand_meshes).expanduser().resolve(), args.hand_side
    )
    max_faces = max(200000, int(len(object_faces) + len(original_hand[1])))
    view_center = None
    view_distance = None
    if args.view == "camera":
        render_fx, render_fy = args.fx, args.fy
        render_cx, render_cy = args.cx, args.cy
    else:
        view_center, scene_extent = scene_bounds(
            object_vertices,
            (original_layout, corrected_layout),
            (original_hand, corrected_hand),
        )
        view_distance = scene_extent * 1.35
        render_fx = render_fy = min(width, height) * 1.05
        render_cx, render_cy = width * 0.5, height * 0.5
        print(
            f"3D view center={view_center.tolist()} "
            f"extent={scene_extent:.4f} distance={view_distance:.4f}"
        )
    renderer = make_renderer(
        width,
        height,
        render_fx,
        render_fy,
        render_cx,
        render_cy,
        device,
        max_faces,
    )
    original_writer = make_writer(out_dir / "original.mp4", args.fps, (width, height))
    corrected_writer = make_writer(
        out_dir / "stage1_corrected.mp4", args.fps, (width, height)
    )
    comparison_writer = make_writer(
        out_dir / "before_after.mp4", args.fps, (width * 2, height)
    )
    object_color = parse_color(args.object_color)
    hand_color = parse_color(args.hand_color)

    try:
        for frame_index, frame_path in enumerate(frames):
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"Failed to read {frame_path}")
            view_spec = None
            if args.view != "camera":
                image = np.full_like(image, 245)
                if args.view == "side":
                    forward = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
                else:
                    angle = 2.0 * np.pi * frame_index / max(len(frames), 1)
                    forward = np.asarray(
                        [np.sin(angle), 0.0, np.cos(angle)], dtype=np.float32
                    )
                view_spec = (view_center, forward, view_distance)
            original = render_overlay(
                image,
                object_vertices,
                object_faces,
                original_layout.get(frame_index),
                original_hand[0][frame_index],
                original_hand[1],
                bool(original_hand[2][frame_index]),
                renderer,
                device,
                object_color,
                hand_color,
                args.alpha,
                view_spec,
            )
            corrected = render_overlay(
                image,
                object_vertices,
                object_faces,
                corrected_layout.get(frame_index),
                corrected_hand[0][frame_index],
                corrected_hand[1],
                bool(corrected_hand[2][frame_index]),
                renderer,
                device,
                object_color,
                hand_color,
                args.alpha,
                view_spec,
            )
            cv2.putText(
                original,
                "Before",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                corrected,
                "Stage 1",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                original,
                f"{frame_index:06d}",
                (16, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                corrected,
                f"{frame_index:06d}",
                (16, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            original_writer.write(original)
            corrected_writer.write(corrected)
            comparison_writer.write(np.concatenate([original, corrected], axis=1))
            print(f"[{frame_index + 1}/{len(frames)}] {frame_path.name}", flush=True)
    finally:
        original_writer.release()
        corrected_writer.release()
        comparison_writer.release()

    summary = {
        "num_frames": len(frames),
        "fps": args.fps,
        "view": args.view,
        "original": str(out_dir / "original.mp4"),
        "corrected": str(out_dir / "stage1_corrected.mp4"),
        "comparison": str(out_dir / "before_after.mp4"),
    }
    (out_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
