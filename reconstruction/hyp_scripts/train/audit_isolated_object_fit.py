#!/usr/bin/env python3
"""Audit initial/fitted object poses against visible masks and RGB-D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

from fit_isolated_object_sequence import (
    camera_to_pytorch3d,
    distribution,
    load_mesh,
    load_pose_rows,
    load_segmentation,
    normalize_frame_id,
    resolve_depth_path,
    target_object_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-json", required=True)
    parser.add_argument("--fitted-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--preview-dir", required=True)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--max-faces", type=int, default=30000)
    parser.add_argument("--hand-dilation-px", type=int, default=7)
    parser.add_argument("--worst-k", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_pose(rows: dict, frame_id: str) -> Optional[np.ndarray]:
    row = rows.get(frame_id)
    if row is None:
        row = rows.get(str(int(frame_id)))
    if row is None or row.get("object_in_camera") is None:
        return None
    pose = np.asarray(row["object_in_camera"], dtype=np.float32)
    if pose.size != 16 or not np.isfinite(pose).all():
        return None
    return pose.reshape(4, 4)


def render_pose(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pose: np.ndarray,
    rasterizer: MeshRasterizer,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = torch.as_tensor(
        pose[:3, :3], dtype=torch.float32, device=vertices.device
    )
    translation = torch.as_tensor(
        pose[:3, 3], dtype=torch.float32, device=vertices.device
    )
    camera_vertices = vertices @ rotation.T + translation
    mesh = Meshes(
        verts=[camera_to_pytorch3d(camera_vertices)],
        faces=[faces],
    )
    with torch.no_grad():
        fragments = rasterizer(mesh)
    mask = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy()
    depth = fragments.zbuf[0, ..., 0].cpu().numpy()
    return mask, depth


def frame_metrics(
    rendered_mask: np.ndarray,
    rendered_depth: np.ndarray,
    object_mask: np.ndarray,
    hand_dilated: np.ndarray,
    observed_depth: np.ndarray,
) -> dict:
    visible_render = rendered_mask & ~hand_dilated
    intersection = visible_render & object_mask
    union = visible_render | object_mask
    valid_render_pixels = int(visible_render.sum())
    object_pixels = int(object_mask.sum())
    valid_depth = (
        intersection
        & np.isfinite(rendered_depth)
        & (rendered_depth > 0.05)
        & (observed_depth > 0.05)
    )
    depth_error = (
        np.abs(rendered_depth[valid_depth] - observed_depth[valid_depth]) * 1000.0
    )
    spill = visible_render & ~object_mask
    return {
        "visible_iou": float(intersection.sum() / max(1, int(union.sum()))),
        "mask_coverage": float(intersection.sum() / max(1, object_pixels)),
        "visible_precision": float(
            intersection.sum() / max(1, valid_render_pixels)
        ),
        "spill_ratio": float(spill.sum() / max(1, valid_render_pixels)),
        "depth_count": int(len(depth_error)),
        "depth_median_mm": (
            float(np.quantile(depth_error, 0.5)) if len(depth_error) else None
        ),
        "depth_p90_mm": (
            float(np.quantile(depth_error, 0.9)) if len(depth_error) else None
        ),
    }


def aggregate(rows: list[dict], prefix: str) -> dict:
    result = {}
    for key in (
        "visible_iou",
        "mask_coverage",
        "visible_precision",
        "spill_ratio",
        "depth_median_mm",
        "depth_p90_mm",
    ):
        values = [
            row[prefix][key]
            for row in rows
            if row[prefix].get(key) is not None
        ]
        result[key] = distribution(np.asarray(values, dtype=np.float64))
    return result


def draw_preview(
    image_path: Path,
    object_mask: np.ndarray,
    initial_mask: np.ndarray,
    fitted_mask: np.ndarray,
    initial: dict,
    fitted: dict,
    output: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    object_mask = cv2.resize(
        object_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    initial_mask = cv2.resize(
        initial_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    fitted_mask = cv2.resize(
        fitted_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    for mask, color in (
        (object_mask, (255, 255, 0)),
        (initial_mask, (255, 0, 255)),
        (fitted_mask, (0, 255, 0)),
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)
    text = (
        f"IoU {initial['visible_iou']:.3f}->{fitted['visible_iou']:.3f}  "
        f"depth {initial.get('depth_median_mm')}->{fitted.get('depth_median_mm')} mm"
    )
    cv2.putText(
        image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )
    cv2.imwrite(str(output), image)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    initial_payload = json.loads(
        Path(args.initial_json).expanduser().resolve().read_text(encoding="utf-8")
    )
    fitted_payload = json.loads(
        Path(args.fitted_json).expanduser().resolve().read_text(encoding="utf-8")
    )
    frame_map = json.loads(
        Path(args.frame_map_json).expanduser().resolve().read_text(encoding="utf-8")
    )
    _, initial_rows = load_pose_rows(initial_payload)
    _, fitted_rows = load_pose_rows(fitted_payload)
    rows = frame_map["frames"]
    stream_dir = Path(frame_map.get("stream_dir") or Path(rows[0]["label_path"]).parent)
    object_id = target_object_id(stream_dir)
    vertices_np, faces_np = load_mesh(
        Path(args.mesh).expanduser().resolve(), args.max_faces
    )
    scale = float(initial_payload.get("source_mesh_scale", 1.0))
    vertices = torch.as_tensor(
        vertices_np * scale, dtype=torch.float32, device=device
    )
    faces = torch.as_tensor(faces_np, dtype=torch.int64, device=device)
    first_segmentation = load_segmentation(Path(rows[0]["label_path"]))
    source_height, source_width = first_segmentation.shape
    width = max(32, int(round(source_width * args.render_scale)))
    height = max(32, int(round(source_height * args.render_scale)))
    intrinsics = np.asarray(initial_payload["intrinsics"], dtype=np.float32).reshape(3, 3)
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
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=max(200000, len(faces)),
        ),
    )
    dilation = max(1, int(round(args.hand_dilation_px * args.render_scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    metrics_rows = []
    preview_cache = {}
    for frame_row in rows:
        frame_id = normalize_frame_id(frame_row["original_frame"])
        initial_pose = resolve_pose(initial_rows, frame_id)
        fitted_pose = resolve_pose(fitted_rows, frame_id)
        if initial_pose is None or fitted_pose is None:
            continue
        segmentation = load_segmentation(Path(frame_row["label_path"]))
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
        hand_dilated = cv2.dilate(hand_mask, kernel, iterations=1).astype(bool)
        observed_depth = cv2.imread(
            str(resolve_depth_path(stream_dir, frame_id)), cv2.IMREAD_UNCHANGED
        )
        observed_depth = cv2.resize(
            observed_depth.astype(np.float32) / 1000.0,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        initial_mask, initial_depth = render_pose(
            vertices, faces, initial_pose, rasterizer
        )
        fitted_mask, fitted_depth = render_pose(
            vertices, faces, fitted_pose, rasterizer
        )
        initial_metrics = frame_metrics(
            initial_mask, initial_depth, object_mask, hand_dilated, observed_depth
        )
        fitted_metrics = frame_metrics(
            fitted_mask, fitted_depth, object_mask, hand_dilated, observed_depth
        )
        metrics_rows.append(
            {
                "frame": frame_id,
                "initial": initial_metrics,
                "fitted": fitted_metrics,
            }
        )
        preview_cache[frame_id] = (
            Path(frame_row["image_path"]),
            object_mask,
            initial_mask,
            fitted_mask,
            initial_metrics,
            fitted_metrics,
        )

    if not metrics_rows:
        raise RuntimeError("No frames were audited")
    ranked = sorted(
        metrics_rows,
        key=lambda row: row["fitted"]["visible_iou"]
        - row["initial"]["visible_iou"],
    )
    preview_dir = Path(args.preview_dir).expanduser().resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)
    for row in ranked[: args.worst_k]:
        values = preview_cache[row["frame"]]
        draw_preview(*values, preview_dir / f"{row['frame']}_worst.jpg")

    audit = {
        "num_frames": len(metrics_rows),
        "initial": aggregate(metrics_rows, "initial"),
        "fitted": aggregate(metrics_rows, "fitted"),
        "worst_fitted_changes": ranked[: args.worst_k],
        "frames": metrics_rows,
        "legend": {
            "cyan": "visible object mask",
            "magenta": "initial FoundationPose render",
            "green": "isolated fitted render",
        },
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "frames"}, indent=2))
    print(f"Wrote: {out_path}")
    print(f"Previews: {preview_dir}")


if __name__ == "__main__":
    main()
