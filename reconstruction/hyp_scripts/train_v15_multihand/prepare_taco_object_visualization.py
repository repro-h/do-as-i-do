#!/usr/bin/env python3
"""Build a metric-unit TACO object cache and optional RGB projection audit."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--triplet", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--tool-id", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--mesh-scale", type=float, default=0.01)
    parser.add_argument("--overlay-count", type=int, default=24)
    return parser.parse_args()


def mesh_data(path, scale):
    mesh = trimesh.load(path, process=False)
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError(f"Expected triangular mesh: {path}")
    return (
        np.asarray(mesh.vertices, dtype=np.float32) * scale,
        np.asarray(mesh.faces, dtype=np.int32),
    )


def project(vertices, pose, intrinsics):
    points = vertices @ pose[:3, :3].T + pose[:3, 3]
    pixels = np.full((len(points), 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    if valid.any():
        projected = points[valid] @ intrinsics.T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def draw_mesh(image, vertices, faces, pose, intrinsics, color):
    pixels, valid = project(vertices, pose, intrinsics)
    height, width = image.shape[:2]
    stride = max(1, len(faces) // 2500)
    for face in faces[::stride]:
        if not valid[face].all():
            continue
        points = pixels[face].round().astype(np.int32)
        if (
            (points[:, 0] < -width).any()
            or (points[:, 0] > 2 * width).any()
            or (points[:, 1] < -height).any()
            or (points[:, 1] > 2 * height).any()
        ):
            continue
        cv2.polylines(image, [points.reshape(-1, 1, 2)], True, color, 1)


def main():
    args = parse_args()
    root = Path(args.taco_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    object_root = root / "Object_Poses" / args.triplet / args.sequence
    mesh_root = root / "object_models_released"
    video_path = (
        root / "Egocentric_RGB_Videos" / args.triplet / args.sequence / "color.mp4"
    )
    camera_root = (
        root / "Egocentric_Camera_Parameters" / args.triplet / args.sequence
    )
    tool_pose = np.asarray(
        np.load(object_root / f"tool_{args.tool_id}.npy", allow_pickle=False),
        dtype=np.float32,
    )
    target_pose = np.asarray(
        np.load(object_root / f"target_{args.target_id}.npy", allow_pickle=False),
        dtype=np.float32,
    )
    if tool_pose.ndim != 3 or tool_pose.shape[1:] != (4, 4):
        raise ValueError(f"Invalid tool pose shape: {tool_pose.shape}")
    if target_pose.shape != tool_pose.shape:
        raise ValueError(
            f"Tool/target pose lengths differ: {tool_pose.shape} vs {target_pose.shape}"
        )
    intrinsics = np.asarray(
        np.loadtxt(camera_root / "egocentric_intrinsic.txt"), dtype=np.float32
    )
    tool_vertices, tool_faces = mesh_data(
        mesh_root / f"{args.tool_id}_cm.obj", args.mesh_scale
    )
    target_vertices, target_faces = mesh_data(
        mesh_root / f"{args.target_id}_cm.obj", args.mesh_scale
    )
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_root / "object_cache.npz",
        frame_indices=np.arange(len(tool_pose), dtype=np.int32),
        tool_pose_camera=tool_pose,
        target_pose_camera=target_pose,
        intrinsics=intrinsics,
        mesh_scale=np.float32(args.mesh_scale),
        tool_mesh=np.asarray(str(mesh_root / f"{args.tool_id}_cm.obj")),
        target_mesh=np.asarray(str(mesh_root / f"{args.target_id}_cm.obj")),
        coordinate_frame=np.asarray("taco_egocentric_camera"),
    )

    capture = cv2.VideoCapture(str(video_path))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    overlay_dir = out_root / "overlay"
    overlay_dir.mkdir(exist_ok=True)
    count = min(max(args.overlay_count, 0), len(tool_pose), video_frames)
    if count:
        indices = np.linspace(0, len(tool_pose) - 1, count, dtype=np.int32)
        for output_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, image = capture.read()
            if not ok:
                continue
            draw_mesh(
                image, tool_vertices, tool_faces, tool_pose[frame_index],
                intrinsics, (0, 180, 255),
            )
            draw_mesh(
                image, target_vertices, target_faces, target_pose[frame_index],
                intrinsics, (255, 120, 0),
            )
            cv2.putText(
                image,
                f"frame={frame_index} tool={args.tool_id} target={args.target_id}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            cv2.imwrite(
                str(overlay_dir / f"{output_index:03d}_frame_{frame_index:06d}.jpg"),
                image,
            )
    capture.release()

    summary = {
        "schema_version": "taco_object_visualization_v1",
        "triplet": args.triplet,
        "sequence": args.sequence,
        "video": str(video_path),
        "video_frames": video_frames,
        "pose_frames": int(len(tool_pose)),
        "frame_counts_match": video_frames == len(tool_pose),
        "tool_id": str(args.tool_id),
        "target_id": str(args.target_id),
        "tool_mesh": str(mesh_root / f"{args.tool_id}_cm.obj"),
        "target_mesh": str(mesh_root / f"{args.target_id}_cm.obj"),
        "mesh_scale": args.mesh_scale,
        "coordinate_frame": "taco_egocentric_camera",
        "overlay_count": len(list(overlay_dir.glob("*.jpg"))),
        "cache": str(out_root / "object_cache.npz"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
