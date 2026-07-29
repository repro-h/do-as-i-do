#!/usr/bin/env python3
"""Visualize calibrated Pi3 hand/object points against tracked meshes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
import viser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--object-pose-json", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--object-label", type=int, required=True)
    parser.add_argument("--hand-label", type=int, default=255)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--full-confidence-threshold", type=float, default=0.1)
    parser.add_argument("--show-full-point-cloud", action="store_true")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--mesh-scale", type=float, default=-1.0)
    parser.add_argument("--object-surface-points", type=int, default=50000)
    parser.add_argument("--point-size", type=float, default=0.0025)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload.get("frame_map", payload))
    if isinstance(rows, dict):
        rows = list(rows.values())
    return sorted(rows, key=lambda row: int(row["output_index"]))


def load_segmentation(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("seg", "segmentation", "label"):
            if key in payload.files:
                return np.asarray(payload[key])
    raise KeyError(f"No segmentation in {path}")


def load_pose(path: Path, frame: int, original_frame: str) -> tuple[np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("by_frame") or payload.get("frames") or payload
    candidates = (f"{frame:06d}", str(original_frame).zfill(6))
    row = None
    if isinstance(rows, dict):
        for key in candidates:
            if key in rows:
                row = rows[key]
                break
    else:
        row = rows[frame]
    if row is None:
        raise KeyError(f"No object pose for frame {frame}")
    value = row
    if isinstance(row, dict):
        value = row.get("object_in_camera") or row.get("pose") or row.get("transform")
    pose = np.asarray(value, dtype=np.float64).reshape(4, 4)
    scale = float(payload.get("source_mesh_scale", payload.get("final_global_scale", 1.0)))
    return pose, scale


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    return mesh


def select_window(audit: dict, frame: int) -> dict:
    candidates = [
        row
        for row in audit["windows"]
        if row["status"] == "ok"
        and int(row["frames"][0]) <= frame <= int(row["frames"][-1])
    ]
    if not candidates:
        raise ValueError(f"No valid calibrated Pi3 window contains frame {frame}")
    return min(
        candidates,
        key=lambda row: abs(
            frame - 0.5 * (int(row["frames"][0]) + int(row["frames"][-1]))
        ),
    )


def main() -> None:
    args = parse_args()
    audit = json.loads(Path(args.audit_json).expanduser().resolve().read_text(encoding="utf-8"))
    rows = load_rows(Path(args.frame_map_json).expanduser().resolve())
    if not 0 <= args.frame < len(rows):
        raise IndexError(f"Frame {args.frame} outside [0, {len(rows) - 1}]")
    row = rows[args.frame]
    window = select_window(audit, args.frame)

    window_path = Path(window["path"])
    with np.load(window_path, allow_pickle=False) as payload:
        frame_indices = np.asarray(payload["frame_indices"], dtype=int)
        local_index = int(np.where(frame_indices == args.frame)[0][0])
        pi3_points = np.asarray(payload["local_points"][local_index], dtype=np.float32)
        confidence = np.asarray(payload["confidence"][local_index], dtype=np.float32)
        K = np.asarray(payload["intrinsics_resized"], dtype=np.float32)
        width, height = np.asarray(payload["resized_wh"], dtype=int)

    ys_grid, xs_grid = np.indices((height, width))
    calibrated_depth = (
        float(window["scale"]) * pi3_points[..., 2]
        + float(window["shift_m"])
    )
    calibrated = np.stack(
        [
            (xs_grid - K[0, 2]) / K[0, 0] * calibrated_depth,
            (ys_grid - K[1, 2]) / K[1, 1] * calibrated_depth,
            calibrated_depth,
        ],
        axis=-1,
    )
    image = cv2.imread(
        str(Path(row["image_path"]).expanduser().resolve()),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise OSError(f"Cannot read RGB frame: {row['image_path']}")
    image = cv2.cvtColor(
        cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    segmentation = load_segmentation(Path(row["label_path"]).expanduser().resolve())
    segmentation = cv2.resize(segmentation, (width, height), interpolation=cv2.INTER_NEAREST)
    finite = np.isfinite(calibrated).all(axis=-1) & (calibrated[..., 2] > 0)
    confident = confidence >= args.confidence_threshold
    full_valid = finite & (
        confidence >= args.full_confidence_threshold
    )
    full_points = calibrated[full_valid]
    full_colors = image[full_valid]
    object_points = calibrated[(segmentation == args.object_label) & finite & confident]
    hand_points = calibrated[(segmentation == args.hand_label) & finite & confident]

    pose, pose_scale = load_pose(
        Path(args.object_pose_json).expanduser().resolve(),
        args.frame,
        str(row["original_frame"]),
    )
    mesh_scale = pose_scale if args.mesh_scale <= 0 else args.mesh_scale
    object_mesh = load_mesh(Path(args.object_mesh).expanduser().resolve(), mesh_scale)
    rng = np.random.default_rng(args.seed)
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    object_surface, _ = trimesh.sample.sample_surface(
        object_mesh, args.object_surface_points
    )
    np.random.set_state(state)
    object_surface = object_surface @ pose[:3, :3].T + pose[:3, 3]

    with np.load(Path(args.hand_npz).expanduser().resolve(), allow_pickle=False) as payload:
        hand_vertices = np.asarray(payload["verts_cam"][args.frame], dtype=np.float32)
        hand_faces = np.asarray(payload["faces"], dtype=np.int64)

    if args.export_dir:
        export_dir = Path(args.export_dir).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        trimesh.points.PointCloud(
            vertices=full_points,
            colors=full_colors,
        ).export(export_dir / f"frame_{args.frame:06d}_full_pointcloud.ply")
        trimesh.points.PointCloud(
            vertices=object_points,
            colors=np.tile(
                np.asarray([[255, 60, 190]], dtype=np.uint8),
                (len(object_points), 1),
            ),
        ).export(export_dir / f"frame_{args.frame:06d}_object_points.ply")
        trimesh.points.PointCloud(
            vertices=hand_points,
            colors=np.tile(
                np.asarray([[255, 210, 40]], dtype=np.uint8),
                (len(hand_points), 1),
            ),
        ).export(export_dir / f"frame_{args.frame:06d}_hand_points.ply")
        trimesh.Trimesh(
            vertices=hand_vertices,
            faces=hand_faces,
            process=False,
        ).export(export_dir / f"frame_{args.frame:06d}_handflow_hand.obj")
        print(f"Exported debug assets: {export_dir}")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    if args.show_full_point_cloud:
        server.scene.add_point_cloud(
            "/pi3_full_rgb_pointcloud",
            points=full_points.astype(np.float32),
            colors=full_colors.astype(np.uint8),
            point_size=args.point_size,
        )
    server.scene.add_point_cloud(
        "/object_mesh_surface",
        points=object_surface.astype(np.float32),
        colors=np.tile(np.asarray([[50, 205, 90]], dtype=np.uint8), (len(object_surface), 1)),
        point_size=args.point_size,
    )
    server.scene.add_point_cloud(
        "/pi3_object_points",
        points=object_points.astype(np.float32),
        colors=np.tile(np.asarray([[255, 60, 190]], dtype=np.uint8), (len(object_points), 1)),
        point_size=args.point_size * 1.35,
    )
    server.scene.add_mesh_simple(
        "/handflow_hand",
        vertices=hand_vertices,
        faces=hand_faces,
        color=(60, 150, 255),
        opacity=0.45,
    )
    server.scene.add_point_cloud(
        "/pi3_hand_points",
        points=hand_points.astype(np.float32),
        colors=np.tile(np.asarray([[255, 210, 40]], dtype=np.uint8), (len(hand_points), 1)),
        point_size=args.point_size * 1.35,
    )
    server.scene.add_frame("/camera", axes_length=0.05, axes_radius=0.002)
    print(f"frame={args.frame:06d}")
    print(f"window={window_path.name}")
    print(f"scale={window['scale']:.6f} shift={window['shift_m'] * 1000:+.3f}mm")
    print(f"Pi3 object points={len(object_points)}")
    print(f"Pi3 hand points={len(hand_points)}")
    print(f"Pi3 full points={len(full_points)}")
    print(f"Viewer: http://localhost:{args.port}")
    print("Green=tracked object mesh surface, magenta=Pi3 object points")
    print("Blue=HandFlow mesh, yellow=Pi3 hand points")
    print("Press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
