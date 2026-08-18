#!/usr/bin/env python3
"""Render V14, DexYCB GT, and Stage2 in a four-view comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

from render_stage1_dexycb_comparison import (
    camera_to_pytorch3d,
    make_renderer,
    make_writer,
    transform_to_view,
)


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
VIEWS = (
    ("Camera", None),
    ("Side", np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)),
    ("Rear", np.asarray([0.0, 0.0, -1.0], dtype=np.float32)),
    ("Top", np.asarray([0.0, -0.94, 0.342], dtype=np.float32)),
)
COLUMNS = (
    ("WiLoR + V14", (0.10, 0.55, 0.95)),
    ("DexYCB GT hand", (0.20, 0.82, 0.35)),
    ("Stage2 refined", (0.95, 0.38, 0.12)),
)
OBJECT_COLOR = (0.68, 0.72, 0.78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--stage2-npz", required=True)
    parser.add_argument("--refined-label", default="Stage2 refined")
    parser.add_argument("--omit-v14", action="store_true")
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument(
        "--dense-root",
        help="Pi3X split root used to recover original-camera intrinsics.",
    )
    parser.add_argument("--out-mp4", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help=(
            "Camera intrinsics used when neither supervision nor query NPZ "
            "contains an intrinsics matrix."
        ),
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def aligned_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {frame_id(value): index for index, value in enumerate(source)}
    missing = [frame_id(value) for value in target if frame_id(value) not in lookup]
    if missing:
        raise KeyError(f"Missing frame IDs: {missing[:10]}")
    return np.asarray([lookup[frame_id(value)] for value in target], dtype=np.int64)


def load_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(loaded).__name__}")
    return (
        np.asarray(loaded.vertices, dtype=np.float32) * float(scale),
        np.asarray(loaded.faces, dtype=np.int64),
    )


def physical_pose(pose: np.ndarray, normalized_left: bool) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    if normalized_left:
        result[:3, :3] = MIRROR_X @ result[:3, :3] @ MIRROR_X
        result[:3, 3] = MIRROR_X @ result[:3, 3]
    return result


def transform_object(vertices: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum("fij,vj->fvi", poses[:, :3, :3], vertices) + poses[:, None, :3, 3]


def robust_scene_bounds(arrays: list[np.ndarray], valid: list[np.ndarray]) -> tuple[np.ndarray, float]:
    samples = []
    for vertices, mask in zip(arrays, valid):
        selected = vertices[mask]
        if len(selected):
            stride = max(1, selected.shape[1] // 1500)
            samples.append(selected[:, ::stride].reshape(-1, 3))
    if not samples:
        raise RuntimeError("No valid geometry for scene bounds")
    points = np.concatenate(samples, axis=0)
    finite = points[np.isfinite(points).all(axis=-1)]
    lower = np.quantile(finite, 0.005, axis=0)
    upper = np.quantile(finite, 0.995, axis=0)
    center = ((lower + upper) * 0.5).astype(np.float32)
    extent = max(float(np.max(upper - lower)), 0.05)
    return center, extent


def render_scene(
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    object_valid: bool,
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    hand_valid: bool,
    renderer,
    device: torch.device,
    hand_color: tuple[float, float, float],
    view_spec: tuple[np.ndarray, np.ndarray, float] | None,
    width: int,
    height: int,
) -> np.ndarray:
    vertices_list = []
    faces_list = []
    colors_list = []
    offset = 0
    geometry = (
        (object_vertices, object_faces, OBJECT_COLOR, object_valid),
        (hand_vertices, hand_faces, hand_color, hand_valid),
    )
    for vertices_np, faces_np, color, enabled in geometry:
        if not enabled or not np.isfinite(vertices_np).all():
            continue
        vertices = torch.as_tensor(vertices_np, dtype=torch.float32, device=device)
        if view_spec is not None:
            vertices = transform_to_view(vertices, *view_spec)
        vertices = camera_to_pytorch3d(vertices)
        faces = torch.as_tensor(faces_np, dtype=torch.int64, device=device)
        vertices_list.append(vertices)
        faces_list.append(faces + offset)
        colors_list.append(
            torch.tensor(color, dtype=torch.float32, device=device).expand(len(vertices), 3)
        )
        offset += len(vertices)

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if not vertices_list:
        return canvas
    mesh = Meshes(
        verts=[torch.cat(vertices_list)],
        faces=[torch.cat(faces_list)],
        textures=TexturesVertex(verts_features=torch.cat(colors_list).unsqueeze(0)),
    )
    with torch.no_grad():
        rendered = renderer(mesh)[0].detach().cpu().numpy()
    color = (rendered[..., :3][..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
    mask = rendered[..., 3] > 0.01
    canvas[mask] = color[mask]
    return canvas


def label_panel(image: np.ndarray, column: str, view: str, frame: str) -> None:
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 38), (245, 245, 245), -1)
    cv2.putText(
        image, column, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, (35, 35, 35), 2, cv2.LINE_AA,
    )
    view_width = cv2.getTextSize(view, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0][0]
    cv2.putText(
        image, view, (width - view_width - 12, 27), cv2.FONT_HERSHEY_SIMPLEX,
        0.60, (80, 80, 80), 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, frame, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (45, 45, 45), 2, cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    output = Path(args.out_mp4).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output exists; pass --force to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json else output.with_suffix(".json")
    )

    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    stage2 = load_npz(Path(args.stage2_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    gt = (
        load_npz(Path(args.gt_hand_npz).expanduser().resolve())
        if args.gt_hand_npz else None
    )
    object_canonical, object_faces = load_mesh(
        Path(args.object_mesh).expanduser().resolve(), args.object_scale
    )

    ids = np.asarray(query["frame_ids"])
    frame_count = len(ids)
    trajectory_index = aligned_indices(trajectory["frame_ids"], ids)
    stage2_index = aligned_indices(stage2["frame_ids"], ids)
    supervision_index = aligned_indices(supervision["frame_ids"], ids)
    hand_side = str(query["hand_side"].item()).lower()
    normalized_left = bool(np.asarray(supervision["normalized_left"]).item())

    query_vertices = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    )
    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_index], dtype=np.float32
    )
    v14_vertices = query_vertices + wrist[:, None]
    stage2_vertices = np.asarray(
        stage2["refined_hand_vertices_camera"][stage2_index], dtype=np.float32
    )
    hand_faces = np.asarray(query["mano_faces"], dtype=np.int64)

    if gt is not None:
        gt_vertices = np.asarray(
            gt[f"{hand_side}_vertices"], dtype=np.float32
        )
        gt_faces = np.asarray(gt[f"{hand_side}_faces"], dtype=np.int64)
        gt_valid = np.asarray(
            gt[f"{hand_side}_valid"]
        ).astype(bool)[:frame_count]
        if len(gt_vertices) < frame_count:
            raise ValueError(
                f"GT hand has {len(gt_vertices)} frames, expected {frame_count}"
            )
        gt_vertices = gt_vertices[:frame_count]

    query_valid = np.asarray(query["model_valid"]).astype(bool)
    prediction_valid = np.asarray(
        trajectory["prediction_valid"][trajectory_index]
    ).astype(bool)
    stage2_valid = np.asarray(
        stage2.get("valid", np.ones(len(stage2_index), dtype=bool))[stage2_index]
        if "valid" in stage2 else np.ones(frame_count, dtype=bool)
    ).astype(bool)
    v14_valid = query_valid & prediction_valid

    poses = np.stack([
        physical_pose(pose, normalized_left)
        for pose in np.asarray(
            supervision["gt_ycb_object_pose"][supervision_index], dtype=np.float32
        )
    ])
    object_vertices = transform_object(object_canonical, poses)
    object_valid = (
        np.asarray(supervision["gt_object_valid"])[supervision_index].astype(bool)
        if "gt_object_valid" in supervision
        else np.ones(frame_count, dtype=bool)
    )

    image_wh = np.asarray(query["image_wh"], dtype=np.int32)
    width, height = (int(value) for value in image_wh[0])
    if not np.all(image_wh == image_wh[0]):
        raise ValueError("Image dimensions change inside the sequence")
    if "intrinsics" in supervision:
        intrinsics = np.asarray(
            supervision["intrinsics"], dtype=np.float32
        ).reshape(3, 3)
        intrinsics_source = "supervision_npz"
    elif "intrinsics" in query:
        intrinsics = np.asarray(query["intrinsics"], dtype=np.float32)
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
        intrinsics = intrinsics.reshape(3, 3)
        intrinsics_source = "query_npz"
    elif args.dense_root is not None:
        dense_stream = (
            Path(args.dense_root).expanduser().resolve()
            / str(query["stream_id"].item())
            / "windows"
        )
        candidates = sorted(dense_stream.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No Pi3X windows available under {dense_stream}"
            )
        dense = load_npz(candidates[0])
        intrinsics = np.asarray(
            dense["intrinsics_resized"], dtype=np.float32
        )
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
        intrinsics = intrinsics.reshape(3, 3).copy()
        resized_wh = np.asarray(dense["resized_wh"], dtype=np.float32).reshape(2)
        original_wh = np.asarray([width, height], dtype=np.float32)
        scale_x = float(original_wh[0] / resized_wh[0])
        scale_y = float(original_wh[1] / resized_wh[1])
        intrinsics[0, 0] *= scale_x
        intrinsics[0, 2] *= scale_x
        intrinsics[1, 1] *= scale_y
        intrinsics[1, 2] *= scale_y
        intrinsics_source = f"pi3x_dense_cache:{candidates[0]}"
    elif args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        intrinsics = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        intrinsics_source = "command_line"
    else:
        raise KeyError(
            "Neither supervision nor query NPZ contains 'intrinsics'; "
            "pass --intrinsics FX FY CX CY"
        )
    if normalized_left:
        intrinsics = intrinsics.copy()
        intrinsics[0, 2] = (width - 1) - intrinsics[0, 2]

    if gt is not None:
        if args.omit_v14:
            columns = (COLUMNS[1], (args.refined_label, COLUMNS[2][1]))
            hands = (gt_vertices, stage2_vertices)
            faces = (gt_faces, hand_faces)
            validity = (gt_valid, stage2_valid)
        else:
            columns = (COLUMNS[0], COLUMNS[1], (args.refined_label, COLUMNS[2][1]))
            hands = (v14_vertices, gt_vertices, stage2_vertices)
            faces = (hand_faces, gt_faces, hand_faces)
            validity = (v14_valid, gt_valid, stage2_valid)
    else:
        if args.omit_v14:
            columns = ((args.refined_label, COLUMNS[2][1]),)
            hands = (stage2_vertices,)
            faces = (hand_faces,)
            validity = (stage2_valid,)
        else:
            columns = (COLUMNS[0], (args.refined_label, COLUMNS[2][1]))
            hands = (v14_vertices, stage2_vertices)
            faces = (hand_faces, hand_faces)
            validity = (v14_valid, stage2_valid)

    center, extent = robust_scene_bounds(
        [object_vertices, *hands], [object_valid, *validity]
    )
    distance = extent * 1.35
    total_faces = len(object_faces) + max(len(value) for value in faces)
    camera_renderer = make_renderer(
        width, height,
        float(intrinsics[0, 0]), float(intrinsics[1, 1]),
        float(intrinsics[0, 2]), float(intrinsics[1, 2]),
        torch.device(args.device), max(200000, total_faces),
    )
    synthetic_renderer = make_renderer(
        width, height,
        min(width, height) * 1.05, min(width, height) * 1.05,
        width * 0.5, height * 0.5,
        torch.device(args.device), max(200000, total_faces),
    )
    writer = make_writer(
        output, args.fps, (width * len(columns), height * len(VIEWS))
    )
    device = torch.device(args.device)

    try:
        for index, raw_id in enumerate(ids):
            rows = []
            current_id = frame_id(raw_id)
            for view_name, forward in VIEWS:
                view_spec = None if forward is None else (center, forward, distance)
                renderer = camera_renderer if forward is None else synthetic_renderer
                panels = []
                for column_index, (column_name, hand_color) in enumerate(columns):
                    panel = render_scene(
                        object_vertices[index],
                        object_faces,
                        bool(object_valid[index]),
                        hands[column_index][index],
                        faces[column_index],
                        bool(validity[column_index][index]),
                        renderer,
                        device,
                        hand_color,
                        view_spec,
                        width,
                        height,
                    )
                    label_panel(panel, column_name, view_name, current_id)
                    panels.append(panel)
                rows.append(np.concatenate(panels, axis=1))
            writer.write(np.concatenate(rows, axis=0))
            print(f"[{index + 1}/{frame_count}] {current_id}", flush=True)
    finally:
        writer.release()

    summary = {
        "method": (
            "v14_gt_stage2_four_view_grid_v1"
            if gt is not None else "v14_stage2_four_view_grid_v1"
        ),
        "output": str(output),
        "frames": frame_count,
        "fps": args.fps,
        "resolution": [width * len(columns), height * len(VIEWS)],
        "rows": [name for name, _ in VIEWS],
        "columns": [name for name, _ in columns],
        "object": "GT YCB mesh with per-frame gt_ycb_object_pose in every panel",
        "hand_side": hand_side,
        "normalized_left_restored_to_physical_camera": normalized_left,
        "intrinsics_source": intrinsics_source,
        "intrinsics": intrinsics.tolist(),
        "scene_center": center.tolist(),
        "scene_extent": extent,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
