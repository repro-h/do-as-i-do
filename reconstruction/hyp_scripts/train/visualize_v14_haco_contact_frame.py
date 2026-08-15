#!/usr/bin/env python3
"""Visualize one V14 WiLoR hand with HACO contact and a GT YCB object."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial import cKDTree


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--frame-id")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def index_for(values: np.ndarray, target: str) -> int:
    normalized = [frame_id(value) for value in values]
    if target not in normalized:
        raise KeyError(f"Frame {target} not found")
    return normalized.index(target)


def load_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return (
        np.asarray(loaded.vertices, dtype=np.float32) * scale,
        np.asarray(loaded.faces, dtype=np.int64),
    )


def physical_pose(pose: np.ndarray, normalized_left: bool) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    if normalized_left:
        result[:3, :3] = MIRROR_X @ result[:3, :3] @ MIRROR_X
        result[:3, 3] = MIRROR_X @ result[:3, 3]
    return result


def main() -> None:
    args = parse_args()
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())

    requested = (
        frame_id(args.frame_id)
        if args.frame_id is not None
        else frame_id(query["frame_ids"][int(args.frame_index)])
    )
    trajectory_index = index_for(trajectory["frame_ids"], requested)
    query_index = index_for(query["frame_ids"], requested)
    supervision_index = index_for(supervision["frame_ids"], requested)
    if frame_id(contact["frame_id"].item()) != requested:
        raise ValueError("HACO contact frame does not match requested frame")
    if "vertices_3d_root_relative_original" not in query:
        raise KeyError(
            "WiLoR cache lacks root-relative vertices; re-export this stream "
            "with the updated export_dexycb_wilor_queries.py"
        )
    if not bool(trajectory["prediction_valid"][trajectory_index]):
        raise RuntimeError(f"V14 prediction is invalid for frame {requested}")

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_index],
        dtype=np.float32,
    )
    hand_vertices = np.asarray(
        query["vertices_3d_root_relative_original"][query_index],
        dtype=np.float32,
    ) + wrist[None]
    hand_faces = np.asarray(query["mano_faces"], dtype=np.int64)
    probability = np.asarray(contact["contact_probability"], dtype=np.float32)
    if len(probability) != len(hand_vertices):
        raise ValueError(
            f"Contact/mesh mismatch: {len(probability)} vs {len(hand_vertices)}"
        )

    object_vertices, object_faces = load_mesh(
        Path(args.object_mesh).expanduser().resolve(), args.object_scale
    )
    if "gt_ycb_object_pose" not in supervision:
        raise KeyError("supervision lacks gt_ycb_object_pose")
    normalized_left = bool(np.asarray(
        supervision.get("normalized_left", False)
    ).item())
    object_pose = physical_pose(
        supervision["gt_ycb_object_pose"][supervision_index], normalized_left
    )
    object_camera = (
        object_vertices @ object_pose[:3, :3].T + object_pose[:3, 3]
    )

    threshold = float(np.asarray(contact["contact_threshold"]).item())
    selected = probability > threshold
    if selected.any():
        nearest = cKDTree(object_camera).query(hand_vertices[selected])[0] * 1000.0
        print(
            "HACO contact-to-GT-object nearest-vertex mm:",
            {
                "count": int(selected.sum()),
                "median": float(np.median(nearest)),
                "p90": float(np.percentile(nearest, 90)),
                "min": float(nearest.min()),
                "max": float(nearest.max()),
            },
        )
    else:
        print("HACO selected no contact vertices")
    print({
        "frame": requested,
        "stream_id": str(trajectory["stream_id"].item()),
        "hand_side": str(query["hand_side"].item()),
        "v14_wrist_camera": wrist.tolist(),
        "haco_threshold": threshold,
        "haco_contact_vertices": int(selected.sum()),
        "normalized_left": normalized_left,
    })

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    threshold_control = server.gui.add_slider(
        "Contact threshold", min=0.0, max=1.0, step=0.01,
        initial_value=threshold,
    )
    point_size = server.gui.add_slider(
        "Contact point size", min=0.001, max=0.02, step=0.001,
        initial_value=0.006,
    )
    handles = []

    def redraw(_) -> None:
        while handles:
            handles.pop().remove()
        handles.append(server.scene.add_mesh_simple(
            "/v14_wilor_hand",
            hand_vertices,
            hand_faces,
            color=(70, 140, 245),
            opacity=0.46,
        ))
        handles.append(server.scene.add_mesh_simple(
            "/gt_ycb_object",
            object_camera,
            object_faces,
            color=(30, 215, 225),
            opacity=0.38,
        ))
        active = probability > float(threshold_control.value)
        if active.any():
            strength = probability[active, None]
            colors = np.concatenate((
                np.full_like(strength, 255.0),
                80.0 * (1.0 - strength),
                40.0 * (1.0 - strength),
            ), axis=1).clip(0, 255).astype(np.uint8)
            handles.append(server.scene.add_point_cloud(
                "/haco_contact",
                points=hand_vertices[active],
                colors=colors,
                point_size=float(point_size.value),
            ))
        handles.append(server.scene.add_point_cloud(
            "/v14_wrist",
            points=wrist[None],
            colors=np.asarray([[255, 255, 255]], dtype=np.uint8),
            point_size=0.012,
        ))

    threshold_control.on_update(redraw)
    point_size.on_update(redraw)
    redraw(None)
    print(f"Viewer: http://localhost:{args.port}")
    print("Blue=V14 WiLoR hand, red=HACO contact, cyan=GT YCB object")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
