#!/usr/bin/env python3
"""Visualize one V14 WiLoR hand with HACO contact and a GT YCB object."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--refined-hand-npz")
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--frame-id")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--summary-json")
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


def load_gt_hand(
    path: Path, hand_side: str, frame_index: int
) -> tuple[np.ndarray, np.ndarray] | None:
    data = load_npz(path)
    side = hand_side.lower()
    vertices_key = f"{side}_vertices"
    faces_key = f"{side}_faces"
    valid_key = f"{side}_valid"
    missing = [
        key for key in (vertices_key, faces_key, valid_key) if key not in data
    ]
    if missing:
        raise KeyError(f"GT hand archive lacks {missing}")
    if frame_index >= len(data[valid_key]):
        raise IndexError(
            f"GT hand has {len(data[valid_key])} frames, requested {frame_index}"
        )
    if not bool(data[valid_key][frame_index]):
        print(f"GT hand is invalid for frame index {frame_index}")
        return None
    vertices = np.asarray(data[vertices_key][frame_index], dtype=np.float32)
    if not np.isfinite(vertices).all():
        print(f"GT hand contains non-finite vertices at frame index {frame_index}")
        return None
    return vertices, np.asarray(data[faces_key], dtype=np.int64)


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
    if "frame_ids" in contact:
        contact_index = index_for(contact["frame_ids"], requested)
        if "contact_valid" in contact and not bool(
            contact["contact_valid"][contact_index]
        ):
            raise RuntimeError(f"HACO contact is invalid for frame {requested}")
    else:
        contact_index = None
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
    probability_all = np.asarray(
        contact["contact_probability"], dtype=np.float32
    )
    probability = (
        probability_all[contact_index]
        if contact_index is not None
        else probability_all
    )
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
    gt_hand = None
    if args.gt_hand_npz:
        gt_hand = load_gt_hand(
            Path(args.gt_hand_npz).expanduser().resolve(),
            str(query["hand_side"].item()),
            query_index,
        )
    refined_hand = None
    if args.refined_hand_npz:
        refined_data = load_npz(
            Path(args.refined_hand_npz).expanduser().resolve()
        )
        if "frame_ids" in refined_data:
            refined_index = index_for(refined_data["frame_ids"], requested)
            refined_hand = np.asarray(
                refined_data["refined_hand_vertices_camera"][refined_index],
                dtype=np.float32,
            )
        else:
            if frame_id(refined_data["frame_id"].item()) != requested:
                raise ValueError("Refined hand frame does not match requested frame")
            refined_hand = np.asarray(
                refined_data["refined_hand_vertices_camera"], dtype=np.float32
            )
        if refined_hand.shape != hand_vertices.shape:
            raise ValueError(
                f"Refined hand shape mismatch: {refined_hand.shape} vs "
                f"{hand_vertices.shape}"
            )

    threshold = float(np.asarray(contact["contact_threshold"]).item())
    selected = probability > threshold
    distance_summary = None
    if selected.any():
        nearest = cKDTree(object_camera).query(hand_vertices[selected])[0] * 1000.0
        distance_summary = {
            "count": int(selected.sum()),
            "median": float(np.median(nearest)),
            "p90": float(np.percentile(nearest, 90)),
            "min": float(nearest.min()),
            "max": float(nearest.max()),
        }
        print("HACO contact-to-GT-object nearest-vertex mm:", distance_summary)
    else:
        print("HACO selected no contact vertices")
    summary = {
        "frame": requested,
        "stream_id": str(trajectory["stream_id"].item()),
        "hand_side": str(query["hand_side"].item()),
        "v14_wrist_camera": wrist.tolist(),
        "haco_threshold": threshold,
        "haco_contact_vertices": int(selected.sum()),
        "haco_probability": {
            "mean": float(probability.mean()),
            "median": float(np.median(probability)),
            "p90": float(np.percentile(probability, 90)),
            "max": float(probability.max()),
        },
        "contact_to_gt_object_nearest_vertex_mm": distance_summary,
        "normalized_left": normalized_left,
        "trajectory_npz": str(Path(args.trajectory_npz).expanduser().resolve()),
        "query_npz": str(Path(args.query_npz).expanduser().resolve()),
        "contact_npz": str(Path(args.contact_npz).expanduser().resolve()),
        "supervision_npz": str(Path(args.supervision_npz).expanduser().resolve()),
        "object_mesh": str(Path(args.object_mesh).expanduser().resolve()),
        "gt_hand_npz": (
            str(Path(args.gt_hand_npz).expanduser().resolve())
            if args.gt_hand_npz else None
        ),
        "refined_hand_npz": (
            str(Path(args.refined_hand_npz).expanduser().resolve())
            if args.refined_hand_npz else None
        ),
    }
    print(summary)
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        temporary.replace(summary_path)
        print(f"Summary: {summary_path}")

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
        if refined_hand is not None:
            handles.append(server.scene.add_mesh_simple(
                "/chamfer_refined_hand",
                refined_hand,
                hand_faces,
                color=(255, 165, 45),
                opacity=0.52,
            ))
        handles.append(server.scene.add_mesh_simple(
            "/gt_ycb_object",
            object_camera,
            object_faces,
            color=(30, 215, 225),
            opacity=0.38,
        ))
        if gt_hand is not None:
            handles.append(server.scene.add_mesh_simple(
                "/gt_dexycb_hand",
                gt_hand[0],
                gt_hand[1],
                color=(55, 215, 95),
                opacity=0.42,
            ))
        active = probability > float(threshold_control.value)
        if active.any():
            contact_vertices = (
                refined_hand if refined_hand is not None else hand_vertices
            )
            strength = probability[active, None]
            colors = np.concatenate((
                np.full_like(strength, 255.0),
                80.0 * (1.0 - strength),
                40.0 * (1.0 - strength),
            ), axis=1).clip(0, 255).astype(np.uint8)
            handles.append(server.scene.add_point_cloud(
                "/haco_contact",
                points=contact_vertices[active],
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
    print(
        "Blue=V14 WiLoR hand, red=HACO contact, "
        "orange=Chamfer refined hand, green=GT DexYCB hand, "
        "cyan=GT YCB object"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
