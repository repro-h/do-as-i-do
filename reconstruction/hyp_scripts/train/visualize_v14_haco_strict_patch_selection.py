#!/usr/bin/env python3
"""Clean Viser audit for strict V14/HACO object-patch selection."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from refine_v14_haco_sequence_contact_containment import mano_contact_region_ids
from select_haco_multiregion_object_contacts_sequence import (
    adjacency,
    strongest_components,
    vertex_normals,
)
from visualize_haco_choir_opposition_candidates import (
    frame_id,
    index_for,
    load_npz,
    physical_pose,
)


COLORS = {
    "palm": (235, 65, 55),
    "index": (25, 180, 245),
    "middle": (245, 135, 25),
    "pinky": (65, 210, 70),
    "ring": (135, 75, 230),
    "thumb": (235, 35, 160),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-npz", required=True)
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--haco-topk", type=int, default=12)
    parser.add_argument("--initial-frame", type=int, default=32)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


def repeated_color(color: tuple[int, int, int], count: int) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.uint8)[None], (count, 1))


def main() -> None:
    args = parse_args()
    if args.haco_topk <= 0:
        raise ValueError("--haco-topk must be positive")
    selection = load_npz(Path(args.selection_npz).expanduser().resolve())
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())

    ids = np.asarray(query["frame_ids"])
    count = len(ids)
    trajectory_indices = np.asarray([
        index_for(trajectory["frame_ids"], frame_id(value)) for value in ids
    ])
    contact_indices = np.asarray([
        index_for(contact["frame_ids"], frame_id(value)) for value in ids
    ])
    supervision_indices = np.asarray([
        index_for(supervision["frame_ids"], frame_id(value)) for value in ids
    ])
    hand = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    ) + np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )[:, None]
    hand_faces = np.asarray(query["mano_faces"], dtype=np.int64)
    valid = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(
            trajectory["prediction_valid"][trajectory_indices]
        ).astype(bool)
    )
    probability = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )
    hand_graph = adjacency(hand_faces, hand.shape[1])

    mesh = trimesh.load(Path(args.object_mesh).expanduser().resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    object_local = np.asarray(mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(mesh.faces, dtype=np.int64)
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    poses = [
        physical_pose(
            supervision["gt_ycb_object_pose"][index], normalized_left
        )
        for index in supervision_indices
    ]

    region_key = (
        "stable_region_names"
        if "stable_region_names" in selection and len(selection["stable_region_names"])
        else "selected_region_names"
    )
    selected_regions = [str(value) for value in selection[region_key]]
    patches = {
        name: np.asarray(
            selection[f"{name}_patch_vertices_canonical"], dtype=np.float32
        )
        for name in selected_regions
    }
    patch_normals = {
        name: np.asarray(
            selection[f"{name}_patch_normals_canonical"], dtype=np.float32
        )
        for name in selected_regions
    }
    print(f"[viser] strict regions: {','.join(selected_regions)}", flush=True)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    initial = max(0, min(count - 1, args.initial_frame))
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=count - 1, step=1, initial_value=initial
    )
    play = server.gui.add_button("Play / Pause")
    fps = server.gui.add_slider(
        "FPS", min=1, max=30, step=1, initial_value=max(args.fps, 1)
    )
    point_size = server.gui.add_slider(
        "Point size", min=0.002, max=0.014, step=0.001, initial_value=0.006
    )
    normal_length = server.gui.add_slider(
        "Normal length", min=0.004, max=0.040, step=0.002, initial_value=0.018
    )
    show_object = server.gui.add_checkbox("GT YCB object", initial_value=True)
    show_hand = server.gui.add_checkbox("V14 hand", initial_value=True)
    show_topk = server.gui.add_checkbox("HACO top-k by region", initial_value=True)
    show_patches = server.gui.add_checkbox("Stable object patches", initial_value=True)
    show_hand_normals = server.gui.add_checkbox("HACO hand normals", initial_value=True)
    show_object_normals = server.gui.add_checkbox(
        "Object patch normals", initial_value=True
    )
    show_pairs = server.gui.add_checkbox("Hand-to-patch pairs", initial_value=True)
    handles = []
    playing = {"value": False}
    suppress = {"value": False}

    def clear() -> None:
        while handles:
            handles.pop().remove()

    def add_normals(
        name: str,
        points: np.ndarray,
        normals: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        length = float(normal_length.value)
        endpoints = points + normals * length
        line_colors = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None], (len(points), 2, 1)
        )
        handles.append(server.scene.add_line_segments(
            name,
            points=np.stack((points, endpoints), axis=1),
            colors=line_colors,
            line_width=3.0,
        ))
        dark = tuple(max(channel - 90, 0) for channel in color)
        handles.append(server.scene.add_point_cloud(
            f"{name}_end",
            points=endpoints,
            colors=repeated_color(dark, len(endpoints)),
            point_size=float(point_size.value) * 0.8,
        ))

    def show_frame(index: int) -> None:
        index = max(0, min(count - 1, int(index)))
        clear()
        pose = poses[index]
        object_vertices = object_local @ pose[:3, :3].T + pose[:3, 3]
        current_hand = hand[index]
        current_hand_normals = vertex_normals(current_hand, hand_faces)
        if show_object.value:
            handles.append(server.scene.add_mesh_simple(
                "/object", vertices=object_vertices, faces=object_faces,
                color=(160, 175, 195), opacity=0.45,
            ))
        if show_hand.value and valid[index]:
            handles.append(server.scene.add_mesh_simple(
                "/v14_hand", vertices=current_hand, faces=hand_faces,
                color=(80, 165, 235), opacity=0.38,
            ))
        print(f"frame={frame_id(ids[index])}", flush=True)
        for region_index, region_name in enumerate(region_names):
            raw = (
                (region_ids == region_index)
                & (probability[index] >= threshold)
            )
            component = strongest_components(
                raw, hand_graph, probability[index], 1
            ) if raw.any() else raw
            component_ids = np.flatnonzero(component)
            if not len(component_ids):
                continue
            topk = min(args.haco_topk, len(component_ids))
            top_ids = component_ids[np.argsort(
                probability[index, component_ids]
            )[-topk:]]
            hand_points = current_hand[top_ids]
            hand_normals = current_hand_normals[top_ids]
            color = COLORS.get(region_name, (220, 190, 30))
            if show_topk.value:
                handles.append(server.scene.add_point_cloud(
                    f"/regions/{region_name}/haco_topk",
                    points=hand_points,
                    colors=repeated_color(color, len(hand_points)),
                    point_size=float(point_size.value) * 1.5,
                ))
            if show_hand_normals.value:
                add_normals(
                    f"/regions/{region_name}/hand_normals",
                    hand_points, hand_normals, color,
                )
            if region_name not in patches:
                print(
                    f"  {region_name}: topk={topk} object_patch=unselected",
                    flush=True,
                )
                continue
            patch_points = patches[region_name] @ pose[:3, :3].T + pose[:3, 3]
            normals = patch_normals[region_name] @ pose[:3, :3].T
            normals /= np.maximum(
                np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8
            )
            pairwise = np.linalg.norm(
                hand_points[:, None] - patch_points[None], axis=-1
            )
            nearest = pairwise.argmin(axis=-1)
            target_points = patch_points[nearest]
            target_normals = normals[nearest]
            normal_dot = np.sum(hand_normals * target_normals, axis=-1)
            if show_patches.value:
                handles.append(server.scene.add_point_cloud(
                    f"/regions/{region_name}/object_patch",
                    points=patch_points,
                    colors=repeated_color(color, len(patch_points)),
                    point_size=float(point_size.value),
                ))
            if show_object_normals.value:
                sample = np.linspace(
                    0, len(patch_points) - 1,
                    min(len(patch_points), 20), dtype=np.int64,
                )
                add_normals(
                    f"/regions/{region_name}/object_normals",
                    patch_points[sample], normals[sample], color,
                )
            if show_pairs.value:
                pair_color = np.tile(
                    np.asarray(color, dtype=np.uint8)[None, None],
                    (len(hand_points), 2, 1),
                )
                handles.append(server.scene.add_line_segments(
                    f"/regions/{region_name}/pairs",
                    points=np.stack((hand_points, target_points), axis=1),
                    colors=pair_color,
                    line_width=2.0,
                ))
            print(
                f"  {region_name}: topk={topk} "
                f"normal_dot median={np.median(normal_dot):.3f} "
                f"max={np.max(normal_dot):.3f}",
                flush=True,
            )

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress["value"]:
            show_frame(int(frame_slider.value))

    @play.on_click
    def _(_) -> None:
        playing["value"] = not playing["value"]

    controls = (
        show_object, show_hand, show_topk, show_patches,
        show_hand_normals, show_object_normals, show_pairs,
    )
    for control in controls:
        control.on_update(lambda _: show_frame(int(frame_slider.value)))
    point_size.on_update(lambda _: show_frame(int(frame_slider.value)))
    normal_length.on_update(lambda _: show_frame(int(frame_slider.value)))

    def playback() -> None:
        while True:
            if playing["value"]:
                value = (int(frame_slider.value) + 1) % count
                suppress["value"] = True
                frame_slider.value = value
                suppress["value"] = False
                show_frame(value)
                time.sleep(1.0 / max(float(fps.value), 1.0))
            else:
                time.sleep(0.05)

    threading.Thread(target=playback, daemon=True).start()
    show_frame(initial)
    print(f"Viewer: http://localhost:{args.port}")
    print("Dark endpoints mark positive normal directions; Ctrl+C stops the viewer.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
