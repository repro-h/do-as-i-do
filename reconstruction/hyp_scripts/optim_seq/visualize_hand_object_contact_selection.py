#!/usr/bin/env python3
"""Visualize per-frame hand-object contact candidates and stable selections."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


GROUP_COLORS = {
    "thumb": (245, 80, 80),
    "index": (255, 170, 45),
    "middle": (65, 200, 105),
    "ring": (65, 150, 245),
    "pinky": (175, 95, 235),
    "palm": (250, 80, 190),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--point-size", type=float, default=0.005)
    return parser.parse_args()


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_json).expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    result_path = audit_path.with_name("contact_selection.npz")
    with np.load(result_path, allow_pickle=False) as raw:
        selection = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(audit["prediction"], allow_pickle=False) as raw:
        hand = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(audit["supervision"], allow_pickle=False) as raw:
        supervision = {key: np.asarray(raw[key]) for key in raw.files}

    hand_vertices = np.asarray(hand["verts_cam"], dtype=np.float32)
    hand_faces = np.asarray(hand["faces"], dtype=np.int64)
    object_pose = np.asarray(supervision["object_pose"], dtype=np.float32)
    normalized_left = audit["coordinate_frame"] == "normalized_camera"
    if normalized_left:
        hand_vertices = hand_vertices.copy()
        hand_vertices[..., 0] *= -1.0
    object_mesh = load_mesh(
        Path(audit["object_mesh"]), float(audit["object_mesh_scale"])
    )
    object_local = np.asarray(object_mesh.vertices, dtype=np.float32)
    if normalized_left:
        object_local = object_local.copy()
        object_local[:, 0] *= -1.0

    semantic = np.asarray(selection["semantic_vertex_indices"], dtype=int)
    candidate = np.asarray(selection["candidate_mask"]).astype(bool)
    contact = np.asarray(selection["contact_mask"]).astype(bool)
    nearest = np.asarray(selection["nearest_object_point"], dtype=np.float32)
    vertex_to_local = {int(value): index for index, value in enumerate(semantic)}
    vertex_colors = np.tile(
        np.asarray([[255, 255, 255]], dtype=np.uint8),
        (hand_vertices.shape[1], 1),
    )
    for name, indices in audit["semantic_groups"].items():
        color = GROUP_COLORS.get(name, (255, 255, 255))
        vertex_colors[np.asarray(indices, dtype=int)] = color

    count = min(len(hand_vertices), len(object_pose), len(contact))
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=count - 1, step=1, initial_value=0
    )
    play_button = server.gui.add_button("Play")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=30, step=1, initial_value=int(args.fps)
    )
    show_candidates = server.gui.add_checkbox(
        "Candidate vertices", initial_value=True
    )
    show_contacts = server.gui.add_checkbox(
        "Stable contacts", initial_value=True
    )
    show_lines = server.gui.add_checkbox(
        "Surface correspondences", initial_value=True
    )
    handles = []
    playing = {"value": False}
    suppress = {"value": False}

    def clear_handles() -> None:
        while handles:
            handles.pop().remove()

    def show_frame(frame: int) -> None:
        frame = max(0, min(count - 1, int(frame)))
        clear_handles()
        pose = object_pose[frame]
        object_vertices = object_local @ pose[:3, :3].T + pose[:3, 3]
        handles.append(
            server.scene.add_mesh_simple(
                "/object",
                vertices=object_vertices,
                faces=np.asarray(object_mesh.faces, dtype=np.int64),
                color=(185, 185, 185),
                opacity=0.38,
            )
        )
        handles.append(
            server.scene.add_mesh_simple(
                "/hand",
                vertices=hand_vertices[frame],
                faces=hand_faces,
                color=(75, 145, 245),
                opacity=0.42,
            )
        )
        candidate_ids = np.flatnonzero(candidate[frame])
        if len(candidate_ids) and show_candidates.value:
            handles.append(
                server.scene.add_point_cloud(
                    "/candidate_vertices",
                    points=hand_vertices[frame, candidate_ids],
                    colors=np.tile(
                        np.asarray([[255, 205, 40]], dtype=np.uint8),
                        (len(candidate_ids), 1),
                    ),
                    point_size=args.point_size * 0.75,
                )
            )
        contact_ids = np.flatnonzero(contact[frame])
        if len(contact_ids) and show_contacts.value:
            handles.append(
                server.scene.add_point_cloud(
                    "/stable_contacts",
                    points=hand_vertices[frame, contact_ids],
                    colors=vertex_colors[contact_ids],
                    point_size=args.point_size * 1.3,
                )
            )
        local_ids = np.asarray(
            [vertex_to_local[int(value)] for value in contact_ids], dtype=int
        )
        if len(local_ids) and show_lines.value:
            object_points = nearest[frame, local_ids]
            hand_points = hand_vertices[frame, contact_ids]
            segments = np.stack([hand_points, object_points], axis=1)
            line_colors = np.tile(
                np.asarray([[[235, 65, 65], [235, 65, 65]]], dtype=np.uint8),
                (len(segments), 1, 1),
            )
            handles.append(
                server.scene.add_line_segments(
                    "/contact_correspondences",
                    points=segments,
                    colors=line_colors,
                    line_width=2.0,
                )
            )
            handles.append(
                server.scene.add_point_cloud(
                    "/object_contact_points",
                    points=object_points,
                    colors=np.tile(
                        np.asarray([[235, 65, 65]], dtype=np.uint8),
                        (len(object_points), 1),
                    ),
                    point_size=args.point_size,
                )
            )
        row = audit["frames"][frame]
        print(
            f"frame={frame:06d} candidates={row['num_candidates']} "
            f"selected={row['num_selected']} "
            f"groups={row['selected_groups']}",
            flush=True,
        )

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress["value"]:
            show_frame(frame_slider.value)

    @play_button.on_click
    def _(_) -> None:
        playing["value"] = not playing["value"]

    for checkbox in (show_candidates, show_contacts, show_lines):
        checkbox.on_update(lambda _: show_frame(frame_slider.value))

    def playback() -> None:
        while True:
            if playing["value"]:
                next_frame = (int(frame_slider.value) + 1) % count
                suppress["value"] = True
                frame_slider.value = next_frame
                suppress["value"] = False
                show_frame(next_frame)
                time.sleep(1.0 / max(float(fps_slider.value), 1.0))
            else:
                time.sleep(0.05)

    threading.Thread(target=playback, daemon=True).start()
    show_frame(0)
    print(f"Viewer: http://localhost:{args.port}")
    print("Yellow=candidates, colored=stable contacts, red=object matches")
    print("Press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
