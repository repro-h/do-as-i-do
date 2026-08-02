#!/usr/bin/env python3
"""Visualize raw, v8, GT, and object-relative target hands."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


COLORS = {
    "raw": (245, 170, 55),
    "v8": (70, 140, 245),
    "gt": (60, 205, 105),
    "target": (235, 70, 190),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-object-faces", type=int, default=120000)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("by_frame") or payload.get("frames") or {}
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    output = {}
    for key, row in iterator:
        if not isinstance(row, dict) or row.get("object_in_camera") is None:
            continue
        frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
        output[frame] = np.asarray(
            row["object_in_camera"], dtype=np.float32
        ).reshape(4, 4)
    return output


def frame_string(value, fallback: int) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else str(fallback)).zfill(6)


def load_object(path: Path, scale: float, max_faces: int):
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max_faces > 0 and len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
    return vertices, faces


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_json).expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    side = audit["hand_side"]
    mesh_key = f"{side}_vertices"
    face_key = f"{side}_faces"
    valid_key = f"{side}_valid"

    mesh_archives = {
        "raw": load_npz(Path(audit["raw_hand_meshes"])),
        "v8": load_npz(Path(audit["v8_hand_meshes"])),
        "gt": load_npz(Path(audit["gt_hand_meshes"])),
        "target": load_npz(Path(audit["target_hand_meshes"])),
    }
    vertices = {
        name: np.asarray(data[mesh_key], dtype=np.float32)
        for name, data in mesh_archives.items()
    }
    faces = {
        name: np.asarray(data[face_key], dtype=np.int64)
        for name, data in mesh_archives.items()
    }
    valid = {
        name: np.asarray(data[valid_key], dtype=bool)
        for name, data in mesh_archives.items()
    }
    target_data = load_npz(Path(audit["target_supervision"]))
    frame_ids = [
        frame_string(value, index)
        for index, value in enumerate(target_data["frame_ids"])
    ]
    filtered_pose = pose_rows(Path(audit["filtered_object_json"]))
    object_local, object_faces = load_object(
        Path(audit["object_mesh"]),
        float(audit["object_mesh_scale"]),
        args.max_object_faces,
    )
    count = min(
        len(frame_ids), *(len(value) for value in vertices.values())
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=count - 1, step=1, initial_value=0
    )
    play_button = server.gui.add_button("Play")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=30, step=1, initial_value=int(args.fps)
    )
    controls = {
        "raw": server.gui.add_checkbox("Raw HandFlow", initial_value=False),
        "v8": server.gui.add_checkbox("V8 hand", initial_value=True),
        "gt": server.gui.add_checkbox("GT hand", initial_value=False),
        "target": server.gui.add_checkbox(
            "Relative target", initial_value=True
        ),
    }
    handles = []
    playing = {"value": False}
    suppress = {"value": False}

    def clear() -> None:
        while handles:
            handles.pop().remove()

    def show_frame(frame: int) -> None:
        frame = max(0, min(count - 1, int(frame)))
        clear()
        frame_id = frame_ids[frame]
        pose = filtered_pose.get(frame_id)
        if pose is not None:
            object_vertices = (
                object_local @ pose[:3, :3].T + pose[:3, 3]
            )
            handles.append(
                server.scene.add_mesh_simple(
                    "/object",
                    vertices=object_vertices,
                    faces=object_faces,
                    color=(185, 185, 185),
                    opacity=0.32,
                )
            )
        for name in ("raw", "v8", "gt", "target"):
            if not controls[name].value or not valid[name][frame]:
                continue
            handles.append(
                server.scene.add_mesh_simple(
                    f"/{name}_hand",
                    vertices=vertices[name][frame],
                    faces=faces[name],
                    color=COLORS[name],
                    opacity=0.48 if name == "target" else 0.32,
                )
            )
        target_valid = bool(target_data["valid"][frame])
        raw_mm = float(np.linalg.norm(
            target_data["raw_target_translation_camera"][frame]
        ) * 1000.0)
        v8_mm = float(np.linalg.norm(
            target_data["v8_target_translation_camera"][frame]
        ) * 1000.0)
        reprojection = float(target_data["target_2d_palm_error_px"][frame])
        print(
            f"frame={frame_id} valid={target_valid} "
            f"raw_to_target={raw_mm:.2f}mm "
            f"v8_to_target={v8_mm:.2f}mm "
            f"target_2d={reprojection:.2f}px",
            flush=True,
        )

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress["value"]:
            show_frame(frame_slider.value)

    @play_button.on_click
    def _(_) -> None:
        playing["value"] = not playing["value"]

    for control in controls.values():
        control.on_update(lambda _: show_frame(frame_slider.value))

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
    print("Raw=orange, V8=blue, GT=green, relative target=magenta")
    print("Press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
