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
    if payload.get("objects") is not None:
        output = {}
        for index, row in enumerate(payload["objects"]):
            local = row.get("local_to_scene") or {}
            quaternion = local.get("quat_wxyz_camera_frame")
            translation = local.get("translation_camera_frame")
            if quaternion is None or translation is None:
                continue
            frame = str(
                row.get("frame_idx", row.get("frame_index", index))
            ).zfill(6)
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = quaternion_wxyz_to_matrix(quaternion)
            matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
            output[frame] = matrix
        return output
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


def quaternion_wxyz_to_matrix(value) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=np.float64).reshape(4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("Zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


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
    gt_ycb_pose = pose_rows(Path(audit["gt_object_json"]))
    object_local, object_faces = load_object(
        Path(audit["object_mesh"]),
        float(audit["object_mesh_scale"]),
        args.max_object_faces,
    )
    gt_ycb_mesh = audit.get("gt_ycb_object_mesh")
    if gt_ycb_mesh:
        gt_ycb_local, gt_ycb_faces = load_object(
            Path(gt_ycb_mesh), 1.0, args.max_object_faces
        )
    else:
        gt_ycb_local, gt_ycb_faces = None, None
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
        "v8": server.gui.add_checkbox("V8 hand", initial_value=False),
        "gt": server.gui.add_checkbox("GT hand", initial_value=False),
        "target": server.gui.add_checkbox(
            "Relative target", initial_value=True
        ),
    }
    object_controls = {
        "sam": server.gui.add_checkbox(
            "Filtered SAM object", initial_value=True
        ),
        "ycb": server.gui.add_checkbox(
            "GT YCB object", initial_value=gt_ycb_local is not None
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
        if pose is not None and object_controls["sam"].value:
            object_vertices = (
                object_local @ pose[:3, :3].T + pose[:3, 3]
            )
            handles.append(
                server.scene.add_mesh_simple(
                    "/object",
                    vertices=object_vertices,
                    faces=object_faces,
                    color=(245, 165, 45),
                    opacity=0.46,
                )
            )
        gt_pose = gt_ycb_pose.get(frame_id)
        if (
            gt_pose is not None
            and gt_ycb_local is not None
            and object_controls["ycb"].value
        ):
            gt_object_vertices = (
                gt_ycb_local @ gt_pose[:3, :3].T + gt_pose[:3, 3]
            )
            handles.append(
                server.scene.add_mesh_simple(
                    "/gt_ycb_object",
                    vertices=gt_object_vertices,
                    faces=gt_ycb_faces,
                    color=(30, 215, 225),
                    opacity=0.28,
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
        raw_delta = target_data["raw_target_translation_camera"][frame]
        raw_mm = float(np.linalg.norm(raw_delta) * 1000.0)
        raw_text = f"{raw_mm:.2f}mm" if np.isfinite(raw_mm) else "n/a"
        v8_delta = target_data["v8_target_translation_camera"][frame]
        v8_mm = float(np.linalg.norm(v8_delta) * 1000.0)
        v8_text = f"{v8_mm:.2f}mm" if np.isfinite(v8_mm) else "n/a"
        reprojection = float(target_data["target_2d_palm_error_px"][frame])
        print(
            f"frame={frame_id} valid={target_valid} "
            f"raw_to_target={raw_text} "
            f"v8_to_target={v8_text} "
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
    for control in object_controls.values():
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
    print(
        "SAM object=amber, GT YCB object=cyan, Raw=orange, "
        "V8=blue, GT hand=green, relative target=magenta"
    )
    print("Press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
