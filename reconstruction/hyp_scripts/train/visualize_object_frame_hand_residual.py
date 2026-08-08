#!/usr/bin/env python3
"""Viser viewer for object-frame hand pose residual predictions."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


COLORS = {
    "prediction": (70, 140, 245),
    "target": (235, 70, 190),
    "gt_hand": (60, 205, 105),
    "sam": (245, 165, 45),
    "sam_gt_pose": (255, 125, 30),
    "gt_ycb": (30, 215, 225),
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", help="SAM3D mesh")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--gt-object-mesh")
    parser.add_argument("--gt-object-scale", type=float, default=1.0)
    parser.add_argument("--gt-object-pose-json")
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-object-faces", type=int, default=120000)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def quaternion_wxyz_to_matrix(value) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 1e-12:
        raise ValueError("zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float32)


def load_pose_rows(path: Path) -> dict[str, np.ndarray]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("objects") is not None:
        rows = payload["objects"]
        iterator = enumerate(rows)
    else:
        rows = payload.get("by_frame") or payload.get("frames") or {}
        iterator = (
            enumerate(rows)
            if isinstance(rows, list)
            else rows.items()
        )
    output = {}
    for key, row in iterator:
        if not isinstance(row, dict):
            continue
        matrix = row.get("object_in_camera")
        if matrix is None:
            local = row.get("local_to_scene") or {}
            quat = local.get("quat_wxyz_camera_frame")
            trans = local.get("translation_camera_frame")
            if quat is not None and trans is not None:
                matrix = np.eye(4, dtype=np.float32)
                matrix[:3, :3] = quaternion_wxyz_to_matrix(quat)
                matrix[:3, 3] = np.asarray(trans, dtype=np.float32)
        if matrix is not None:
            frame = str(
                row.get("frame", row.get("frame_id", row.get("frame_idx", key)))
            )
            digits = "".join(c for c in frame if c.isdigit())
            output[(digits[-6:] if digits else str(key)).zfill(6)] = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    return output


def mesh_data(path: Path, scale: float, max_faces: int):
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max_faces > 0 and len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
    return vertices, faces


def rigid_delta_camera(
    object_pose: np.ndarray,
    initial_t: np.ndarray,
    initial_r: np.ndarray,
    predicted_t: np.ndarray,
    predicted_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert object-frame T_pred*T_initial^-1 into a camera-frame delta."""
    r_co = object_pose[:, :3, :3]
    t_co = object_pose[:, :3, 3]
    r_delta_o = predicted_r @ np.swapaxes(initial_r, -1, -2)
    t_delta_o = predicted_t - np.einsum("nij,nj->ni", r_delta_o, initial_t)
    r_delta_c = r_co @ r_delta_o @ np.swapaxes(r_co, -1, -2)
    t_delta_c = (
        t_co
        - np.einsum("nij,nj->ni", r_delta_c, t_co)
        + np.einsum("nij,nj->ni", r_co, t_delta_o)
    )
    return r_delta_c, t_delta_c


def apply_delta(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    return vertices @ rotation.T + translation[None]


def main() -> None:
    options = args()
    prediction_path = Path(options.prediction_npz).expanduser().resolve()
    supervision_path = Path(options.supervision_npz).expanduser().resolve()
    prediction = load_npz(prediction_path)
    supervision = load_npz(supervision_path)
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )

    raw_path = Path(str(prediction["handflow_camera_result"].item()))
    raw = load_npz(raw_path)
    raw_vertices = np.asarray(raw["verts_cam"], dtype=np.float32).copy()
    faces = np.asarray(raw["faces"], dtype=np.int64)
    saved_vertices = np.asarray(prediction["verts_cam"], dtype=np.float32).copy()
    if normalized_left:
        # Supervision poses live in the mirrored left-normalized camera frame.
        raw_vertices[..., 0] *= -1.0
        saved_vertices[..., 0] *= -1.0
        faces = faces[:, [0, 2, 1]]
    pose_key = "filtered_object_pose" if "filtered_object_pose" in supervision else "object_pose"
    object_pose = np.asarray(supervision[pose_key], dtype=np.float32)
    gt_sam_pose = np.asarray(
        supervision.get("gt_sam_object_pose", np.full_like(object_pose, np.nan)),
        dtype=np.float32,
    )
    gt_ycb_pose_supervision = np.asarray(
        supervision.get("gt_ycb_object_pose", np.full_like(object_pose, np.nan)),
        dtype=np.float32,
    )
    canonical_sam_to_ycb = np.asarray(
        supervision.get("canonical_sam_to_ycb", np.eye(4)),
        dtype=np.float32,
    )
    initial_t = np.asarray(prediction["initial_translation_object"], dtype=np.float32)
    initial_r = np.asarray(prediction["initial_rotation_object"], dtype=np.float32)
    predicted_t = np.asarray(prediction["predicted_translation_object"], dtype=np.float32)
    predicted_r = np.asarray(prediction["predicted_rotation_object"], dtype=np.float32)
    valid = np.asarray(
        prediction.get("camera_mesh_correction_valid", np.ones(len(raw_vertices))),
        dtype=bool,
    )
    frame_ids = np.asarray(prediction.get("frame_ids", np.arange(len(raw_vertices))))
    count = min(
        len(raw_vertices), len(saved_vertices), len(object_pose), len(valid), len(frame_ids)
    )
    raw_vertices = raw_vertices[:count]
    saved_vertices = saved_vertices[:count]
    object_pose = object_pose[:count]
    valid = valid[:count]
    frame_ids = frame_ids[:count]

    delta_r, delta_t = rigid_delta_camera(
        object_pose[:count], initial_t[:count], initial_r[:count],
        predicted_t[:count], predicted_r[:count],
    )
    delta_vertices = np.stack(
        [apply_delta(raw_vertices[i], delta_r[i], delta_t[i]) for i in range(count)]
    ).astype(np.float32)

    gt_vertices = None
    gt_faces = None
    gt_valid = np.zeros(count, dtype=bool)
    if options.gt_hand_npz:
        gt = load_npz(Path(options.gt_hand_npz).expanduser().resolve())
        side_value = raw.get("hand_side", np.asarray("right"))
        side = str(np.asarray(side_value).item()).lower()
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported hand side: {side!r}")
        gt_vertices = np.asarray(
            gt[f"{side}_vertices"], dtype=np.float32
        )[:count].copy()
        gt_faces = np.asarray(gt[f"{side}_faces"], dtype=np.int64)
        if normalized_left:
            gt_vertices[..., 0] *= -1.0
            gt_faces = gt_faces[:, [0, 2, 1]]
        gt_valid = np.asarray(
            gt.get(f"{side}_valid", np.ones(len(gt_vertices), dtype=bool)),
            dtype=bool,
        )[:count]

    sam_vertices = sam_faces = None
    if options.object_mesh:
        sam_vertices, sam_faces = mesh_data(
            Path(options.object_mesh).expanduser().resolve(),
            options.object_scale,
            options.max_object_faces,
        )

    sam_vertices_ycb = None
    if sam_vertices is not None:
        sam_vertices_ycb = (
            sam_vertices @ canonical_sam_to_ycb[:3, :3].T
            + canonical_sam_to_ycb[:3, 3]
        ).astype(np.float32)

    gt_ycb_vertices = gt_ycb_faces = None
    gt_ycb_poses = {}
    if options.gt_object_mesh:
        gt_ycb_vertices, gt_ycb_faces = mesh_data(
            Path(options.gt_object_mesh).expanduser().resolve(),
            options.gt_object_scale,
            options.max_object_faces,
        )
    if options.gt_object_pose_json:
        gt_ycb_poses = load_pose_rows(Path(options.gt_object_pose_json).expanduser().resolve())
    print(
        "reference assets:",
        "camera_frame=", "normalized_left" if normalized_left else "physical",
        "SAM_mesh=", sam_vertices is not None,
        "GT_YCB_mesh=", gt_ycb_vertices is not None,
        "GT_YCB_poses=", len(gt_ycb_poses),
        "GT-SAM_poses=", int(np.isfinite(gt_sam_pose).all(axis=(1, 2)).sum()),
        "GT_hand=", gt_vertices is not None,
    )

    server = viser.ViserServer(port=options.port)
    server.scene.set_up_direction("-y")
    slider = server.gui.add_slider("Frame", min=0, max=count - 1, step=1, initial_value=0)
    play = server.gui.add_button("Play")
    fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=int(options.fps))
    controls = {
        "prediction": server.gui.add_checkbox("Prediction hand", initial_value=True),
        "target": server.gui.add_checkbox("Supervision target", initial_value=True),
        "gt_hand": server.gui.add_checkbox("GT hand", initial_value=gt_vertices is not None),
        "sam": server.gui.add_checkbox("SAM3D object", initial_value=sam_vertices is not None),
        "sam_gt_pose": server.gui.add_checkbox(
            "SAM3D at GT-SAM pose",
            initial_value=sam_vertices is not None
            and bool(np.isfinite(gt_sam_pose).any()),
        ),
        "gt_ycb": server.gui.add_checkbox("GT YCB object", initial_value=gt_ycb_vertices is not None),
    }
    handles = []
    playing = {"value": False}

    def clear():
        while handles:
            handles.pop().remove()

    def show(index: int):
        index = max(0, min(count - 1, int(index)))
        clear()
        if controls["sam"].value and sam_vertices is not None:
            handles.append(server.scene.add_mesh_simple(
                "/sam_object", sam_vertices @ object_pose[index, :3, :3].T + object_pose[index, :3, 3],
                sam_faces, color=COLORS["sam"], opacity=0.45,
            ))
        if (
            controls["sam_gt_pose"].value
            and sam_vertices_ycb is not None
            and np.isfinite(gt_ycb_pose_supervision[index]).all()
        ):
            handles.append(server.scene.add_mesh_simple(
                "/sam_gt_pose",
                sam_vertices_ycb @ gt_ycb_pose_supervision[index, :3, :3].T
                + gt_ycb_pose_supervision[index, :3, 3],
                sam_faces,
                color=COLORS["sam_gt_pose"],
                opacity=0.34,
            ))
        target_vertices = apply_delta(
            raw_vertices[index],
            rigid_delta_camera(
                object_pose[index:index+1],
                initial_t[index:index+1], initial_r[index:index+1],
                np.asarray(prediction["target_translation_object"][index:index+1], dtype=np.float32),
                np.asarray(prediction["target_rotation_object"][index:index+1], dtype=np.float32),
            )[0][0],
            rigid_delta_camera(
                object_pose[index:index+1],
                initial_t[index:index+1], initial_r[index:index+1],
                np.asarray(prediction["target_translation_object"][index:index+1], dtype=np.float32),
                np.asarray(prediction["target_rotation_object"][index:index+1], dtype=np.float32),
            )[1][0],
        )
        entries = [
            ("prediction", delta_vertices[index], faces, 0.38),
            ("target", target_vertices, faces, 0.38),
        ]
        if gt_vertices is not None:
            entries.append(("gt_hand", gt_vertices[index], gt_faces, 0.32))
        for name, vertices, mesh_faces, opacity in entries:
            visible = gt_valid[index] if name == "gt_hand" else True
            if controls[name].value and visible:
                handles.append(server.scene.add_mesh_simple(
                    f"/{name}_hand", vertices, mesh_faces,
                    color=COLORS[name], opacity=opacity,
                ))
        gt_pose = (
            gt_ycb_pose_supervision[index]
            if np.isfinite(gt_ycb_pose_supervision[index]).all()
            else gt_ycb_poses.get(str(frame_ids[index]).zfill(6))
        )
        if controls["gt_ycb"].value and gt_ycb_vertices is not None and gt_pose is not None:
            handles.append(server.scene.add_mesh_simple(
                "/gt_ycb_object",
                gt_ycb_vertices @ gt_pose[:3, :3].T + gt_pose[:3, 3],
                gt_ycb_faces,
                color=COLORS["gt_ycb"], opacity=0.32,
            ))
        saved_err = np.linalg.norm(saved_vertices[index].mean(0) - delta_vertices[index].mean(0)) * 1000
        print(
            f"frame={str(frame_ids[index])} valid={bool(valid[index])} "
            f"saved_vs_delta_center={saved_err:.2f}mm",
            flush=True,
        )

    @slider.on_update
    def _(_):
        show(slider.value)

    @play.on_click
    def _(_):
        playing["value"] = not playing["value"]

    for control in controls.values():
        control.on_update(lambda _: show(slider.value))

    def playback():
        while True:
            if playing["value"]:
                slider.value = (int(slider.value) + 1) % count
                time.sleep(1.0 / max(float(fps.value), 1.0))
            else:
                time.sleep(0.05)

    threading.Thread(target=playback, daemon=True).start()
    show(0)
    print(f"Viewer: http://localhost:{options.port}")
    print(
        "GT hand=green, prediction=blue, target=magenta, "
        "SAM(filtered)=amber, SAM(GT-SAM pose)=orange, GT YCB=cyan"
    )
    print("Press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
