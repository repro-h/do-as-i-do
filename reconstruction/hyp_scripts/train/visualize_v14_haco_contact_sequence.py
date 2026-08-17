#!/usr/bin/env python3
"""Interactively inspect V14/Stage1/Stage2 contact over a full sequence."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from refine_v14_haco_one_way_chamfer import physical_pose
from refine_v14_haco_sequence_chamfer import aligned_indices, frame_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--stage2-npz")
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--gt-hand-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh: {type(loaded).__name__}")
    return (
        np.asarray(loaded.vertices, dtype=np.float32) * float(scale),
        np.asarray(loaded.faces, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    stage1 = load_npz(Path(args.stage1_npz).expanduser().resolve())
    stage2 = (
        load_npz(Path(args.stage2_npz).expanduser().resolve())
        if args.stage2_npz else None
    )
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    gt = load_npz(Path(args.gt_hand_npz).expanduser().resolve())

    ids = np.asarray(query["frame_ids"])
    count = len(ids)
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    contact_indices = aligned_indices(contact["frame_ids"], ids)
    stage1_indices = aligned_indices(stage1["frame_ids"], ids)
    stage2_indices = (
        aligned_indices(stage2["frame_ids"], ids)
        if stage2 is not None else None
    )
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices], dtype=np.float32
    )
    v14 = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    ) + wrist[:, None]
    stage1_vertices = np.asarray(
        stage1["refined_hand_vertices_camera"][stage1_indices], dtype=np.float32
    )
    stage2_vertices = (
        np.asarray(
            stage2["refined_hand_vertices_camera"][stage2_indices],
            dtype=np.float32,
        )
        if stage2 is not None and stage2_indices is not None else None
    )
    hand_faces = np.asarray(query["mano_faces"], dtype=np.int64)
    probability = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())

    side = str(query["hand_side"].item()).lower()
    gt_vertices = np.asarray(gt[f"{side}_vertices"], dtype=np.float32)[:count]
    gt_faces = np.asarray(gt[f"{side}_faces"], dtype=np.int64)
    gt_valid = np.asarray(gt[f"{side}_valid"]).astype(bool)[:count]

    object_local, object_faces = load_mesh(
        Path(args.object_mesh).expanduser().resolve(), args.object_scale
    )
    normalized_left = bool(
        np.asarray(supervision.get("normalized_left", False)).item()
    )
    object_vertices = np.empty(
        (count, len(object_local), 3), dtype=np.float32
    )
    for output_index, supervision_index in enumerate(supervision_indices):
        pose = physical_pose(
            supervision["gt_ycb_object_pose"][supervision_index],
            normalized_left,
        )
        object_vertices[output_index] = (
            object_local @ pose[:3, :3].T + pose[:3, 3]
        )

    stage1_inside = np.asarray(
        stage1["refined_object_vertex_inside_capped_mano"][stage1_indices]
    ).astype(bool)
    stage2_inside = (
        np.asarray(
            stage2["refined_object_vertex_inside_capped_mano"][stage2_indices]
        ).astype(bool)
        if stage2 is not None and stage2_indices is not None else None
    )
    stage1_inside_count = stage1_inside.sum(axis=1)
    stage2_inside_count = (
        stage2_inside.sum(axis=1)
        if stage2_inside is not None else np.zeros(count, dtype=np.int32)
    )
    valid = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    initial = max(0, min(count - 1, int(args.initial_frame)))
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=count - 1, step=1, initial_value=initial
    )
    play_button = server.gui.add_button("Play / Pause")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=30, step=1, initial_value=int(args.fps)
    )
    threshold_slider = server.gui.add_slider(
        "HACO threshold", min=0.0, max=1.0, step=0.01,
        initial_value=threshold,
    )
    point_size = server.gui.add_slider(
        "Point size", min=0.001, max=0.015, step=0.001,
        initial_value=0.005,
    )
    controls = {
        "object": server.gui.add_checkbox("GT YCB object", initial_value=True),
        "v14": server.gui.add_checkbox("V14 WiLoR hand", initial_value=False),
        "stage1": server.gui.add_checkbox(
            "Stage1 rigid hand", initial_value=stage2_vertices is None
        ),
        "stage2": server.gui.add_checkbox(
            "Stage2 local hand", initial_value=stage2_vertices is not None
        ),
        "gt": server.gui.add_checkbox("DexYCB GT hand", initial_value=True),
        "stage2_contact": server.gui.add_checkbox(
            "HACO contacts on refined hand", initial_value=True
        ),
        "gt_contact": server.gui.add_checkbox(
            "HACO vertex indices on GT", initial_value=False
        ),
        "stage1_inside": server.gui.add_checkbox(
            "Stage1 contained YCB vertices",
            initial_value=stage2_inside is None,
        ),
        "stage2_inside": server.gui.add_checkbox(
            "Stage2 contained YCB vertices",
            initial_value=stage2_inside is not None,
        ),
    }
    handles = []
    playing = {"value": False}
    suppress = {"value": False}

    def clear() -> None:
        while handles:
            handles.pop().remove()

    def add_hand(
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[int, int, int],
        opacity: float,
    ) -> None:
        handles.append(server.scene.add_mesh_simple(
            name, vertices=vertices, faces=faces, color=color, opacity=opacity
        ))

    def show_frame(value: int) -> None:
        index = max(0, min(count - 1, int(value)))
        clear()
        if controls["object"].value:
            handles.append(server.scene.add_mesh_simple(
                "/gt_ycb_object",
                vertices=object_vertices[index],
                faces=object_faces,
                color=(35, 205, 220),
                opacity=0.34,
            ))
        if controls["v14"].value and valid[index]:
            add_hand("/v14_wilor_hand", v14[index], hand_faces, (65, 135, 245), 0.34)
        if controls["stage1"].value and valid[index]:
            add_hand(
                "/stage1_rigid_hand", stage1_vertices[index], hand_faces,
                (245, 155, 35), 0.40,
            )
        if (
            controls["stage2"].value
            and stage2_vertices is not None
            and valid[index]
        ):
            add_hand(
                "/stage2_local_hand", stage2_vertices[index], hand_faces,
                (220, 65, 190), 0.48,
            )
        if controls["gt"].value and gt_valid[index]:
            add_hand(
                "/dexycb_gt_hand", gt_vertices[index], gt_faces,
                (50, 215, 90), 0.38,
            )

        active = probability[index] > float(threshold_slider.value)
        if active.any():
            strength = probability[index, active, None]
            contact_colors = np.concatenate((
                np.full_like(strength, 255.0),
                150.0 * (1.0 - strength),
                30.0 * (1.0 - strength),
            ), axis=1).clip(0, 255).astype(np.uint8)
            if controls["stage2_contact"].value:
                contact_vertices = (
                    stage2_vertices[index]
                    if stage2_vertices is not None
                    else stage1_vertices[index]
                )
                handles.append(server.scene.add_point_cloud(
                    "/refined_haco_contacts",
                    points=contact_vertices[active],
                    colors=contact_colors,
                    point_size=float(point_size.value),
                ))
            if controls["gt_contact"].value and gt_valid[index]:
                handles.append(server.scene.add_point_cloud(
                    "/gt_haco_vertex_indices",
                    points=gt_vertices[index, active],
                    colors=np.tile(
                        np.asarray([[255, 225, 35]], dtype=np.uint8),
                        (int(active.sum()), 1),
                    ),
                    point_size=float(point_size.value),
                ))
        if controls["stage1_inside"].value and stage1_inside[index].any():
            selected = object_vertices[index, stage1_inside[index]]
            handles.append(server.scene.add_point_cloud(
                "/stage1_contained_object_vertices",
                points=selected,
                colors=np.tile(
                    np.asarray([[255, 125, 25]], dtype=np.uint8),
                    (len(selected), 1),
                ),
                point_size=float(point_size.value) * 0.8,
            ))
        if (
            controls["stage2_inside"].value
            and stage2_inside is not None
            and stage2_inside[index].any()
        ):
            selected = object_vertices[index, stage2_inside[index]]
            handles.append(server.scene.add_point_cloud(
                "/stage2_contained_object_vertices",
                points=selected,
                colors=np.tile(
                    np.asarray([[255, 25, 25]], dtype=np.uint8),
                    (len(selected), 1),
                ),
                point_size=float(point_size.value) * 0.9,
            ))
        print(
            f"frame={frame_id(ids[index])} index={index} "
            f"haco={int(active.sum())} "
            f"stage1_inside={int(stage1_inside_count[index])} "
            f"stage2_inside={int(stage2_inside_count[index])} "
            f"gt_valid={bool(gt_valid[index])}",
            flush=True,
        )

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress["value"]:
            show_frame(int(frame_slider.value))

    @play_button.on_click
    def _(_) -> None:
        playing["value"] = not playing["value"]

    for control in controls.values():
        control.on_update(lambda _: show_frame(int(frame_slider.value)))
    threshold_slider.on_update(lambda _: show_frame(int(frame_slider.value)))
    point_size.on_update(lambda _: show_frame(int(frame_slider.value)))

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
    show_frame(initial)
    print(f"Viewer: http://localhost:{args.port}")
    print(
        "Blue=V14, orange=Stage1, magenta=Stage2, green=GT hand, "
        "cyan=GT YCB, warm=HACO contacts, red=contained YCB vertices"
    )
    print("Press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
