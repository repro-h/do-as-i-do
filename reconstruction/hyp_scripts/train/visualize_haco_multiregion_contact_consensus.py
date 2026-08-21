#!/usr/bin/env python3
"""Visualize fixed multi-region YCB contact patches across a sequence."""

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
    select_hand_vertices_key,
    strongest_components,
    vertex_normals,
)
from visualize_haco_choir_opposition_candidates import (
    frame_id,
    index_for,
    load_npz,
    physical_pose,
)
from visualize_haco_multiregion_object_contacts import PALETTE, colors, lighter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-npz", required=True)
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--hand-npz")
    parser.add_argument("--hand-vertices-key")
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--haco-anchor-topk", type=int, default=12)
    parser.add_argument("--haco-anchor-min-vertices", type=int, default=3)
    return parser.parse_args()


def probability_colors(values: np.ndarray, threshold: float) -> np.ndarray:
    normalized = np.clip(
        (values - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    )
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * np.abs(normalized - 0.5), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    return np.rint(np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    selection = load_npz(Path(args.selection_npz).expanduser().resolve())
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())

    ids = np.asarray(query["frame_ids"])
    count = len(ids)
    trajectory_indices = np.asarray(
        [index_for(trajectory["frame_ids"], frame_id(value)) for value in ids]
    )
    contact_indices = np.asarray(
        [index_for(contact["frame_ids"], frame_id(value)) for value in ids]
    )
    supervision_indices = np.asarray(
        [index_for(supervision["frame_ids"], frame_id(value)) for value in ids]
    )
    hand = (
        np.asarray(query["vertices_3d_root_relative_original"], dtype=np.float32)
        + np.asarray(
            trajectory["predicted_wrist_camera"][trajectory_indices],
            dtype=np.float32,
        )[:, None]
    )
    hand_label = "V14 hand"
    if args.hand_npz:
        hand_data = load_npz(Path(args.hand_npz).expanduser().resolve())
        hand_indices = np.asarray([
            index_for(hand_data["frame_ids"], frame_id(value)) for value in ids
        ])
        hand_key = select_hand_vertices_key(
            hand_data, args.hand_vertices_key
        )
        hand = np.asarray(
            hand_data[hand_key][hand_indices], dtype=np.float32
        )
        hand_label = f"Selection hand: {hand_key}"
    hand_faces = np.asarray(query["mano_faces"], dtype=np.int64)
    valid = (
        np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
    )
    probability = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    threshold = float(np.asarray(contact["contact_threshold"]).item())
    region_ids, region_names = mano_contact_region_ids(
        args.mano_data_dir, str(query["hand_side"].item()).lower()
    )
    mano_graph = adjacency(hand_faces, 778)

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
            supervision["gt_ycb_object_pose"][supervision_index],
            normalized_left,
        )
        for supervision_index in supervision_indices
    ]
    region_key = (
        "stable_region_names"
        if "stable_region_names" in selection
        and len(selection["stable_region_names"])
        else "selected_region_names"
    )
    selected_regions = [str(value) for value in selection[region_key]]
    print(
        f"[viser] fixed patch regions ({region_key}): "
        f"{','.join(selected_regions)}",
        flush=True,
    )
    translation_consistent = set(
        str(value) for value in selection.get(
            "translation_consistent_region_names", np.asarray([])
        )
    )
    opposition_pairs = [
        (str(pair[0]), str(pair[1]))
        for pair in selection.get(
            "automatic_opposition_region_pairs", np.empty((0, 2))
        )
    ]
    print(
        "[viser] translation-consistent regions: "
        f"{','.join(sorted(translation_consistent)) or '-'}",
        flush=True,
    )
    print(f"[viser] automatic opposition pairs: {opposition_pairs}", flush=True)
    patches = {
        name: np.asarray(
            selection[f"{name}_patch_vertices_canonical"], dtype=np.float32
        )
        for name in selected_regions
    }
    observation_lookup: dict[tuple[str, str], int] = {}
    for observation_frame, observation_region, vertex_id in zip(
        selection["observation_frame_ids"],
        selection["observation_region_names"],
        selection["observation_selected_vertex_ids"],
    ):
        observation_lookup[(str(observation_frame), str(observation_region))] = int(vertex_id)
    vote_lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    if "observation_translation_votes_camera" in selection:
        for observation_frame, observation_region, center, vote in zip(
            selection["observation_frame_ids"],
            selection["observation_region_names"],
            selection["observation_hand_region_centers_camera"],
            selection["observation_translation_votes_camera"],
        ):
            vote_lookup[(str(observation_frame), str(observation_region))] = (
                np.asarray(center, dtype=np.float32),
                np.asarray(vote, dtype=np.float32),
            )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    initial = max(0, min(count - 1, args.initial_frame))
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=count - 1, step=1, initial_value=initial
    )
    play_button = server.gui.add_button("Play / Pause")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=30, step=1, initial_value=max(1, args.fps)
    )
    point_size = server.gui.add_slider(
        "Point size", min=0.001, max=0.015, step=0.001, initial_value=0.006
    )
    show_object = server.gui.add_checkbox("GT YCB object", initial_value=True)
    show_hand = server.gui.add_checkbox(hand_label, initial_value=True)
    show_haco = server.gui.add_checkbox("HACO components", initial_value=True)
    show_anchors = server.gui.add_checkbox(
        "Selected high-probability anchors", initial_value=True
    )
    probability_heatmap = server.gui.add_checkbox(
        "HACO probability heatmap", initial_value=True
    )
    patch_facing_only = server.gui.add_checkbox(
        "Patch-facing HACO only", initial_value=False
    )
    show_patches = server.gui.add_checkbox("Fixed object patches", initial_value=True)
    show_observations = server.gui.add_checkbox(
        "Sampled frame observations", initial_value=True
    )
    show_votes = server.gui.add_checkbox("Translation votes", initial_value=True)
    show_opposition = server.gui.add_checkbox(
        "Automatic opposition", initial_value=True
    )
    handles = []
    playing = {"value": False}
    suppress = {"value": False}

    def clear() -> None:
        while handles:
            handles.pop().remove()

    def show_frame(index: int) -> None:
        index = max(0, min(count - 1, int(index)))
        clear()
        pose = poses[index]
        object_vertices = object_local @ pose[:3, :3].T + pose[:3, 3]
        hand_normals = vertex_normals(hand[index], hand_faces)
        requested = frame_id(ids[index])
        if show_object.value:
            handles.append(server.scene.add_mesh_simple(
                "/object",
                vertices=object_vertices,
                faces=object_faces,
                color=(170, 180, 195),
                opacity=0.55,
            ))
        if show_hand.value and valid[index]:
            handles.append(server.scene.add_mesh_simple(
                "/selection_hand",
                vertices=hand[index],
                faces=hand_faces,
                color=(80, 175, 245),
                opacity=0.45,
            ))
        for region_index, name in enumerate(region_names):
            color = PALETTE[name]
            raw_mask = (
                (region_ids == region_index)
                & (probability[index] >= threshold)
            )
            component = strongest_components(
                raw_mask, mano_graph, probability[index], 1
            ) if raw_mask.any() else raw_mask
            displayed_component = component.copy()
            compatibility = np.full(len(hand[index]), np.nan, dtype=np.float32)
            if name in patches and component.any():
                patch_normals_key = f"{name}_patch_normals_canonical"
                if patch_normals_key in selection:
                    patch_normal = np.asarray(
                        selection[patch_normals_key], dtype=np.float32
                    ).mean(axis=0) @ pose[:3, :3].T
                    patch_normal /= max(float(np.linalg.norm(patch_normal)), 1e-12)
                    compatibility[component] = (
                        hand_normals[component] @ patch_normal
                    )
                    if patch_facing_only.value:
                        displayed_component &= compatibility <= -0.2
            if show_haco.value and component.any():
                displayed_probability = probability[index, displayed_component]
                point_colors = (
                    probability_colors(displayed_probability, threshold)
                    if probability_heatmap.value
                    else colors(int(displayed_component.sum()), lighter(color))
                )
                if displayed_component.any():
                    handles.append(server.scene.add_point_cloud(
                        f"/haco/{name}",
                        points=hand[index, displayed_component],
                        colors=point_colors,
                        point_size=float(point_size.value) * 0.75,
                    ))
                compatible = component & (compatibility <= -0.2)
                incompatible = component & np.isfinite(compatibility) & ~compatible
                if compatible.any() or incompatible.any():
                    compatible_probability = probability[index, compatible]
                    incompatible_probability = probability[index, incompatible]
                    print(
                        f"  {name}: facing={int(compatible.sum())} "
                        f"p50={float(np.median(compatible_probability)) if compatible.any() else float('nan'):.3f} "
                        f"other={int(incompatible.sum())} "
                        f"p50={float(np.median(incompatible_probability)) if incompatible.any() else float('nan'):.3f}",
                        flush=True,
                    )
            if (
                show_anchors.value
                and name in patches
                and int(component.sum()) >= args.haco_anchor_min_vertices
            ):
                component_ids = np.flatnonzero(component)
                anchor_count = min(args.haco_anchor_topk, len(component_ids))
                order = np.argsort(probability[index, component_ids])[-anchor_count:]
                anchor_ids = component_ids[order]
                handles.append(server.scene.add_point_cloud(
                    f"/haco_anchor/{name}",
                    points=hand[index, anchor_ids],
                    colors=colors(anchor_count, (20, 20, 20)),
                    point_size=float(point_size.value) * 1.5,
                ))
            if name not in patches:
                continue
            patch_camera = patches[name] @ pose[:3, :3].T + pose[:3, 3]
            if show_patches.value:
                handles.append(server.scene.add_point_cloud(
                    f"/fixed_patch/{name}",
                    points=patch_camera,
                    colors=colors(len(patch_camera), color),
                    point_size=float(point_size.value),
                ))
            observation_id = observation_lookup.get((requested, name))
            if show_observations.value and observation_id is not None:
                handles.append(server.scene.add_point_cloud(
                    f"/observation/{name}",
                    points=object_vertices[[observation_id]],
                    colors=colors(1, color),
                    point_size=float(point_size.value) * 1.8,
                ))
            vote = vote_lookup.get((requested, name))
            if show_votes.value and vote is not None:
                center, offset = vote
                vote_color = color if name in translation_consistent else (255, 190, 30)
                handles.append(server.scene.add_line_segments(
                    f"/translation_vote/{name}",
                    points=np.stack([center, center + offset])[None],
                    colors=np.asarray([[vote_color, vote_color]], dtype=np.uint8),
                    line_width=3.0,
                ))
        if show_opposition.value:
            for first, second in opposition_pairs:
                if first not in patches or second not in patches:
                    continue
                first_center = (
                    patches[first] @ pose[:3, :3].T + pose[:3, 3]
                ).mean(axis=0)
                second_center = (
                    patches[second] @ pose[:3, :3].T + pose[:3, 3]
                ).mean(axis=0)
                handles.append(server.scene.add_line_segments(
                    f"/automatic_opposition/{first}_{second}",
                    points=np.stack([first_center, second_center])[None],
                    colors=np.asarray(
                        [[[255, 235, 40], [255, 235, 40]]], dtype=np.uint8
                    ),
                    line_width=4.0,
                ))
        print(
            f"frame={requested} index={index} "
            f"sampled={any(frame == requested for frame, _ in observation_lookup)}",
            flush=True,
        )

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress["value"]:
            show_frame(int(frame_slider.value))

    @play_button.on_click
    def _(_) -> None:
        playing["value"] = not playing["value"]

    for control in (
        point_size,
        show_object,
        show_hand,
        show_haco,
        show_anchors,
        probability_heatmap,
        patch_facing_only,
        show_patches,
        show_observations,
        show_votes,
        show_opposition,
    ):
        control.on_update(lambda _: show_frame(int(frame_slider.value)))

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
    print(f"Viewer: http://localhost:{args.port}", flush=True)
    print("Press Ctrl+C to stop", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
