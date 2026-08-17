#!/usr/bin/env python3
"""Cap the fixed MANO wrist boundary and inspect the virtual faces in Viser."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
import viser

from refine_v14_haco_one_way_chamfer import load_npz, physical_pose, write_npz
from refine_v14_haco_sequence_chamfer import aligned_indices, frame_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--supervision-npz")
    parser.add_argument("--object-mesh")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args()


def directed_boundary_loop(faces: np.ndarray) -> np.ndarray:
    directed_edges = []
    for face in np.asarray(faces, dtype=np.int64):
        directed_edges.extend((
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ))
    counts = Counter(tuple(sorted(edge)) for edge in directed_edges)
    boundary = [
        edge for edge in directed_edges if counts[tuple(sorted(edge))] == 1
    ]
    if not boundary:
        raise RuntimeError("MANO mesh has no boundary; it may already be watertight")

    successors: dict[int, int] = {}
    predecessors: dict[int, int] = {}
    for start, end in boundary:
        if start in successors or end in predecessors:
            raise RuntimeError(
                "Boundary is not a single consistently oriented manifold loop"
            )
        successors[start] = end
        predecessors[end] = start
    if set(successors) != set(predecessors):
        raise RuntimeError("Boundary edge directions do not form a closed loop")

    first = boundary[0][0]
    loop = [first]
    current = first
    for _ in range(len(boundary) - 1):
        current = successors[current]
        if current == first:
            raise RuntimeError("Boundary loop closed before consuming all edges")
        loop.append(current)
    if successors[current] != first:
        raise RuntimeError("Boundary loop did not close")
    if len(set(loop)) != len(boundary):
        raise RuntimeError("Boundary contains repeated vertices")
    return np.asarray(loop, dtype=np.int64)


def cap_faces(boundary_loop: np.ndarray, center_index: int) -> np.ndarray:
    faces = []
    for index, start in enumerate(boundary_loop):
        end = boundary_loop[(index + 1) % len(boundary_loop)]
        # Reverse the existing boundary-edge direction to preserve winding.
        faces.append((int(end), int(start), int(center_index)))
    return np.asarray(faces, dtype=np.int64)


def edge_incidence(faces: np.ndarray) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for face in np.asarray(faces, dtype=np.int64):
        for start, end in (
            (face[0], face[1]), (face[1], face[2]), (face[2], face[0])
        ):
            counts[tuple(sorted((int(start), int(end))))] += 1
    return counts


def load_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return (
        np.asarray(mesh.vertices, dtype=np.float32) * scale,
        np.asarray(mesh.faces, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    hand_data = load_npz(Path(args.hand_npz).expanduser().resolve())
    requested = frame_id(args.frame_id)
    hand_index = int(aligned_indices(
        hand_data["frame_ids"], np.asarray([requested])
    )[0])
    vertices_key = (
        "refined_hand_vertices_camera"
        if "refined_hand_vertices_camera" in hand_data
        else "stage1_hand_vertices_camera"
    )
    vertices = np.asarray(
        hand_data[vertices_key][hand_index], dtype=np.float32
    )
    faces = np.asarray(hand_data["mano_faces"], dtype=np.int64)
    boundary = directed_boundary_loop(faces)
    center = vertices[boundary].mean(axis=0)
    capped_vertices = np.concatenate((vertices, center[None]), axis=0)
    virtual_faces = cap_faces(boundary, len(vertices))
    capped_faces = np.concatenate((faces, virtual_faces), axis=0)
    incidence = edge_incidence(capped_faces)
    nonmanifold_edges = [edge for edge, count in incidence.items() if count != 2]

    capped_mesh = trimesh.Trimesh(
        vertices=capped_vertices, faces=capped_faces, process=False
    )
    summary = {
        "method": "fixed_topology_mano_wrist_center_fan_cap_v1",
        "frame_id": requested,
        "vertices_key": vertices_key,
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "boundary_vertices": int(len(boundary)),
        "cap_faces": int(len(virtual_faces)),
        "capped_vertices": int(len(capped_vertices)),
        "capped_faces": int(len(capped_faces)),
        "nonmanifold_edges": int(len(nonmanifold_edges)),
        "is_watertight": bool(capped_mesh.is_watertight),
        "is_winding_consistent": bool(capped_mesh.is_winding_consistent),
        "signed_volume": float(capped_mesh.volume),
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    write_npz(output_path, {
        "frame_id": np.asarray(requested),
        "original_vertices_camera": vertices,
        "original_faces": faces,
        "boundary_vertex_indices": boundary,
        "boundary_vertices_camera": vertices[boundary],
        "cap_center_camera": center,
        "capped_vertices_camera": capped_vertices,
        "cap_faces": virtual_faces,
        "capped_faces": capped_faces,
        "is_watertight": np.asarray(capped_mesh.is_watertight),
        "is_winding_consistent": np.asarray(capped_mesh.is_winding_consistent),
        "signed_volume": np.float32(capped_mesh.volume),
        "method": np.asarray(summary["method"]),
    })
    summary_path = Path(
        args.out_json or output_path.with_suffix(".json")
    ).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not capped_mesh.is_watertight or nonmanifold_edges:
        raise RuntimeError("Virtual wrist cap did not produce a watertight mesh")

    object_camera = None
    object_faces = None
    if args.supervision_npz and args.object_mesh:
        supervision = load_npz(
            Path(args.supervision_npz).expanduser().resolve()
        )
        supervision_index = int(aligned_indices(
            supervision["frame_ids"], np.asarray([requested])
        )[0])
        object_vertices, object_faces = load_mesh(
            Path(args.object_mesh).expanduser().resolve(), args.object_scale
        )
        normalized_left = bool(np.asarray(
            supervision.get("normalized_left", False)
        ).item())
        pose = physical_pose(
            supervision["gt_ycb_object_pose"][supervision_index],
            normalized_left,
        )
        object_camera = object_vertices @ pose[:3, :3].T + pose[:3, 3]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_mesh_simple(
        "/mano/original",
        vertices,
        faces,
        color=(75, 145, 245),
        opacity=0.32,
    )
    cap_visual_faces = np.concatenate(
        (virtual_faces, virtual_faces[:, ::-1]), axis=0
    )
    server.scene.add_mesh_simple(
        "/mano/virtual_wrist_cap",
        capped_vertices,
        cap_visual_faces,
        color=(245, 65, 170),
        opacity=0.9,
    )
    boundary_points = vertices[boundary]
    segments = np.stack(
        (boundary_points, np.roll(boundary_points, -1, axis=0)), axis=1
    )
    server.scene.add_line_segments(
        "/mano/wrist_boundary",
        points=segments,
        colors=np.tile(
            np.asarray([[[255, 220, 40], [255, 220, 40]]], dtype=np.uint8),
            (len(segments), 1, 1),
        ),
        line_width=3.0,
    )
    server.scene.add_point_cloud(
        "/mano/wrist_boundary_vertices",
        points=boundary_points,
        colors=np.tile(
            np.asarray([[255, 220, 40]], dtype=np.uint8),
            (len(boundary_points), 1),
        ),
        point_size=0.004,
    )
    server.scene.add_point_cloud(
        "/mano/cap_center",
        points=center[None],
        colors=np.asarray([[255, 30, 30]], dtype=np.uint8),
        point_size=0.008,
    )
    if object_camera is not None and object_faces is not None:
        server.scene.add_mesh_simple(
            "/gt_ycb_object",
            object_camera,
            object_faces,
            color=(30, 215, 225),
            opacity=0.2,
        )
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Viewer: http://localhost:{args.port}")
    print("Blue=MANO, magenta=virtual cap, yellow=wrist loop, red=cap center")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
