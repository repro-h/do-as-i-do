#!/usr/bin/env python3
"""Detect original YCB mesh vertices contained by a wrist-capped MANO mesh."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from refine_v14_haco_one_way_chamfer import (
    distribution,
    load_npz,
    physical_pose,
    write_npz,
)
from refine_v14_haco_sequence_chamfer import aligned_indices, frame_id
from visualize_capped_mano_wrist import (
    cap_faces,
    directed_boundary_loop,
    edge_incidence,
    load_mesh,
)


RAY_DIRECTIONS = np.asarray(
    [
        [1.0, 0.37139067, 0.127831],
        [-0.219439, 1.0, 0.413117],
        [0.287771, -0.193337, 1.0],
    ],
    dtype=np.float64,
)
RAY_DIRECTIONS /= np.linalg.norm(RAY_DIRECTIONS, axis=-1, keepdims=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--vertices-key")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--point-chunk", type=int, default=256)
    parser.add_argument("--visualize-frame")
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args()


def select_vertices_key(data: dict[str, np.ndarray], requested: str | None) -> str:
    if requested:
        if requested not in data:
            raise KeyError(f"Hand archive lacks vertices key {requested!r}")
        return requested
    for key in (
        "refined_hand_vertices_camera",
        "stage1_hand_vertices_camera",
        "initial_hand_vertices_camera",
    ):
        if key in data:
            return key
    raise KeyError("Could not find camera-space MANO vertices in hand archive")


def ray_parity(
    points: np.ndarray,
    triangles: np.ndarray,
    direction: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Return odd/even ray-triangle intersection parity for each point."""
    vertex0 = triangles[:, 0]
    edge1 = triangles[:, 1] - vertex0
    edge2 = triangles[:, 2] - vertex0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ti,ti->t", edge1, h)
    usable = np.abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    result = np.zeros(len(points), dtype=bool)
    epsilon = 1e-10

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        offset = points[start:end, None, :] - vertex0[None, :, :]
        barycentric_u = np.einsum("pti,ti->pt", offset, h) * inverse[None]
        q = np.cross(offset, edge1[None, :, :])
        barycentric_v = np.einsum("i,pti->pt", direction, q) * inverse[None]
        distance = np.einsum("ti,pti->pt", edge2, q) * inverse[None]
        hit = (
            usable[None]
            & (barycentric_u >= -epsilon)
            & (barycentric_v >= -epsilon)
            & (barycentric_u + barycentric_v <= 1.0 + epsilon)
            & (distance > epsilon)
        )
        result[start:end] = (hit.sum(axis=1) % 2) == 1
    return result


def contains_points_vote(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    votes = np.stack(
        [
            ray_parity(points, triangles, direction, chunk_size)
            for direction in RAY_DIRECTIONS
        ],
        axis=0,
    )
    return votes.sum(axis=0) >= 2, votes


def transformed_object_vertices(
    local_vertices: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    return local_vertices @ pose[:3, :3].T + pose[:3, 3]


def main() -> None:
    args = parse_args()
    if args.point_chunk <= 0:
        raise ValueError("--point-chunk must be positive")

    hand = load_npz(Path(args.hand_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    vertices_key = select_vertices_key(hand, args.vertices_key)
    ids = np.asarray(hand["frame_ids"])
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)
    hand_vertices = np.asarray(hand[vertices_key], dtype=np.float32)
    mano_faces = np.asarray(hand["mano_faces"], dtype=np.int64)
    object_vertices, object_faces = load_mesh(
        Path(args.object_mesh).expanduser().resolve(), args.object_scale
    )
    boundary = directed_boundary_loop(mano_faces)
    virtual_faces = cap_faces(boundary, hand_vertices.shape[1])
    capped_faces = np.concatenate((mano_faces, virtual_faces), axis=0)
    incidence = edge_incidence(capped_faces)
    nonmanifold = sum(count != 2 for count in incidence.values())
    if nonmanifold:
        raise RuntimeError(f"Capped MANO has {nonmanifold} nonmanifold edges")

    normalized_left = bool(np.asarray(
        supervision.get("normalized_left", False)
    ).item())
    object_valid_key = next(
        (
            key for key in ("gt_object_valid", "object_valid")
            if key in supervision
        ),
        None,
    )
    valid = np.isfinite(hand_vertices).all(axis=(1, 2))
    if object_valid_key is not None:
        valid &= np.asarray(
            supervision[object_valid_key][supervision_indices]
        ).astype(bool)

    inside_mask = np.zeros(
        (len(ids), len(object_vertices)), dtype=bool
    )
    ray_votes = np.zeros(
        (len(ids), len(object_vertices)), dtype=np.uint8
    )
    inside_count = np.zeros(len(ids), dtype=np.int32)
    disagreement_count = np.zeros(len(ids), dtype=np.int32)
    watertight = np.zeros(len(ids), dtype=bool)
    winding_consistent = np.zeros(len(ids), dtype=bool)
    signed_volume = np.full(len(ids), np.nan, dtype=np.float32)

    for output_index, supervision_index in enumerate(supervision_indices):
        if not valid[output_index]:
            continue
        center = hand_vertices[output_index, boundary].mean(axis=0)
        capped_vertices = np.concatenate(
            (hand_vertices[output_index], center[None]), axis=0
        )
        capped_mesh = trimesh.Trimesh(
            vertices=capped_vertices,
            faces=capped_faces,
            process=False,
        )
        watertight[output_index] = capped_mesh.is_watertight
        winding_consistent[output_index] = capped_mesh.is_winding_consistent
        signed_volume[output_index] = capped_mesh.volume
        if not watertight[output_index] or not winding_consistent[output_index]:
            raise RuntimeError(
                f"Frame {frame_id(ids[output_index])} capped MANO is invalid"
            )

        pose = physical_pose(
            supervision["gt_ycb_object_pose"][supervision_index],
            normalized_left,
        )
        object_camera = transformed_object_vertices(object_vertices, pose)
        contained, votes = contains_points_vote(
            object_camera,
            capped_vertices,
            capped_faces,
            args.point_chunk,
        )
        vote_count = votes.sum(axis=0).astype(np.uint8)
        inside_mask[output_index] = contained
        ray_votes[output_index] = vote_count
        inside_count[output_index] = int(contained.sum())
        disagreement_count[output_index] = int(
            ((vote_count > 0) & (vote_count < len(RAY_DIRECTIONS))).sum()
        )
        print(
            f"[{output_index + 1}/{len(ids)}] {frame_id(ids[output_index])} "
            f"inside={inside_count[output_index]}/{len(object_vertices)} "
            f"ray_disagreement={disagreement_count[output_index]}"
        )

    valid_counts = inside_count[valid]
    valid_disagreement = disagreement_count[valid]
    top_indices = np.flatnonzero(valid)[
        np.argsort(valid_counts)[-10:][::-1]
    ]
    summary = {
        "method": "original_ycb_vertices_inside_capped_mano_ray_vote_v1",
        "vertices_key": vertices_key,
        "frames": int(len(ids)),
        "valid_frames": int(valid.sum()),
        "object_vertices": int(len(object_vertices)),
        "object_faces": int(len(object_faces)),
        "mano_boundary_vertices": int(len(boundary)),
        "ray_directions": int(len(RAY_DIRECTIONS)),
        "inside_vertices_per_frame": distribution(valid_counts),
        "inside_fraction_per_frame": distribution(
            valid_counts.astype(np.float64) / max(len(object_vertices), 1)
        ),
        "ray_disagreement_vertices_per_frame": distribution(valid_disagreement),
        "frames_with_inside_vertices": int((valid_counts > 0).sum()),
        "top_frames": [
            {
                "frame_id": frame_id(ids[index]),
                "inside_vertices": int(inside_count[index]),
                "inside_fraction": float(
                    inside_count[index] / max(len(object_vertices), 1)
                ),
                "ray_disagreement_vertices": int(disagreement_count[index]),
            }
            for index in top_indices
        ],
        "warning": (
            "Containment is evaluated only at original YCB mesh vertices; "
            "triangle-only intersections can still be missed."
        ),
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    write_npz(output_path, {
        "frame_ids": ids,
        "valid": valid,
        "object_vertices_canonical": object_vertices,
        "object_faces": object_faces,
        "mano_faces": mano_faces,
        "wrist_boundary_vertex_indices": boundary,
        "virtual_cap_faces": virtual_faces,
        "capped_mano_faces": capped_faces,
        "object_vertex_inside_capped_mano": inside_mask,
        "object_vertex_inside_ray_votes": ray_votes,
        "inside_object_vertices": inside_count,
        "ray_disagreement_vertices": disagreement_count,
        "capped_mano_watertight": watertight,
        "capped_mano_winding_consistent": winding_consistent,
        "capped_mano_signed_volume": signed_volume,
        "vertices_key": np.asarray(vertices_key),
        "method": np.asarray(summary["method"]),
    })
    summary_path = Path(
        args.out_json or output_path.with_suffix(".json")
    ).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")

    if args.visualize_frame is None:
        return

    import viser

    requested = frame_id(args.visualize_frame)
    hand_index = int(aligned_indices(ids, np.asarray([requested]))[0])
    supervision_index = int(supervision_indices[hand_index])
    center = hand_vertices[hand_index, boundary].mean(axis=0)
    capped_vertices = np.concatenate(
        (hand_vertices[hand_index], center[None]), axis=0
    )
    pose = physical_pose(
        supervision["gt_ycb_object_pose"][supervision_index], normalized_left
    )
    object_camera = transformed_object_vertices(object_vertices, pose)
    contained = inside_mask[hand_index]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_mesh_simple(
        "/mano/original",
        hand_vertices[hand_index],
        mano_faces,
        color=(75, 145, 245),
        opacity=0.28,
    )
    cap_visual_faces = np.concatenate(
        (virtual_faces, virtual_faces[:, ::-1]), axis=0
    )
    server.scene.add_mesh_simple(
        "/mano/virtual_wrist_cap",
        capped_vertices,
        cap_visual_faces,
        color=(245, 65, 170),
        opacity=0.75,
    )
    server.scene.add_mesh_simple(
        "/ycb/object",
        object_camera,
        object_faces,
        color=(30, 215, 225),
        opacity=0.2,
    )
    if contained.any():
        server.scene.add_point_cloud(
            "/ycb/vertices_inside_capped_mano",
            points=object_camera[contained],
            colors=np.tile(
                np.asarray([[255, 30, 30]], dtype=np.uint8),
                (int(contained.sum()), 1),
            ),
            point_size=0.004,
        )
    print(f"Viewer: http://localhost:{args.port}")
    print(
        f"Frame {requested}: red={int(contained.sum())}/"
        f"{len(contained)} original YCB vertices inside capped MANO"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
