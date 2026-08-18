#!/usr/bin/env python3
"""Detect sequence contact phases from HACO and initial hand-object geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-sequence-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--gt-hand-npz")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--object-samples", type=int, default=8192)
    parser.add_argument("--phase-topk", type=int, default=16)
    parser.add_argument("--enter-gap-mm", type=float, default=15.0)
    parser.add_argument("--exit-gap-mm", type=float, default=25.0)
    parser.add_argument("--min-near-contacts", type=int, default=8)
    parser.add_argument("--exit-min-near-contacts", type=int, default=3)
    parser.add_argument("--enter-patience", type=int, default=3)
    parser.add_argument("--exit-patience", type=int, default=5)
    parser.add_argument("--boundary-ramp", type=int, default=3)
    parser.add_argument("--gt-contact-mm", type=float, default=5.0)
    parser.add_argument("--gt-exit-mm", type=float, default=8.0)
    parser.add_argument("--gt-min-vertices", type=int, default=5)
    parser.add_argument("--use-object-dynamic-phase", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def aligned_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {frame_id(value): index for index, value in enumerate(source)}
    result = []
    for value in target:
        key = frame_id(value)
        if key not in lookup:
            raise KeyError(f"Frame {key} missing from aligned source")
        result.append(lookup[key])
    return np.asarray(result, dtype=np.int64)


def load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64) * scale,
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def deterministic_surface_samples(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    cdf = np.cumsum(np.maximum(np.linalg.norm(cross, axis=-1), 1e-12))
    targets = (np.arange(count, dtype=np.float64) + 0.5) / count * cdf[-1]
    selected = np.searchsorted(cdf, targets).clip(0, len(triangles) - 1)
    sequence = np.arange(count, dtype=np.float64)
    u = np.mod((sequence + 0.5) * 0.7548776662466927, 1.0)
    v = np.mod((sequence + 0.5) * 0.5698402909980532, 1.0)
    sqrt_u = np.sqrt(np.maximum(u, 1e-8))
    barycentric = np.stack(
        [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v], axis=-1
    )
    return (triangles[selected] * barycentric[:, :, None]).sum(axis=1).astype(
        np.float32
    )


def physical_pose(pose: np.ndarray, normalized_left: bool) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    if normalized_left:
        result[:3, :3] = MIRROR_X @ result[:3, :3] @ MIRROR_X
        result[:3, 3] = MIRROR_X @ result[:3, 3]
    return result


def hysteresis(
    enter: np.ndarray,
    stay: np.ndarray,
    enter_patience: int,
    exit_patience: int,
) -> np.ndarray:
    output = np.zeros(len(enter), dtype=bool)
    active = False
    positive_run = 0
    negative_run = 0
    start = 0
    for index in range(len(enter)):
        if not active:
            positive_run = positive_run + 1 if enter[index] else 0
            if positive_run >= enter_patience:
                start = index - enter_patience + 1
                output[start:index + 1] = True
                active = True
                negative_run = 0
        else:
            output[index] = True
            negative_run = 0 if stay[index] else negative_run + 1
            if negative_run >= exit_patience:
                output[index - exit_patience + 1:index + 1] = False
                active = False
                positive_run = 0
    return output


def segments(mask: np.ndarray, frame_ids: np.ndarray) -> list[dict[str, object]]:
    result = []
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            result.append({
                "start_index": int(start),
                "end_index": int(end),
                "start_frame": frame_id(frame_ids[start]),
                "end_frame": frame_id(frame_ids[end]),
                "frames": int(end - start + 1),
            })
            start = None
    return result


def ramped_gate(mask: np.ndarray, width: int) -> np.ndarray:
    gate = mask.astype(np.float32)
    if width <= 0:
        return gate
    for segment in segments(mask, np.arange(len(mask)).astype(str)):
        start = int(segment["start_index"])
        end = int(segment["end_index"])
        length = end - start + 1
        ramp = min(width, max(0, length // 2))
        for offset in range(ramp):
            value = float(offset + 1) / float(ramp + 1)
            if start > 0:
                gate[start + offset] = min(gate[start + offset], value)
            if end < len(mask) - 1:
                gate[end - offset] = min(gate[end - offset], value)
    return gate


def object_dynamic_mask(
    supervision: dict[str, np.ndarray], frame_count: int
) -> tuple[np.ndarray, str | None, list[dict[str, object]]]:
    mask = np.zeros(frame_count, dtype=bool)
    if "filtered_object_json" not in supervision:
        return mask, None, []
    path = Path(str(supervision["filtered_object_json"].item())).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Object motion segmentation not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("dynamic_segments")
    if rows is None:
        rows = payload.get("tapir_motion_segmentation", {}).get(
            "dynamic_segments"
        )
    if rows is None:
        raise KeyError(f"No dynamic_segments in {path}")
    for row in rows:
        begin, end = (int(value) for value in row["output_frames"])
        begin = max(0, begin)
        end = min(frame_count - 1, end)
        if begin <= end:
            mask[begin:end + 1] = True
    return mask, str(path), rows


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    query = load_npz(Path(args.query_npz).expanduser().resolve())
    contact = load_npz(Path(args.contact_sequence_npz).expanduser().resolve())
    supervision = load_npz(Path(args.supervision_npz).expanduser().resolve())
    ids = np.asarray(query["frame_ids"])
    trajectory_indices = aligned_indices(trajectory["frame_ids"], ids)
    contact_indices = aligned_indices(contact["frame_ids"], ids)
    supervision_indices = aligned_indices(supervision["frame_ids"], ids)
    count = len(ids)

    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_indices],
        dtype=np.float32,
    )
    hand = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    ) + wrist[:, None]
    probability = np.asarray(
        contact["contact_probability"][contact_indices], dtype=np.float32
    )
    contact_mask = np.asarray(
        contact["contact_mask"][contact_indices]
    ).astype(bool)
    valid = (
        np.asarray(trajectory["prediction_valid"][trajectory_indices]).astype(bool)
        & np.asarray(query["model_valid"]).astype(bool)
        & np.asarray(contact["contact_valid"][contact_indices]).astype(bool)
    )

    mesh = load_mesh(Path(args.object_mesh).expanduser().resolve(), args.object_scale)
    local_object = deterministic_surface_samples(mesh, args.object_samples)
    normalized_left = bool(np.asarray(
        supervision.get("normalized_left", False)
    ).item())
    poses = np.stack([
        physical_pose(supervision["gt_ycb_object_pose"][index], normalized_left)
        for index in supervision_indices
    ])
    gap_topk = np.full(count, np.nan, dtype=np.float32)
    gap_median = np.full(count, np.nan, dtype=np.float32)
    near_contacts = np.zeros(count, dtype=np.int32)
    exit_near_contacts = np.zeros(count, dtype=np.int32)
    active_contacts = contact_mask.sum(axis=1).astype(np.int32)
    probability_mean = np.full(count, np.nan, dtype=np.float32)
    object_points_by_frame = []
    for index in range(count):
        pose = poses[index]
        object_points = local_object @ pose[:3, :3].T + pose[:3, 3]
        object_points_by_frame.append(object_points)
        selected = contact_mask[index]
        if not valid[index] or not selected.any():
            continue
        distances = cKDTree(object_points).query(hand[index, selected])[0] * 1000.0
        sorted_distance = np.sort(distances)
        take = min(args.phase_topk, len(sorted_distance))
        gap_topk[index] = float(np.median(sorted_distance[:take]))
        gap_median[index] = float(np.median(distances))
        near_contacts[index] = int((distances <= args.enter_gap_mm).sum())
        exit_near_contacts[index] = int((distances <= args.exit_gap_mm).sum())
        probability_mean[index] = float(probability[index, selected].mean())

    enter = (
        valid
        & (gap_topk <= args.enter_gap_mm)
        & (near_contacts >= args.min_near_contacts)
    )
    stay = (
        valid
        & (gap_topk <= args.exit_gap_mm)
        & (exit_near_contacts >= args.exit_min_near_contacts)
    )
    geometry_phase = hysteresis(
        enter, stay, args.enter_patience, args.exit_patience
    )
    geometry_gate = ramped_gate(geometry_phase, args.boundary_ramp)
    dynamic_phase = np.zeros(count, dtype=bool)
    dynamic_source = None
    dynamic_segments: list[dict[str, object]] = []
    if args.use_object_dynamic_phase:
        dynamic_phase, dynamic_source, dynamic_segments = object_dynamic_mask(
            supervision, count
        )
    predicted_phase = geometry_phase | dynamic_phase
    predicted_gate = np.maximum(
        geometry_gate, dynamic_phase.astype(np.float32)
    )

    gt_phase = np.zeros(count, dtype=bool)
    gt_gate = np.zeros(count, dtype=np.float32)
    gt_near_vertices = np.full(count, -1, dtype=np.int32)
    gt_min_distance = np.full(count, np.nan, dtype=np.float32)
    if args.gt_hand_npz:
        gt_data = load_npz(Path(args.gt_hand_npz).expanduser().resolve())
        side = str(query["hand_side"].item()).lower()
        gt_vertices = np.asarray(gt_data[f"{side}_vertices"], dtype=np.float32)
        gt_valid = np.asarray(gt_data[f"{side}_valid"]).astype(bool)
        gt_enter = np.zeros(count, dtype=bool)
        gt_stay = np.zeros(count, dtype=bool)
        for index in range(min(count, len(gt_valid))):
            if not gt_valid[index] or not np.isfinite(gt_vertices[index]).all():
                continue
            distances = cKDTree(object_points_by_frame[index]).query(
                gt_vertices[index]
            )[0] * 1000.0
            gt_near_vertices[index] = int((distances <= args.gt_contact_mm).sum())
            gt_min_distance[index] = float(distances.min())
            gt_enter[index] = gt_near_vertices[index] >= args.gt_min_vertices
            gt_stay[index] = int((distances <= args.gt_exit_mm).sum()) >= max(
                1, args.gt_min_vertices // 2
            )
        gt_phase = hysteresis(
            gt_enter, gt_stay, args.enter_patience, args.exit_patience
        )
        gt_gate = ramped_gate(gt_phase, args.boundary_ramp)

    predicted_segments = segments(predicted_phase, ids)
    gt_segments = segments(gt_phase, ids)
    intersection = int((predicted_phase & gt_phase).sum())
    union = int((predicted_phase | gt_phase).sum())
    summary = {
        "method": (
            "v14_haco_geometry_object_dynamic_contact_phase_v2"
            if args.use_object_dynamic_phase
            else "v14_haco_geometry_contact_phase_v1"
        ),
        "stream_id": str(query["stream_id"].item()),
        "frames": count,
        "valid_frames": int(valid.sum()),
        "predicted_contact_frames": int(predicted_phase.sum()),
        "predicted_segments": predicted_segments,
        "geometry_contact_frames": int(geometry_phase.sum()),
        "geometry_segments": segments(geometry_phase, ids),
        "object_dynamic_frames": int(dynamic_phase.sum()),
        "object_dynamic_segments": dynamic_segments,
        "object_dynamic_source": dynamic_source,
        "oracle_contact_frames": int(gt_phase.sum()),
        "oracle_segments": gt_segments,
        "phase_iou": float(intersection / union) if union else None,
        "thresholds": {
            "enter_gap_mm": args.enter_gap_mm,
            "exit_gap_mm": args.exit_gap_mm,
            "min_near_contacts": args.min_near_contacts,
            "phase_topk": args.phase_topk,
            "gt_contact_mm": args.gt_contact_mm,
            "gt_min_vertices": args.gt_min_vertices,
        },
    }
    output_path = Path(args.out_npz).expanduser().resolve()
    write_npz(output_path, {
        "frame_ids": ids,
        "valid": valid,
        "haco_active_contacts": active_contacts,
        "haco_probability_mean": probability_mean,
        "contact_gap_topk_mm": gap_topk,
        "contact_gap_median_mm": gap_median,
        "near_contact_vertices": near_contacts,
        "exit_near_contact_vertices": exit_near_contacts,
        "predicted_contact_phase": predicted_phase,
        "predicted_contact_gate": predicted_gate,
        "geometry_contact_phase": geometry_phase,
        "geometry_contact_gate": geometry_gate,
        "object_dynamic_phase": dynamic_phase,
        "gt_near_contact_vertices": gt_near_vertices,
        "gt_min_contact_distance_mm": gt_min_distance,
        "oracle_contact_phase": gt_phase,
        "oracle_contact_gate": gt_gate,
        "stream_id": np.asarray(str(query["stream_id"].item())),
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


if __name__ == "__main__":
    main()
