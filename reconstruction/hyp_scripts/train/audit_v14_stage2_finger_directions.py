#!/usr/bin/env python3
"""Audit per-finger Stage2 push directions and realized MANO motion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


JOINT_NAMES = ("mcp", "pip", "dip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-npz", type=Path, required=True)
    parser.add_argument(
        "--frames",
        nargs="*",
        default=[],
        help="Frame IDs such as 000035. Defaults to frames with a push direction.",
    )
    parser.add_argument("--joint-limit-deg", type=float, default=16.0)
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 1e-8 else np.zeros(3)


def finite_percentile(values: np.ndarray, percentiles=(10, 50, 90)) -> str:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return "-"
    return str(np.round(np.percentile(values, percentiles), 3).tolist())


def main() -> None:
    args = parse_args()
    with np.load(args.stage2_npz.expanduser().resolve(), allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    required = {
        "frame_ids",
        "initial_hand_vertices_camera",
        "refined_hand_vertices_camera",
        "contact_region_names",
        "contact_region_id",
        "contact_normal_region_direction_camera",
        "joint_rotation_delta_rotvec",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise KeyError(f"Stage2 archive is missing keys: {missing}")

    frame_ids = [str(value) for value in payload["frame_ids"]]
    region_names = [str(value) for value in payload["contact_region_names"]]
    region_ids = payload["contact_region_id"].astype(np.int64)
    before = payload["initial_hand_vertices_camera"].astype(np.float64)
    after = payload["refined_hand_vertices_camera"].astype(np.float64)
    movement = after - before
    directions = payload["contact_normal_region_direction_camera"].astype(np.float64)
    references = payload.get(
        "contact_normal_patch_side_reference_camera",
        np.zeros_like(directions),
    ).astype(np.float64)
    joint_delta = np.linalg.norm(
        payload["joint_rotation_delta_rotvec"].astype(np.float64), axis=-1
    ) * 180.0 / np.pi

    initial_labels = payload.get("initial_inside_object_region_id")
    refined_labels = payload.get("refined_inside_object_region_id")
    gate = payload.get("contact_normal_pushout_gate")

    if args.frames:
        unknown = sorted(set(args.frames) - set(frame_ids))
        if unknown:
            raise KeyError(f"Unknown frame IDs: {unknown}")
        frame_indices = [frame_ids.index(value) for value in args.frames]
    else:
        active = np.linalg.norm(directions, axis=-1).max(axis=1) > 1e-6
        frame_indices = np.flatnonzero(active).tolist()

    print("Legend: axial motion is along push; tangent is perpendicular to push.")
    print("Flags: REVERSED, MOVED_BACKWARD, TANGENT_DOMINANT, SATURATED.\n")

    for frame_index in frame_indices:
        print(f"===== {frame_ids[frame_index]} =====")
        for region_index, region_name in enumerate(region_names):
            vertex_mask = region_ids == region_index
            direction = unit(directions[frame_index, region_index])
            reference = unit(references[frame_index, region_index])
            has_direction = np.linalg.norm(direction) > 0
            if not vertex_mask.any() and not has_direction:
                continue

            region_move = movement[frame_index, vertex_mask]
            if has_direction:
                axial = region_move @ direction * 1000.0
                tangent = np.linalg.norm(
                    region_move - (region_move @ direction)[:, None] * direction,
                    axis=-1,
                ) * 1000.0
            else:
                axial = np.full(int(vertex_mask.sum()), np.nan)
                tangent = np.full(int(vertex_mask.sum()), np.nan)

            patch_dot = (
                float(direction @ reference)
                if has_direction and np.linalg.norm(reference) > 0
                else float("nan")
            )
            inside_before = (
                int((initial_labels[frame_index] == region_index).sum())
                if initial_labels is not None
                else -1
            )
            inside_after = (
                int((refined_labels[frame_index] == region_index).sum())
                if refined_labels is not None
                else -1
            )
            gated = (
                int(gate[frame_index, vertex_mask].sum())
                if gate is not None
                else -1
            )

            joint_start = region_index * 3 - 3 if region_name == "thumb" else None
            finger_order = {"index": 0, "middle": 1, "pinky": 2, "ring": 3, "thumb": 4}
            finger_index = finger_order.get(region_name)
            joints = (
                joint_delta[frame_index, finger_index * 3 : finger_index * 3 + 3]
                if finger_index is not None
                else np.asarray([])
            )
            del joint_start

            flags = []
            if np.isfinite(patch_dot) and patch_dot < 0:
                flags.append("REVERSED")
            if has_direction and np.nanmedian(axial) < -0.25:
                flags.append("MOVED_BACKWARD")
            if has_direction and np.nanmedian(tangent) > max(1.0, np.nanmedian(axial)):
                flags.append("TANGENT_DOMINANT")
            if joints.size and np.any(joints >= args.joint_limit_deg - 0.1):
                flags.append("SATURATED")

            print(
                f"{region_name:6s} inside={inside_before}->{inside_after} "
                f"gated={gated} patch_dot={patch_dot:.3f}"
            )
            print(
                "       push=", np.round(direction, 3).tolist(),
                "patch=", np.round(reference, 3).tolist(),
            )
            print(
                f"       axial_mm(p10/med/p90)={finite_percentile(axial)} "
                f"tangent_mm={finite_percentile(tangent)}"
            )
            if joints.size:
                joint_text = ", ".join(
                    f"{name}={value:.2f}" for name, value in zip(JOINT_NAMES, joints)
                )
                print(f"       joints_deg: {joint_text}")
            print("       flags:", ", ".join(flags) if flags else "OK")
        print()


if __name__ == "__main__":
    main()
