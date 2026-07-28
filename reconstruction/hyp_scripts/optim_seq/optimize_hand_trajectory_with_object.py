#!/usr/bin/env python3
"""Suppress hand trajectory glitches using a filtered object trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--object-pose-json", required=True)
    parser.add_argument("--segmentation-audit", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--trajectory-mode",
        choices=("symmetric_median", "causal_hold"),
        default="symmetric_median",
    )
    parser.add_argument("--median-window", type=int, default=7)
    parser.add_argument("--causal-min-history", type=int, default=6)
    parser.add_argument("--causal-jump-mm", type=float, default=8.0)
    parser.add_argument(
        "--causal-start-frame",
        type=int,
        default=-1,
        help="Do not classify jumps before this output frame; -1 uses the segment start.",
    )
    parser.add_argument("--w-relative-velocity", type=float, default=4.0)
    parser.add_argument("--w-relative-acceleration", type=float, default=2.0)
    parser.add_argument("--w-anchor", type=float, default=1.0)
    parser.add_argument("--w-correction-acceleration", type=float, default=1.0)
    parser.add_argument("--max-translation-mm", type=float, default=30.0)
    parser.add_argument("--boundary-blend-frames", type=int, default=4)
    parser.add_argument("--loss", choices=("linear", "soft_l1", "huber"), default="soft_l1")
    parser.add_argument("--f-scale-mm", type=float, default=3.0)
    return parser.parse_args()


def pose_rows(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        payload.get("by_frame")
        or payload.get("poses")
        or payload.get("frames")
        or payload
    )
    iterator = rows.items() if isinstance(rows, dict) else enumerate(rows)
    output = {}
    for key, row in iterator:
        frame = str(key).zfill(6)
        value = row
        if isinstance(row, dict):
            frame = str(row.get("frame", row.get("frame_id", key))).zfill(6)
            value = (
                row.get("object_in_camera")
                or row.get("pose")
                or row.get("transform")
            )
        if value is None:
            continue
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.size == 16 and np.isfinite(matrix).all():
            output[frame] = matrix.reshape(4, 4)
    return output


def dynamic_segments(path: Path) -> list[tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("dynamic_segments")
    if rows is None:
        rows = payload.get("tapir_motion_segmentation", {}).get("dynamic_segments")
    if rows is None:
        raise KeyError(f"No dynamic_segments in {path}")
    return [
        tuple(int(value) for value in row["output_frames"])
        for row in rows
    ]


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    output = np.empty_like(values)
    for index in range(len(values)):
        output[index] = np.median(
            values[max(0, index - radius) : min(len(values), index + radius + 1)],
            axis=0,
        )
    return output


def causal_hold_filter(
    values: np.ndarray,
    edge_frames: np.ndarray,
    window: int,
    min_history: int,
    jump_threshold: float,
    start_frame: int,
) -> tuple[np.ndarray, list[dict]]:
    output = values.copy()
    history: list[np.ndarray] = []
    events = []
    for index, value in enumerate(values):
        can_test = (
            len(history) >= min_history
            and (start_frame < 0 or int(edge_frames[index]) >= start_frame)
        )
        baseline = (
            np.median(np.stack(history[-window:]), axis=0)
            if history
            else value
        )
        deviation = float(np.linalg.norm(value - baseline))
        if can_test and deviation > jump_threshold:
            output[index] = baseline
            events.append(
                {
                    "edge_frames": [
                        int(edge_frames[index]),
                        int(edge_frames[index] + 1),
                    ],
                    "raw_relative_velocity_mm": (value * 1000.0).tolist(),
                    "baseline_relative_velocity_mm": (
                        baseline * 1000.0
                    ).tolist(),
                    "deviation_mm": deviation * 1000.0,
                }
            )
        history.append(output[index])
    return output, events


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(hand_path, allow_pickle=False) as source:
        hand = {key: source[key] for key in source.files}
    vertices = np.asarray(hand["verts_cam"], dtype=np.float64)
    valid = np.asarray(
        hand.get("pred_valid", np.ones(len(vertices))), dtype=bool
    )
    frame_ids = np.asarray(
        hand.get("frame_ids", [f"{index:06d}" for index in range(len(vertices))]
        )
    ).astype(str)
    if len(frame_ids) != len(vertices):
        frame_ids = np.asarray([f"{index:06d}" for index in range(len(vertices))])

    poses = pose_rows(Path(args.object_pose_json).expanduser().resolve())
    object_centers = np.full((len(vertices), 3), np.nan, dtype=np.float64)
    object_rotations = np.full((len(vertices), 3, 3), np.nan, dtype=np.float64)
    for index, frame in enumerate(frame_ids):
        pose = poses.get(str(frame).zfill(6))
        if pose is not None:
            object_centers[index] = pose[:3, 3]
            object_rotations[index] = pose[:3, :3]
    valid &= np.isfinite(vertices).all(axis=(1, 2))
    valid &= np.isfinite(object_centers).all(axis=1)
    valid &= np.isfinite(object_rotations).all(axis=(1, 2))
    hand_centers = np.nanmean(vertices, axis=1)
    relative = np.einsum(
        "tji,tj->ti",
        object_rotations,
        hand_centers - object_centers,
    )

    segments = dynamic_segments(
        Path(args.segmentation_audit).expanduser().resolve()
    )
    correction = np.zeros((len(vertices), 3), dtype=np.float64)
    segment_audits = []
    max_translation = args.max_translation_mm / 1000.0

    for segment_index, (raw_start, raw_end) in enumerate(segments):
        start = max(0, raw_start)
        end = min(len(vertices) - 1, raw_end)
        indices = np.arange(start, end + 1)
        indices = indices[valid[indices]]
        if len(indices) < 3:
            segment_audits.append(
                {
                    "segment_index": segment_index,
                    "output_frames": [start, end],
                    "status": "too_short",
                }
            )
            continue
        if np.any(np.diff(indices) != 1):
            raise ValueError(
                f"Dynamic segment {start}-{end} contains invalid internal frames"
            )

        segment_relative = relative[indices]
        relative_velocity = np.diff(segment_relative, axis=0)
        causal_events: list[dict] = []
        if args.trajectory_mode == "causal_hold":
            target_velocity, causal_events = causal_hold_filter(
                relative_velocity,
                indices[:-1],
                args.median_window,
                args.causal_min_history,
                args.causal_jump_mm / 1000.0,
                args.causal_start_frame,
            )
        else:
            target_velocity = median_filter(
                relative_velocity, args.median_window
            )
        count = len(indices)
        segment_rotations = object_rotations[indices]

        def residual(flat: np.ndarray) -> np.ndarray:
            delta = flat.reshape(count, 3)
            delta_local = np.einsum(
                "tji,tj->ti", segment_rotations, delta
            )
            corrected_relative = segment_relative + delta_local
            corrected_velocity = np.diff(corrected_relative, axis=0)
            terms = [
                np.sqrt(args.w_anchor) * delta,
                np.sqrt(args.w_relative_velocity)
                * (corrected_velocity - target_velocity),
            ]
            if count >= 3:
                relative_acceleration = np.diff(corrected_relative, n=2, axis=0)
                target_acceleration = np.diff(target_velocity, axis=0)
                terms.append(
                    np.sqrt(args.w_relative_acceleration)
                    * (relative_acceleration - target_acceleration)
                )
                terms.append(
                    np.sqrt(args.w_correction_acceleration)
                    * np.diff(delta, n=2, axis=0)
                )
            return np.concatenate([term.reshape(-1) for term in terms])

        result = least_squares(
            residual,
            np.zeros(count * 3, dtype=np.float64),
            bounds=(-max_translation, max_translation),
            loss=args.loss,
            f_scale=args.f_scale_mm / 1000.0,
        )
        delta = result.x.reshape(count, 3)

        blend = max(0, args.boundary_blend_frames)
        if blend and start > 0:
            length = min(blend, count)
            weights = np.linspace(0.0, 1.0, length + 1)[1:]
            delta[:length] *= weights[:, None]
        if blend and end < len(vertices) - 1:
            length = min(blend, count)
            weights = np.linspace(1.0, 0.0, length + 1)[:-1]
            delta[-length:] *= weights[:, None]

        correction[indices] = delta
        corrected_relative = segment_relative + np.einsum(
            "tji,tj->ti", segment_rotations, delta
        )
        before_velocity = np.linalg.norm(np.diff(segment_relative, axis=0), axis=1)
        after_velocity = np.linalg.norm(np.diff(corrected_relative, axis=0), axis=1)
        before_acceleration = np.linalg.norm(
            np.diff(segment_relative, n=2, axis=0), axis=1
        )
        after_acceleration = np.linalg.norm(
            np.diff(corrected_relative, n=2, axis=0), axis=1
        )
        segment_audits.append(
            {
                "segment_index": segment_index,
                "output_frames": [start, end],
                "status": "ok",
                "solver": {
                    "success": bool(result.success),
                    "message": result.message,
                    "cost": float(result.cost),
                    "nfev": int(result.nfev),
                },
                "trajectory_mode": args.trajectory_mode,
                "causal_events": causal_events,
                "relative_velocity_mm": {
                    "before": distribution(before_velocity * 1000.0),
                    "after": distribution(after_velocity * 1000.0),
                },
                "relative_acceleration_mm": {
                    "before": distribution(before_acceleration * 1000.0),
                    "after": distribution(after_acceleration * 1000.0),
                },
                "translation_mm": distribution(
                    np.linalg.norm(delta, axis=1) * 1000.0
                ),
            }
        )

    corrected_vertices = vertices + correction[:, None, :]
    output = dict(hand)
    output["verts_cam"] = corrected_vertices.astype(np.float32)
    output["object_guided_translation_camera"] = correction.astype(np.float32)
    output["object_guided_predicted"] = (
        np.linalg.norm(correction, axis=1) > 1e-9
    )
    output["object_guided_source"] = np.asarray(str(hand_path))
    if "hand_center_cam" in output:
        output["hand_center_cam"] = (
            np.asarray(output["hand_center_cam"]) + correction
        ).astype(np.float32)
    result_path = out_dir / "hand_camera_result_object_guided.npz"
    np.savez_compressed(result_path, **output)

    correction_norm = np.linalg.norm(correction, axis=1) * 1000.0
    audit = {
        "hand_npz": str(hand_path),
        "object_pose_json": str(
            Path(args.object_pose_json).expanduser().resolve()
        ),
        "segmentation_audit": str(
            Path(args.segmentation_audit).expanduser().resolve()
        ),
        "result": str(result_path),
        "settings": vars(args),
        "num_frames": len(vertices),
        "num_dynamic_segments": len(segments),
        "translation_mm": distribution(correction_norm),
        "static_translation_max_mm": float(
            correction_norm[correction_norm <= 1e-9].max(initial=0.0)
        ),
        "segments": segment_audits,
        "per_frame": [
            {
                "frame": str(frame_ids[index]).zfill(6),
                "dynamic": bool(
                    any(start <= index <= end for start, end in segments)
                ),
                "translation_xyz_mm": (correction[index] * 1000.0).tolist(),
                "translation_mm": float(correction_norm[index]),
            }
            for index in range(len(vertices))
        ],
    }
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items() if key not in {"segments", "per_frame"}}, indent=2))
    for row in segment_audits:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
