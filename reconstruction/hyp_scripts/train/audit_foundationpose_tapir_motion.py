#!/usr/bin/env python3
"""Compare FoundationPose frame motion with TAPIR depth-at-t PnP motion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--tapir-npz", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-plot")
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--boundary-change-mm", type=float, default=4.0)
    parser.add_argument("--lag-mismatch-mm", type=float, default=5.0)
    parser.add_argument("--min-pnp-inliers", type=int, default=12)
    return parser.parse_args()


def normalize_frame(value: object) -> str:
    value = str(value)
    if value.startswith("color_"):
        value = value.split("_")[-1]
    return value.zfill(6)


def pose_rows(payload: dict) -> dict:
    for key in ("by_frame", "frames"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    raise KeyError("FoundationPose JSON has no by_frame/frames dictionary")


def resolve_pose(rows: dict, frame_id: str) -> np.ndarray | None:
    row = rows.get(frame_id)
    if row is None:
        row = rows.get(str(int(frame_id)))
    if row is None or row.get("object_in_camera") is None:
        return None
    value = np.asarray(row["object_in_camera"], dtype=np.float64)
    if value.size != 16 or not np.isfinite(value).all():
        return None
    return value.reshape(4, 4)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def transform_metrics(transform: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(transform[:3, 3]) * 1000.0),
        rotation_angle_deg(transform[:3, :3]),
    )


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def median_filter(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index in range(len(values)):
        begin = max(0, index - 1)
        end = min(len(values), index + 2)
        window = values[begin:end]
        finite = window[np.isfinite(window)]
        if len(finite):
            result[index] = np.median(finite)
    return result


def lag_scores(
    reference: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    max_lag: int,
) -> list[dict]:
    result = []
    count = len(reference)
    for lag in range(-max_lag, max_lag + 1):
        indices = np.arange(count)
        candidate_indices = indices + lag
        selected = (
            valid
            & (candidate_indices >= 0)
            & (candidate_indices < count)
        )
        left = reference[selected]
        right = candidate[candidate_indices[selected]]
        if len(left) < 3:
            continue
        correlation = (
            float(np.corrcoef(left, right)[0, 1])
            if np.std(left) > 1e-8 and np.std(right) > 1e-8
            else None
        )
        result.append(
            {
                "lag_frames": lag,
                "meaning": (
                    "FoundationPose delayed"
                    if lag > 0
                    else "FoundationPose leads"
                    if lag < 0
                    else "aligned"
                ),
                "count": int(len(left)),
                "mae_mm": float(np.mean(np.abs(left - right))),
                "correlation": correlation,
            }
        )
    return result


def save_plot(
    path: Path,
    tapir_speed: np.ndarray,
    foundationpose_speed: np.ndarray,
    candidates: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 4))
    indices = np.arange(len(tapir_speed))
    axis.plot(indices, tapir_speed, label="TAPIR PnP", color="#00a67d")
    axis.plot(
        indices,
        foundationpose_speed,
        label="FoundationPose",
        color="#d1495b",
    )
    for candidate in candidates:
        color = "#1976d2" if candidate["type"] == "start_lag" else "#ef6c00"
        axis.axvline(candidate["pair_index"], color=color, alpha=0.35)
    axis.set_xlabel("frame pair t -> t+1")
    axis.set_ylabel("translation speed (mm/frame)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    fp_payload = json.loads(
        Path(args.foundationpose_json).expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    frame_map = json.loads(
        Path(args.frame_map_json).expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    rows = pose_rows(fp_payload)
    mapped_frames = [
        normalize_frame(row["original_frame"]) for row in frame_map["frames"]
    ]

    with np.load(
        Path(args.tapir_npz).expanduser().resolve(), allow_pickle=True
    ) as payload:
        pairs = np.asarray(payload["frame_pairs"], dtype=np.int64)
        tapir_transforms = np.asarray(
            payload["relative_transform_pnp"], dtype=np.float64
        )
        pnp_status = np.asarray(payload["pnp_status"]).astype(str)
        pnp_inliers = np.asarray(payload["pnp_inlier_mask"]).sum(axis=1)

    pair_count = len(pairs)
    tapir_speed = np.full(pair_count, np.nan, dtype=np.float64)
    tapir_rotation = np.full(pair_count, np.nan, dtype=np.float64)
    fp_speed = np.full(pair_count, np.nan, dtype=np.float64)
    fp_rotation = np.full(pair_count, np.nan, dtype=np.float64)
    translation_error = np.full(pair_count, np.nan, dtype=np.float64)
    rotation_error = np.full(pair_count, np.nan, dtype=np.float64)
    direction_cosine = np.full(pair_count, np.nan, dtype=np.float64)
    valid = np.zeros(pair_count, dtype=bool)
    records = []

    for pair_index, (first_index, second_index) in enumerate(pairs):
        first_frame = mapped_frames[int(first_index)]
        second_frame = mapped_frames[int(second_index)]
        first_pose = resolve_pose(rows, first_frame)
        second_pose = resolve_pose(rows, second_frame)
        tapir = tapir_transforms[pair_index]
        _, tapir_rotation[pair_index] = transform_metrics(tapir)
        if first_pose is not None and second_pose is not None:
            foundationpose = second_pose @ np.linalg.inv(first_pose)
            _, fp_rotation[pair_index] = transform_metrics(foundationpose)
            first_center = first_pose[:3, 3]
            second_center = second_pose[:3, 3]
            tapir_center = (
                tapir[:3, :3] @ first_center + tapir[:3, 3]
            )
            tapir_vector = tapir_center - first_center
            fp_vector = second_center - first_center
            tapir_speed[pair_index] = (
                np.linalg.norm(tapir_vector) * 1000.0
            )
            fp_speed[pair_index] = np.linalg.norm(fp_vector) * 1000.0
            translation_error[pair_index] = (
                np.linalg.norm(tapir_center - second_center) * 1000.0
            )
            rotation_residual = (
                tapir[:3, :3] @ foundationpose[:3, :3].T
            )
            rotation_error[pair_index] = rotation_angle_deg(
                rotation_residual
            )
            denominator = np.linalg.norm(tapir_vector) * np.linalg.norm(fp_vector)
            if denominator > 1e-10:
                direction_cosine[pair_index] = float(
                    np.dot(tapir_vector, fp_vector) / denominator
                )
        valid[pair_index] = (
            pnp_status[pair_index] == "ok"
            and pnp_inliers[pair_index] >= args.min_pnp_inliers
            and np.isfinite(fp_speed[pair_index])
        )
        records.append(
            {
                "pair_index": pair_index,
                "output_pair": [int(first_index), int(second_index)],
                "original_pair": [first_frame, second_frame],
                "valid": bool(valid[pair_index]),
                "pnp_status": pnp_status[pair_index],
                "pnp_inliers": int(pnp_inliers[pair_index]),
                "tapir_center_motion_mm": (
                    float(tapir_speed[pair_index])
                    if np.isfinite(tapir_speed[pair_index])
                    else None
                ),
                "foundationpose_center_motion_mm": (
                    float(fp_speed[pair_index])
                    if np.isfinite(fp_speed[pair_index])
                    else None
                ),
                "center_prediction_error_mm": (
                    float(translation_error[pair_index])
                    if np.isfinite(translation_error[pair_index])
                    else None
                ),
                "tapir_rotation_deg": float(tapir_rotation[pair_index]),
                "foundationpose_rotation_deg": (
                    float(fp_rotation[pair_index])
                    if np.isfinite(fp_rotation[pair_index])
                    else None
                ),
                "rotation_error_deg": (
                    float(rotation_error[pair_index])
                    if np.isfinite(rotation_error[pair_index])
                    else None
                ),
                "translation_direction_cosine": (
                    float(direction_cosine[pair_index])
                    if np.isfinite(direction_cosine[pair_index])
                    else None
                ),
            }
        )

    tapir_smoothed = median_filter(tapir_speed)
    fp_smoothed = median_filter(fp_speed)
    tapir_change = np.diff(tapir_smoothed, prepend=tapir_smoothed[0])
    candidates = []
    for index in range(1, pair_count):
        if not valid[index] or not valid[index - 1]:
            continue
        mismatch = tapir_smoothed[index] - fp_smoothed[index]
        if (
            tapir_change[index] >= args.boundary_change_mm
            and mismatch >= args.lag_mismatch_mm
        ):
            candidate_type = "start_lag"
        elif (
            tapir_change[index] <= -args.boundary_change_mm
            and mismatch <= -args.lag_mismatch_mm
        ):
            candidate_type = "stop_lag"
        else:
            continue
        candidates.append(
            {
                "type": candidate_type,
                "pair_index": index,
                "output_pair": records[index]["output_pair"],
                "original_pair": records[index]["original_pair"],
                "tapir_speed_mm": float(tapir_smoothed[index]),
                "foundationpose_speed_mm": float(fp_smoothed[index]),
                "tapir_speed_change_mm": float(tapir_change[index]),
                "speed_mismatch_mm": float(mismatch),
                "center_prediction_error_mm": records[index][
                    "center_prediction_error_mm"
                ],
                "rotation_error_deg": records[index]["rotation_error_deg"],
            }
        )

    scores = lag_scores(tapir_smoothed, fp_smoothed, valid, args.max_lag)
    best_lag = min(scores, key=lambda row: row["mae_mm"]) if scores else None
    summary = {
        "settings": vars(args),
        "translation_comparison": (
            "TAPIR PnP transform applied to the FoundationPose object center "
            "at t, compared with the FoundationPose object center at t+1"
        ),
        "num_pairs": pair_count,
        "num_valid_pairs": int(valid.sum()),
        "tapir_center_motion_mm": distribution(tapir_speed[valid]),
        "foundationpose_center_motion_mm": distribution(fp_speed[valid]),
        "center_prediction_error_mm": distribution(translation_error[valid]),
        "rotation_error_deg": distribution(rotation_error[valid]),
        "translation_direction_cosine": distribution(direction_cosine[valid]),
        "lag_scores": scores,
        "best_lag": best_lag,
        "boundary_candidates": candidates,
        "pairs": records,
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.out_plot:
        save_plot(
            Path(args.out_plot).expanduser().resolve(),
            tapir_smoothed,
            fp_smoothed,
            candidates,
        )
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
