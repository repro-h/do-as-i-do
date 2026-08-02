#!/usr/bin/env python3
"""Audit object-transported hand translation targets across streams."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--worst-k", type=int, default=20)
    return parser.parse_args()


def distribution(values: list[np.ndarray], suffix: str = "mm") -> dict:
    arrays = [np.asarray(value).reshape(-1) for value in values if np.asarray(value).size]
    if not arrays:
        return {"count": 0}
    array = np.concatenate(arrays).astype(np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        f"median_{suffix}": float(np.median(array)),
        f"p90_{suffix}": float(np.quantile(array, 0.9)),
        f"p99_{suffix}": float(np.quantile(array, 0.99)),
        f"max_{suffix}": float(np.max(array)),
    }


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        [
            intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2],
            intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2],
        ],
        axis=-1,
    )


def scalar(archive, key: str, default: str = "unknown") -> str:
    if key not in archive.files:
        return default
    value = np.asarray(archive[key]).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def summarize(rows: list[dict]) -> dict:
    return {
        "num_streams": len(rows),
        "num_valid_frames": int(sum(row["num_valid"] for row in rows)),
        "object_center_error": distribution(
            [row["object_center_error_mm"] for row in rows]
        ),
        "required_hand_correction": distribution(
            [row["required_hand_correction_mm"] for row in rows]
        ),
        "target_gt_projection_shift": distribution(
            [row["target_gt_projection_shift_px"] for row in rows], "px"
        ),
        "target_correction_step": distribution(
            [row["target_correction_step_mm"] for row in rows]
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.supervision_root).expanduser().resolve()
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No supervision NPZ files in {root}")

    stream_rows = []
    failures = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as raw:
                pred_hand = np.asarray(raw["pred_hand_center"], dtype=np.float64)
                pred_object = np.asarray(raw["pred_object_center"], dtype=np.float64)
                gt_hand = np.asarray(raw["gt_hand_center"], dtype=np.float64)
                gt_object = np.asarray(raw["gt_object_center"], dtype=np.float64)
                valid = np.asarray(raw["relative_supervision_valid"], dtype=bool)
                intrinsics = np.asarray(raw["intrinsics"], dtype=np.float64)
                side = scalar(raw, "hand_side")
                object_name = scalar(raw, "object_name")
                stream_id = scalar(raw, "stream_id", path.stem)
                pose_source = scalar(raw, "object_pose_source", "not_recorded")

            count = min(
                len(pred_hand), len(pred_object), len(gt_hand), len(gt_object), len(valid)
            )
            pred_hand = pred_hand[:count]
            pred_object = pred_object[:count]
            gt_hand = gt_hand[:count]
            gt_object = gt_object[:count]
            valid = valid[:count]
            finite = (
                np.isfinite(pred_hand).all(axis=1)
                & np.isfinite(pred_object).all(axis=1)
                & np.isfinite(gt_hand).all(axis=1)
                & np.isfinite(gt_object).all(axis=1)
            )
            valid &= finite
            if not valid.any():
                raise ValueError("No valid relative supervision frames")

            # Move the GT hand by the current-vs-GT object center displacement.
            object_shift = pred_object - gt_object
            target_hand = gt_hand + object_shift
            correction = target_hand - pred_hand
            object_error = np.linalg.norm(object_shift[valid], axis=-1) * 1000.0
            correction_error = np.linalg.norm(correction[valid], axis=-1) * 1000.0
            projection_shift = np.linalg.norm(
                project(target_hand[valid], intrinsics)
                - project(gt_hand[valid], intrinsics),
                axis=-1,
            )
            pair_valid = valid[1:] & valid[:-1]
            correction_step = (
                np.linalg.norm(correction[1:] - correction[:-1], axis=-1)[pair_valid]
                * 1000.0
            )
            stream_rows.append(
                {
                    "stream_id": stream_id,
                    "hand_side": side,
                    "object_name": object_name,
                    "supervision_npz": str(path),
                    "object_pose_source": pose_source,
                    "num_valid": int(valid.sum()),
                    "object_center_error_mm": object_error,
                    "required_hand_correction_mm": correction_error,
                    "target_gt_projection_shift_px": projection_shift,
                    "target_correction_step_mm": correction_step,
                    "object_center_error_median_mm": float(np.median(object_error)),
                    "required_hand_correction_median_mm": float(
                        np.median(correction_error)
                    ),
                    "projection_shift_median_px": float(np.median(projection_shift)),
                }
            )
        except Exception as error:
            failures.append(
                {"path": str(path), "error": f"{type(error).__name__}: {error}"}
            )

    groups = defaultdict(list)
    for row in stream_rows:
        groups[f"side:{row['hand_side']}"].append(row)
        groups[f"object:{row['object_name']}"].append(row)

    serializable_rows = []
    for row in stream_rows:
        serializable_rows.append(
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, np.ndarray)
            }
        )
    worst = sorted(
        serializable_rows,
        key=lambda row: (
            row["projection_shift_median_px"],
            row["object_center_error_median_mm"],
        ),
        reverse=True,
    )[: args.worst_k]
    payload = {
        "supervision_root": str(root),
        "num_files": len(paths),
        "num_completed": len(stream_rows),
        "num_failed": len(failures),
        "aggregate": summarize(stream_rows),
        "by_group": {
            name: summarize(rows) for name, rows in sorted(groups.items())
        },
        "worst_streams": worst,
        "failures": failures,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    compact = {
        key: value
        for key, value in payload.items()
        if key not in {"by_group", "worst_streams", "failures"}
    }
    print(json.dumps(compact, indent=2))
    print("\nWorst streams:")
    for row in worst:
        print(
            row["stream_id"],
            f"object={row['object_name']}",
            f"side={row['hand_side']}",
            f"object_mm={row['object_center_error_median_mm']:.2f}",
            f"correction_mm={row['required_hand_correction_median_mm']:.2f}",
            f"projection_px={row['projection_shift_median_px']:.2f}",
        )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
