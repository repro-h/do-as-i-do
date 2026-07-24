#!/usr/bin/env python3
"""Audit compact Stage-1 rigid-refinement supervision before training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True, metavar="NAME=ROOT")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--large-residual-mm", type=float, default=50.0)
    return parser.parse_args()


def parse_split(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--split must use NAME=ROOT, got: {value}")
    name, root = value.split("=", 1)
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return name, path


def summary(values: list[np.ndarray], scale: float = 1.0) -> dict:
    if not values:
        return {"count": 0, "median": None, "p90": None, "p99": None, "max": None}
    array = np.concatenate(values).astype(np.float64) * scale
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        (
            intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2],
            intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2],
        ),
        axis=1,
    )


def append_norm(target: dict[str, list[np.ndarray]], key: str, values: np.ndarray):
    if len(values):
        target[key].append(np.linalg.norm(values, axis=-1))


def audit_root(root: Path, large_residual_m: float) -> dict:
    metrics: dict[str, list[np.ndarray]] = defaultdict(list)
    by_side: dict[str, dict[str, list[np.ndarray]]] = {
        "left": defaultdict(list),
        "right": defaultdict(list),
    }
    failures = []
    large_streams = []
    files = sorted(root.glob("*.npz"))

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as raw:
                hand = np.asarray(raw["pred_hand_center"], dtype=np.float32)
                obj = np.asarray(raw["pred_object_center"], dtype=np.float32)
                gt_hand = np.asarray(raw["gt_hand_center"], dtype=np.float32)
                gt_obj = np.asarray(raw["gt_object_center"], dtype=np.float32)
                hand_mask = np.asarray(raw["hand_supervision_valid"]).astype(bool)
                object_mask = np.asarray(raw["object_supervision_valid"]).astype(bool)
                relative_mask = np.asarray(raw["relative_supervision_valid"]).astype(bool)
                intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32)
                side = str(np.asarray(raw["hand_side"]).item())
                stream_id = str(np.asarray(raw["stream_id"]).item())

            hand_delta = gt_hand - hand
            object_delta = gt_obj - obj
            relative_delta = hand_delta - object_delta
            append_norm(metrics, "hand_residual_m", hand_delta[hand_mask])
            append_norm(metrics, "object_residual_m", object_delta[object_mask])
            append_norm(metrics, "relative_residual_m", relative_delta[relative_mask])
            append_norm(by_side[side], "hand_residual_m", hand_delta[hand_mask])
            append_norm(
                by_side[side], "relative_residual_m", relative_delta[relative_mask]
            )

            if hand_mask.any():
                hand_projection = np.linalg.norm(
                    project(hand[hand_mask], intrinsics)
                    - project(gt_hand[hand_mask], intrinsics),
                    axis=1,
                )
                metrics["hand_projection_px"].append(hand_projection)
                by_side[side]["hand_projection_px"].append(hand_projection)
            if object_mask.any():
                metrics["object_projection_px"].append(
                    np.linalg.norm(
                        project(obj[object_mask], intrinsics)
                        - project(gt_obj[object_mask], intrinsics),
                        axis=1,
                    )
                )

            relative_norm = np.linalg.norm(relative_delta[relative_mask], axis=-1)
            if len(relative_norm) and float(relative_norm.max()) >= large_residual_m:
                large_streams.append(
                    {
                        "stream_id": stream_id,
                        "max_relative_residual_mm": float(relative_norm.max() * 1000.0),
                        "large_frame_count": int(
                            (relative_norm >= large_residual_m).sum()
                        ),
                    }
                )
        except Exception as error:
            failures.append(
                {"path": str(path), "error": f"{type(error).__name__}: {error}"}
            )

    output = {
        "root": str(root),
        "num_files": len(files),
        "num_failed": len(failures),
        "metrics": {
            key.replace("_m", "_mm"): summary(values, 1000.0)
            if key.endswith("_m")
            else summary(values)
            for key, values in sorted(metrics.items())
        },
        "metrics_by_side": {
            side: {
                key.replace("_m", "_mm"): summary(values, 1000.0)
                if key.endswith("_m")
                else summary(values)
                for key, values in sorted(side_metrics.items())
            }
            for side, side_metrics in by_side.items()
        },
        "num_large_relative_streams": len(large_streams),
        "largest_relative_streams": sorted(
            large_streams,
            key=lambda row: row["max_relative_residual_mm"],
            reverse=True,
        )[:50],
        "failures": failures,
    }
    relative = output["metrics"].get("relative_residual_mm", {})
    output["large_relative_frame_fraction"] = (
        sum(row["large_frame_count"] for row in large_streams)
        / max(int(relative.get("count") or 0), 1)
    )
    return output


def main() -> None:
    args = parse_args()
    splits = [parse_split(value) for value in args.split]
    payload = {
        "large_residual_mm": args.large_residual_mm,
        "splits": {
            name: audit_root(root, args.large_residual_mm / 1000.0)
            for name, root in splits
        },
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
