#!/usr/bin/env python3
"""Audit side-dependent labels and input quality for the V11.2 pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train_v11_2_handflow_latent_pi3x_ray_residual import make_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    return parser.parse_args()


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def residual_error(rows: list[dict], correction: float) -> dict:
    target = np.asarray([row["target_ray"] for row in rows], dtype=np.float64)
    initial = np.asarray([row["initial"] for row in rows], dtype=np.float64)
    target_t = np.asarray([row["target"] for row in rows], dtype=np.float64)
    ray = initial / np.maximum(np.linalg.norm(initial, axis=-1, keepdims=True), 1e-8)
    corrected = initial + correction * ray
    before = np.linalg.norm(initial - target_t, axis=-1)
    after = np.linalg.norm(corrected - target_t, axis=-1)
    return {
        "correction_mm": float(correction * 1000.0),
        "ray_after_median_mm": float(np.median(np.abs(target - correction)) * 1000.0),
        "translation_after_median_mm": float(np.median(after) * 1000.0),
        "degraded_fraction": float(np.mean(after > before + 1e-6)),
    }


def main() -> None:
    args = parse_args()
    dataset_args = SimpleNamespace(
        neighborhood_size=args.neighborhood_size,
        min_confidence=args.min_confidence,
    )
    dataset = make_dataset(
        args.windows,
        args.global_root,
        args.dense_root,
        args.handflow_root,
        dataset_args,
    )

    frames: dict[tuple[int, int], dict] = {}
    for sample in dataset:
        stream_indices = sample["stream_index"].numpy()
        frame_indices = sample["frame_index"].numpy()
        for time_index in range(len(frame_indices)):
            if not bool(sample["valid"][time_index]):
                continue
            key = (
                int(stream_indices[time_index]),
                int(frame_indices[time_index]),
            )
            if key in frames:
                continue
            initial = sample["initial_t"][time_index].numpy().astype(np.float64)
            target = sample["target_t"][time_index].numpy().astype(np.float64)
            ray = initial / max(float(np.linalg.norm(initial)), 1e-8)
            target_ray = float(np.dot(target - initial, ray))
            token_valid = sample["neighborhood_valid"][time_index].numpy()
            metadata = sample["neighborhood_metadata"][time_index].numpy()
            confidence = metadata[..., -1][token_valid]
            latent = sample["handflow_translation_latent"][time_index].numpy()
            frames[key] = {
                "side": "left" if int(sample["side"][time_index]) == 0 else "right",
                "stream_index": key[0],
                "frame_index": key[1],
                "initial": initial,
                "target": target,
                "target_ray": target_ray,
                "initial_error": float(np.linalg.norm(target - initial)),
                "token_valid_fraction": float(np.mean(token_valid)),
                "joint_valid_fraction": float(np.mean(token_valid.any(axis=-1))),
                "confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
                "latent": latent.astype(np.float64),
                "latent_norm": float(np.linalg.norm(latent)),
            }

    rows = list(frames.values())
    if not rows:
        raise RuntimeError("No valid frames")
    global_constant = float(np.median([row["target_ray"] for row in rows]))
    output = {
        "windows": len(dataset),
        "unique_valid_frames": len(rows),
        "global_constant": residual_error(rows, global_constant),
        "sides": {},
    }
    for side in ("left", "right"):
        selected = [row for row in rows if row["side"] == side]
        target_ray = [row["target_ray"] * 1000.0 for row in selected]
        side_constant = float(np.median([row["target_ray"] for row in selected]))
        latent = np.stack([row["latent"] for row in selected])
        by_stream: dict[int, list[dict]] = defaultdict(list)
        for row in selected:
            by_stream[row["stream_index"]].append(row)
        temporal_delta = []
        for stream_rows in by_stream.values():
            stream_rows.sort(key=lambda row: row["frame_index"])
            if len(stream_rows) > 1:
                values = np.stack([row["latent"] for row in stream_rows])
                temporal_delta.extend(np.linalg.norm(np.diff(values, axis=0), axis=-1))
        output["sides"][side] = {
            "streams": len(by_stream),
            "frames": len(selected),
            "target_ray_mm": distribution(target_ray),
            "target_positive_fraction": float(np.mean(np.asarray(target_ray) > 0.0)),
            "initial_error_mm": distribution([
                row["initial_error"] * 1000.0 for row in selected
            ]),
            "token_valid_fraction": distribution([
                row["token_valid_fraction"] for row in selected
            ]),
            "joint_valid_fraction": distribution([
                row["joint_valid_fraction"] for row in selected
            ]),
            "valid_token_confidence": distribution([
                row["confidence"] for row in selected
            ]),
            "latent_norm": distribution([
                row["latent_norm"] for row in selected
            ]),
            "latent_dimension_mean_abs": float(np.mean(np.abs(latent.mean(axis=0)))),
            "latent_dimension_std_median": float(np.median(latent.std(axis=0))),
            "latent_temporal_delta": distribution(list(temporal_delta)),
            "global_constant_result": residual_error(selected, global_constant),
            "side_constant_result": residual_error(selected, side_constant),
        }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
