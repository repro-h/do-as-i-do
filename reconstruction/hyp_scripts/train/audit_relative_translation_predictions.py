#!/usr/bin/env python3
"""Audit relative-translation predictions by hand side and target magnitude."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from apply_relative_hand_translation_refiner import quality_mask


BINS = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, np.inf))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--relative-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def decode(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def stats(chunks: list[np.ndarray]) -> dict:
    arrays = [np.asarray(chunk, dtype=np.float64) for chunk in chunks if len(chunk)]
    if not arrays:
        return {"count": 0}
    values = np.concatenate(arrays) * 1000.0
    return {
        "count": int(len(values)),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.quantile(values, 0.9)),
        "max_mm": float(values.max()),
    }


def summarize_group(group: dict[str, list[np.ndarray]]) -> dict:
    initial = np.concatenate(group["initial"]) if group["initial"] else np.empty(0)
    corrected = (
        np.concatenate(group["corrected"]) if group["corrected"] else np.empty(0)
    )
    improved = corrected < initial
    return {
        "initial": stats([initial]),
        "corrected": stats([corrected]),
        "improved": int(improved.sum()),
        "degraded": int((corrected > initial).sum()),
        "degraded_fraction": (
            float((corrected > initial).mean()) if len(corrected) else None
        ),
    }


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    relative_root = Path(args.relative_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["args"]

    groups = defaultdict(lambda: defaultdict(list))
    stream_rows = []
    for result_path in sorted(
        prediction_root.glob("*/handflow_camera_result_relative_refined.npz")
    ):
        stream_id = result_path.parent.name
        supervision_path = relative_root / f"{stream_id}.npz"
        if not supervision_path.is_file():
            continue
        with np.load(supervision_path, allow_pickle=False) as archive:
            raw = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(result_path, allow_pickle=False) as archive:
            correction = np.asarray(
                archive["relative_translation_camera"], dtype=np.float64
            )
            observed = np.asarray(
                archive["relative_translation_predicted"], dtype=bool
            )

        count = min(len(correction), len(raw["frame_ids"]))
        target = (
            raw["gt_hand_center"][:count]
            - raw["gt_object_center"][:count]
            - raw["pred_hand_center"][:count]
            + raw["pred_object_center"][:count]
        ).astype(np.float64)
        valid = quality_mask(raw, config)[:count] & observed[:count]
        initial = np.linalg.norm(target, axis=-1)
        corrected = np.linalg.norm(target - correction[:count], axis=-1)
        magnitude_mm = initial * 1000.0
        side = decode(raw["hand_side"])

        for label in ("all", side):
            groups[label]["initial"].append(initial[valid])
            groups[label]["corrected"].append(corrected[valid])
        for lower, upper in BINS:
            bin_mask = valid & (magnitude_mm >= lower) & (magnitude_mm < upper)
            bin_name = f"{int(lower)}_{'inf' if not np.isfinite(upper) else int(upper)}mm"
            for label in (f"all/{bin_name}", f"{side}/{bin_name}"):
                groups[label]["initial"].append(initial[bin_mask])
                groups[label]["corrected"].append(corrected[bin_mask])

        stream_initial = initial[valid]
        stream_corrected = corrected[valid]
        stream_rows.append(
            {
                "stream_id": stream_id,
                "hand_side": side,
                "count": int(valid.sum()),
                "initial_median_mm": (
                    float(np.median(stream_initial) * 1000.0)
                    if len(stream_initial)
                    else None
                ),
                "corrected_median_mm": (
                    float(np.median(stream_corrected) * 1000.0)
                    if len(stream_corrected)
                    else None
                ),
                "degraded_fraction": (
                    float((stream_corrected > stream_initial).mean())
                    if len(stream_corrected)
                    else None
                ),
            }
        )

    summary = {
        "prediction_root": str(prediction_root),
        "relative_root": str(relative_root),
        "checkpoint": str(checkpoint_path),
        "groups": {
            name: summarize_group(group) for name, group in sorted(groups.items())
        },
        "streams": stream_rows,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"groups": summary["groups"]}, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
