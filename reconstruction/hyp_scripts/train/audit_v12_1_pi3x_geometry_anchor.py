#!/usr/bin/env python3
"""Audit whether cached Pi3X points directly observe GT hand ray depth."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_v10_pi3x_hand_neighborhood_depth import HandNeighborhoodDataset
from train_v9_2_pi3x_feature_trajectory_depth import distribution


CANDIDATES = ("center", "median", "confidence_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--val-windows", required=True)
    parser.add_argument("--global-train-root", required=True)
    parser.add_argument("--global-val-root", required=True)
    parser.add_argument("--dense-train-root", required=True)
    parser.add_argument("--dense-val-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    return parser.parse_args()


def make_dataset(
    windows: str,
    global_root: str,
    dense_root: str,
    args: argparse.Namespace,
) -> HandNeighborhoodDataset:
    return HandNeighborhoodDataset(
        Path(windows),
        Path(global_root),
        Path(dense_root),
        args.neighborhood_size,
        args.min_confidence,
    )


def collect(
    dataset: HandNeighborhoodDataset,
    args: argparse.Namespace,
    split: str,
) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    records: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for batch in tqdm(loader, desc=split):
        pred = batch["pred_joints"].numpy()
        target = batch["target_joints"].numpy()
        points = batch["neighborhood_points"].numpy()
        metadata = batch["neighborhood_metadata"].numpy()
        neighbors_valid = batch["neighborhood_valid"].numpy()
        frame_valid = batch["valid"].numpy()
        stream = batch["stream_index"].numpy()
        frame = batch["frame_index"].numpy()
        side = batch["side"].numpy()
        observed = batch["observed"].numpy()
        metric_valid = batch.get("metric_scalar_valid")
        if metric_valid is None:
            raise KeyError(
                "metric_scalar_valid is missing; the manifest still points "
                "to a cache without exported metric features"
            )
        metric_valid = metric_valid.numpy()

        for b, t in zip(*np.nonzero(frame_valid & metric_valid)):
            wrist = pred[b, t, 0]
            wrist_norm = float(np.linalg.norm(wrist))
            if not np.isfinite(wrist_norm) or wrist_norm <= 1e-6:
                continue
            ray = wrist / wrist_norm
            target_depth = float(np.dot(target[b, t, 0], ray))
            if not np.isfinite(target_depth) or target_depth <= 1e-6:
                continue

            point_depth = np.linalg.norm(points[b, t, 0], axis=-1)
            valid = neighbors_valid[b, t, 0] & np.isfinite(point_depth)
            offsets = metadata[b, t, 0, :, :2]
            center_index = int(np.argmin(np.linalg.norm(offsets, axis=-1)))
            values = {name: math.nan for name in CANDIDATES}
            if valid.any():
                values["median"] = float(np.median(point_depth[valid]))
                confidence = np.clip(
                    metadata[b, t, 0, :, -1].astype(np.float64), 0.0, None
                )
                weights = confidence * valid
                if weights.sum() > 1e-8:
                    values["confidence_mean"] = float(
                        np.sum(point_depth * weights) / weights.sum()
                    )
            if valid[center_index]:
                values["center"] = float(point_depth[center_index])

            records[(int(stream[b, t]), int(frame[b, t]))].append({
                "side": int(side[b, t]),
                "observed": bool(observed[b, t]),
                "initial": wrist_norm,
                "target": target_depth,
                **values,
            })

    unique = []
    for rows in records.values():
        row = rows[0].copy()
        for name in ("initial", "target", *CANDIDATES):
            values = np.asarray([value[name] for value in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[name] = float(np.mean(finite)) if len(finite) else math.nan
        unique.append(row)
    return unique


def fit_affine(rows: list[dict], candidate: str) -> tuple[float, float]:
    x = np.asarray([row[candidate] for row in rows], dtype=np.float64)
    y = np.asarray([row["target"] for row in rows], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        raise RuntimeError(f"Not enough valid {candidate} anchors")
    design = np.stack((x[valid], np.ones(valid.sum())), axis=-1)
    slope, bias = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return float(slope), float(bias)


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None
    x, y = x[valid], y[valid]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(
    rows: list[dict],
    candidate: str,
    slope: float,
    bias: float,
) -> dict:
    initial = np.asarray([row["initial"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    anchor = np.asarray([row[candidate] for row in rows], dtype=np.float64)
    valid = np.isfinite(initial) & np.isfinite(target) & np.isfinite(anchor)
    initial, target, anchor = initial[valid], target[valid], anchor[valid]
    predicted = slope * anchor + bias
    before = np.abs(initial - target)
    after = np.abs(predicted - target)
    oracle = np.minimum(before, after)
    return {
        "count": int(valid.sum()),
        "availability_fraction": float(valid.mean()) if len(valid) else 0.0,
        "slope": slope,
        "bias_mm": bias * 1000.0,
        "initial_ray_depth": distribution([before]),
        "anchor_ray_depth": distribution([after]),
        "oracle_select_ray_depth": distribution([oracle]),
        "degraded_fraction": float(np.mean(after > before + 1e-6)),
        "improved_fraction": float(np.mean(after < before - 1e-6)),
        "residual_correlation": pearson(
            predicted - initial,
            target - initial,
        ),
    }


def evaluate_groups(
    rows: list[dict],
    candidate: str,
    slope: float,
    bias: float,
) -> dict:
    groups = {
        "all": rows,
        "left": [row for row in rows if row["side"] == 0],
        "right": [row for row in rows if row["side"] == 1],
        "observed": [row for row in rows if row["observed"]],
        "unobserved": [row for row in rows if not row["observed"]],
    }
    return {
        name: evaluate(group, candidate, slope, bias)
        for name, group in groups.items() if group
    }


def main() -> None:
    args = parse_args()
    train_data = make_dataset(
        args.train_windows,
        args.global_train_root,
        args.dense_train_root,
        args,
    )
    val_data = make_dataset(
        args.val_windows,
        args.global_val_root,
        args.dense_val_root,
        args,
    )
    train_rows = collect(train_data, args, "train")
    val_rows = collect(val_data, args, "val")
    report = {
        "description": (
            "Pi3X local_points are already multiplied by outputs['metric']; "
            "this audit tests their radial depth without a learned feature model."
        ),
        "train_unique_frames": len(train_rows),
        "val_unique_frames": len(val_rows),
        "candidates": {},
    }
    for candidate in CANDIDATES:
        slope, bias = fit_affine(train_rows, candidate)
        report["candidates"][candidate] = {
            "raw": evaluate_groups(val_rows, candidate, 1.0, 0.0),
            "train_affine": evaluate_groups(
                val_rows, candidate, slope, bias
            ),
        }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
