#!/usr/bin/env python3
"""Apply Stage-1 translation residuals and aggregate overlapping windows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_stage1_rigid_refiner import RigidTemporalRefiner, load_jsonl, load_supervision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def window_features(raw: dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
    hand = np.nan_to_num(
        np.asarray(raw["pred_hand_center"][start:end], dtype=np.float32)
    )
    obj = np.nan_to_num(
        np.asarray(raw["pred_object_center"][start:end], dtype=np.float32)
    )
    rot6d = np.nan_to_num(
        np.asarray(raw["pred_object_rot6d"][start:end], dtype=np.float32)
    )
    hand_valid = np.asarray(raw["pred_hand_valid"][start:end]).astype(bool)
    object_valid = np.asarray(raw["pred_object_valid"][start:end]).astype(bool)
    hand_velocity = np.zeros_like(hand)
    object_velocity = np.zeros_like(obj)
    hand_velocity[1:] = hand[1:] - hand[:-1]
    object_velocity[1:] = obj[1:] - obj[:-1]
    return np.concatenate(
        [
            hand,
            obj,
            hand - obj,
            hand_velocity,
            object_velocity,
            rot6d,
            hand_valid[:, None],
            object_valid[:, None],
        ],
        axis=1,
    ).astype(np.float32)


class InferenceWindows(Dataset):
    def __init__(self, path: Path):
        self.rows = load_jsonl(path)
        if not self.rows:
            raise RuntimeError(f"No windows in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        raw = load_supervision(row["supervision_npz"])
        return {
            "features": torch.from_numpy(window_features(raw, start, end)),
            "stream_id": row["stream_id"],
            "supervision_npz": row["supervision_npz"],
            "start": start,
            "end": end,
        }


def quantiles(values: list[np.ndarray]) -> dict:
    if not values:
        return {"count": 0, "median_mm": None, "p90_mm": None}
    array = np.concatenate(values).astype(np.float64)
    if not len(array):
        return {"count": 0, "median_mm": None, "p90_mm": None}
    return {
        "count": int(len(array)),
        "median_mm": float(np.quantile(array, 0.5) * 1000.0),
        "p90_mm": float(np.quantile(array, 0.9) * 1000.0),
        "max_mm": float(array.max() * 1000.0),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["args"]
    model = RigidTemporalRefiner(
        int(config["hidden_dim"]),
        int(config["layers"]),
        int(config["heads"]),
        float(config["dropout"]),
        str(config.get("mode", "joint")),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    max_residual = float(config["max_residual_mm"]) / 1000.0

    dataset = InferenceWindows(Path(args.windows).expanduser().resolve())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    sums: dict[str, dict[str, np.ndarray]] = {}
    paths = {}
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(args.device)
            hand_delta, object_delta = model(features, max_residual)
            hand_delta = hand_delta.cpu().numpy()
            object_delta = object_delta.cpu().numpy()
            for batch_index, stream_id in enumerate(batch["stream_id"]):
                path = str(batch["supervision_npz"][batch_index])
                raw = load_supervision(path)
                count = len(raw["frame_ids"])
                if stream_id not in sums:
                    sums[stream_id] = {
                        "hand": np.zeros((count, 3), dtype=np.float64),
                        "object": np.zeros((count, 3), dtype=np.float64),
                        "count": np.zeros(count, dtype=np.int64),
                    }
                    paths[stream_id] = path
                start = int(batch["start"][batch_index])
                end = int(batch["end"][batch_index])
                sums[stream_id]["hand"][start:end] += hand_delta[batch_index]
                sums[stream_id]["object"][start:end] += object_delta[batch_index]
                sums[stream_id]["count"][start:end] += 1

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    aggregate = defaultdict(list)
    stream_rows = []
    for stream_id, values in sorted(sums.items()):
        out_path = out_root / stream_id / "stage1_rigid_prediction.npz"
        if out_path.is_file() and not args.overwrite:
            continue
        raw = load_supervision(paths[stream_id])
        prediction_count = values["count"]
        predicted = prediction_count > 0
        denominator = np.maximum(prediction_count, 1)[:, None]
        hand_delta = (values["hand"] / denominator).astype(np.float32)
        object_delta = (values["object"] / denominator).astype(np.float32)
        corrected_hand = np.asarray(raw["pred_hand_center"], dtype=np.float32) + hand_delta
        corrected_object = (
            np.asarray(raw["pred_object_center"], dtype=np.float32) + object_delta
        )
        hand_mask = predicted & np.asarray(raw["hand_supervision_valid"]).astype(bool)
        object_mask = predicted & np.asarray(raw["object_supervision_valid"]).astype(bool)
        relative_mask = (
            predicted & np.asarray(raw["relative_supervision_valid"]).astype(bool)
        )
        gt_hand = np.asarray(raw["gt_hand_center"], dtype=np.float32)
        gt_object = np.asarray(raw["gt_object_center"], dtype=np.float32)
        pred_hand = np.asarray(raw["pred_hand_center"], dtype=np.float32)
        pred_object = np.asarray(raw["pred_object_center"], dtype=np.float32)
        metrics = {}
        definitions = (
            ("hand", pred_hand, corrected_hand, gt_hand, hand_mask),
            ("object", pred_object, corrected_object, gt_object, object_mask),
            (
                "relative",
                pred_hand - pred_object,
                corrected_hand - corrected_object,
                gt_hand - gt_object,
                relative_mask,
            ),
        )
        for name, initial, corrected, target, mask in definitions:
            initial_error = np.linalg.norm(initial[mask] - target[mask], axis=-1)
            corrected_error = np.linalg.norm(corrected[mask] - target[mask], axis=-1)
            metrics[f"initial_{name}"] = quantiles([initial_error])
            metrics[f"corrected_{name}"] = quantiles([corrected_error])
            aggregate[f"initial_{name}"].append(initial_error)
            aggregate[f"corrected_{name}"].append(corrected_error)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            frame_ids=np.asarray(raw["frame_ids"]),
            hand_delta=hand_delta,
            object_delta=object_delta,
            corrected_hand_center=corrected_hand,
            corrected_object_center=corrected_object,
            prediction_count=prediction_count,
            predicted=predicted,
            stream_id=np.asarray(stream_id),
            checkpoint=np.asarray(str(checkpoint_path)),
        )
        stream_rows.append(
            {
                "stream_id": stream_id,
                "prediction": str(out_path),
                "num_predicted": int(predicted.sum()),
                "metrics": metrics,
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_total": float(checkpoint["val_total"]),
        "windows": str(Path(args.windows).expanduser().resolve()),
        "num_windows": len(dataset),
        "num_streams": len(stream_rows),
        "aggregate_metrics": {
            key: quantiles(value) for key, value in sorted(aggregate.items())
        },
        "streams": stream_rows,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "streams"}, indent=2))
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
