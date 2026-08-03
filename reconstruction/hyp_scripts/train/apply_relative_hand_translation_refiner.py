#!/usr/bin/env python3
"""Apply the compact relative hand translation refiner to overlapping windows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_relative_hand_translation_refiner import (
    RelativeTranslationRefiner,
    RelativeWindowDataset,
    load_jsonl,
    load_npz,
    project,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--relative-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(
        self,
        windows: Path,
        global_root: Path,
        config: dict,
        relative_root: Optional[Path] = None,
    ):
        self.rows = load_jsonl(windows)
        self.relative_root = relative_root
        self.base = RelativeWindowDataset(
            windows,
            global_root,
            SimpleNamespace(**config),
            relative_root,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        sample = self.base[index]
        supervision_path = (
            self.relative_root / f"{row['stream_id']}.npz"
            if self.relative_root is not None
            else Path(row["supervision_npz"])
        ).resolve()
        return {
            "features": sample["features"],
            "stream_id": row["stream_id"],
            "supervision_npz": str(supervision_path),
            "start": int(row["start"]),
            "end": int(row["end"]),
        }


def blend_weights(length: int) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype=np.float64)
    return np.maximum(np.hanning(length + 2)[1:-1], 0.05).astype(np.float64)


def distribution(values: list[np.ndarray]) -> dict:
    arrays = [value for value in values if len(value)]
    if not arrays:
        return {"count": 0}
    value = np.concatenate(arrays).astype(np.float64) * 1000.0
    return {
        "count": int(len(value)),
        "median_mm": float(np.median(value)),
        "p90_mm": float(np.quantile(value, 0.9)),
        "max_mm": float(np.max(value)),
    }


def quality_mask(raw: dict[str, np.ndarray], config: dict) -> np.ndarray:
    pred_hand = np.asarray(raw["pred_hand_center"], dtype=np.float64)
    pred_object = np.asarray(raw["pred_object_center"], dtype=np.float64)
    gt_hand = np.asarray(raw["gt_hand_center"], dtype=np.float64)
    gt_object = np.asarray(raw["gt_object_center"], dtype=np.float64)
    intrinsics = np.asarray(raw["intrinsics"], dtype=np.float64)
    valid = np.asarray(raw["relative_supervision_valid"], dtype=bool).copy()
    object_delta = gt_object - pred_object
    relative_delta = (gt_hand - gt_object) - (pred_hand - pred_object)
    target_hand = gt_hand - object_delta
    projection_shift = np.linalg.norm(
        project(target_hand, intrinsics) - project(gt_hand, intrinsics), axis=-1
    )
    valid &= (
        np.linalg.norm(relative_delta, axis=-1)
        <= float(config["max_correction_mm"]) / 1000.0
    )
    valid &= (
        np.linalg.norm(object_delta, axis=-1)
        <= float(config["max_object_center_error_mm"]) / 1000.0
    )
    valid &= projection_shift <= float(config["max_target_projection_shift_px"])
    return valid


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["args"]
    model = RelativeTranslationRefiner(
        int(checkpoint["input_dim"]),
        int(config["hidden_dim"]),
        int(config["layers"]),
        float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()

    windows = Path(args.windows).expanduser().resolve()
    dataset = InferenceDataset(
        windows,
        Path(args.global_root).expanduser().resolve(),
        config,
        (
            Path(args.relative_root).expanduser().resolve()
            if args.relative_root
            else None
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    sums: dict[str, dict[str, np.ndarray]] = {}
    supervision_paths = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="apply relative translation", dynamic_ncols=True):
            correction = model(
                batch["features"].to(args.device),
                float(config["max_correction_mm"]) / 1000.0,
            ).cpu().numpy()
            for index, stream_id in enumerate(batch["stream_id"]):
                supervision_path = str(batch["supervision_npz"][index])
                raw = load_npz(supervision_path)
                count = len(raw["frame_ids"])
                if stream_id not in sums:
                    sums[stream_id] = {
                        "correction": np.zeros((count, 3), dtype=np.float64),
                        "weight": np.zeros(count, dtype=np.float64),
                    }
                    supervision_paths[stream_id] = supervision_path
                start = int(batch["start"][index])
                end = int(batch["end"][index])
                weight = blend_weights(end - start)
                sums[stream_id]["correction"][start:end] += (
                    correction[index] * weight[:, None]
                )
                sums[stream_id]["weight"][start:end] += weight

    out_root = Path(args.out_root).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    aggregate = defaultdict(list)
    aggregate_counts = defaultdict(int)
    stream_rows = []
    for stream_id, values in sorted(sums.items()):
        stream_out = out_root / stream_id
        result_path = stream_out / "handflow_camera_result_relative_refined.npz"
        if result_path.is_file() and not args.overwrite:
            continue
        raw = load_npz(supervision_paths[stream_id])
        weight = values["weight"]
        correction = (
            values["correction"] / np.maximum(weight[:, None], 1e-8)
        ).astype(np.float32)
        observed = (
            np.asarray(raw["pred_hand_valid"], dtype=bool)
            & np.asarray(raw["pred_object_valid"], dtype=bool)
            & (weight > 0.0)
        )
        correction[~observed] = 0.0
        pred_hand = np.asarray(raw["pred_hand_center"], dtype=np.float32)
        pred_object = np.asarray(raw["pred_object_center"], dtype=np.float32)
        gt_hand = np.asarray(raw["gt_hand_center"], dtype=np.float32)
        gt_object = np.asarray(raw["gt_object_center"], dtype=np.float32)
        corrected_hand = pred_hand + correction
        initial_relative = pred_hand - pred_object
        corrected_relative = corrected_hand - pred_object
        target_relative = gt_hand - gt_object
        evaluate = quality_mask(raw, config) & observed
        initial_error = np.linalg.norm(
            initial_relative[evaluate] - target_relative[evaluate], axis=-1
        )
        corrected_error = np.linalg.norm(
            corrected_relative[evaluate] - target_relative[evaluate], axis=-1
        )
        aggregate["initial_relative"].append(initial_error)
        aggregate["corrected_relative"].append(corrected_error)
        correction_magnitude = np.linalg.norm(correction[evaluate], axis=-1)
        aggregate["correction_magnitude"].append(correction_magnitude)
        improved = corrected_error < initial_error
        aggregate_counts["evaluated"] += int(len(initial_error))
        aggregate_counts["improved"] += int(improved.sum())
        aggregate_counts["degraded"] += int((corrected_error > initial_error).sum())

        handflow_path = handflow_root / stream_id / "handflow_camera_result.npz"
        with np.load(handflow_path, allow_pickle=False) as archive:
            handflow = {key: np.asarray(archive[key]) for key in archive.files}
        vertices = np.asarray(handflow["verts_cam"], dtype=np.float32)
        count = min(len(vertices), len(correction))
        output = dict(handflow)
        output["verts_cam"] = vertices[:count] + correction[:count, None]
        output["hand_center_cam"] = corrected_hand[:count]
        output["relative_translation_camera"] = correction[:count]
        output["relative_translation_predicted"] = observed[:count]
        output["relative_translation_checkpoint"] = np.asarray(str(checkpoint_path))
        output["relative_translation_model_version"] = np.asarray(
            checkpoint["model_version"]
        )
        stream_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result_path, **output)
        metrics = {
            "initial_relative": distribution([initial_error]),
            "corrected_relative": distribution([corrected_error]),
            "correction_magnitude": distribution([correction_magnitude]),
            "num_frames": count,
            "num_predicted": int(observed[:count].sum()),
            "num_evaluated": int(evaluate[:count].sum()),
            "num_improved": int(improved.sum()),
            "num_degraded": int((corrected_error > initial_error).sum()),
            "degraded_fraction": float(
                (corrected_error > initial_error).mean()
            ) if len(corrected_error) else None,
        }
        (stream_out / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        stream_rows.append(
            {"stream_id": stream_id, "result": str(result_path), "metrics": metrics}
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_version": checkpoint["model_version"],
        "windows": str(windows),
        "num_windows": len(dataset),
        "num_streams": len(stream_rows),
        "aggregate_metrics": {
            key: distribution(value) for key, value in sorted(aggregate.items())
        },
        "aggregate_counts": {
            **aggregate_counts,
            "degraded_fraction": (
                aggregate_counts["degraded"] / aggregate_counts["evaluated"]
                if aggregate_counts["evaluated"]
                else None
            ),
        },
        "streams": stream_rows,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "streams"}, indent=2))


if __name__ == "__main__":
    main()
