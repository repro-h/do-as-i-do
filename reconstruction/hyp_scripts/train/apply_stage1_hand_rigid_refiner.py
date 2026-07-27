#!/usr/bin/env python3
"""Apply the hand-only temporal rigid refiner with overlap blending."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_stage1_hand_rigid_refiner import (
    HandRigidTemporalRefiner,
    HandRigidWindowDataset,
    axis_angle_to_matrix,
    load_jsonl,
    load_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, windows: Path, max_target_center_m: float):
        self.rows = load_jsonl(windows)
        self.base = HandRigidWindowDataset(windows, max_target_center_m)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = self.base[index]
        return {
            "features": sample["features"],
            "stream_id": row["stream_id"],
            "supervision_npz": row["supervision_npz"],
            "start": int(row["start"]),
            "end": int(row["end"]),
        }


def blend_weights(length: int) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype=np.float64)
    values = np.hanning(length + 2)[1:-1]
    return np.maximum(values, 0.05).astype(np.float64)


def quantiles(values: list[np.ndarray]) -> dict:
    if not values:
        return {"count": 0, "median_mm": None, "p90_mm": None}
    array = np.concatenate(values).astype(np.float64)
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
    model = HandRigidTemporalRefiner(
        int(checkpoint["input_dim"]),
        int(config["hidden_dim"]),
        int(config["layers"]),
        int(config["heads"]),
        float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    max_translation = float(config["max_translation_mm"]) / 1000.0
    max_rotation = np.deg2rad(float(config["max_rotation_deg"]))
    windows_path = Path(args.windows).expanduser().resolve()
    dataset = InferenceDataset(
        windows_path,
        float(config["max_target_center_mm"]) / 1000.0,
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
        for batch in tqdm(loader, desc="apply hand rigid", dynamic_ncols=True):
            features = batch["features"].to(args.device)
            translation, rotation = model(
                features, max_translation, max_rotation
            )
            translation = translation.cpu().numpy()
            rotation = rotation.cpu().numpy()
            for batch_index, stream_id in enumerate(batch["stream_id"]):
                path = str(batch["supervision_npz"][batch_index])
                raw = load_npz(path)
                frame_count = len(raw["frame_ids"])
                if stream_id not in sums:
                    sums[stream_id] = {
                        "translation": np.zeros((frame_count, 3), dtype=np.float64),
                        "rotation": np.zeros((frame_count, 3), dtype=np.float64),
                        "weight": np.zeros(frame_count, dtype=np.float64),
                    }
                    supervision_paths[stream_id] = path
                start = int(batch["start"][batch_index])
                end = int(batch["end"][batch_index])
                weights = blend_weights(end - start)
                sums[stream_id]["translation"][start:end] += (
                    translation[batch_index] * weights[:, None]
                )
                sums[stream_id]["rotation"][start:end] += (
                    rotation[batch_index] * weights[:, None]
                )
                sums[stream_id]["weight"][start:end] += weights

    handflow_root = Path(args.handflow_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    aggregate = defaultdict(list)
    stream_rows = []
    for stream_id, values in sorted(sums.items()):
        stream_out = out_root / stream_id
        result_path = stream_out / "handflow_camera_result_stage1_hand_rigid.npz"
        if result_path.is_file() and not args.overwrite:
            continue
        supervision = load_npz(supervision_paths[stream_id])
        weights = values["weight"]
        predicted = weights > 0.0
        denominator = np.maximum(weights, 1e-8)[:, None]
        translation = (values["translation"] / denominator).astype(np.float32)
        rotation_vector = (values["rotation"] / denominator).astype(np.float32)
        rotation_matrix = axis_angle_to_matrix(
            torch.from_numpy(rotation_vector)
        ).numpy().astype(np.float32)

        handflow_path = (
            handflow_root / stream_id / "handflow_camera_result.npz"
        )
        with np.load(handflow_path, allow_pickle=False) as raw:
            handflow = {key: np.asarray(raw[key]) for key in raw.files}
        vertices = np.asarray(handflow["verts_cam"], dtype=np.float32)
        count = min(len(vertices), len(translation))
        vertices = vertices[:count]
        center = np.asarray(
            supervision["pred_hand_center"][:count], dtype=np.float32
        )
        local = vertices - center[:, None]
        corrected = np.einsum(
            "tij,tvj->tvi", rotation_matrix[:count], local
        )
        corrected += center[:, None] + translation[:count, None]
        corrected_center = center + translation[:count]
        valid = (
            np.asarray(supervision["supervision_valid"][:count]).astype(bool)
            & predicted[:count]
        )
        supervision_valid = np.asarray(
            supervision["supervision_valid"][:count]
        ).astype(bool)
        uncovered_valid = supervision_valid & ~predicted[:count]
        gt_center = np.asarray(
            supervision["gt_hand_center"][:count], dtype=np.float32
        )
        initial_error = np.linalg.norm(center[valid] - gt_center[valid], axis=-1)
        corrected_error = np.linalg.norm(
            corrected_center[valid] - gt_center[valid], axis=-1
        )
        aggregate["initial_center"].append(initial_error)
        aggregate["corrected_center"].append(corrected_error)
        metrics = {
            "initial_center": quantiles([initial_error]),
            "corrected_center": quantiles([corrected_error]),
            "num_frames": count,
            "num_predicted": int(predicted[:count].sum()),
            "num_evaluated": int(valid.sum()),
            "num_uncovered_valid": int(uncovered_valid.sum()),
            "uncovered_valid_frames": np.flatnonzero(
                uncovered_valid
            ).astype(int).tolist(),
        }
        stream_out.mkdir(parents=True, exist_ok=True)
        output = dict(handflow)
        output["verts_cam"] = corrected
        output["hand_center_cam"] = corrected_center
        output["stage1_translation"] = translation[:count]
        output["stage1_rotation_vector"] = rotation_vector[:count]
        output["stage1_rotation_matrix"] = rotation_matrix[:count]
        output["stage1_predicted"] = predicted[:count]
        output["stage1_checkpoint"] = np.asarray(str(checkpoint_path))
        output["stage1_source_handflow"] = np.asarray(str(handflow_path))
        np.savez_compressed(result_path, **output)
        metrics_path = stream_out / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        stream_rows.append(
            {
                "stream_id": stream_id,
                "result": str(result_path),
                "metrics": metrics,
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_total": float(checkpoint["val_total"]),
        "windows": str(windows_path),
        "num_windows": len(dataset),
        "num_streams": len(stream_rows),
        "aggregate_metrics": {
            key: quantiles(value) for key, value in sorted(aggregate.items())
        },
        "streams": stream_rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "streams"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
