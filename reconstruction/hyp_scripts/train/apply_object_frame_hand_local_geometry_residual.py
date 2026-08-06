#!/usr/bin/env python3
"""Apply the hand-local Pi3X SE(3) residual model to overlapping windows."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import ObjectFrameWindowDataset
from train_object_frame_hand_pose_local_geometry_residual import (
    HandLocalGeometryResidualModel,
    localize_token_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--pi3x-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def blend_weights(length: int) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype=np.float64)
    return np.maximum(np.hanning(length + 2)[1:-1], 0.05).astype(np.float64)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = Namespace(**checkpoint["args"])
    windows = Path(args.windows).expanduser().resolve()
    pi3x_root = Path(args.pi3x_root).expanduser().resolve()
    rows = load_jsonl(windows)
    object_names = list(checkpoint.get("object_names", []))
    object_to_index = {name: index for index, name in enumerate(object_names)}
    dataset_args = Namespace(
        translation_noise_mm=0.0,
        rotation_noise_deg=0.0,
        initial_pose_dropout=0.0,
    )
    dataset = ObjectFrameWindowDataset(
        windows,
        dataset_args,
        augment=False,
        object_to_index=object_to_index,
        pi3x_root=pi3x_root,
    )
    sample = dataset[0]
    model = HandLocalGeometryResidualModel(
        int(checkpoint["local_hand_dim"]),
        int(checkpoint["pi3x_feature_dim"]),
        int(checkpoint["pi3x_metadata_dim"]),
        int(config.pi3x_relation_dim),
        int(config.pi3x_heads),
        int(config.hidden_dim),
        int(config.layers),
        float(config.dropout),
        float(config.max_normalized_residual),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    accumulated: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="apply local SE3 residual", dynamic_ncols=True):
            initial_r = batch["initial_rotation"].to(device)
            hand_metadata, key_metadata = localize_token_metadata(
                batch["hand_token_metadata"].to(device),
                batch["hand_token_valid"].to(device),
                batch["key_token_metadata"].to(device),
                initial_r,
            )
            delta_local, delta_r = model(
                batch["local_hand_features"].to(device),
                batch["hand_token_features"].to(device),
                hand_metadata,
                batch["hand_token_valid"].to(device),
                batch["key_token_features"].to(device),
                key_metadata,
                batch["key_token_valid"].to(device),
                batch["key_token_types"].to(device),
            )
            initial_t = batch["initial_translation"].to(device)
            delta_object = torch.einsum(
                "btc,btdc->btd", delta_local, initial_r
            )
            predicted_t = initial_t + delta_object
            predicted_r = initial_r @ delta_r
            values = {
                "predicted_t": predicted_t.cpu().numpy(),
                "predicted_r": predicted_r.cpu().numpy(),
                "delta_local": delta_local.cpu().numpy(),
                "delta_r": delta_r.cpu().numpy(),
                "initial_t": batch["initial_translation"].numpy(),
                "initial_r": batch["initial_rotation"].numpy(),
                "target_t": batch["target_translation"].numpy(),
                "target_r": batch["target_rotation"].numpy(),
                "scale": batch["object_scale"].numpy(),
                "valid_t": batch["valid_translation"].numpy(),
                "valid_r": batch["valid_rotation"].numpy(),
            }
            for index, row_index in enumerate(batch["dataset_index"].numpy()):
                row = rows[int(row_index)]
                stream_id = str(row["stream_id"])
                start, end = int(row["start"]), int(row["end"])
                length = end - start
                if stream_id not in accumulated:
                    frame_count = len(
                        np.load(row["supervision_npz"], allow_pickle=False)["frame_ids"]
                    )
                    accumulated[stream_id] = {
                        "weight": np.zeros(frame_count, dtype=np.float64),
                        **{
                            name: np.zeros(
                                (frame_count,) + value[index].shape[1:],
                                dtype=np.float64,
                            )
                            for name, value in values.items()
                            if name not in ("valid_t", "valid_r")
                        },
                        "valid_t": np.zeros(frame_count, dtype=bool),
                        "valid_r": np.zeros(frame_count, dtype=bool),
                    }
                target = accumulated[stream_id]
                weight = blend_weights(length)
                target["weight"][start:end] += weight
                for name, value in values.items():
                    if name in ("valid_t", "valid_r"):
                        target[name][start:end] |= value[index]
                    else:
                        target[name][start:end] += value[index] * weight.reshape(
                            (length,) + (1,) * (value[index].ndim - 1)
                        )

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stream_rows = []
    all_initial_t, all_predicted_t = [], []
    all_initial_r, all_predicted_r = [], []
    for stream_id, data in sorted(accumulated.items()):
        stream_out = out_root / stream_id
        result_path = stream_out / "hand_object_pose_local_residual.npz"
        if result_path.is_file() and not args.overwrite:
            continue
        weight = np.maximum(data.pop("weight"), 1e-8)
        for name, value in data.items():
            if value.ndim >= 1 and value.shape[0] == len(weight):
                if name not in ("valid_t", "valid_r"):
                    data[name] = value / weight.reshape(
                        (len(weight),) + (1,) * (value.ndim - 1)
                    )
        valid_t = data["valid_t"]
        valid_r = data["valid_r"]
        initial_t = data["initial_t"] * data["scale"][:, None]
        predicted_t = data["predicted_t"] * data["scale"][:, None]
        target_t = data["target_t"] * data["scale"][:, None]
        t_mask = valid_t & np.isfinite(target_t).all(axis=-1)
        r_mask = valid_r
        if t_mask.any():
            all_initial_t.append(np.linalg.norm(initial_t[t_mask] - target_t[t_mask], axis=-1))
            all_predicted_t.append(np.linalg.norm(predicted_t[t_mask] - target_t[t_mask], axis=-1))
        if r_mask.any():
            def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                rel = np.einsum("tji,tjk->tik", a, b)
                cosine = (np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0
                return np.arccos(np.clip(cosine, -1.0, 1.0))
            all_initial_r.append(angle(data["initial_r"][r_mask], data["target_r"][r_mask]))
            all_predicted_r.append(angle(data["predicted_r"][r_mask], data["target_r"][r_mask]))
        stream_out.mkdir(parents=True, exist_ok=True)
        output = dict(data)
        output["predicted_translation_object"] = predicted_t.astype(np.float32)
        output["initial_translation_object"] = initial_t.astype(np.float32)
        output["target_translation_object"] = target_t.astype(np.float32)
        output["predicted_rotation_object"] = data["predicted_r"].astype(np.float32)
        output["initial_rotation_object"] = data["initial_r"].astype(np.float32)
        output["target_rotation_object"] = data["target_r"].astype(np.float32)
        output["checkpoint"] = np.asarray(str(checkpoint_path))
        output["model_version"] = np.asarray(checkpoint["model_version"])
        np.savez_compressed(result_path, **output)
        stream_rows.append({"stream_id": stream_id, "result": str(result_path)})

    def dist(groups: list[np.ndarray], scale: float) -> dict:
        if not groups:
            return {"count": 0}
        value = np.concatenate(groups) * scale
        return {
            "count": int(value.size),
            "median_mm": float(np.median(value)),
            "p90_mm": float(np.quantile(value, 0.9)),
            "max_mm": float(np.max(value)),
        }

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_version": checkpoint["model_version"],
        "windows": str(windows),
        "num_windows": len(rows),
        "num_streams": len(stream_rows),
        "aggregate_metrics": {
            "initial_translation": dist(all_initial_t, 1000.0),
            "predicted_translation": dist(all_predicted_t, 1000.0),
            "initial_rotation": dist(all_initial_r, 180.0 / np.pi),
            "predicted_rotation": dist(all_predicted_r, 180.0 / np.pi),
        },
        "streams": stream_rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "streams"}, indent=2))


if __name__ == "__main__":
    main()
