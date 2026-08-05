#!/usr/bin/env python3
"""Audit de-overlapped object-frame absolute hand-pose predictions."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_object_frame_hand_pose_baseline import (
    AbsoluteObjectFramePoseModel,
    ObjectFrameWindowDataset,
)
from train_object_frame_hand_pose_selector import PoseCandidateSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--pi3x-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selector-checkpoint")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def rotation_average(values: list[np.ndarray]) -> np.ndarray:
    matrix = np.mean(np.stack(values), axis=0)
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def rotation_error_deg(prediction: np.ndarray, target: np.ndarray) -> float:
    relative = prediction.T @ target
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def summarize(entries: list[dict]) -> dict:
    initial_t, predicted_t = [], []
    initial_r, predicted_r = [], []
    oracle_t, oracle_r = [], []
    selected_t, selected_r = [], []
    translation_improved = rotation_improved = rotation_count = 0
    for entry in entries:
        if entry["valid_translation"]:
            old = float(np.linalg.norm(entry["initial_t"] - entry["target_t"]))
            new = float(np.linalg.norm(entry["predicted_t"] - entry["target_t"]))
            initial_t.append(old * 1000.0)
            predicted_t.append(new * 1000.0)
            oracle_t.append(min(old, new) * 1000.0)
            if "selected_t" in entry:
                selected = float(
                    np.linalg.norm(entry["selected_t"] - entry["target_t"])
                )
                selected_t.append(selected * 1000.0)
            translation_improved += int(new < old)
        if entry["valid_rotation"]:
            old = rotation_error_deg(entry["initial_r"], entry["target_r"])
            new = rotation_error_deg(entry["predicted_r"], entry["target_r"])
            initial_r.append(old)
            predicted_r.append(new)
            oracle_r.append(min(old, new))
            if "selected_r" in entry:
                selected_r.append(
                    rotation_error_deg(entry["selected_r"], entry["target_r"])
                )
            rotation_improved += int(new < old)
            rotation_count += 1
    output = {
        "frames": len(entries),
        "initial_translation_mm": distribution(initial_t),
        "predicted_translation_mm": distribution(predicted_t),
        "oracle_translation_mm": distribution(oracle_t),
        "translation_improved_fraction": (
            translation_improved / len(initial_t) if initial_t else None
        ),
        "initial_rotation_deg": distribution(initial_r),
        "predicted_rotation_deg": distribution(predicted_r),
        "oracle_rotation_deg": distribution(oracle_r),
        "rotation_improved_fraction": (
            rotation_improved / rotation_count if rotation_count else None
        ),
    }
    if selected_t:
        output["selected_translation_mm"] = distribution(selected_t)
    if selected_r:
        output["selected_rotation_deg"] = distribution(selected_r)
    return output


def main() -> None:
    args = parse_args()
    windows = Path(args.windows).expanduser().resolve()
    pi3x_root = Path(args.pi3x_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = Namespace(**checkpoint["args"])
    config.translation_noise_mm = 0.0
    config.rotation_noise_deg = 0.0
    config.initial_pose_dropout = 0.0
    object_names = list(checkpoint["object_names"])
    object_to_index = {name: index for index, name in enumerate(object_names)}
    dataset = ObjectFrameWindowDataset(
        windows,
        config,
        augment=False,
        object_to_index=object_to_index,
        pi3x_root=pi3x_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = AbsoluteObjectFramePoseModel(
        checkpoint["input_dim"],
        config.hidden_dim,
        config.layers,
        config.dropout,
        config.max_normalized_translation,
        len(object_names),
        checkpoint.get("object_embedding_dim", 0),
        checkpoint.get("pi3x_feature_dim", 0),
        checkpoint.get("pi3x_metadata_dim", 0),
        checkpoint.get("pi3x_relation_dim", 128),
        checkpoint.get("pi3x_heads", 8),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()
    selector = None
    selector_checkpoint_path = None
    if args.selector_checkpoint:
        selector_checkpoint_path = (
            Path(args.selector_checkpoint).expanduser().resolve()
        )
        selector_checkpoint = torch.load(
            selector_checkpoint_path, map_location="cpu"
        )
        selector_config = Namespace(**selector_checkpoint["args"])
        selector = PoseCandidateSelector(
            config.hidden_dim,
            selector_config.hidden_dim,
            len(object_names),
            selector_config.object_embedding_dim,
            selector_config.side_embedding_dim,
            selector_config.dropout,
        )
        selector.load_state_dict(selector_checkpoint["selector_state"])
        selector.to(device).eval()

    accumulated: dict[tuple[str, int], dict] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="audit"):
            indices = batch["dataset_index"].numpy()
            device_batch = {key: value.to(device) for key, value in batch.items()}
            model_output = model(
                device_batch["features"],
                device_batch["object_index"],
                device_batch.get("hand_token_features"),
                device_batch.get("hand_token_metadata"),
                device_batch.get("hand_token_valid"),
                device_batch.get("key_token_features"),
                device_batch.get("key_token_metadata"),
                device_batch.get("key_token_valid"),
                device_batch.get("key_token_types"),
                return_context=selector is not None,
            )
            if selector is None:
                predicted_t, predicted_r = model_output
                selected_t = selected_r = None
            else:
                predicted_t, predicted_r, context = model_output
                translation_logits, rotation_logits = selector(
                    context,
                    device_batch["initial_translation"],
                    predicted_t,
                    device_batch["initial_rotation"],
                    predicted_r,
                    device_batch["object_index"],
                    device_batch["hand_side_index"],
                )
                translation_choose = translation_logits >= 0
                rotation_choose = rotation_logits >= 0
                selected_t = torch.where(
                    translation_choose[..., None],
                    predicted_t,
                    device_batch["initial_translation"],
                )
                selected_r = torch.where(
                    rotation_choose[..., None, None],
                    predicted_r,
                    device_batch["initial_rotation"],
                )
            predicted_t = (
                predicted_t * device_batch["object_scale"][..., None]
            ).cpu().numpy()
            predicted_r = predicted_r.cpu().numpy()
            if selected_t is not None:
                selected_t = (
                    selected_t * device_batch["object_scale"][..., None]
                ).cpu().numpy()
                selected_r = selected_r.cpu().numpy()
            initial_t = (
                device_batch["initial_translation"]
                * device_batch["object_scale"][..., None]
            ).cpu().numpy()
            target_t = (
                device_batch["target_translation"]
                * device_batch["object_scale"][..., None]
            ).cpu().numpy()
            initial_r = device_batch["initial_rotation"].cpu().numpy()
            target_r = device_batch["target_rotation"].cpu().numpy()
            valid_t = device_batch["valid_translation"].cpu().numpy()
            valid_r = device_batch["valid_rotation"].cpu().numpy()

            for batch_index, dataset_index in enumerate(indices):
                row = dataset.rows[int(dataset_index)]
                stream_id = str(row["stream_id"])
                start = int(row["start"])
                for offset in range(predicted_t.shape[1]):
                    key = (stream_id, start + offset)
                    entry = accumulated.setdefault(
                        key,
                        {
                            "object_name": str(row["object_name"]),
                            "hand_side": str(row["hand_side"]),
                            "predicted_t_values": [],
                            "predicted_r_values": [],
                            "selected_t_values": [],
                            "selected_r_values": [],
                            "initial_t": initial_t[batch_index, offset],
                            "target_t": target_t[batch_index, offset],
                            "initial_r": initial_r[batch_index, offset],
                            "target_r": target_r[batch_index, offset],
                            "valid_translation": bool(valid_t[batch_index, offset]),
                            "valid_rotation": bool(valid_r[batch_index, offset]),
                        },
                    )
                    entry["predicted_t_values"].append(
                        predicted_t[batch_index, offset]
                    )
                    entry["predicted_r_values"].append(
                        predicted_r[batch_index, offset]
                    )
                    if selected_t is not None:
                        entry["selected_t_values"].append(
                            selected_t[batch_index, offset]
                        )
                        entry["selected_r_values"].append(
                            selected_r[batch_index, offset]
                        )

    entries = []
    for entry in accumulated.values():
        entry["predicted_t"] = np.mean(
            np.stack(entry.pop("predicted_t_values")), axis=0
        )
        entry["predicted_r"] = rotation_average(
            entry.pop("predicted_r_values")
        )
        selected_t_values = entry.pop("selected_t_values")
        selected_r_values = entry.pop("selected_r_values")
        if selected_t_values:
            entry["selected_t"] = np.mean(
                np.stack(selected_t_values), axis=0
            )
            entry["selected_r"] = rotation_average(selected_r_values)
        entries.append(entry)

    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        groups["all"].append(entry)
        groups[f"object:{entry['object_name']}"].append(entry)
        groups[f"side:{entry['hand_side']}"].append(entry)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selector_checkpoint": (
            str(selector_checkpoint_path)
            if selector_checkpoint_path is not None
            else None
        ),
        "windows": str(windows),
        "num_unique_frames": len(entries),
        "groups": {
            name: summarize(values) for name, values in sorted(groups.items())
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
