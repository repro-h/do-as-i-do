#!/usr/bin/env python3
"""Build balanced mixed-dataset smoke manifests with explicit cache paths."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-streams-per-dataset", type=int, default=2)
    parser.add_argument("--val-streams-per-dataset", type=int, default=1)
    parser.add_argument("--windows-per-stream", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path):
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evenly_spaced(rows, count):
    rows = sorted(rows, key=lambda row: (int(row["start"]), int(row["end"])))
    if count <= 0 or len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def attach_paths(row, dataset, visibility_root, track_root):
    row = dict(row)
    stream = str(row["stream_id"])
    row["dataset"] = dataset
    row["visibility_npz"] = str(
        (visibility_root / stream / "visibility_cache.npz").resolve()
    )
    row["tracks_npz"] = str((track_root / stream / "tracks.npz").resolve())
    return row


def supervised_instances(rows, visibility_path, tracks_path):
    requested_frames = {
        int(frame) for row in rows for frame in row["frame_indices"]
    }
    with np.load(str(visibility_path), allow_pickle=False) as visibility:
        visibility_frames = np.asarray(visibility["frame_indices"], dtype=np.int64)
        visibility_valid = np.asarray(visibility["visibility_valid"], dtype=bool)
    with np.load(str(tracks_path), allow_pickle=False) as tracks:
        track_frames = np.asarray(tracks["frame_indices"], dtype=np.int64)
        observation_valid = np.asarray(tracks["observation_valid"], dtype=bool)
        target_valid = np.asarray(tracks["target_valid"], dtype=bool)
    if visibility_valid.ndim == 1:
        visibility_valid = visibility_valid[:, None]
    if observation_valid.ndim == 1:
        observation_valid = observation_valid[:, None]
    if target_valid.ndim == 1:
        target_valid = target_valid[:, None]
    visibility_index = {
        int(frame): index for index, frame in enumerate(visibility_frames)
    }
    track_index = {int(frame): index for index, frame in enumerate(track_frames)}
    count = 0
    for frame in requested_frames:
        if frame not in visibility_index or frame not in track_index:
            continue
        visible = visibility_valid[visibility_index[frame]]
        observed = observation_valid[track_index[frame]]
        target = target_valid[track_index[frame]]
        hands = min(len(visible), len(observed), len(target))
        count += int((visible[:hands] & observed[:hands] & target[:hands]).sum())
    return count


def select_split(entry, split, stream_limit, windows_per_stream, rng):
    manifest_value = entry.get(f"{split}_windows")
    if not manifest_value:
        return [], {"available": False, "selected_streams": [], "windows": 0}
    visibility_root = Path(entry[f"visibility_{split}_root"]).expanduser().resolve()
    track_root = Path(entry[f"track_{split}_root"]).expanduser().resolve()
    grouped = defaultdict(list)
    for row in load_jsonl(manifest_value):
        grouped[str(row["stream_id"])].append(row)

    complete = []
    rejected = {}
    for stream, rows in sorted(grouped.items()):
        visibility = visibility_root / stream / "visibility_cache.npz"
        tracks = track_root / stream / "tracks.npz"
        missing = []
        if not visibility.is_file():
            missing.append("visibility")
        if not tracks.is_file():
            missing.append("tracks")
        if missing:
            rejected[stream] = missing
        else:
            complete.append(stream)

    rng.shuffle(complete)
    selected_rows = []
    validated_streams = []
    supervised_by_stream = {}
    for stream in complete:
        if stream_limit > 0 and len(validated_streams) >= stream_limit:
            break
        selected = evenly_spaced(grouped[stream], windows_per_stream)
        missing_labels = sorted({
            str(Path(path).expanduser())
            for row in selected for path in row.get("label_paths", [])
            if not Path(path).expanduser().is_file()
        })
        if missing_labels:
            rejected[stream] = [f"selected_labels:{len(missing_labels)}"]
            continue
        visibility = visibility_root / stream / "visibility_cache.npz"
        tracks = track_root / stream / "tracks.npz"
        supervised = supervised_instances(selected, visibility, tracks)
        if supervised == 0:
            rejected[stream] = ["no_detector_supervised_instances"]
            continue
        validated_streams.append(stream)
        supervised_by_stream[stream] = supervised
        for row in selected:
            selected_rows.append(attach_paths(
                row, entry["name"], visibility_root, track_root
            ))
    report = {
        "available": True,
        "manifest": str(Path(manifest_value).expanduser().resolve()),
        "candidate_streams": len(grouped),
        "complete_streams": len(complete),
        "selected_streams": validated_streams,
        "supervised_instances_by_stream": supervised_by_stream,
        "windows": len(selected_rows),
        "rejected_streams": rejected,
    }
    return selected_rows, report


def write_jsonl(path, rows, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = config.get("datasets", config)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Config must contain a non-empty datasets list")
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"Dataset names must be unique: {names}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    outputs = {"train": [], "val": []}
    report = {"config": str(config_path), "datasets": {}}
    for entry in entries:
        dataset_report = {}
        for split, limit in (
            ("train", args.train_streams_per_dataset),
            ("val", args.val_streams_per_dataset),
        ):
            rows, split_report = select_split(
                entry, split, limit, args.windows_per_stream, rng
            )
            outputs[split].extend(rows)
            dataset_report[split] = split_report
        report["datasets"][entry["name"]] = dataset_report

    rng.shuffle(outputs["train"])
    outputs["val"].sort(
        key=lambda row: (row["dataset"], row["stream_id"], row["start"])
    )
    for split in ("train", "val"):
        path = out_dir / f"{split}_windows.jsonl"
        if outputs[split]:
            write_jsonl(path, outputs[split], args.overwrite)
        report[f"{split}_windows"] = len(outputs[split])
        report[f"{split}_output"] = str(path) if outputs[split] else None
    report_path = out_dir / "selection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not outputs["train"] or not outputs["val"]:
        raise RuntimeError(
            "Mixed smoke needs non-empty train and validation rows; inspect "
            f"{report_path} for rejected streams and cache roots"
        )


if __name__ == "__main__":
    main()
