#!/usr/bin/env python3
"""Batch-convert extracted OakInk2 sequences and merge V15 manifests."""

import argparse
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


CAMERAS = (
    "egocentric",
    "allocentric_top",
    "allocentric_left",
    "allocentric_right",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oakink2-root", required=True)
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--sequence-list")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=["egocentric"])
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument(
        "--val-subjects", nargs="+",
        help="Fixed validation subjects; otherwise reuse an existing status.json split.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument("--overlay-count", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subject_id(sequence):
    if "__" not in sequence or "++" not in sequence:
        raise ValueError(f"Cannot extract OakInk2 subject from {sequence}")
    return sequence.split("__", 1)[1].split("++", 1)[0]


def discover_sequences(root, sequence_list):
    data_root = root / "data"
    annotation_root = root / "anno_preview"
    available = {
        path.stem: path for path in annotation_root.glob("*.pkl")
        if (data_root / path.stem).is_dir()
    }
    if sequence_list:
        requested = [
            line.strip() for line in Path(sequence_list).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} requested sequences lack data or annotation: "
                + ", ".join(missing[:5])
            )
        return requested, available
    return sorted(available), available


def subject_split(sequences, val_fraction):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between zero and one")
    groups = defaultdict(list)
    for sequence in sequences:
        groups[subject_id(sequence)].append(sequence)
    subjects = sorted(groups)
    if len(subjects) < 2:
        raise ValueError("At least two subjects are required for a split")
    target = max(1, round(len(sequences) * val_fraction))
    candidates = []
    for count in range(1, len(subjects)):
        for selected in itertools.combinations(subjects, count):
            size = sum(len(groups[subject]) for subject in selected)
            candidates.append((abs(size - target), count, selected))
    val_subjects = set(min(candidates)[2])
    split = {
        sequence: ("val" if subject_id(sequence) in val_subjects else "train")
        for sequence in sequences
    }
    return split, sorted(val_subjects)


def output_is_current(manifest):
    if not manifest.is_file():
        return False
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or not rows[0].get("label_paths"):
        return False
    label = Path(rows[0]["label_paths"][0])
    try:
        with np.load(label, allow_pickle=False) as data:
            return {"seg", "joint_in_frame", "observation_valid"}.issubset(data.files)
    except (OSError, KeyError, ValueError):
        return False


def resolve_split(sequences, out_root, val_fraction, val_subjects=None):
    status_path = out_root / "status.json"
    previous_subjects = None
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        previous_subjects = previous.get("val_subjects")
        if not previous_subjects:
            raise ValueError(f"Existing status has no validation subjects: {status_path}")
    if val_subjects is not None and previous_subjects is not None:
        if set(val_subjects) != set(previous_subjects):
            raise ValueError(
                "Validation subjects differ from the existing output; "
                "use a new output root for a different split"
            )
    fixed = val_subjects if val_subjects is not None else previous_subjects
    if fixed is None:
        return subject_split(sequences, val_fraction)
    fixed = sorted(set(fixed))
    split = {
        sequence: ("val" if subject_id(sequence) in fixed else "train")
        for sequence in sequences
    }
    return split, fixed


def main():
    args = parse_args()
    oak_root = Path(args.oakink2_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    sequences, annotations = discover_sequences(oak_root, args.sequence_list)
    split_by_sequence, val_subjects = resolve_split(
        sequences, out_root, args.val_fraction, args.val_subjects
    )
    converter = Path(__file__).with_name("prepare_oakink2_v15.py")
    merged = {"train": [], "val": []}
    failures = []

    for index, sequence in enumerate(sequences, 1):
        split = split_by_sequence[sequence]
        for camera in args.cameras:
            stream_id = f"{sequence}__oakink2_{camera}"
            stream_out = out_root / "streams" / split / stream_id
            manifest = stream_out / f"{split}_windows.jsonl"
            print(
                f"[{index}/{len(sequences)}] {split} {sequence} camera={camera}",
                flush=True,
            )
            if not args.overwrite and output_is_current(manifest):
                print("  cached", flush=True)
            else:
                command = [
                    sys.executable, "-u", str(converter),
                    "--sequence-dir", str(oak_root / "data" / sequence),
                    "--annotation-pkl", str(annotations[sequence]),
                    "--mano-model-folder", args.mano_model_folder,
                    "--out-dir", str(stream_out),
                    "--split", split,
                    "--camera", camera,
                    "--frame-stride", str(args.frame_stride),
                    "--window-size", str(args.window_size),
                    "--window-stride", str(args.window_stride),
                    "--min-valid-frames", str(args.min_valid_frames),
                    "--overlay-count", str(args.overlay_count),
                    "--overwrite",
                ]
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as error:
                    failures.append({"stream_id": stream_id, "returncode": error.returncode})
                    continue
            if output_is_current(manifest):
                merged[split].extend(
                    json.loads(line) for line in manifest.read_text().splitlines()
                    if line.strip()
                )

    manifest_root = out_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split, rows in merged.items():
        rows.sort(key=lambda row: (row["stream_id"], row["start"]))
        with (manifest_root / f"{split}_windows.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    status = {
        "sequences": len(sequences),
        "cameras": args.cameras,
        "train_sequences": sum(value == "train" for value in split_by_sequence.values()),
        "val_sequences": sum(value == "val" for value in split_by_sequence.values()),
        "val_subjects": val_subjects,
        "train_windows": len(merged["train"]),
        "val_windows": len(merged["val"]),
        "failures": failures,
    }
    (out_root / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
