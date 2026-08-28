#!/usr/bin/env python3
"""Audit TACO V1 availability and make deterministic official-split samples."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TEST_SPLITS = ("test_1", "test_2", "test_3", "test_4")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--official-split", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count-per-split", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_key(seed, *values):
    text = "|".join([str(seed), *map(str, values)]).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def load_split(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            sequence, split = row[0].strip(), row[1].strip()
            if split in {"train", *TEST_SPLITS}:
                result[sequence] = split
    return result


def frame_count(path):
    try:
        value = np.load(path, mmap_mode="r")
        return int(value.shape[0])
    except (OSError, ValueError, IndexError):
        return 0


def discover(root, official):
    rows = []
    video_root = root / "Egocentric_RGB_Videos"
    for video in sorted(video_root.glob("*/*/color.mp4")):
        triplet = video.parent.parent.name
        sequence = video.parent.name
        hand_root = root / "Hand_Poses" / triplet / sequence
        camera_root = root / "Egocentric_Camera_Parameters" / triplet / sequence
        extrinsics = camera_root / "egocentric_frame_extrinsic.npy"
        intrinsics = camera_root / "egocentric_intrinsic.txt"
        required = [
            video, extrinsics, intrinsics,
            hand_root / "left_hand.pkl", hand_root / "left_hand_shape.pkl",
            hand_root / "right_hand.pkl", hand_root / "right_hand_shape.pkl",
        ]
        rows.append({
            "sequence": sequence,
            "triplet": triplet,
            "official_split": official.get(sequence, "unknown"),
            "frames": frame_count(extrinsics),
            "video": str(video),
            "hand_root": str(hand_root),
            "camera_root": str(camera_root),
            "complete": all(path.is_file() for path in required),
        })
    return rows


def diverse_sample(rows, count, seed, preferred_triplets=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row["triplet"]].append(row)
    for triplet in groups:
        groups[triplet].sort(
            key=lambda row: stable_key(seed, triplet, row["sequence"])
        )
    triplets = sorted(
        groups,
        key=lambda value: (
            0 if preferred_triplets and value in preferred_triplets else 1,
            stable_key(seed, value),
        ),
    )
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for triplet in triplets:
            if depth < len(groups[triplet]):
                selected.append(groups[triplet][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def write_jsonl(path, rows, local_split):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = dict(row)
            value["local_split"] = local_split
            handle.write(json.dumps(value, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    root = Path(args.taco_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    official = load_split(args.official_split)
    rows = discover(root, official)
    complete = [row for row in rows if row["complete"] and row["frames"] > 0]
    by_split = {
        split: [row for row in complete if row["official_split"] == split]
        for split in ("train", *TEST_SPLITS)
    }

    train = diverse_sample(by_split["train"], args.train_count, args.seed)
    train_ids = {row["sequence"] for row in train}
    remaining = [row for row in by_split["train"] if row["sequence"] not in train_ids]
    train_triplets = {row["triplet"] for row in train}
    val = diverse_sample(
        remaining, args.val_count, args.seed + 1,
        preferred_triplets=train_triplets,
    )
    tests = {
        split: diverse_sample(
            by_split[split], args.test_count_per_split,
            args.seed + 10 + index,
        )
        for index, split in enumerate(TEST_SPLITS)
    }

    selection_root = out_root / "selections"
    write_jsonl(selection_root / "train_32.jsonl", train, "train")
    write_jsonl(selection_root / "val_8.jsonl", val, "val")
    for split, selected in tests.items():
        write_jsonl(selection_root / f"{split}_8.jsonl", selected, split)

    split_counts = {
        split: {
            "available_sequences": len(selected),
            "frames": int(sum(row["frames"] for row in selected)),
        }
        for split, selected in by_split.items()
    }
    report = {
        "taco_root": str(root),
        "official_split": str(Path(args.official_split).resolve()),
        "discovered_ego_sequences": len(rows),
        "complete_ego_sequences": len(complete),
        "unknown_split_sequences": sum(row["official_split"] == "unknown" for row in rows),
        "official_split_counts": split_counts,
        "selection": {
            "train": {"sequences": len(train), "frames": sum(row["frames"] for row in train)},
            "val": {"sequences": len(val), "frames": sum(row["frames"] for row in val)},
            **{
                split: {"sequences": len(selected), "frames": sum(row["frames"] for row in selected)}
                for split, selected in tests.items()
            },
        },
        "train_triplets": len({row["triplet"] for row in train}),
        "val_triplets": len({row["triplet"] for row in val}),
        "incomplete": [row for row in rows if not row["complete"] or row["frames"] <= 0],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
