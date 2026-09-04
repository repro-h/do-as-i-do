#!/usr/bin/env python3
"""Compare track/GT joint queries with matched detector joint queries."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset import auxiliary_cache_path, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--visibility-root")
    parser.add_argument("--track-root")
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def distribution(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0, "median_px": None, "p90_px": None,
            "p95_px": None, "p99_px": None, "max_px": None,
        }
    return {
        "count": int(len(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "p95_px": float(np.percentile(values, 95)),
        "p99_px": float(np.percentile(values, 99)),
        "max_px": float(np.max(values)),
    }


def side_name(code):
    return "left" if code == 0 else "right" if code == 1 else "unknown"


def main():
    args = parse_args()
    rows = load_jsonl(args.windows)
    visibility_root = (
        None if args.visibility_root is None
        else Path(args.visibility_root).expanduser().resolve()
    )
    track_root = (
        None if args.track_root is None
        else Path(args.track_root).expanduser().resolve()
    )

    streams = {}
    for row in rows:
        key = (str(row.get("dataset", "unknown")), str(row["stream_id"]))
        streams.setdefault(key, row)

    groups = defaultdict(lambda: {"all": [], "wrist": []})
    matched_instances = total_track_instances = track_id_mismatches = 0
    missing_detector_uv = []

    for (dataset, stream), row in sorted(streams.items()):
        track_path = auxiliary_cache_path(
            row, "tracks_npz", track_root, "tracks.npz"
        )
        visibility_path = auxiliary_cache_path(
            row, "visibility_npz", visibility_root, "visibility_cache.npz"
        )
        with np.load(track_path, allow_pickle=False) as tracks, np.load(
            visibility_path, allow_pickle=False
        ) as detector:
            if "detector_joint_uv" not in detector.files:
                missing_detector_uv.append(stream)
                continue
            track_frames = np.asarray(tracks["frame_indices"], dtype=np.int64)
            detector_frames = np.asarray(detector["frame_indices"], dtype=np.int64)
            detector_index = {
                int(frame): index for index, frame in enumerate(detector_frames)
            }
            gt_uv = np.asarray(tracks["joint_uv"], dtype=np.float32)
            gt_valid = np.asarray(tracks["joint_valid"], dtype=bool)
            track_valid = np.asarray(tracks["track_valid"], dtype=bool)
            hand_side = np.asarray(tracks["hand_side"], dtype=np.int8)
            track_ids = np.asarray(tracks["track_ids"], dtype=np.int64)
            detector_uv = np.asarray(detector["detector_joint_uv"], dtype=np.float32)
            detector_valid = np.asarray(detector["visibility_valid"], dtype=bool)
            detector_ids = (
                np.asarray(detector["track_ids"], dtype=np.int64)
                if "track_ids" in detector.files else None
            )

            for track_offset, frame in enumerate(track_frames):
                detector_offset = detector_index.get(int(frame))
                if detector_offset is None:
                    continue
                hands = min(gt_uv.shape[1], detector_uv.shape[1])
                for hand in range(hands):
                    if not track_valid[track_offset, hand]:
                        continue
                    total_track_instances += 1
                    if not detector_valid[detector_offset, hand]:
                        continue
                    if detector_ids is not None and (
                        detector_ids[detector_offset, hand]
                        != track_ids[track_offset, hand]
                    ):
                        track_id_mismatches += 1
                        continue
                    valid = gt_valid[track_offset, hand].copy()
                    valid &= np.isfinite(
                        detector_uv[detector_offset, hand]
                    ).all(axis=-1)
                    if not valid.any():
                        continue
                    matched_instances += 1
                    error = np.linalg.norm(
                        detector_uv[detector_offset, hand]
                        - gt_uv[track_offset, hand],
                        axis=-1,
                    )
                    side = side_name(int(hand_side[track_offset, hand]))
                    for key in ("overall", f"dataset:{dataset}", f"side:{side}"):
                        groups[key]["all"].extend(error[valid].tolist())
                        if valid[0]:
                            groups[key]["wrist"].append(float(error[0]))

    report = {
        "windows": str(Path(args.windows).expanduser().resolve()),
        "streams": len(streams),
        "track_instances": total_track_instances,
        "matched_detector_instances": matched_instances,
        "detector_instance_coverage": (
            matched_instances / total_track_instances
            if total_track_instances else 0.0
        ),
        "track_id_mismatches": track_id_mismatches,
        "streams_missing_detector_joint_uv": missing_detector_uv,
        "groups": {
            key: {
                "all_joints": distribution(values["all"]),
                "wrist": distribution(values["wrist"]),
            }
            for key, values in sorted(groups.items())
        },
    }
    output = Path(args.out_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
