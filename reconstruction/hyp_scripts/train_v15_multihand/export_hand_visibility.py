#!/usr/bin/env python3
"""Export 21-joint visibility probabilities for DexYCB streams.

Run this script in the hand_visibility_detector environment. For DexYCB's
single annotated hand, detections are associated to GT 2D joints only to choose
the matching instance; the exported probabilities remain model predictions.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--detector-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone", default="wilor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hand-confidence", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path):
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"--windows must point to a JSONL file, got: {manifest}"
        )
    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_stream_frames(rows):
    streams = defaultdict(dict)
    for row in rows:
        side = str(row.get("hand_side_metadata_only", "unknown"))
        for frame, image, label in zip(
            row["frame_indices"], row["image_paths"], row["label_paths"]
        ):
            streams[row["stream_id"]][int(frame)] = (image, label, side)
    return streams


def select_detection(results, label_path, expected_side):
    if not results:
        return None, float("nan")
    with np.load(label_path, allow_pickle=False) as data:
        target = np.asarray(data["joint_2d"], dtype=np.float32)[0]
    valid = np.isfinite(target).all(axis=-1)
    candidates = []
    for result in results:
        predicted = np.asarray(result.keypoints_2d, dtype=np.float32)
        if predicted.shape != target.shape:
            continue
        distance = float(np.linalg.norm(predicted[valid] - target[valid], axis=-1).mean())
        side_matches = (
            expected_side not in ("left", "right")
            or bool(result.is_right) == (expected_side == "right")
        )
        candidates.append((not side_matches, distance, -float(result.bbox_conf), result))
    if not candidates:
        return None, float("nan")
    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3], candidates[0][1]


def main():
    args = parse_args()
    detector_root = Path(args.detector_root).expanduser().resolve()
    sys.path.insert(0, str(detector_root / "src"))
    from hand_visibility_detector import HandVisibilityPipeline

    streams = unique_stream_frames(load_rows(args.windows))
    pipeline = HandVisibilityPipeline(
        device=args.device,
        vis_checkpoint=args.checkpoint,
        backbone=args.backbone,
        hand_conf=args.hand_confidence,
    )
    out_root = Path(args.out_root).expanduser().resolve()
    completed = failed = 0
    for stream_id, records in tqdm(sorted(streams.items()), desc="streams"):
        stream_out = out_root / stream_id
        output = stream_out / "visibility_cache.npz"
        if output.is_file() and not args.overwrite:
            completed += 1
            continue
        stream_out.mkdir(parents=True, exist_ok=True)
        frame_indices = np.asarray(sorted(records), dtype=np.int64)
        visibility = np.full((len(frame_indices), 21), 0.5, dtype=np.float32)
        valid = np.zeros(len(frame_indices), dtype=bool)
        bbox_confidence = np.zeros(len(frame_indices), dtype=np.float32)
        match_error = np.full(len(frame_indices), np.nan, dtype=np.float32)
        try:
            for offset, frame in enumerate(frame_indices):
                image_path, label_path, side = records[int(frame)]
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue
                results = pipeline.predict(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                result, error = select_detection(results, label_path, side)
                if result is None:
                    continue
                value = np.asarray(result.visibility, dtype=np.float32)
                if value.shape != (21,) or not np.isfinite(value).all():
                    continue
                visibility[offset] = np.clip(value, 0.0, 1.0)
                valid[offset] = True
                bbox_confidence[offset] = float(result.bbox_conf)
                match_error[offset] = error
            np.savez_compressed(
                output,
                cache_version=np.asarray("hand_visibility_detector_v1"),
                stream_id=np.asarray(stream_id),
                frame_indices=frame_indices,
                joint_visibility=visibility.astype(np.float16),
                visibility_valid=valid,
                bbox_confidence=bbox_confidence,
                matched_keypoint_error_px=match_error,
                joint_order=np.asarray(
                    "wrist_thumb_index_middle_ring_pinky_mano21"
                ),
                source_checkpoint=np.asarray(args.checkpoint or "hub_default"),
            )
            with (stream_out / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump({
                    "stream_id": stream_id,
                    "frames": int(len(frame_indices)),
                    "valid": int(valid.sum()),
                    "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
                    "match_error_median_px": (
                        float(np.nanmedian(match_error)) if valid.any() else None
                    ),
                    "output": str(output),
                }, handle, indent=2)
            completed += 1
        except Exception as error:
            failed += 1
            print(f"FAILED {stream_id}: {type(error).__name__}: {error}")
    print(json.dumps({
        "streams": len(streams), "completed": completed, "failed": failed,
        "out_root": str(out_root),
    }, indent=2))


if __name__ == "__main__":
    main()
