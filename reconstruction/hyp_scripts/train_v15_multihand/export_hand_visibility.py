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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--status-json")
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


def valid_cache(path, stream_id, expected_frames):
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "frame_indices", "joint_visibility", "visibility_valid",
                "bbox_confidence", "detector_joint_uv",
            }
            if not required.issubset(data.files):
                return False
            frames = np.asarray(data["frame_indices"], dtype=np.int64)
            visibility = np.asarray(data["joint_visibility"])
            valid = np.asarray(data["visibility_valid"])
            if str(data["stream_id"].item()) != stream_id:
                return False
            if not np.array_equal(frames, expected_frames):
                return False
            if visibility.shape != (len(frames), 21) or valid.shape != (len(frames),):
                return False
            return bool(np.isfinite(visibility).all())
    except (OSError, KeyError, ValueError):
        return False


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
    streams = unique_stream_frames(load_rows(args.windows))
    items = sorted(streams.items())
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index is outside the shard range")
    items = items[args.shard_index::args.num_shards]
    out_root = Path(args.out_root).expanduser().resolve()
    pending = []
    cached = []
    for stream_id, records in items:
        expected = np.asarray(sorted(records), dtype=np.int64)
        output = out_root / stream_id / "visibility_cache.npz"
        if not args.overwrite and valid_cache(output, stream_id, expected):
            cached.append(stream_id)
        else:
            pending.append((stream_id, records))

    pipeline = None
    if pending:
        detector_root = Path(args.detector_root).expanduser().resolve()
        sys.path.insert(0, str(detector_root / "src"))
        from hand_visibility_detector import HandVisibilityPipeline

        pipeline = HandVisibilityPipeline(
            device=args.device,
            vis_checkpoint=args.checkpoint,
            backbone=args.backbone,
            hand_conf=args.hand_confidence,
        )

    completed = list(cached)
    failures = []
    for stream_id, records in tqdm(pending, desc="streams"):
        stream_out = out_root / stream_id
        output = stream_out / "visibility_cache.npz"
        stream_out.mkdir(parents=True, exist_ok=True)
        frame_indices = np.asarray(sorted(records), dtype=np.int64)
        visibility = np.full((len(frame_indices), 21), 0.5, dtype=np.float32)
        valid = np.zeros(len(frame_indices), dtype=bool)
        bbox_confidence = np.zeros(len(frame_indices), dtype=np.float32)
        match_error = np.full(len(frame_indices), np.nan, dtype=np.float32)
        detector_joint_uv = np.full(
            (len(frame_indices), 21, 2), np.nan, dtype=np.float32
        )
        detector_bbox_xyxy = np.full(
            (len(frame_indices), 4), np.nan, dtype=np.float32
        )
        detector_is_right = np.zeros(len(frame_indices), dtype=bool)
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
                detector_joint_uv[offset] = np.asarray(
                    result.keypoints_2d, dtype=np.float32
                )
                detector_bbox_xyxy[offset] = np.asarray(
                    result.hand_bbox, dtype=np.float32
                )[:4]
                detector_is_right[offset] = bool(result.is_right)
            np.savez_compressed(
                output,
                cache_version=np.asarray("hand_visibility_detector_v1"),
                stream_id=np.asarray(stream_id),
                frame_indices=frame_indices,
                joint_visibility=visibility.astype(np.float16),
                visibility_valid=valid,
                bbox_confidence=bbox_confidence,
                matched_keypoint_error_px=match_error,
                detector_joint_uv=detector_joint_uv,
                detector_bbox_xyxy=detector_bbox_xyxy,
                detector_is_right=detector_is_right,
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
            completed.append(stream_id)
        except Exception as error:
            failures.append({"stream_id": stream_id, "error": repr(error)})
            print(f"FAILED {stream_id}: {type(error).__name__}: {error}")
    status = {
        "windows": str(Path(args.windows).expanduser().resolve()),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "streams": len(items),
        "completed": len(completed),
        "cached": len(cached),
        "failed": len(failures),
        "completed_streams": completed,
        "failures": failures,
        "out_root": str(out_root),
    }
    if args.status_json:
        status_path = Path(args.status_json).expanduser().resolve()
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
