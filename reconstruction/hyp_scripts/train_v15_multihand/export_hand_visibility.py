#!/usr/bin/env python3
"""Export per-joint visibility for stable single- or multi-hand slots."""

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
    parser.add_argument("--track-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone", default="wilor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hand-confidence", type=float, default=0.3)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--max-match-distance-px", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--status-json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path):
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"--windows must be a JSONL file: {manifest}")
    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_stream_frames(rows):
    streams = defaultdict(dict)
    for row in rows:
        side = row.get(
            "hand_sides_metadata_only",
            row.get("hand_side_metadata_only", "unknown"),
        )
        for frame, image, label in zip(
            row["frame_indices"], row["image_paths"], row["label_paths"]
        ):
            streams[row["stream_id"]][int(frame)] = (image, label, side)
    return streams


def valid_cache(path, stream_id, expected_frames, max_hands, require_multihand):
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
            multi_shape = visibility.shape == (len(frames), max_hands, 21)
            multi_shape &= valid.shape == (len(frames), max_hands)
            legacy_shape = visibility.shape == (len(frames), 21)
            legacy_shape &= valid.shape == (len(frames),)
            if not multi_shape and (require_multihand or not legacy_shape):
                return False
            return bool(np.isfinite(visibility).all())
    except (OSError, KeyError, ValueError):
        return False


def side_code(value):
    value = str(value).lower()
    return 0 if value == "left" else 1 if value == "right" else -1


def label_targets(label_path, side_metadata, max_hands):
    with np.load(label_path, allow_pickle=False) as data:
        target = np.asarray(data["joint_2d"], dtype=np.float32)
        joint_in_frame = (
            np.asarray(data["joint_in_frame"], dtype=bool)
            if "joint_in_frame" in data.files else None
        )
    if target.ndim == 2:
        target = target[None]
    target = target[:max_hands]
    if joint_in_frame is not None:
        if joint_in_frame.ndim == 1:
            joint_in_frame = joint_in_frame[None]
        joint_in_frame = joint_in_frame[:len(target)]
        if joint_in_frame.shape != target.shape[:-1]:
            raise ValueError(
                f"joint_in_frame shape {joint_in_frame.shape} does not match "
                f"joint_2d shape {target.shape}"
            )
        target = target.copy()
        target[~joint_in_frame] = np.nan
    valid = np.isfinite(target).all(axis=-1).any(axis=-1)
    values = side_metadata if isinstance(side_metadata, list) else [side_metadata]
    sides = np.full(len(target), -1, dtype=np.int8)
    for index, value in enumerate(values[:len(target)]):
        sides[index] = side_code(value)
    return target, valid, sides


def match_detections(results, targets, target_valid, sides, max_distance):
    pairs = []
    for slot in np.flatnonzero(target_valid):
        finite = np.isfinite(targets[slot]).all(axis=-1)
        if not finite.any():
            continue
        for result_index, result in enumerate(results):
            predicted = np.asarray(result.keypoints_2d, dtype=np.float32)
            if predicted.shape != (21, 2):
                continue
            distance = float(np.mean(np.linalg.norm(
                predicted[finite] - targets[slot, finite], axis=-1
            )))
            side_mismatch = (
                sides[slot] >= 0
                and bool(result.is_right) != bool(sides[slot])
            )
            cost = distance + float(side_mismatch) * max_distance
            pairs.append((cost, distance, slot, result_index))
    matches = {}
    used_results = set()
    for _, distance, slot, result_index in sorted(pairs):
        if distance > max_distance:
            continue
        if slot in matches or result_index in used_results:
            continue
        matches[slot] = (results[result_index], distance)
        used_results.add(result_index)
    return matches


def main():
    args = parse_args()
    streams = unique_stream_frames(load_rows(args.windows))
    items = sorted(streams.items())
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    items = items[args.shard_index::args.num_shards]
    out_root = Path(args.out_root).expanduser().resolve()
    track_root = (
        None if not args.track_root
        else Path(args.track_root).expanduser().resolve()
    )
    pending, cached = [], []
    for stream_id, records in items:
        expected = np.asarray(sorted(records), dtype=np.int64)
        output = out_root / stream_id / "visibility_cache.npz"
        if not args.overwrite and valid_cache(
            output, stream_id, expected, args.max_hands,
            require_multihand=track_root is not None,
        ):
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
        shape = (len(frame_indices), args.max_hands)
        visibility = np.full((*shape, 21), 0.5, dtype=np.float32)
        valid = np.zeros(shape, dtype=bool)
        bbox_confidence = np.zeros(shape, dtype=np.float32)
        match_error = np.full(shape, np.nan, dtype=np.float32)
        detector_joint_uv = np.full((*shape, 21, 2), np.nan, dtype=np.float32)
        detector_bbox_xyxy = np.full((*shape, 4), np.nan, dtype=np.float32)
        detector_is_right = np.zeros(shape, dtype=bool)
        track_ids = np.full(shape, -1, dtype=np.int64)
        track_cache = None
        track_index = {}
        try:
            if track_root is not None:
                track_cache = np.load(
                    str(track_root / stream_id / "tracks.npz"),
                    allow_pickle=False,
                )
                track_index = {
                    int(frame): offset for offset, frame in enumerate(
                        np.asarray(track_cache["frame_indices"], dtype=np.int64)
                    )
                }
            for offset, frame in enumerate(frame_indices):
                image_path, label_path, sides_metadata = records[int(frame)]
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue
                results = pipeline.predict(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                if track_cache is None:
                    targets, target_valid, sides = label_targets(
                        label_path, sides_metadata, args.max_hands
                    )
                elif int(frame) in track_index:
                    source = track_index[int(frame)]
                    targets = np.asarray(
                        track_cache["joint_uv"][source], dtype=np.float32
                    )[:args.max_hands]
                    target_valid = np.asarray(
                        track_cache["observation_valid"][source], dtype=bool
                    )[:args.max_hands]
                    sides = np.asarray(
                        track_cache["hand_side"][source], dtype=np.int8
                    )[:args.max_hands]
                    track_ids[offset, :len(targets)] = np.asarray(
                        track_cache["track_ids"][source], dtype=np.int64
                    )[:args.max_hands]
                else:
                    continue
                matches = match_detections(
                    results, targets, target_valid, sides,
                    args.max_match_distance_px,
                )
                for slot, (result, error) in matches.items():
                    value = np.asarray(result.visibility, dtype=np.float32)
                    if value.shape != (21,) or not np.isfinite(value).all():
                        continue
                    visibility[offset, slot] = np.clip(value, 0.0, 1.0)
                    valid[offset, slot] = True
                    bbox_confidence[offset, slot] = float(result.bbox_conf)
                    match_error[offset, slot] = error
                    detector_joint_uv[offset, slot] = np.asarray(
                        result.keypoints_2d, dtype=np.float32
                    )
                    detector_bbox_xyxy[offset, slot] = np.asarray(
                        result.hand_bbox, dtype=np.float32
                    )[:4]
                    detector_is_right[offset, slot] = bool(result.is_right)
            np.savez_compressed(
                output,
                cache_version=np.asarray("hand_visibility_detector_multihand_v2"),
                stream_id=np.asarray(stream_id),
                frame_indices=frame_indices,
                joint_visibility=visibility.astype(np.float16),
                visibility_valid=valid,
                bbox_confidence=bbox_confidence,
                matched_keypoint_error_px=match_error,
                detector_joint_uv=detector_joint_uv,
                detector_bbox_xyxy=detector_bbox_xyxy,
                detector_is_right=detector_is_right,
                track_ids=track_ids,
                joint_order=np.asarray("wrist_thumb_index_middle_ring_pinky_mano21"),
                source_checkpoint=np.asarray(args.checkpoint or "hub_default"),
            )
            summary = {
                "stream_id": stream_id,
                "frames": int(len(frame_indices)),
                "max_hands": args.max_hands,
                "valid_instances": int(valid.sum()),
                "valid_fraction": float(valid.mean()) if valid.size else 0.0,
                "match_error_median_px": (
                    float(np.nanmedian(match_error)) if valid.any() else None
                ),
                "output": str(output),
            }
            (stream_out / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            completed.append(stream_id)
        except Exception as error:
            failures.append({"stream_id": stream_id, "error": repr(error)})
            print(f"FAILED {stream_id}: {type(error).__name__}: {error}")
        finally:
            if track_cache is not None:
                track_cache.close()

    status = {
        "windows": str(Path(args.windows).expanduser().resolve()),
        "track_root": None if track_root is None else str(track_root),
        "max_hands": args.max_hands,
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
