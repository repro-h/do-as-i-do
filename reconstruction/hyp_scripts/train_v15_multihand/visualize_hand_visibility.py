#!/usr/bin/env python3
"""Render detector visibility, detector joints and DexYCB GT joints."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--visibility-root", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def records_for_stream(path, stream_id):
    records = {}
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("stream_id")) != stream_id:
                continue
            for frame, image, label in zip(
                row["frame_indices"], row["image_paths"], row["label_paths"]
            ):
                records[int(frame)] = (image, label)
    if not records:
        raise RuntimeError(f"No frames for {stream_id} in {path}")
    return records


def visibility_color(probability):
    probability = float(np.clip(probability, 0.0, 1.0))
    return (40, int(255 * probability), int(255 * (1.0 - probability)))


def point(value):
    return tuple(np.round(value).astype(int).tolist())


def draw_skeleton(image, joints, visibility):
    finite = np.isfinite(joints).all(axis=-1)
    for first, second in BONES:
        if not (finite[first] and finite[second]):
            continue
        probability = 0.5 * (visibility[first] + visibility[second])
        cv2.line(
            image, point(joints[first]), point(joints[second]),
            visibility_color(probability), 2, cv2.LINE_AA,
        )
    for index, value in enumerate(joints):
        if not finite[index]:
            continue
        cv2.circle(
            image, point(value), 4, visibility_color(visibility[index]),
            -1, cv2.LINE_AA,
        )
        cv2.putText(
            image, str(index), (point(value)[0] + 4, point(value)[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA,
        )


def main():
    args = parse_args()
    records = records_for_stream(args.windows, args.stream_id)
    cache_path = (
        Path(args.visibility_root).expanduser().resolve()
        / args.stream_id / "visibility_cache.npz"
    )
    with np.load(str(cache_path), allow_pickle=False) as cache:
        required = ("detector_joint_uv", "detector_bbox_xyxy")
        missing = [key for key in required if key not in cache.files]
        if missing:
            raise KeyError(
                f"Cache lacks {missing}; re-export visibility with --overwrite"
            )
        frame_indices = np.asarray(cache["frame_indices"], dtype=np.int64)
        visibility = np.asarray(cache["joint_visibility"], dtype=np.float32)
        valid = np.asarray(cache["visibility_valid"], dtype=bool)
        confidence = np.asarray(cache["bbox_confidence"], dtype=np.float32)
        error = np.asarray(cache["matched_keypoint_error_px"], dtype=np.float32)
        detector_uv = np.asarray(cache["detector_joint_uv"], dtype=np.float32)
        boxes = np.asarray(cache["detector_bbox_xyxy"], dtype=np.float32)
        is_right = np.asarray(cache["detector_is_right"], dtype=bool)
    cache_index = {int(frame): offset for offset, frame in enumerate(frame_indices)}

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty; pass --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(set(records) & set(cache_index))
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    writer = None
    rendered = 0
    for frame in frames:
        image_path, label_path = records[frame]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        offset = cache_index[frame]
        with np.load(label_path, allow_pickle=False) as label:
            gt_uv = np.asarray(label["joint_2d"], dtype=np.float32)[0]
        # GT is cyan and intentionally not visibility-coloured.
        for first, second in BONES:
            cv2.line(
                image, point(gt_uv[first]), point(gt_uv[second]),
                (255, 220, 0), 1, cv2.LINE_AA,
            )
        for value in gt_uv:
            cv2.circle(image, point(value), 2, (255, 220, 0), -1, cv2.LINE_AA)
        if valid[offset]:
            draw_skeleton(image, detector_uv[offset], visibility[offset])
            x1, y1, x2, y2 = point(boxes[offset, :2]) + point(boxes[offset, 2:])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        text = (
            f"frame={frame:06d} valid={int(valid[offset])} "
            f"side={'R' if is_right[offset] else 'L'} "
            f"bbox_conf={confidence[offset]:.3f} match={error[offset]:.1f}px "
            f"vis_mean={visibility[offset].mean():.3f}"
        )
        cv2.putText(
            image, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (20, 20, 20), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image, "cyan=GT  red/green=detector visibility",
            (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"frame_{frame:06d}.jpg"), image)
        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(
                str(out_dir / "visibility.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height),
            )
        writer.write(image)
        rendered += 1
    if writer is not None:
        writer.release()
    print(json.dumps({
        "stream_id": args.stream_id,
        "rendered_frames": rendered,
        "output": str(out_dir),
        "video": str(out_dir / "visibility.mp4"),
    }, indent=2))


if __name__ == "__main__":
    main()

