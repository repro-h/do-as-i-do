#!/usr/bin/env python3
"""Audit DexYCB hand-detector gate failures with raw detector overlays."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from export_hand_visibility import label_targets, match_detections


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
    parser.add_argument("--track-root")
    parser.add_argument("--detector-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backbone", default="wilor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hand-confidence", type=float, default=0.3)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--max-match-distance-px", type=float, default=120.0)
    parser.add_argument("--failure-count", type=int, default=64)
    parser.add_argument("--control-count", type=int, default=16)
    parser.add_argument("--max-per-stream", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-columns", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path):
    streams = defaultdict(dict)
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            side = row.get(
                "hand_sides_metadata_only",
                row.get("hand_side_metadata_only", "unknown"),
            )
            for frame, image, label in zip(
                row["frame_indices"], row["image_paths"], row["label_paths"]
            ):
                streams[str(row["stream_id"])][int(frame)] = (
                    str(image), str(label), side,
                )
    return streams


def load_targets(stream_id, records, track_root, max_hands):
    targets = {}
    if track_root is None:
        for frame, (_, label, side) in records.items():
            uv, valid, sides = label_targets(label, side, max_hands)
            targets[frame] = (uv, valid, sides)
        return targets

    track_path = track_root / stream_id / "tracks.npz"
    with np.load(str(track_path), allow_pickle=False) as track:
        frames = np.asarray(track["frame_indices"], dtype=np.int64)
        uv = np.asarray(track["joint_uv"], dtype=np.float32)
        valid = np.asarray(track["observation_valid"], dtype=bool)
        sides = np.asarray(track["hand_side"], dtype=np.int8)
        target_valid = (
            np.asarray(track["target_valid"], dtype=bool)
            if "target_valid" in track.files else np.ones_like(valid)
        )
        track_valid = (
            np.asarray(track["track_valid"], dtype=bool)
            if "track_valid" in track.files else np.ones_like(valid)
        )
        for offset, frame in enumerate(frames):
            count = min(max_hands, uv.shape[1])
            targets[int(frame)] = (
                uv[offset, :count],
                valid[offset, :count]
                & target_valid[offset, :count]
                & track_valid[offset, :count],
                sides[offset, :count],
            )
    return targets


def collect_candidates(streams, visibility_root, track_root, max_hands):
    failed, controls = [], []
    cache_stats = Counter()
    for stream_id, records in tqdm(sorted(streams.items()), desc="scan caches"):
        cache_path = visibility_root / stream_id / "visibility_cache.npz"
        if not cache_path.is_file():
            cache_stats["missing_cache_streams"] += 1
            continue
        targets = load_targets(stream_id, records, track_root, max_hands)
        with np.load(str(cache_path), allow_pickle=False) as cache:
            frames = np.asarray(cache["frame_indices"], dtype=np.int64)
            detector_valid = np.asarray(cache["visibility_valid"], dtype=bool)
        if detector_valid.ndim == 1:
            detector_valid = detector_valid[:, None]
        for offset, frame_value in enumerate(frames):
            frame = int(frame_value)
            if frame not in records or frame not in targets:
                cache_stats["frame_alignment_misses"] += 1
                continue
            _, target_valid, sides = targets[frame]
            count = min(max_hands, len(target_valid), detector_valid.shape[1])
            for slot in range(count):
                if not target_valid[slot]:
                    continue
                item = {
                    "stream_id": stream_id,
                    "frame": frame,
                    "slot": slot,
                    "side": int(sides[slot]),
                    "cache_valid": bool(detector_valid[offset, slot]),
                }
                cache_stats["target_instances"] += 1
                if item["cache_valid"]:
                    cache_stats["cache_observed"] += 1
                    controls.append(item)
                else:
                    cache_stats["cache_missing"] += 1
                    failed.append(item)
    return failed, controls, cache_stats


def stratified_sample(items, count, max_per_stream, seed):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    selected = []
    per_stream = Counter()
    for item in shuffled:
        stream = item["stream_id"]
        if max_per_stream > 0 and per_stream[stream] >= max_per_stream:
            continue
        selected.append(item)
        per_stream[stream] += 1
        if len(selected) >= count:
            break
    return selected


def finite_result_joints(result):
    joints = np.asarray(result.keypoints_2d, dtype=np.float32)
    return joints if joints.shape == (21, 2) else None


def mean_joint_distance(predicted, target):
    finite = np.isfinite(predicted).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if not finite.any():
        return float("inf")
    return float(np.mean(np.linalg.norm(predicted[finite] - target[finite], axis=-1)))


def classify(item, results, targets, target_valid, sides, max_distance):
    matches = match_detections(
        results, targets, target_valid, sides, max_distance,
    )
    slot = item["slot"]
    distances = []
    for index, result in enumerate(results):
        joints = finite_result_joints(result)
        if joints is None:
            continue
        distances.append((mean_joint_distance(joints, targets[slot]), index))
    distances.sort()
    closest_distance = distances[0][0] if distances else None
    closest_index = distances[0][1] if distances else None

    if item["cache_valid"]:
        category = "matched_control" if slot in matches else "control_rerun_missing"
    elif not results:
        category = "no_raw_detection"
    elif slot in matches:
        result = matches[slot][0]
        visibility = np.asarray(result.visibility)
        category = (
            "cache_miss_rerun_match"
            if visibility.shape == (21,) and np.isfinite(visibility).all()
            else "invalid_visibility_output"
        )
    elif closest_distance is None:
        category = "invalid_detector_keypoints"
    elif closest_distance > max_distance:
        category = "detector_geometry_far"
    else:
        assigned_slots = {
            result_index: matched_slot
            for matched_slot, (matched, _) in matches.items()
            for result_index, result in enumerate(results)
            if result is matched
        }
        if closest_index in assigned_slots:
            category = "assignment_conflict"
        else:
            expected_right = sides[slot] == 1
            detected_right = bool(results[closest_index].is_right)
            category = (
                "side_disagreement" if sides[slot] >= 0
                and detected_right != expected_right
                else "unmatched_other"
            )
    return category, closest_distance, closest_index, matches


def point(value):
    return tuple(np.round(value).astype(int).tolist())


def draw_skeleton(image, joints, color, width=2, radius=3):
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape != (21, 2):
        return
    finite = np.isfinite(joints).all(axis=-1)
    for first, second in BONES:
        if finite[first] and finite[second]:
            cv2.line(
                image, point(joints[first]), point(joints[second]),
                color, width, cv2.LINE_AA,
            )
    for value in joints[finite]:
        cv2.circle(image, point(value), radius, color, -1, cv2.LINE_AA)


def outlined_text(image, text, origin, scale=0.45, color=(255, 255, 255)):
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (15, 15, 15), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 1, cv2.LINE_AA,
    )


def render_case(image, item, targets, target_valid, sides, results, category,
                closest_distance, closest_index):
    gt_colors = [(255, 220, 0), (255, 0, 220), (220, 255, 0), (0, 220, 255)]
    for slot in np.flatnonzero(target_valid):
        color = gt_colors[int(slot) % len(gt_colors)]
        width = 3 if int(slot) == item["slot"] else 1
        draw_skeleton(image, targets[slot], color, width=width, radius=3)
        wrist = targets[slot, 0]
        if np.isfinite(wrist).all():
            outlined_text(
                image, f"GT slot={slot} {'R' if sides[slot] == 1 else 'L'}",
                (point(wrist)[0] + 6, point(wrist)[1] - 6), color=color,
            )
    for index, result in enumerate(results):
        color = (80, 255, 80) if bool(result.is_right) else (80, 140, 255)
        joints = finite_result_joints(result)
        if joints is not None:
            draw_skeleton(image, joints, color, width=2, radius=2)
        box = np.asarray(result.hand_bbox, dtype=np.float32).reshape(-1)
        if len(box) >= 4 and np.isfinite(box[:4]).all():
            x1, y1, x2, y2 = np.round(box[:4]).astype(int).tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            outlined_text(
                image,
                f"D{index} {'R' if bool(result.is_right) else 'L'} "
                f"conf={float(result.bbox_conf):.2f}",
                (x1, max(18, y1 - 5)), color=color,
            )
    distance_text = "n/a" if closest_distance is None else f"{closest_distance:.1f}px"
    lines = [
        f"{category}",
        f"frame={item['frame']} slot={item['slot']} cache={int(item['cache_valid'])}",
        f"raw detections={len(results)} closest=D{closest_index} {distance_text}",
        "thick cyan/magenta=GT; green/orange=raw detector",
    ]
    for index, text in enumerate(lines):
        outlined_text(image, text, (12, 24 + 22 * index), scale=0.43)
    return image


def make_contact_sheet(paths, output, columns):
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    cell_width = 480
    resized = []
    for image in images:
        scale = cell_width / image.shape[1]
        resized.append(cv2.resize(
            image, (cell_width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        ))
    cell_height = max(image.shape[0] for image in resized)
    rows = (len(resized) + columns - 1) // columns
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, image in enumerate(resized):
        row, column = divmod(index, columns)
        sheet[
            row * cell_height:row * cell_height + image.shape[0],
            column * cell_width:(column + 1) * cell_width,
        ] = image
    cv2.imwrite(str(output), sheet)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty; pass --overwrite: {out_dir}")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    streams = load_records(args.windows)
    visibility_root = Path(args.visibility_root).expanduser().resolve()
    track_root = (
        None if not args.track_root
        else Path(args.track_root).expanduser().resolve()
    )
    failed, controls, cache_stats = collect_candidates(
        streams, visibility_root, track_root, args.max_hands,
    )
    selected = stratified_sample(
        failed, args.failure_count, args.max_per_stream, args.seed,
    )
    selected += stratified_sample(
        controls, args.control_count, args.max_per_stream, args.seed + 1,
    )

    detector_root = Path(args.detector_root).expanduser().resolve()
    sys.path.insert(0, str(detector_root / "src"))
    from hand_visibility_detector import HandVisibilityPipeline

    pipeline = HandVisibilityPipeline(
        device=args.device,
        vis_checkpoint=args.checkpoint,
        backbone=args.backbone,
        hand_conf=args.hand_confidence,
    )

    target_cache = {}
    cases, rendered_paths = [], []
    category_counts = Counter()
    for case_index, item in enumerate(tqdm(selected, desc="rerun detector")):
        stream_id = item["stream_id"]
        frame = item["frame"]
        records = streams[stream_id]
        if stream_id not in target_cache:
            target_cache[stream_id] = load_targets(
                stream_id, records, track_root, args.max_hands,
            )
        targets, target_valid, sides = target_cache[stream_id][frame]
        image_path, _, _ = records[frame]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            category_counts["image_read_failure"] += 1
            continue
        results = pipeline.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        category, closest_distance, closest_index, _ = classify(
            item, results, targets, target_valid, sides,
            args.max_match_distance_px,
        )
        category_counts[category] += 1
        filename = (
            f"{case_index:03d}_{category}_{stream_id}_"
            f"f{frame:06d}_s{item['slot']}.jpg"
        )
        output_path = frames_dir / filename
        rendered = render_case(
            image, item, targets, target_valid, sides, results, category,
            closest_distance, closest_index,
        )
        cv2.imwrite(str(output_path), rendered)
        rendered_paths.append(output_path)
        cases.append({
            **item,
            "category": category,
            "raw_detections": len(results),
            "closest_distance_px": closest_distance,
            "closest_detection_index": closest_index,
            "image_path": image_path,
            "overlay_path": str(output_path),
        })

    make_contact_sheet(
        rendered_paths, out_dir / "contact_sheet.jpg", args.contact_columns,
    )
    target_count = cache_stats["target_instances"]
    report = {
        "windows": str(Path(args.windows).expanduser().resolve()),
        "visibility_root": str(visibility_root),
        "track_root": None if track_root is None else str(track_root),
        "cache_stats": dict(cache_stats),
        "cache_coverage": (
            cache_stats["cache_observed"] / target_count if target_count else 0.0
        ),
        "available_failure_candidates": len(failed),
        "available_control_candidates": len(controls),
        "audited_cases": len(cases),
        "category_counts": dict(category_counts),
        "cases": cases,
        "contact_sheet": str(out_dir / "contact_sheet.jpg"),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
