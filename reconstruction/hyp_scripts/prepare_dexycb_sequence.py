#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def _mask_geometry(mask: np.ndarray) -> dict[str, float]:
    mask_bool = np.asarray(mask).astype(bool)
    height, width = mask_bool.shape
    area_pixels = int(mask_bool.sum())
    if area_pixels == 0:
        return {
            "area_ratio": 0.0,
            "touches_border": 1.0,
            "largest_component_ratio": 0.0,
            "border_clearance_ratio": 0.0,
        }
    ys, xs = np.nonzero(mask_bool)
    clearance = min(xs.min(), ys.min(), width - 1 - xs.max(), height - 1 - ys.max())
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), 8)
    largest = (
        float(stats[1:, cv2.CC_STAT_AREA].max()) / float(area_pixels)
        if num_labels > 1
        else 0.0
    )
    return {
        "area_ratio": float(area_pixels / max(1, height * width)),
        "touches_border": float(clearance <= 0),
        "largest_component_ratio": largest,
        "border_clearance_ratio": float(max(0, clearance) / max(1.0, min(height, width))),
    }


def score_hand_object_anchor_frame(
    object_mask: np.ndarray,
    hand_mask: np.ndarray,
    contact_radius_px: int,
) -> dict[str, float]:
    obj = np.asarray(object_mask).astype(bool)
    hand = np.asarray(hand_mask).astype(bool)
    if obj.shape != hand.shape or obj.ndim != 2:
        raise ValueError(f"Expected matching 2D masks, got object={obj.shape}, hand={hand.shape}")
    obj_stats = _mask_geometry(obj)
    hand_stats = _mask_geometry(hand)
    kernel_size = max(1, int(contact_radius_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    obj_boundary = obj & ~cv2.erode(
        obj.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    hand_near = cv2.dilate(hand.astype(np.uint8), kernel, iterations=1).astype(bool)
    contact_ratio = float((obj_boundary & hand_near).sum() / max(1, int(obj_boundary.sum())))
    return {
        "object_area_ratio": obj_stats["area_ratio"],
        "hand_area_ratio": hand_stats["area_ratio"],
        "object_touches_border": obj_stats["touches_border"],
        "hand_touches_border": hand_stats["touches_border"],
        "object_largest_component_ratio": obj_stats["largest_component_ratio"],
        "object_border_clearance_ratio": obj_stats["border_clearance_ratio"],
        "hand_object_boundary_contact_ratio": contact_ratio,
    }


def rank_hand_object_anchor_rows(
    rows: list[dict],
    min_object_area_ratio: float,
    min_hand_area_ratio: float,
) -> list[dict]:
    if not rows:
        return []

    def ranks(key: str) -> np.ndarray:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.linspace(0.0, 1.0, len(values)) if len(values) > 1 else 1.0
        return result

    object_area_rank = ranks("object_area_ratio")
    hand_area_rank = ranks("hand_area_ratio")
    ranked: list[dict] = []
    for index, source in enumerate(rows):
        row = dict(source)
        valid = (
            row["object_area_ratio"] >= min_object_area_ratio
            and row["hand_area_ratio"] >= min_hand_area_ratio
            and row["object_touches_border"] < 0.5
        )
        score = (
            1.50 * object_area_rank[index]
            + 0.35 * hand_area_rank[index]
            + 0.50 * row["object_largest_component_ratio"]
            + 0.30 * row["object_border_clearance_ratio"]
            - 1.20 * row["hand_object_boundary_contact_ratio"]
            - 1.00 * row["object_touches_border"]
            - 0.25 * row["hand_touches_border"]
        )
        row["object_area_rank"] = float(object_area_rank[index])
        row["hand_area_rank"] = float(hand_area_rank[index])
        row["valid_anchor_candidate"] = float(valid)
        row["anchor_score"] = float(score if valid else score - 10.0)
        ranked.append(row)
    return sorted(ranked, key=lambda row: row["anchor_score"], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one DexYCB camera stream for do-as-i-do without running SAM3."
    )
    parser.add_argument("--stream_dir", required=True)
    parser.add_argument("--object_model_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--init_frame", default=None, help="Original DexYCB frame token.")
    parser.add_argument("--contact_radius_px", type=int, default=7)
    parser.add_argument("--min_object_area_ratio", type=float, default=0.005)
    parser.add_argument("--min_hand_area_ratio", type=float, default=0.002)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_object_names(model_root: Path) -> dict[int, str]:
    numbered: list[tuple[int, str]] = []
    for path in sorted(model_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            numbered.append((int(path.name.split("_", 1)[0]), path.name))
        except ValueError:
            continue
    numbered.sort()
    if not numbered:
        raise RuntimeError(f"No numbered YCB model directories found in {model_root}")
    # DexYCB uses contiguous class IDs; YCB model directories use original IDs.
    return {class_id: name for class_id, (_, name) in enumerate(numbered, start=1)}


def frame_token(path: Path) -> str:
    if not path.stem.startswith("color_"):
        raise ValueError(f"Unexpected DexYCB image name: {path.name}")
    return path.stem.split("_", 1)[1]


def load_segmentation(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "seg" not in payload:
            raise KeyError(f"Missing seg in {path}")
        seg = np.asarray(payload["seg"])
    if seg.ndim == 3 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim != 2:
        raise ValueError(f"Expected 2D segmentation in {path}, got {seg.shape}")
    return seg


def main() -> None:
    args = parse_args()
    stream_dir = Path(args.stream_dir).expanduser().resolve()
    model_root = Path(args.object_model_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    meta_path = stream_dir.parent / "meta.yml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing DexYCB metadata: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle) or {}
    ycb_ids = list(meta.get("ycb_ids", []) or [])
    grasp_index = int(meta.get("ycb_grasp_ind", 0))
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(f"Invalid ycb_grasp_ind={grasp_index} for ycb_ids={ycb_ids}")
    target_id = int(ycb_ids[grasp_index])
    object_names = load_object_names(model_root)
    if target_id not in object_names:
        raise KeyError(f"DexYCB class ID {target_id} has no model directory in {model_root}")
    object_name = object_names[target_id]

    images = sorted(stream_dir.glob("color_*.jpg")) or sorted(stream_dir.glob("color_*.png"))
    selected_images: list[Path] = []
    stride = max(1, int(args.frame_stride))
    for image in images:
        value = int(frame_token(image))
        if args.start_frame is not None and value < args.start_frame:
            continue
        if args.end_frame is not None and value > args.end_frame:
            continue
        selected_images.append(image)
    selected_images = selected_images[::stride]
    if not selected_images:
        raise RuntimeError(f"No selected color frames in {stream_dir}")

    frames_dir = out_dir / "all_frames"
    masks_root = out_dir / "video_segmentation" / "masks"
    if out_dir.exists() and not args.overwrite and (out_dir / "dexycb_frame_map.json").exists():
        raise FileExistsError(f"Output already prepared: {out_dir}; pass --overwrite to replace files")
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    frame_map: list[dict] = []
    first_image = cv2.imread(str(selected_images[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(f"Cannot read image: {selected_images[0]}")
    height, width = first_image.shape[:2]
    video_path = out_dir / "dexycb_sequence.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")

    try:
        for output_index, image_path in enumerate(selected_images):
            original_frame = frame_token(image_path)
            label_path = stream_dir / f"labels_{original_frame}.npz"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for selected frame: {label_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            if image.shape[:2] != (height, width):
                raise ValueError(f"Frame size changed at {image_path}: {image.shape[:2]}")
            seg = load_segmentation(label_path)
            object_mask = np.asarray(seg == target_id, dtype=np.uint8)
            hand_mask = np.asarray(seg == 255, dtype=np.uint8)

            output_frame = f"{output_index:06d}"
            frame_path = frames_dir / f"{output_frame}.png"
            root_frame_path = out_dir / f"{output_index:04d}.png"
            mask_dir = masks_root / f"frame_{output_frame}_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{object_name}.png"
            cv2.imwrite(str(frame_path), image)
            cv2.imwrite(str(root_frame_path), image)
            cv2.imwrite(str(mask_path), object_mask * 255)
            writer.write(image)

            metrics = score_hand_object_anchor_frame(
                object_mask.astype(bool),
                hand_mask.astype(bool),
                contact_radius_px=args.contact_radius_px,
            )
            row = {
                "frame": output_frame,
                "output_index": output_index,
                "original_frame": original_frame,
                "image_path": str(image_path),
                "label_path": str(label_path),
                **metrics,
            }
            rows.append(row)
            frame_map.append(
                {
                    "output_index": output_index,
                    "output_frame": output_frame,
                    "original_frame": original_frame,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "mask_path": str(mask_path),
                }
            )
    finally:
        writer.release()

    ranked = rank_hand_object_anchor_rows(
        rows,
        min_object_area_ratio=args.min_object_area_ratio,
        min_hand_area_ratio=args.min_hand_area_ratio,
    )
    valid = [row for row in ranked if row["valid_anchor_candidate"] > 0.5]
    if not valid:
        raise RuntimeError("No valid initialization frame after preparing the stream")
    if args.init_frame is None:
        init_row = valid[0]
    else:
        matches = [row for row in rows if row["original_frame"] == str(args.init_frame).zfill(6)]
        if not matches:
            raise ValueError(f"Requested original init frame is not selected: {args.init_frame}")
        init_row = matches[0]

    config = {
        "frame_number": int(init_row["output_index"]),
        "object_names": [object_name],
        "anchor_hand": str((meta.get("mano_sides") or ["right"])[0]),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    payload = {
        "source": "dexycb_ground_truth_segmentation",
        "stream_dir": str(stream_dir),
        "video_path": str(video_path),
        "object_name": object_name,
        "target_dexycb_class_id": target_id,
        "num_frames": len(frame_map),
        "fps": float(args.fps),
        "frame_stride": stride,
        "init_frame": init_row,
        "top_init_candidates": valid[:10],
        "frames": frame_map,
    }
    map_path = out_dir / "dexycb_frame_map.json"
    map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**config, "frame_map": str(map_path), "video": str(video_path)}, indent=2))


if __name__ == "__main__":
    main()
