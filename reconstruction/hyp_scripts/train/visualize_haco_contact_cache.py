#!/usr/bin/env python3
"""Render per-frame HACO contact diagnostics from an existing cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--haco-root", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--contact-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backbone", default="hamer")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def frame_id(value: object) -> str:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (digits[-6:] if digits else text).zfill(6)


def expanded_xywh(box: np.ndarray, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(4)
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    width *= ratio
    height *= ratio
    return np.asarray(
        [
            center_x - width * 0.5,
            center_y - height * 0.5,
            width,
            height,
        ],
        dtype=np.float32,
    )


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    haco_root = Path(args.haco_root).expanduser().resolve()
    if not haco_root.is_dir():
        raise FileNotFoundError(haco_root)
    sys.path.insert(0, str(haco_root))
    os.chdir(haco_root)

    from lib.core.config import cfg, update_config

    update_config(
        backbone_type=args.backbone,
        exp_dir="experiments_contact_visualization",
    )
    from lib.utils.preprocessing import augmentation_contact
    from lib.utils.vis_utils import ContactRenderer

    query_path = Path(args.query_npz).expanduser().resolve()
    contact_path = Path(args.contact_npz).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    query = load_npz(query_path)
    contact = load_npz(contact_path)

    query_ids = [frame_id(value) for value in query["frame_ids"]]
    contact_ids = [frame_id(value) for value in contact["frame_ids"]]
    contact_lookup = {value: index for index, value in enumerate(contact_ids)}
    missing = [value for value in query_ids if value not in contact_lookup]
    if missing:
        raise KeyError(f"Contact cache lacks frames: {missing[:10]}")

    mirrored = bool(
        np.asarray(query["canonical_right_horizontal_mirror"]).item()
    )
    box_key = (
        "bbox_xyxy_canonical_right" if mirrored else "bbox_xyxy_original"
    )
    model_valid = np.asarray(query["model_valid"]).astype(bool)
    contact_valid = np.asarray(
        contact.get("contact_valid", np.ones(len(contact_ids), dtype=bool))
    ).astype(bool)
    masks = np.asarray(contact["contact_mask"]).astype(bool)
    probabilities = np.asarray(
        contact["contact_probability"], dtype=np.float32
    )
    renderer = ContactRenderer()

    rendered = 0
    invalid = 0
    skipped = 0
    for index in range(0, len(query_ids), args.stride):
        current_id = query_ids[index]
        output_path = output_dir / "contact" / f"{current_id}.png"
        detection_path = output_dir / "detection" / f"{current_id}.png"
        if (
            not args.overwrite
            and output_path.is_file()
            and detection_path.is_file()
        ):
            skipped += 1
            continue

        image_path = Path(str(query["image_paths"][index])).expanduser()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if mirrored:
            image = np.ascontiguousarray(image[:, ::-1])

        box = np.asarray(query[box_key][index], dtype=np.float32)
        crop_box = expanded_xywh(
            box, cfg.DATASET.ho_big_bbox_expand_ratio
        )
        crop, *_ = augmentation_contact(
            image, crop_box, "test", enforce_flip=False
        )
        contact_index = contact_lookup[current_id]
        finite = bool(np.isfinite(probabilities[contact_index]).all())
        valid = bool(
            model_valid[index] and contact_valid[contact_index] and finite
        )

        if valid:
            rendered_image = renderer.render_contact(
                crop[..., ::-1], masks[contact_index], mode="demo"
            )
            label = (
                f"{current_id} contact_vertices="
                f"{int(masks[contact_index].sum())}"
            )
            cv2.putText(
                rendered_image,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            rendered += 1
        else:
            rendered_image = cv2.resize(
                crop[..., ::-1], cfg.MODEL.input_img_shape, interpolation=cv2.INTER_CUBIC
            )
            cv2.putText(
                rendered_image,
                f"{current_id} HACO INVALID / NON-FINITE",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            invalid += 1
        write_image(output_path, rendered_image)

        detection = image[..., ::-1].copy()
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = (0, 255, 0) if valid else (0, 0, 255)
        cv2.rectangle(detection, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            detection,
            f"HACO input {current_id} valid={int(valid)}",
            (max(0, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        write_image(detection_path, detection)

    summary = {
        "stream_id": str(query["stream_id"].item()),
        "query_npz": str(query_path),
        "contact_npz": str(contact_path),
        "output_dir": str(output_dir),
        "frames": len(query_ids),
        "stride": args.stride,
        "rendered_valid_frames": rendered,
        "rendered_invalid_frames": invalid,
        "cached_frames": skipped,
        "mirrored_to_canonical_right": mirrored,
        "contact_dir": str(output_dir / "contact"),
        "detection_dir": str(output_dir / "detection"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
