#!/usr/bin/env python3
"""Audit visible-hand metric depth without reading object pixels as hand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--mask-erode-px", type=int, default=3)
    parser.add_argument("--max-rays", type=int, default=2048)
    parser.add_argument("--min-visible-pixels", type=int, default=64)
    parser.add_argument("--min-ray-hits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_frame_map(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload.get("frame_map", payload))
    if isinstance(rows, dict):
        rows = list(rows.values())
    return sorted(rows, key=lambda row: int(row["output_index"]))


def load_segmentation(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("seg", "segmentation", "label"):
            if key in payload.files:
                return np.asarray(payload[key])
    raise KeyError(f"No segmentation array in {path}")


def depth_path(label_path: Path, frame: str) -> Path | None:
    for name in (
        f"aligned_depth_to_color_{frame}.png",
        f"depth_{frame}.png",
        f"aligned_depth_{frame}.png",
    ):
        candidate = label_path.parent / name
        if candidate.is_file():
            return candidate
    return None


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    size = pixels * 2 + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def raycast(
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    max_rays: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) > max_rays:
        selected = rng.choice(len(xs), size=max_rays, replace=False)
        xs, ys = xs[selected], ys[selected]
    if not len(xs):
        return np.empty((0, 3)), xs, ys
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    directions = np.stack(
        [(xs - cx) / fx, (ys - cy) / fy, np.ones(len(xs))], axis=1
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hits, ray_ids, _ = mesh.ray.intersects_location(
        np.zeros_like(directions), directions, multiple_hits=False
    )
    return hits, xs[ray_ids], ys[ray_ids]


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(hand_path, allow_pickle=False) as payload:
        vertices = np.asarray(payload["verts_cam"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
        valid = np.asarray(
            payload["pred_valid"]
            if "pred_valid" in payload.files
            else np.ones(len(vertices)),
            dtype=bool,
        )
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None], len(vertices), axis=0)
    if intrinsics.shape[0] == 1:
        intrinsics = np.repeat(intrinsics, len(vertices), axis=0)

    rows = load_frame_map(frame_map_path)
    rng = np.random.default_rng(args.seed)
    records = []
    for row in rows:
        index = int(row["output_index"])
        original_frame = str(row["original_frame"]).zfill(6)
        record = {
            "output_index": index,
            "frame": f"{index:06d}",
            "original_frame": original_frame,
            "status": "invalid",
        }
        try:
            if index >= len(vertices) or not valid[index]:
                raise ValueError("invalid_hand")
            label_path = Path(row["label_path"]).expanduser().resolve()
            selected_depth = depth_path(label_path, original_frame)
            if selected_depth is None:
                raise ValueError("missing_depth")
            segmentation = load_segmentation(label_path)
            depth_raw = cv2.imread(str(selected_depth), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise ValueError("cannot_read_depth")
            if depth_raw.shape != segmentation.shape:
                depth_raw = cv2.resize(
                    depth_raw,
                    (segmentation.shape[1], segmentation.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            depth = depth_raw.astype(np.float64) * args.depth_scale
            visible_mask = erode(
                (segmentation == 255) & np.isfinite(depth) & (depth > 0),
                args.mask_erode_px,
            )
            visible_pixels = int(visible_mask.sum())
            if visible_pixels < args.min_visible_pixels:
                raise ValueError(f"visible_pixels_{visible_pixels}")

            hits, hit_u, hit_v = raycast(
                vertices[index],
                faces,
                visible_mask,
                intrinsics[min(index, len(intrinsics) - 1)],
                args.max_rays,
                rng,
            )
            if len(hits) < args.min_ray_hits:
                raise ValueError(f"ray_hits_{len(hits)}")
            observed_depth = depth[hit_v, hit_u]
            residual_mm = (observed_depth - hits[:, 2]) * 1000.0
            finite = np.isfinite(residual_mm)
            residual_mm = residual_mm[finite]
            if len(residual_mm) < args.min_ray_hits:
                raise ValueError(f"valid_depth_hits_{len(residual_mm)}")

            median = float(np.median(residual_mm))
            mad = float(np.median(np.abs(residual_mm - median)))
            inlier = np.abs(residual_mm - median) <= max(3.0 * mad, 3.0)
            inlier_residual = residual_mm[inlier]
            if len(inlier_residual) < args.min_ray_hits:
                raise ValueError(f"depth_inliers_{len(inlier_residual)}")
            visible_reference = max(
                visible_pixels,
                int(np.quantile([visible_pixels], 0.9)),
            )
            confidence = min(1.0, len(inlier_residual) / args.max_rays)
            confidence *= float(np.exp(-mad / 20.0))
            record.update(
                {
                    "status": "ok",
                    "depth_path": str(selected_depth),
                    "visible_hand_pixels": visible_pixels,
                    "ray_hits": int(len(hits)),
                    "depth_inliers": int(len(inlier_residual)),
                    "depth_residual_mm": distribution(inlier_residual),
                    "recommended_camera_z_mm": float(
                        np.median(inlier_residual)
                    ),
                    "depth_mad_mm": mad,
                    "confidence": confidence,
                }
            )
        except Exception as error:
            record["reason"] = str(error)
        records.append(record)
        print(
            f"[{index + 1}/{len(rows)}] {record['frame']} {record['status']}"
            + (
                f" dz={record['recommended_camera_z_mm']:+.2f}mm"
                f" visible={record['visible_hand_pixels']}"
                f" confidence={record['confidence']:.3f}"
                if record["status"] == "ok"
                else f" reason={record.get('reason')}"
            )
        )

    valid_records = [row for row in records if row["status"] == "ok"]
    visible_counts = np.asarray(
        [row["visible_hand_pixels"] for row in valid_records], dtype=float
    )
    reference_pixels = (
        float(np.quantile(visible_counts, 0.9)) if len(visible_counts) else 0.0
    )
    for row in records:
        row["visible_fraction"] = (
            min(1.0, row.get("visible_hand_pixels", 0) / reference_pixels)
            if reference_pixels > 0
            else 0.0
        )
        if row["status"] == "ok":
            row["confidence"] *= row["visible_fraction"]

    output = {
        "hand_npz": str(hand_path),
        "frame_map_json": str(frame_map_path),
        "settings": vars(args),
        "num_frames": len(records),
        "num_valid_depth_frames": len(valid_records),
        "visible_reference_pixels": reference_pixels,
        "recommended_camera_z_mm": distribution(
            [
                row["recommended_camera_z_mm"]
                for row in records
                if row["status"] == "ok"
            ]
        ),
        "frames": records,
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "frames"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
