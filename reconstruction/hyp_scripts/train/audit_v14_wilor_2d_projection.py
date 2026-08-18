#!/usr/bin/env python3
"""Overlay DexYCB GT, WiLoR queries, and V14-projected hand joints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--dense-root")
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--out-image", required=True)
    parser.add_argument("--out-json")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def frame_text(value: object) -> str:
    text = str(value)
    return text.zfill(6) if text.isdigit() else text


def index_by_frame(frame_ids: np.ndarray, requested: str) -> int:
    normalized = [frame_text(value) for value in frame_ids]
    if requested not in normalized:
        raise KeyError(f"Frame {requested} not found")
    return normalized.index(requested)


def load_intrinsics(
    args: argparse.Namespace,
    query: dict[str, np.ndarray],
    stream_id: str,
) -> tuple[np.ndarray, str]:
    if "intrinsics" in query:
        matrix = np.asarray(query["intrinsics"], dtype=np.float32)
        return matrix[0] if matrix.ndim == 3 else matrix.reshape(3, 3), "query"
    if args.dense_root:
        candidates = sorted(
            (Path(args.dense_root) / stream_id / "windows").glob("*.npz")
        )
        if not candidates:
            raise FileNotFoundError(f"No dense windows for {stream_id}")
        dense = load_npz(candidates[0])
        matrix = np.asarray(dense["intrinsics_resized"], dtype=np.float32)
        matrix = (matrix[0] if matrix.ndim == 3 else matrix).reshape(3, 3).copy()
        resized_wh = np.asarray(dense["resized_wh"], dtype=np.float32).reshape(2)
        original_wh = np.asarray(query["image_wh"], dtype=np.float32).reshape(-1, 2)[0]
        matrix[0] *= original_wh[0] / resized_wh[0]
        matrix[1] *= original_wh[1] / resized_wh[1]
        return matrix, str(candidates[0])
    if args.intrinsics:
        fx, fy, cx, cy = args.intrinsics
        return np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32), "cli"
    raise KeyError("Pass --dense-root or --intrinsics FX FY CX CY")


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = points[:, 2:3]
    pixels = points @ intrinsics.T
    pixels = pixels[:, :2] / np.maximum(depth, 1e-8)
    pixels[depth[:, 0] <= 1e-6] = np.nan
    return pixels


def draw_skeleton(
    image: np.ndarray,
    joints: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for first, second in EDGES:
        if valid[first] and valid[second]:
            cv2.line(
                image, tuple(np.rint(joints[first]).astype(int)),
                tuple(np.rint(joints[second]).astype(int)), color, thickness,
                cv2.LINE_AA,
            )
    for index, point in enumerate(joints):
        if valid[index]:
            cv2.circle(
                image, tuple(np.rint(point).astype(int)),
                3 if index else 5, color, -1, cv2.LINE_AA,
            )


def metrics(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    error = np.linalg.norm(predicted - target, axis=-1)[valid]
    return {
        "count": int(len(error)),
        "median_px": float(np.median(error)),
        "p90_px": float(np.percentile(error, 90)),
        "max_px": float(error.max()),
        "wrist_px": float(np.linalg.norm(predicted[0] - target[0])),
    }


def main() -> None:
    args = parse_args()
    query_path = Path(args.query_npz).expanduser().resolve()
    query = load_npz(query_path)
    trajectory = load_npz(Path(args.trajectory_npz).expanduser().resolve())
    requested = frame_text(args.frame_id)
    query_index = index_by_frame(query["frame_ids"], requested)
    trajectory_index = index_by_frame(trajectory["frame_ids"], requested)
    stream_id = str(np.asarray(query["stream_id"]).item())

    image_path = Path(str(query["image_paths"][query_index])).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    label_path = image_path.with_name(
        image_path.name.replace("color_", "labels_")
    ).with_suffix(".npz")
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    label = load_npz(label_path)
    gt = np.asarray(label["joint_2d"], dtype=np.float32).reshape(21, 2)
    gt_valid = np.isfinite(gt).all(axis=-1) & ~np.all(np.isclose(gt, -1), axis=-1)

    wilor = np.asarray(query["joints_uv_full_original"][query_index], dtype=np.float32)
    relative = np.asarray(
        query["joints_3d_root_relative_original"][query_index], dtype=np.float32
    )
    wrist = np.asarray(
        trajectory["predicted_wrist_camera"][trajectory_index], dtype=np.float32
    )
    intrinsics, intrinsics_source = load_intrinsics(args, query, stream_id)
    v14 = project(relative + wrist[None], intrinsics)
    valid = gt_valid & np.isfinite(wilor).all(axis=-1) & np.isfinite(v14).all(axis=-1)

    draw_skeleton(image, gt, valid, (0, 210, 0), 3)
    draw_skeleton(image, wilor, valid, (0, 0, 255), 2)
    draw_skeleton(image, v14, valid, (255, 100, 0), 2)
    lines = [
        "GT green | WiLoR red | V14 blue",
        f"WiLoR median {metrics(wilor, gt, valid)['median_px']:.2f}px",
        f"V14 median {metrics(v14, gt, valid)['median_px']:.2f}px",
    ]
    for line_index, text in enumerate(lines):
        position = (12, 26 + line_index * 23)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    output = Path(args.out_image).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to write {output}")
    summary = {
        "frame_id": requested,
        "image": str(image_path),
        "label": str(label_path),
        "intrinsics_source": intrinsics_source,
        "wilor_vs_gt": metrics(wilor, gt, valid),
        "v14_vs_gt": metrics(v14, gt, valid),
        "wilor_vs_v14": metrics(wilor, v14, valid),
        "output": str(output),
    }
    if args.out_json:
        json_path = Path(args.out_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
