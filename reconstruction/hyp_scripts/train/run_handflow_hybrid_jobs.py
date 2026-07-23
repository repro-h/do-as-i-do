#!/usr/bin/env python3
"""Run HandFlow on hybrid-manifest streams and cache compact camera-space results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--handflow-python", required=True)
    parser.add_argument("--fm-ckpt", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_intrinsics(pose_json: Path) -> np.ndarray:
    payload = load_json(pose_json)
    value = payload.get("intrinsics", payload.get("K"))
    if value is None:
        raise KeyError(f"No intrinsics/K in {pose_json}")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4,):
        fx, fy, cx, cy = matrix
        matrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid intrinsics shape/value in {pose_json}: {matrix.shape}")
    return matrix


def frame_paths(stream_dir: Path) -> list[Path]:
    paths = sorted(stream_dir.glob("color_*.jpg"))
    if not paths:
        paths = sorted(stream_dir.glob("color_*.png"))
    if not paths:
        raise FileNotFoundError(f"No color frames in {stream_dir}")
    return paths


def write_video(paths: list[Path], path: Path, mirror: bool, fps: float = 30.0) -> tuple[int, int]:
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(paths[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temporary video: {path}")
    try:
        for frame_path in paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(frame_path)
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Frame dimensions changed at {frame_path}")
            writer.write(cv2.flip(frame, 1) if mirror else frame)
    finally:
        writer.release()
    return width, height


def scalar_text(value) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else array.tolist())


def adapt_result(
    raw_path: Path,
    out_path: Path,
    record: dict,
    original_intrinsics: np.ndarray,
    mirrored_left: bool,
) -> dict:
    with np.load(raw_path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}

    if "verts_cam" not in payload or "faces" not in payload:
        raise KeyError(f"HandFlow output lacks verts_cam/faces: {raw_path}")
    vertices = np.asarray(payload["verts_cam"], dtype=np.float32).copy()
    faces = np.asarray(payload["faces"], dtype=np.int64).copy()
    if mirrored_left:
        vertices[..., 0] *= -1.0
        faces = faces[..., [0, 2, 1]]

    valid = np.asarray(
        payload.get("pred_valid", np.ones(vertices.shape[0], dtype=bool))
    ).astype(bool)
    centers = np.nanmean(vertices, axis=1).astype(np.float32)
    output = {
        "verts_cam": vertices,
        "faces": faces,
        "pred_valid": valid,
        "hand_center_cam": centers,
        "intrinsics": original_intrinsics.astype(np.float32),
        "hand_side": np.asarray(record["hand_side"]),
        "stream_id": np.asarray(record["stream_id"]),
        "object_name": np.asarray(record["object_name"]),
        "mirrored_left_input": np.asarray(mirrored_left),
        "source": np.asarray("handflow_hybrid_cache_v1"),
    }
    # Keep compact MANO parameters for right/direct runs. Mirrored right-hand
    # parameters are not valid left-MANO parameters, so they are explicitly raw.
    for key in ("pose", "trans", "betas"):
        if key in payload:
            output[f"handflow_raw_{key}"] = payload[key]
    np.savez_compressed(out_path, **output)
    return {
        "num_frames": int(vertices.shape[0]),
        "num_valid": int(valid.sum()),
        "num_vertices": int(vertices.shape[1]),
        "mirrored_left_input": mirrored_left,
        "raw_side": scalar_text(payload.get("side", "")),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    handflow_python = Path(args.handflow_python).expanduser().resolve()
    checkpoint = Path(args.fm_ckpt).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    status_path = Path(args.status_json).expanduser().resolve()
    demo_path = handflow_root / "scripts" / "demo.py"
    for path in (manifest_path, handflow_python, checkpoint, demo_path):
        if not path.exists():
            raise FileNotFoundError(path)

    records = load_jsonl(manifest_path)
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    records = records[args.shard_index :: args.num_shards]
    if args.limit > 0:
        records = records[: args.limit]
    out_root.mkdir(parents=True, exist_ok=True)
    state = {
        "manifest": str(manifest_path),
        "out_root": str(out_root),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_requested": len(records),
        "completed": {},
        "failed": {},
    }

    for index, record in enumerate(records, start=1):
        stream_id = record["stream_id"]
        stream_out = out_root / stream_id
        result_path = stream_out / "handflow_camera_result.npz"
        log_path = stream_out / "handflow.log"
        if result_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(records)}] {stream_id}: cached")
            state["completed"][stream_id] = {
                "result": str(result_path),
                "cached": True,
            }
            write_json_atomic(status_path, state)
            continue

        stream_out.mkdir(parents=True, exist_ok=True)
        raw_result = stream_out / "handflow_raw_result.npz"
        pose_json = Path(record["foundationpose_json"])
        stream_dir = Path(record["stream_dir"])
        intrinsics = resolve_intrinsics(pose_json)
        is_left = record["hand_side"] == "left"
        print(
            f"[{index}/{len(records)}] {stream_id} "
            f"side={record['hand_side']} object={record['object_name']}"
        )
        try:
            paths = frame_paths(stream_dir)
            with tempfile.TemporaryDirectory(prefix="handflow_hybrid_") as temporary:
                video_path = Path(temporary) / "input.mp4"
                width, _ = write_video(paths, video_path, mirror=is_left)
                run_intrinsics = intrinsics.copy()
                if is_left:
                    run_intrinsics[0, 2] = (width - 1) - run_intrinsics[0, 2]
                fx, fy = run_intrinsics[0, 0], run_intrinsics[1, 1]
                cx, cy = run_intrinsics[0, 2], run_intrinsics[1, 2]
                command = [
                    str(handflow_python),
                    "-u",
                    str(demo_path),
                    "--input",
                    str(video_path),
                    "--fm_ckpt",
                    str(checkpoint),
                    "--intrinsics",
                    f"{fx},{fy},{cx},{cy}",
                    "--fix_camera",
                    "--side",
                    "right",
                    "--output_dir",
                    str(stream_out),
                    "--save_npz",
                    str(raw_result),
                    "--device",
                    args.device,
                ]
                with log_path.open("w", encoding="utf-8") as log:
                    log.write("command: " + " ".join(command) + "\n")
                    log.flush()
                    subprocess.run(
                        command,
                        cwd=handflow_root,
                        env=os.environ.copy(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            metrics = adapt_result(
                raw_result,
                result_path,
                record,
                intrinsics,
                mirrored_left=is_left,
            )
            if not args.keep_raw and raw_result.is_file():
                raw_result.unlink()
            if not args.keep_videos:
                for name in ("overlay.mp4", "ortho.mp4"):
                    video = stream_out / name
                    if video.is_file():
                        video.unlink()
            state["completed"][stream_id] = {
                "result": str(result_path),
                "log": str(log_path),
                **metrics,
            }
            print(f"  done: {result_path}")
        except Exception as error:
            state["failed"][stream_id] = {
                "error": f"{type(error).__name__}: {error}",
                "log": str(log_path),
            }
            print(f"  failed: {type(error).__name__}: {error}")
        write_json_atomic(status_path, state)

    state["num_completed"] = len(state["completed"])
    state["num_failed"] = len(state["failed"])
    write_json_atomic(status_path, state)
    print(
        json.dumps(
            {
                "num_requested": state["num_requested"],
                "num_completed": state["num_completed"],
                "num_failed": state["num_failed"],
                "status_json": str(status_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
