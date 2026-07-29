#!/usr/bin/env python3
"""Export Pi3X geometry feature caches for a small manifest pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-script", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--feature-dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_token(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1].zfill(6)


def prepare_frame_map(record: dict, out_path: Path) -> tuple[Path, int]:
    stream_dir = Path(record["stream_dir"]).expanduser().resolve()
    images = sorted(stream_dir.glob("color_*.jpg")) or sorted(
        stream_dir.glob("color_*.png")
    )
    if not images:
        raise FileNotFoundError(f"No RGB frames in {stream_dir}")

    meta_path = stream_dir.parent / "meta.yml"
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}
    ycb_ids = list(metadata.get("ycb_ids", []) or [])
    grasp_index = int(metadata.get("ycb_grasp_ind", 0))
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(
            f"Invalid ycb_grasp_ind={grasp_index} for {meta_path}"
        )
    object_label = int(ycb_ids[grasp_index])

    frames = []
    for output_index, image_path in enumerate(images):
        token = frame_token(image_path)
        label_path = stream_dir / f"labels_{token}.npz"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        frames.append(
            {
                "output_index": output_index,
                "output_frame": f"{output_index:06d}",
                "original_frame": token,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
            }
        )
    payload = {
        "source": "dexycb_manifest_pilot",
        "stream_id": record["stream_id"],
        "stream_dir": str(stream_dir),
        "object_name": record.get("object_name"),
        "hand_side": record.get("hand_side"),
        "target_dexycb_class_id": object_label,
        "num_frames": len(frames),
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path, object_label


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    export_script = Path(args.export_script).expanduser().resolve()
    rows = load_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No records in {manifest_path}")

    completed = []
    failures = []
    status = {
        "manifest": str(manifest_path),
        "num_requested": len(rows),
        "num_completed": 0,
        "num_failed": 0,
        "completed": completed,
        "failures": failures,
    }
    for index, record in enumerate(rows):
        stream_id = record["stream_id"]
        stream_out = out_root / stream_id
        summary_path = stream_out / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            print(
                f"[{index + 1}/{len(rows)}] cached {stream_id}",
                flush=True,
            )
            completed.append(stream_id)
        else:
            try:
                frame_map_path, object_label = prepare_frame_map(
                    record,
                    stream_out / "dexycb_frame_map.json",
                )
                hand_path = (
                    handflow_root / stream_id / "handflow_camera_result.npz"
                )
                if not hand_path.is_file():
                    raise FileNotFoundError(hand_path)
                command = [
                    sys.executable,
                    "-u",
                    str(export_script),
                    "--frame-map-json",
                    str(frame_map_path),
                    "--hand-npz",
                    str(hand_path),
                    "--hand-uni-root",
                    args.hand_uni_root,
                    "--pi3-root",
                    args.pi3_root,
                    "--checkpoint",
                    args.checkpoint,
                    "--out-dir",
                    str(stream_out),
                    "--object-label",
                    str(object_label),
                    "--window-size",
                    str(args.window_size),
                    "--window-stride",
                    str(args.window_stride),
                    "--pixel-limit",
                    str(args.pixel_limit),
                    "--confidence-threshold",
                    str(args.confidence_threshold),
                    "--feature-dtype",
                    args.feature_dtype,
                    "--device",
                    args.device,
                ]
                if args.overwrite:
                    command.append("--overwrite")
                print(
                    f"[{index + 1}/{len(rows)}] {stream_id} "
                    f"object_label={object_label}",
                    flush=True,
                )
                subprocess.run(command, check=True)
                completed.append(stream_id)
            except Exception as error:
                failures.append(
                    {"stream_id": stream_id, "error": repr(error)}
                )
                print(f"FAILED {stream_id}: {error}", flush=True)

        status["num_completed"] = len(completed)
        status["num_failed"] = len(failures)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "status.json").write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
