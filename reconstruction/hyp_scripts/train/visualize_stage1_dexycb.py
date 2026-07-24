#!/usr/bin/env python3
"""Prepare and launch before/after Stage-1 DexYCB Viser viewers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--object-model-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--viewer-python", required=True)
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--original-port", type=int, default=8095)
    parser.add_argument("--corrected-port", type=int, default=8096)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relative_median(row: dict, stage: str) -> Optional[float]:
    value = (row.get("metrics") or {}).get(f"{stage}_relative", {}).get("median_mm")
    return float(value) if value is not None else None


def select_stream(
    records: dict[str, dict], summary: dict, requested_stream_id: Optional[str]
) -> tuple[dict, dict]:
    predictions = {
        row["stream_id"]: row
        for row in summary.get("streams", [])
        if row.get("stream_id")
    }
    if requested_stream_id:
        if requested_stream_id not in records:
            raise KeyError(f"Stream is not in manifest: {requested_stream_id}")
        if requested_stream_id not in predictions:
            raise KeyError(f"Stream has no Stage-1 prediction: {requested_stream_id}")
        return records[requested_stream_id], predictions[requested_stream_id]

    candidates = []
    for stream_id, prediction in predictions.items():
        if stream_id not in records:
            continue
        before = relative_median(prediction, "initial")
        after = relative_median(prediction, "corrected")
        if before is None or after is None or after >= before:
            continue
        candidates.append((abs(before - 29.822) + abs(after - 22.378), stream_id))
    if not candidates:
        raise RuntimeError(
            "No stream with improved relative median was found in prediction summary"
        )
    stream_id = min(candidates)[1]
    return records[stream_id], predictions[stream_id]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_frame_map(stream_dir: Path, frames_dir: Path, out_path: Path) -> None:
    images = sorted(stream_dir.glob("color_*.jpg"))
    if not images:
        images = sorted(stream_dir.glob("color_*.png"))
    if not images:
        raise RuntimeError(f"No color frames found in {stream_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_index, image_path in enumerate(images):
        original_frame = image_path.stem.split("_")[-1]
        destination = frames_dir / f"{output_index:06d}{image_path.suffix.lower()}"
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(image_path.resolve())
        rows.append(
            {
                "output_index": output_index,
                "original_frame": original_frame,
                "image_path": str(image_path.resolve()),
                "label_path": str(
                    (stream_dir / f"labels_{original_frame}.npz").resolve()
                ),
            }
        )
    out_path.write_text(
        json.dumps(
            {
                "stream_dir": str(stream_dir),
                "num_frames": len(rows),
                "frames": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_intrinsics(handflow_path: Path, foundationpose_path: Path) -> np.ndarray:
    with np.load(handflow_path, allow_pickle=True) as data:
        for key in ("K", "intrinsics", "camera_intrinsics"):
            if key not in data:
                continue
            value = np.asarray(data[key])
            if value.ndim == 3:
                value = value[0]
            if value.size == 9:
                return value.reshape(3, 3).astype(np.float64)
    value = np.asarray(load_json(foundationpose_path).get("intrinsics"))
    if value.size != 9:
        raise KeyError("No 3x3 intrinsics in HandFlow NPZ or FoundationPose JSON")
    return value.reshape(3, 3).astype(np.float64)


def launch(command: list[str], log_path: Path, pid_path: Path) -> int:
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return process.pid


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[3]
    manifest_path = Path(args.manifest).expanduser().resolve()
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    records = {row["stream_id"]: row for row in load_jsonl(manifest_path)}
    summary = load_json(prediction_root / "summary.json")
    record, prediction = select_stream(records, summary, args.stream_id)

    stream_id = record["stream_id"]
    stream_out = out_root / stream_id
    frames_dir = stream_out / "frames"
    frame_map = stream_out / "dexycb_frame_map.json"
    gt_out = stream_out / "gt"
    original_out = stream_out / "original"
    corrected_out = stream_out / "stage1_corrected"
    for path in (stream_out, frames_dir, gt_out, original_out, corrected_out):
        path.mkdir(parents=True, exist_ok=True)

    prediction_path = Path(
        prediction.get(
            "prediction",
            prediction_root / stream_id / "stage1_rigid_prediction.npz",
        )
    ).resolve()
    handflow_path = (handflow_root / stream_id / "handflow_camera_result.npz").resolve()
    foundationpose_path = Path(record["foundationpose_json"]).resolve()
    object_mesh = Path(record["sam3d_glb"]).resolve()
    gt_object_mesh = (
        Path(args.object_model_root).expanduser().resolve()
        / record["object_name"]
        / "textured_simple.obj"
    )
    selected = {
        **record,
        "stage1_prediction": str(prediction_path),
        "stage1_metrics": prediction.get("metrics", {}),
    }
    (stream_out / "selected_stream.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    required = (
        handflow_path,
        foundationpose_path,
        prediction_path,
        object_mesh,
        gt_object_mesh,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.force_prepare or not frame_map.is_file():
        prepare_frame_map(Path(record["stream_dir"]), frames_dir, frame_map)

    gt_hand = gt_out / "dexycb_gt_hand_meshes.npz"
    gt_layout = gt_out / "dexycb_gt_object_layout_camera_frame.json"
    if args.force_prepare or not gt_hand.is_file() or not gt_layout.is_file():
        run(
            [
                sys.executable,
                str(repository / "reconstruction/hyp_scripts/prepare_dexycb_gt_visualization.py"),
                "--frame-map-json",
                str(frame_map),
                "--mano-data-dir",
                str(Path(args.mano_data_dir).expanduser().resolve()),
                "--object-model-root",
                str(Path(args.object_model_root).expanduser().resolve()),
                "--out-dir",
                str(gt_out),
            ]
        )

    adapter = repository / (
        "reconstruction/hyp_scripts/"
        "prepare_foundationpose_handflow_visualization.py"
    )
    common = [
        sys.executable,
        str(adapter),
        "--foundationpose-json",
        str(foundationpose_path),
        "--frame-map-json",
        str(frame_map),
        "--handflow-npz",
        str(handflow_path),
        "--hand-side",
        record["hand_side"],
        "--invalid-hand-mode",
        "keep",
    ]
    original_layout = original_out / "foundationpose_layout_camera_frame.json"
    corrected_layout = corrected_out / "foundationpose_layout_camera_frame.json"
    if args.force_prepare or not original_layout.is_file():
        run(common + ["--out-dir", str(original_out)])
    if args.force_prepare or not corrected_layout.is_file():
        run(
            common
            + [
                "--stage1-prediction",
                str(prediction_path),
                "--out-dir",
                str(corrected_out),
            ]
        )

    intrinsics = load_intrinsics(handflow_path, foundationpose_path)
    camera = {
        "fx": float(intrinsics[0, 0]),
        "fy": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
    }
    (stream_out / "camera.json").write_text(
        json.dumps(camera, indent=2), encoding="utf-8"
    )

    before = relative_median(prediction, "initial")
    after = relative_median(prediction, "corrected")
    print(f"Selected: {stream_id}")
    print(f"Relative median: {before:.3f} -> {after:.3f} mm")
    print(f"Prepared: {stream_out}")
    if args.prepare_only:
        return

    viewer = repository / "reconstruction/scripts/visualize_3d.py"
    viewer_common = [
        str(Path(args.viewer_python).expanduser().resolve()),
        "-u",
        str(viewer),
        "--frames-dir",
        str(frames_dir),
        "--mesh",
        str(object_mesh),
        "--hands",
        record["hand_side"],
        "--scale",
        str(record["foundationpose_source_mesh_scale"]),
        "--translation-scale",
        "1.0",
        "--gt-hand-meshes",
        str(gt_hand),
        "--gt-object-layout-json",
        str(gt_layout),
        "--gt-object-mesh",
        str(gt_object_mesh),
        "--gt-object-scale",
        "1.0",
        "--fx",
        str(camera["fx"]),
        "--fy",
        str(camera["fy"]),
        "--cx",
        str(camera["cx"]),
        "--cy",
        str(camera["cy"]),
        "--width",
        "640",
        "--height",
        "480",
        "--fps",
        "30",
        "--frustum-scale",
        "0.15",
    ]
    original_command = viewer_common + [
        "--layout-json",
        str(original_layout),
        "--hand-meshes",
        str(original_out / "all_hand_meshes_handflow.npz"),
        "--port",
        str(args.original_port),
    ]
    corrected_command = viewer_common + [
        "--layout-json",
        str(corrected_layout),
        "--hand-meshes",
        str(corrected_out / "all_hand_meshes_handflow.npz"),
        "--port",
        str(args.corrected_port),
    ]
    original_pid = launch(
        original_command,
        stream_out / "viser_original.log",
        stream_out / "viser_original.pid",
    )
    corrected_pid = launch(
        corrected_command,
        stream_out / "viser_corrected.log",
        stream_out / "viser_corrected.pid",
    )
    print(f"Original PID={original_pid}: http://localhost:{args.original_port}")
    print(f"Corrected PID={corrected_pid}: http://localhost:{args.corrected_port}")


if __name__ == "__main__":
    main()
