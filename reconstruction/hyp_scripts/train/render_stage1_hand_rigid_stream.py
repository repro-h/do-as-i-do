#!/usr/bin/env python3
"""Prepare and render one hand-rigid validation stream as a 3D triptych."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--filtered-object-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--object-model-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--corrected-hand-npz",
        help="Optional corrected hand NPZ; overrides --prediction-root lookup.",
    )
    parser.add_argument(
        "--corrected-label",
        default="Stage1 Hand + Filtered Object",
    )
    parser.add_argument(
        "--gt-python",
        default=sys.executable,
        help="Python environment used to decode DexYCB MANO ground truth.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--four-view-grid",
        action="store_true",
        help=(
            "Render camera, side, rear, and top rows with columns ordered "
            "as raw FoundationPose, DexYCB GT, and final correction."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(command: list[str], expected: list[Path]) -> None:
    if expected and all(path.is_file() for path in expected):
        return
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Command did not create: {missing}")


def prepare_frame_map(stream_dir: Path, frames_dir: Path, path: Path) -> dict:
    images = sorted(stream_dir.glob("color_*.jpg")) or sorted(
        stream_dir.glob("color_*.png")
    )
    if not images:
        raise FileNotFoundError(f"No color frames in {stream_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_index, image in enumerate(images):
        frame = image.stem.rsplit("_", 1)[-1].zfill(6)
        destination = frames_dir / f"{output_index:06d}{image.suffix.lower()}"
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(image.resolve())
        rows.append(
            {
                "output_index": output_index,
                "original_frame": frame,
                "image_path": str(image.resolve()),
                "label_path": str(
                    (stream_dir / f"labels_{frame}.npz").resolve()
                ),
            }
        )
    payload = {
        "stream_dir": str(stream_dir),
        "num_frames": len(rows),
        "frames": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def stack_view_triptychs(
    sources: list[tuple[str, Path]],
    output: Path,
    fps: float,
) -> None:
    captures = [cv2.VideoCapture(str(path)) for _, path in sources]
    try:
        if not all(capture.isOpened() for capture in captures):
            raise RuntimeError(f"Cannot open one of: {[str(path) for _, path in sources]}")
        widths = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) for capture in captures]
        heights = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) for capture in captures]
        if len(set(widths)) != 1 or len(set(heights)) != 1:
            raise RuntimeError(f"View video dimensions differ: {list(zip(widths, heights))}")
        width, height = widths[0], heights[0]
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height * len(captures)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {output}")
        try:
            frame_index = 0
            while True:
                rows = []
                for (label, _), capture in zip(sources, captures):
                    ok, frame = capture.read()
                    if not ok:
                        rows = []
                        break
                    cv2.putText(
                        frame,
                        label,
                        (16, 66),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (40, 40, 40),
                        2,
                        cv2.LINE_AA,
                    )
                    rows.append(frame)
                if not rows:
                    break
                writer.write(np.concatenate(rows, axis=0))
                frame_index += 1
            if frame_index == 0:
                raise RuntimeError("No frames were stacked")
        finally:
            writer.release()
    finally:
        for capture in captures:
            capture.release()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[3]
    records = {
        row["stream_id"]: row
        for row in load_jsonl(Path(args.manifest).expanduser().resolve())
    }
    if args.stream_id not in records:
        raise KeyError(f"Stream is not in manifest: {args.stream_id}")
    record = records[args.stream_id]
    handflow_root = Path(args.handflow_root).expanduser().resolve()
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    filtered_root = Path(args.filtered_object_root).expanduser().resolve()
    stream_out = (
        Path(args.out_root).expanduser().resolve() / args.stream_id
    )
    frames_dir = stream_out / "frames"
    frame_map = stream_out / "dexycb_frame_map.json"
    original_out = stream_out / "original"
    corrected_out = stream_out / "corrected"
    gt_out = stream_out / "gt"
    render_out = stream_out / "render_camera_clean"
    for path in (
        stream_out, frames_dir, original_out, corrected_out, gt_out, render_out
    ):
        path.mkdir(parents=True, exist_ok=True)

    if args.force or not frame_map.is_file():
        prepare_frame_map(Path(record["stream_dir"]), frames_dir, frame_map)
    original_handflow = (
        handflow_root / args.stream_id / "handflow_camera_result.npz"
    )
    corrected_handflow = (
        Path(args.corrected_hand_npz).expanduser().resolve()
        if args.corrected_hand_npz
        else (
            prediction_root
            / args.stream_id
            / "handflow_camera_result_stage1_hand_rigid.npz"
        )
    )
    raw_pose = Path(record["foundationpose_json"]).expanduser().resolve()
    filtered_pose = (
        filtered_root
        / args.split
        / args.stream_id
        / "segmented_ekf_rts"
        / "foundationpose_segmented_ekf_rts.json"
    )
    object_mesh = Path(record["sam3d_glb"]).expanduser().resolve()
    gt_object_mesh = (
        Path(args.object_model_root).expanduser().resolve()
        / record["object_name"]
        / "textured_simple.obj"
    )
    for path in (
        original_handflow, corrected_handflow, raw_pose, filtered_pose,
        object_mesh, gt_object_mesh
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    adapter = (
        repository
        / "reconstruction/hyp_scripts/"
        "prepare_foundationpose_handflow_visualization.py"
    )
    for handflow, pose_json, out_dir in (
        (original_handflow, raw_pose, original_out),
        (corrected_handflow, filtered_pose, corrected_out),
    ):
        expected = [
            out_dir / "foundationpose_layout_camera_frame.json",
            out_dir / "all_hand_meshes_handflow.npz",
        ]
        if args.force:
            expected = []
        run(
            [
                sys.executable,
                str(adapter),
                "--foundationpose-json",
                str(pose_json),
                "--frame-map-json",
                str(frame_map),
                "--handflow-npz",
                str(handflow),
                "--out-dir",
                str(out_dir),
                "--hand-side",
                record["hand_side"],
                "--invalid-hand-mode",
                "keep",
            ],
            expected,
        )

    gt_hand = gt_out / "dexycb_gt_hand_meshes.npz"
    gt_layout = gt_out / "dexycb_gt_object_layout_camera_frame.json"
    run(
        [
            str(Path(args.gt_python).expanduser().resolve()),
            str(
                repository
                / "reconstruction/hyp_scripts/"
                "prepare_dexycb_gt_visualization.py"
            ),
            "--frame-map-json",
            str(frame_map),
            "--mano-data-dir",
            str(Path(args.mano_data_dir).expanduser().resolve()),
            "--object-model-root",
            str(Path(args.object_model_root).expanduser().resolve()),
            "--out-dir",
            str(gt_out),
        ],
        [] if args.force else [gt_hand, gt_layout],
    )
    with np.load(original_handflow, allow_pickle=False) as raw:
        intrinsic = np.asarray(raw["intrinsics"], dtype=np.float64).reshape(3, 3)
    render_script = (
        repository
        / "reconstruction/hyp_scripts/train/"
        "render_stage1_dexycb_comparison.py"
    )
    render_base = [
            sys.executable,
            "-u",
            str(render_script),
            "--frames-dir",
            str(frames_dir),
            "--mesh",
            str(object_mesh),
            "--original-layout",
            str(original_out / "foundationpose_layout_camera_frame.json"),
            "--original-hand-meshes",
            str(original_out / "all_hand_meshes_handflow.npz"),
            "--corrected-layout",
            str(corrected_out / "foundationpose_layout_camera_frame.json"),
            "--corrected-hand-meshes",
            str(corrected_out / "all_hand_meshes_handflow.npz"),
            "--gt-object-layout",
            str(gt_layout),
            "--gt-object-mesh",
            str(gt_object_mesh),
            "--gt-hand-meshes",
            str(gt_hand),
            "--hand-side",
            record["hand_side"],
            "--fx",
            str(float(intrinsic[0, 0])),
            "--fy",
            str(float(intrinsic[1, 1])),
            "--cx",
            str(float(intrinsic[0, 2])),
            "--cy",
            str(float(intrinsic[1, 2])),
            "--fps",
            str(args.fps),
            "--gt-separate-panel",
            "--original-label",
            "Original HandFlow + Raw FoundationPose",
            "--corrected-label",
            args.corrected_label,
            "--gt-label",
            "DexYCB Ground Truth",
            "--device",
            args.device,
        ]
    views = (
        [
            ("Camera view", "camera_clean"),
            ("Side view", "side"),
            ("Rear view", "rear"),
            ("Top view", "top"),
        ]
        if args.four_view_grid
        else [("Camera view", "camera_clean")]
    )
    view_outputs = []
    for view_label, view_name in views:
        view_out = (
            stream_out / f"render_{view_name}"
            if args.four_view_grid
            else render_out
        )
        view_out.mkdir(parents=True, exist_ok=True)
        triptych = view_out / "foundationpose_gt_rts.mp4"
        run(
            render_base + ["--out-dir", str(view_out), "--view", view_name],
            [] if args.force else [triptych],
        )
        view_outputs.append((view_label, triptych))
    grid_output = None
    if args.four_view_grid:
        grid_output = stream_out / "grid_4views_x_3methods.mp4"
        if args.force or not grid_output.is_file():
            stack_view_triptychs(view_outputs, grid_output, args.fps)
    summary = {
        "stream_id": args.stream_id,
        "hand_side": record["hand_side"],
        "object_name": record["object_name"],
        "original_handflow": str(original_handflow),
        "raw_foundationpose": str(raw_pose),
        "corrected_handflow": str(corrected_handflow),
        "filtered_object_pose": str(filtered_pose),
        "four_view_grid": str(grid_output) if grid_output else None,
        "view_triptychs": {
            label: str(path) for label, path in view_outputs
        },
    }
    (stream_out / "render_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
