#!/usr/bin/env python3
"""Prepare and render one hand-rigid validation stream as a 3D triptych."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
        "--gt-python",
        default=sys.executable,
        help="Python environment used to decode DexYCB MANO ground truth.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
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
        prediction_root
        / args.stream_id
        / "handflow_camera_result_stage1_hand_rigid.npz"
    )
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
        original_handflow, corrected_handflow, filtered_pose,
        object_mesh, gt_object_mesh
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    adapter = (
        repository
        / "reconstruction/hyp_scripts/"
        "prepare_foundationpose_handflow_visualization.py"
    )
    for handflow, out_dir in (
        (original_handflow, original_out),
        (corrected_handflow, corrected_out),
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
                str(filtered_pose),
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
    comparison = render_out / "before_after.mp4"
    run(
        [
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
            "--out-dir",
            str(render_out),
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
            "--view",
            "camera_clean",
            "--gt-separate-panel",
            "--original-label",
            "Original HandFlow + Frozen Object",
            "--corrected-label",
            "Stage1 Hand Rigid + Frozen Object",
            "--gt-label",
            "DexYCB Ground Truth",
            "--device",
            args.device,
        ],
        [] if args.force else [comparison],
    )
    summary = {
        "stream_id": args.stream_id,
        "hand_side": record["hand_side"],
        "object_name": record["object_name"],
        "original_handflow": str(original_handflow),
        "corrected_handflow": str(corrected_handflow),
        "filtered_object_pose": str(filtered_pose),
        "comparison": str(comparison),
    }
    (stream_out / "render_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
