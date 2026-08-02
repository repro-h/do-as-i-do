#!/usr/bin/env python3
"""Prepare and open one foreground Viser for a v8 DexYCB stream."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument(
        "--prediction-root",
        help="Override the default v8 prediction root.",
    )
    parser.add_argument(
        "--prediction-filename",
        default="handflow_camera_result_pi3x_depth_refined.npz",
    )
    parser.add_argument(
        "--output-root",
        help="Override the prepared visualization output root.",
    )
    parser.add_argument(
        "--object-pose-json",
        default=None,
        help=(
            "Optional object pose JSON override. The JSON must express the "
            "SAM3D mesh pose in camera coordinates, for example the "
            "DexYCB GT pose converted into the SAM canonical frame."
        ),
    )
    parser.add_argument(
        "--comparison-object-pose-json",
        default=None,
        help=(
            "Optional second pose JSON rendered with the same SAM3D mesh. "
            "This replaces the default DexYCB CAD GT object overlay."
        ),
    )
    parser.add_argument(
        "--viewer-python",
        default=os.environ.get(
            "VIS_PYTHON",
            "/home/mengxiangting/nas/mengxt/anaconda3/envs/"
            "sam3d-objects/bin/python",
        ),
    )
    parser.add_argument(
        "--mano-data-dir",
        default=(
            "/home/mengxiangting/nas/mengxt/Projects/"
            "Pi3_WiLoR_Hand/mano_data/mano"
        ),
    )
    parser.add_argument(
        "--object-model-root",
        default="/mnt/nas/wuke/HumanData/DexYCB/models",
    )
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--show-camera",
        action="store_true",
        help="Show the camera axes, image plane, and frustum.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[3]
    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    stream_id = "__".join(sequence_dir.parts[-3:])

    hybrid_root = (
        repository / "reconstruction/data/dexycb/hybrid_training_v1"
    )
    stage_root = hybrid_root / "stage1_global_hand_v2"
    manifest = hybrid_root / "manifests/val.jsonl"
    handflow_root = hybrid_root / "handflow_cache/val_v1/streams"
    prediction_root = (
        Path(args.prediction_root).expanduser().resolve()
        if args.prediction_root
        else stage_root / "predictions/pi3x_ray_depth_only_full_v1"
    )
    filtered_object = (
        hybrid_root
        / "object_motion_filter_v2_compact/val"
        / stream_id
        / "segmented_ekf_rts/foundationpose_segmented_ekf_rts.json"
    )
    object_pose_json = (
        Path(args.object_pose_json).expanduser().resolve()
        if args.object_pose_json
        else filtered_object
    )
    comparison_object_pose_json = (
        Path(args.comparison_object_pose_json).expanduser().resolve()
        if args.comparison_object_pose_json
        else None
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else stage_root / "visualization/pi3x_ray_depth_only_full_v1_val"
    )
    prediction = (
        prediction_root
        / stream_id
        / args.prediction_filename
    )
    required_paths = [manifest, object_pose_json, prediction]
    if comparison_object_pose_json is not None:
        required_paths.append(comparison_object_pose_json)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(prediction, allow_pickle=False) as data:
        if "relative_translation_model_version" in data.files:
            objective = str(data["relative_translation_model_version"].item())
            checkpoint = str(data["relative_translation_checkpoint"].item())
        else:
            objective = (
                str(data["pi3x_depth_objective"].item())
                if "pi3x_depth_objective" in data.files
                else "not_recorded"
            )
            checkpoint = (
                str(data["pi3x_depth_checkpoint"].item())
                if "pi3x_depth_checkpoint" in data.files
                else "not_recorded"
            )

    records = {
        row["stream_id"]: row for row in load_jsonl(manifest)
    }
    if stream_id not in records:
        raise KeyError(f"Stream is not in val manifest: {stream_id}")
    record = records[stream_id]

    prepare_script = Path(__file__).with_name(
        "visualize_stage1_dexycb.py"
    )
    prepare_command = [
        sys.executable,
        "-u",
        str(prepare_script),
        "--manifest",
        str(manifest),
        "--prediction-root",
        str(prediction_root),
        "--handflow-root",
        str(handflow_root),
        "--mano-data-dir",
        str(Path(args.mano_data_dir).expanduser().resolve()),
        "--object-model-root",
        str(Path(args.object_model_root).expanduser().resolve()),
        "--out-root",
        str(output_root),
        "--viewer-python",
        str(Path(args.viewer_python).expanduser().resolve()),
        "--stream-id",
        stream_id,
        "--foundationpose-json",
        str(object_pose_json),
        "--prepare-only",
    ]
    if args.force_prepare:
        prepare_command.append("--force-prepare")
    subprocess.run(prepare_command, check=True)

    stream_out = output_root / stream_id
    comparison_layout = None
    comparison_scale = None
    if comparison_object_pose_json is not None:
        comparison_out = stream_out / "comparison_object"
        comparison_prepare_script = (
            repository
            / "reconstruction/hyp_scripts/"
            "prepare_foundationpose_handflow_visualization.py"
        )
        comparison_command = [
            sys.executable,
            "-u",
            str(comparison_prepare_script),
            "--foundationpose-json",
            str(comparison_object_pose_json),
            "--frame-map-json",
            str(stream_out / "dexycb_frame_map.json"),
            "--handflow-npz",
            str(prediction),
            "--hand-side",
            record["hand_side"],
            "--invalid-hand-mode",
            "keep",
            "--out-dir",
            str(comparison_out),
        ]
        subprocess.run(comparison_command, check=True)
        comparison_layout = (
            comparison_out / "foundationpose_layout_camera_frame.json"
        )
        comparison_payload = json.loads(
            comparison_object_pose_json.read_text(encoding="utf-8")
        )
        comparison_scale = float(
            comparison_payload.get(
                "source_mesh_scale",
                record["foundationpose_source_mesh_scale"],
            )
        )
    camera = json.loads(
        (stream_out / "camera.json").read_text(encoding="utf-8")
    )
    corrected = stream_out / "stage1_corrected"
    gt_out = stream_out / "gt"
    viewer_script = repository / "reconstruction/scripts/visualize_3d.py"
    viewer_command = [
        str(Path(args.viewer_python).expanduser().resolve()),
        "-u",
        str(viewer_script),
        "--frames-dir",
        str(stream_out / "frames"),
        "--layout-json",
        str(corrected / "foundationpose_layout_camera_frame.json"),
        "--mesh",
        str(Path(record["sam3d_glb"]).expanduser().resolve()),
        "--hand-meshes",
        str(corrected / "all_hand_meshes_handflow.npz"),
        "--gt-hand-meshes",
        str(gt_out / "dexycb_gt_hand_meshes.npz"),
        "--gt-object-layout-json",
        str(
            comparison_layout
            if comparison_layout is not None
            else gt_out / "dexycb_gt_object_layout_camera_frame.json"
        ),
        "--gt-object-mesh",
        str(
            Path(record["sam3d_glb"]).expanduser().resolve()
            if comparison_layout is not None
            else (
                Path(args.object_model_root).expanduser().resolve()
                / record["object_name"]
                / "textured_simple.obj"
            )
        ),
        "--gt-object-scale",
        str(comparison_scale if comparison_scale is not None else 1.0),
        "--scale",
        str(record["foundationpose_source_mesh_scale"]),
        "--translation-scale",
        "1.0",
        "--hands",
        record["hand_side"],
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
        str(args.fps),
        "--frustum-scale",
        "0.15",
        "--port",
        str(args.port),
    ]
    if not args.show_camera:
        viewer_command.append("--hide-camera")
    print(f"Stream: {stream_id}")
    print(f"Objective: {objective}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Object pose: {object_pose_json}")
    if comparison_object_pose_json is not None:
        print(f"Comparison object pose: {comparison_object_pose_json}")
    print(f"Viewer: http://localhost:{args.port}")
    print("Press Ctrl+C to stop.", flush=True)
    subprocess.run(viewer_command, check=True)


if __name__ == "__main__":
    main()
