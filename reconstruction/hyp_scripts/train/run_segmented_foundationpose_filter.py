#!/usr/bin/env python3
"""Run TAPIR motion segmentation and segmented FoundationPose EKF/RTS."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    data_root = repository / "reconstruction/data/dexycb"
    hybrid_root = data_root / "hybrid_training_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--passed-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--manifest",
        default=str(hybrid_root / "manifests/train.jsonl"),
    )
    parser.add_argument(
        "--handflow-root",
        default=str(hybrid_root / "handflow_cache/train_v1/streams"),
    )
    parser.add_argument(
        "--out-root",
        default=str(hybrid_root / "object_motion_filter_v1"),
    )
    parser.add_argument(
        "--tapir-root",
        default=str(repository / "reconstruction/modules/tapnet"),
    )
    parser.add_argument(
        "--tapir-checkpoint",
        default=str(
            repository
            / "reconstruction/weights/tapnet/bootstapir_checkpoint_v2.pt"
        ),
    )
    parser.add_argument("--tapir-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--boundary-blend-frames", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--prepare-gt", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Skip visualization adapters and remove temporary frames, TAPIR "
            "tracks, plots, and CSV files after a successful run."
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stream_id_from_passed_dir(path: Path) -> str:
    parts = path.resolve().parts
    if len(parts) < 3:
        raise ValueError(f"Cannot parse stream from {path}")
    subject, sequence, camera = parts[-3:]
    return f"{subject}__{sequence}__{camera}"


def prepare_frame_map(
    stream_dir: Path, frames_dir: Path, out_path: Path
) -> dict:
    images = sorted(stream_dir.glob("color_*.jpg"))
    if not images:
        images = sorted(stream_dir.glob("color_*.png"))
    if not images:
        raise RuntimeError(f"No color frames found in {stream_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_index, image_path in enumerate(images):
        original_frame = image_path.stem.rsplit("_", 1)[-1]
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
    payload = {
        "stream_dir": str(stream_dir.resolve()),
        "num_frames": len(rows),
        "frames": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_intrinsics(handflow_path: Path, foundationpose_path: Path) -> np.ndarray:
    with np.load(handflow_path, allow_pickle=True) as payload:
        for key in ("K", "intrinsics", "camera_intrinsics"):
            if key not in payload:
                continue
            value = np.asarray(payload[key])
            if value.ndim == 3:
                value = value[0]
            if value.size == 9:
                return value.reshape(3, 3).astype(np.float64)
    foundationpose = json.loads(
        foundationpose_path.read_text(encoding="utf-8")
    )
    value = np.asarray(foundationpose.get("intrinsics"))
    if value.size != 9:
        raise KeyError("No 3x3 intrinsics in HandFlow or FoundationPose")
    return value.reshape(3, 3).astype(np.float64)


def run(
    command: list[str],
    log_path: Path,
    expected: list[Path],
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    if not overwrite and expected and all(path.is_file() for path in expected):
        print(f"[cached] {expected[0]}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(
            f"Command failed ({process.returncode}): {log_path}\n{tail}"
        )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Command did not create expected files: {missing}")


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[3]
    scripts = repository / "reconstruction/hyp_scripts/train"
    passed_dir = Path(args.passed_dir).expanduser().resolve()
    if not passed_dir.is_dir():
        raise FileNotFoundError(passed_dir)
    stream_id = stream_id_from_passed_dir(passed_dir)
    manifest_path = Path(args.manifest).expanduser().resolve()
    records = {
        row["stream_id"]: row for row in load_jsonl(manifest_path)
    }
    if stream_id not in records:
        raise KeyError(f"{stream_id} is not present in {manifest_path}")
    record = records[stream_id]
    stream_dir = Path(record["stream_dir"]).expanduser().resolve()
    foundationpose_json = (
        Path(record["foundationpose_json"]).expanduser().resolve()
    )
    handflow_path = (
        Path(args.handflow_root).expanduser().resolve()
        / stream_id
        / "handflow_camera_result.npz"
    )
    tapir_root = Path(args.tapir_root).expanduser().resolve()
    tapir_checkpoint = Path(args.tapir_checkpoint).expanduser().resolve()
    required = [
        stream_dir,
        foundationpose_json,
        handflow_path,
        tapir_root,
        tapir_checkpoint,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir = (
        Path(args.out_root).expanduser().resolve()
        / args.split
        / stream_id
    )
    frames_dir = out_dir / "frames"
    frame_map_json = out_dir / "dexycb_frame_map.json"
    camera_json = out_dir / "camera.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_map = (
        prepare_frame_map(stream_dir, frames_dir, frame_map_json)
        if args.overwrite or not frame_map_json.is_file()
        else json.loads(frame_map_json.read_text(encoding="utf-8"))
    )
    intrinsic = load_intrinsics(handflow_path, foundationpose_json)
    camera_json.write_text(
        json.dumps(
            {
                "fx": float(intrinsic[0, 0]),
                "fy": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "K": intrinsic.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selected = {
        **record,
        "passed_dir": str(passed_dir),
        "foundationpose_json": str(foundationpose_json),
        "handflow_npz": str(handflow_path),
        "num_frames": len(frame_map["frames"]),
    }
    (out_dir / "selected_stream.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    tracks_dir = out_dir / "tapir_tracks"
    tracks_npz = tracks_dir / "tapir_object_3d_tracks.npz"
    tracks_summary = tracks_dir / "summary.json"
    track_command = [
        args.tapir_python,
        str(scripts / "export_tapir_object_3d_tracks.py"),
        "--frames-dir",
        str(frames_dir),
        "--object",
        record["object_name"],
        "--frame-map-json",
        str(frame_map_json),
        "--intrinsics-json",
        str(camera_json),
        "--checkpoint",
        str(tapir_checkpoint),
        "--out-npz",
        str(tracks_npz),
        "--out-summary",
        str(tracks_summary),
        "--num-points",
        str(args.num_points),
        "--device",
        args.device,
    ]
    if args.preview:
        track_command.extend(["--preview-dir", str(tracks_dir / "previews")])
    tapir_env = os.environ.copy()
    tapir_env["PYTHONPATH"] = (
        str(tapir_root)
        + os.pathsep
        + tapir_env.get("PYTHONPATH", "")
    )
    run(
        track_command,
        tracks_dir / "run.log",
        [tracks_npz, tracks_summary],
        env=tapir_env,
        cwd=tapir_root,
        overwrite=args.overwrite,
    )

    motion_dir = out_dir / "motion_audit"
    motion_json = motion_dir / "audit.json"
    motion_csv = motion_dir / "pairs.csv"
    run(
        [
            sys.executable,
            str(scripts / "audit_foundationpose_tapir_motion.py"),
            "--foundationpose-json",
            str(foundationpose_json),
            "--frame-map-json",
            str(frame_map_json),
            "--tapir-npz",
            str(tracks_npz),
            "--out-json",
            str(motion_json),
            "--out-csv",
            str(motion_csv),
            "--out-plot",
            str(motion_dir / "speed_comparison.png"),
        ],
        motion_dir / "run.log",
        [motion_json, motion_csv],
        overwrite=args.overwrite,
    )

    segment_dir = out_dir / "motion_segmentation"
    static_json = segment_dir / "foundationpose_static_consolidated.json"
    segment_audit = segment_dir / "audit.json"
    run(
        [
            sys.executable,
            str(scripts / "segment_and_consolidate_foundationpose_tapir.py"),
            "--foundationpose-json",
            str(foundationpose_json),
            "--frame-map-json",
            str(frame_map_json),
            "--tapir-npz",
            str(tracks_npz),
            "--out-json",
            str(static_json),
            "--out-audit",
            str(segment_audit),
            "--out-plot",
            str(segment_dir / "motion_segments.png"),
            "--enter-translation-mm",
            "6",
            "--enter-rotation-deg",
            "1.2",
            "--exit-translation-mm",
            "3",
            "--exit-rotation-deg",
            "0.6",
            "--enter-frames",
            "2",
            "--exit-frames",
            "4",
            "--stationary-window",
            "6",
            "--exit-net-translation-mm",
            "8",
            "--exit-net-rotation-deg",
            "1.5",
            "--median-window",
            "3",
            "--min-static-frames",
            "5",
            "--static-trim-frames",
            "1",
        ],
        segment_dir / "run.log",
        [static_json, segment_audit],
        overwrite=args.overwrite,
    )

    filter_dir = out_dir / "segmented_ekf_rts"
    ekf_json = filter_dir / "foundationpose_segmented_ekf.json"
    rts_json = filter_dir / "foundationpose_segmented_ekf_rts.json"
    rts_audit = filter_dir / "foundationpose_segmented_ekf_rts_audit.json"
    run(
        [
            sys.executable,
            str(scripts / "filter_foundationpose_ekf_rts.py"),
            "--foundationpose-json",
            str(static_json),
            "--segmentation-audit-json",
            str(segment_audit),
            "--out-ekf-json",
            str(ekf_json),
            "--out-rts-json",
            str(rts_json),
            "--fps",
            "30",
            "--translation-measurement-mm",
            "4",
            "--translation-acceleration-mps2",
            "0.6",
            "--rotation-measurement-deg",
            "3",
            "--angular-acceleration-deg-s2",
            "90",
            "--boundary-blend-frames",
            str(args.boundary_blend_frames),
        ],
        filter_dir / "run.log",
        [ekf_json, rts_json, rts_audit],
        overwrite=args.overwrite,
    )

    visualization = out_dir / "visualization"
    if not args.compact:
        adapter = (
            repository
            / "reconstruction/hyp_scripts/"
            "prepare_foundationpose_handflow_visualization.py"
        )
        for name, pose_json in (
            ("input", foundationpose_json),
            ("static_shared", static_json),
            ("ekf", ekf_json),
            ("rts", rts_json),
        ):
            adapter_out = visualization / name
            run(
                [
                    sys.executable,
                    str(adapter),
                    "--foundationpose-json",
                    str(pose_json),
                    "--frame-map-json",
                    str(frame_map_json),
                    "--handflow-npz",
                    str(handflow_path),
                    "--out-dir",
                    str(adapter_out),
                    "--hand-side",
                    record["hand_side"],
                    "--invalid-hand-mode",
                    "keep",
                ],
                adapter_out / "run.log",
                [
                    adapter_out / "foundationpose_layout_camera_frame.json",
                    adapter_out / "all_hand_meshes_handflow.npz",
                ],
                overwrite=args.overwrite,
            )

    gt_dir = out_dir / "gt"
    if args.prepare_gt:
        run(
            [
                sys.executable,
                str(
                    repository
                    / "reconstruction/hyp_scripts/"
                    "prepare_dexycb_gt_visualization.py"
                ),
                "--frame-map-json",
                str(frame_map_json),
                "--mano-data-dir",
                str(Path(args.mano_data_dir).expanduser().resolve()),
                "--object-model-root",
                str(Path(args.object_model_root).expanduser().resolve()),
                "--out-dir",
                str(gt_dir),
            ],
            gt_dir / "run.log",
            [
                gt_dir / "dexycb_gt_hand_meshes.npz",
                gt_dir / "dexycb_gt_object_layout_camera_frame.json",
            ],
            overwrite=args.overwrite,
        )

    summary = {
        "stream_id": stream_id,
        "record": record,
        "uses_custom_trained_checkpoint": False,
        "foundationpose_json": str(foundationpose_json),
        "tapir_checkpoint": str(tapir_checkpoint),
        "compact": args.compact,
        "tracks_npz": None if args.compact else str(tracks_npz),
        "motion_audit": str(motion_json),
        "segmentation_audit": str(segment_audit),
        "static_consolidated_json": str(static_json),
        "ekf_json": str(ekf_json),
        "rts_json": str(rts_json),
        "rts_audit": str(rts_audit),
        "visualization": (
            None
            if args.compact
            else {
                name: str(visualization / name)
                for name in ("input", "static_shared", "ekf", "rts")
            }
        ),
        "gt": str(gt_dir) if args.prepare_gt else None,
    }
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if args.compact:
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(tracks_dir, ignore_errors=True)
        for path in (
            frame_map_json,
            camera_json,
            out_dir / "selected_stream.json",
            motion_csv,
            motion_dir / "speed_comparison.png",
            segment_dir / "motion_segments.png",
        ):
            if path.is_file() or path.is_symlink():
                path.unlink()
    print(json.dumps(summary, indent=2))
    print(f"Done: {summary_path}")


if __name__ == "__main__":
    main()
