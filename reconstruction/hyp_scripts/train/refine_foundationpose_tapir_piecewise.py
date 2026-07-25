#!/usr/bin/env python3
"""Run TAPIR pose-graph refinement independently on segmented motion spans."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-json", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--tapir-npz", required=True)
    parser.add_argument("--motion-audit-json", required=True)
    parser.add_argument("--segmentation-audit-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-audit", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--prior-translation-mm", type=float, default=12.0)
    parser.add_argument("--prior-rotation-deg", type=float, default=8.0)
    parser.add_argument("--edge-translation-mm", type=float, default=4.0)
    parser.add_argument("--edge-rotation-deg", type=float, default=3.0)
    parser.add_argument("--correction-smooth-mm", type=float, default=3.0)
    parser.add_argument("--correction-smooth-deg", type=float, default=2.0)
    parser.add_argument("--low-speed-mm", type=float, default=4.0)
    parser.add_argument("--high-speed-mm", type=float, default=15.0)
    parser.add_argument("--low-speed-smooth-multiplier", type=float, default=2.5)
    parser.add_argument("--candidate-edge-multiplier", type=float, default=2.0)
    parser.add_argument("--max-translation-mm", type=float, default=40.0)
    parser.add_argument("--max-rotation-deg", type=float, default=20.0)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--robust-scale", type=float, default=2.0)
    return parser.parse_args()


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"Pose-graph segment failed with return code "
            f"{process.returncode}: {log_path}\n{tail}"
        )


def main() -> None:
    args = parse_args()
    segmentation_path = (
        Path(args.segmentation_audit_json).expanduser().resolve()
    )
    segmentation = json.loads(
        segmentation_path.read_text(encoding="utf-8")
    )
    dynamic_segments = segmentation.get("dynamic_segments", [])
    if not dynamic_segments:
        raise RuntimeError("Segmentation audit contains no dynamic segments")
    num_frames = int(segmentation["num_frames"])
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    refiner = Path(__file__).with_name(
        "refine_foundationpose_with_tapir_pose_graph.py"
    )
    current_json = (
        Path(args.foundationpose_json).expanduser().resolve()
    )
    segment_results = []

    for segment_index, segment in enumerate(dynamic_segments):
        dynamic_begin, dynamic_end = [
            int(value) for value in segment["output_frames"]
        ]
        optimization_start = max(0, dynamic_begin - 1)
        optimization_end = min(num_frames - 1, dynamic_end + 1)
        free_end = dynamic_end >= num_frames - 1
        segment_dir = work_dir / f"segment_{segment_index:02d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_json = segment_dir / "foundationpose_poses_refined.json"
        segment_audit = segment_dir / "audit.json"
        segment_log = segment_dir / "run.log"
        command = [
            sys.executable,
            str(refiner),
            "--foundationpose-json",
            str(current_json),
            "--frame-map-json",
            str(Path(args.frame_map_json).expanduser().resolve()),
            "--tapir-npz",
            str(Path(args.tapir_npz).expanduser().resolve()),
            "--motion-audit-json",
            str(Path(args.motion_audit_json).expanduser().resolve()),
            "--out-json",
            str(segment_json),
            "--out-audit",
            str(segment_audit),
            "--start-frame",
            str(optimization_start),
            "--end-frame",
            str(optimization_end),
            "--prior-translation-mm",
            str(args.prior_translation_mm),
            "--prior-rotation-deg",
            str(args.prior_rotation_deg),
            "--edge-translation-mm",
            str(args.edge_translation_mm),
            "--edge-rotation-deg",
            str(args.edge_rotation_deg),
            "--correction-smooth-mm",
            str(args.correction_smooth_mm),
            "--correction-smooth-deg",
            str(args.correction_smooth_deg),
            "--adaptive-smoothing",
            "--low-speed-mm",
            str(args.low_speed_mm),
            "--high-speed-mm",
            str(args.high_speed_mm),
            "--low-speed-smooth-multiplier",
            str(args.low_speed_smooth_multiplier),
            "--candidate-edge-multiplier",
            str(args.candidate_edge_multiplier),
            "--max-translation-mm",
            str(args.max_translation_mm),
            "--max-rotation-deg",
            str(args.max_rotation_deg),
            "--max-nfev",
            str(args.max_nfev),
            "--robust-scale",
            str(args.robust_scale),
        ]
        if free_end:
            command.append("--free-end")
        print(
            f"[{segment_index + 1}/{len(dynamic_segments)}] "
            f"dynamic=[{dynamic_begin}, {dynamic_end}] "
            f"optimization=[{optimization_start}, {optimization_end}] "
            f"free_end={free_end}",
            flush=True,
        )
        run(command, segment_log)
        audit = json.loads(segment_audit.read_text(encoding="utf-8"))
        segment_results.append(
            {
                "segment_index": segment_index,
                "dynamic_frames": [dynamic_begin, dynamic_end],
                "optimization_interval": [
                    optimization_start,
                    optimization_end,
                ],
                "free_end": free_end,
                "output_json": str(segment_json),
                "audit_json": str(segment_audit),
                "log": str(segment_log),
                "solver": audit["solver"],
                "tapir_edge_center_error_mm": audit[
                    "tapir_edge_center_error_mm"
                ],
                "tapir_edge_rotation_error_deg": audit[
                    "tapir_edge_rotation_error_deg"
                ],
                "translation_correction_mm": audit[
                    "translation_correction_mm"
                ],
                "rotation_correction_deg": audit[
                    "rotation_correction_deg"
                ],
            }
        )
        current_json = segment_json

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(current_json, out_json)
    audit = {
        "settings": vars(args),
        "source_foundationpose_json": str(
            Path(args.foundationpose_json).expanduser().resolve()
        ),
        "segmentation_audit_json": str(segmentation_path),
        "out_json": str(out_json),
        "num_dynamic_segments": len(dynamic_segments),
        "segments": segment_results,
    }
    out_audit = Path(args.out_audit).expanduser().resolve()
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
