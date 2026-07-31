#!/usr/bin/env python3
"""Batch render selected DexYCB streams with the single-stream renderer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--sequence-dir",
        action="append",
        default=[],
        help="Repeat this argument once for each sequence directory.",
    )
    parser.add_argument(
        "--sequence-list",
        help="Optional text file containing one sequence directory per line.",
    )
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--filtered-object-root", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--object-model-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--video-only-root",
        help=(
            "Optional directory that receives only the final MP4, grouped "
            "under one stream-ID directory per sequence."
        ),
    )
    parser.add_argument("--gt-python", default=sys.executable)
    parser.add_argument("--split", default="val")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--corrected-label", default="Stage1 Hand + Filtered Object"
    )
    parser.add_argument("--four-view-grid", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def stream_id_from_sequence_dir(value: str) -> str:
    path = Path(value).expanduser().resolve()
    if len(path.parts) < 3:
        raise ValueError(f"Invalid sequence directory: {path}")
    return "__".join(path.parts[-3:])


def main() -> None:
    args = parse_args()
    sequence_dirs = list(args.sequence_dir)
    if args.sequence_list:
        list_path = Path(args.sequence_list).expanduser().resolve()
        sequence_dirs.extend(
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    sequence_dirs = list(dict.fromkeys(sequence_dirs))
    if not sequence_dirs:
        raise ValueError(
            "Provide at least one --sequence-dir or --sequence-list"
        )

    renderer = Path(__file__).with_name(
        "render_stage1_hand_rigid_stream.py"
    ).resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    video_only_root = None
    if args.video_only_root:
        video_only_root = Path(args.video_only_root).expanduser().resolve()
        video_only_root.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable,
        "-u",
        str(renderer),
        "--manifest",
        args.manifest,
        "--handflow-root",
        args.handflow_root,
        "--prediction-root",
        args.prediction_root,
        "--filtered-object-root",
        args.filtered_object_root,
        "--mano-data-dir",
        args.mano_data_dir,
        "--object-model-root",
        args.object_model_root,
        "--out-root",
        args.out_root,
        "--gt-python",
        args.gt_python,
        "--split",
        args.split,
        "--fps",
        str(args.fps),
        "--device",
        args.device,
        "--corrected-label",
        args.corrected_label,
    ]
    if args.four_view_grid:
        common.append("--four-view-grid")
    if args.force:
        common.append("--force")

    completed = []
    failures = []
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        stream_id = stream_id_from_sequence_dir(sequence_dir)
        print(
            f"[{index}/{len(sequence_dirs)}] {stream_id}", flush=True
        )
        try:
            subprocess.run(
                common + ["--sequence-dir", sequence_dir], check=True
            )
            if video_only_root is not None:
                source_video = (
                    out_root
                    / stream_id
                    / "grid_4views_x_3methods.mp4"
                )
                if not source_video.is_file():
                    raise FileNotFoundError(source_video)
                video_dir = video_only_root / stream_id
                video_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_video, video_dir / source_video.name)
            completed.append(stream_id)
        except Exception as error:
            row = {
                "stream_id": stream_id,
                "sequence_dir": sequence_dir,
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(row)
            print(f"  failed: {row['error']}", flush=True)
            if args.fail_fast:
                break

    summary = {
        "num_requested": len(sequence_dirs),
        "num_completed": len(completed),
        "num_failed": len(failures),
        "video_only_root": (
            str(video_only_root) if video_only_root is not None else None
        ),
        "completed": completed,
        "failures": failures,
    }
    summary_path = out_root / "batch_render_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
