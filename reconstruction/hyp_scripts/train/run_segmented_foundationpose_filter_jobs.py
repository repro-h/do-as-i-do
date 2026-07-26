#!/usr/bin/env python3
"""Run the segmented FoundationPose filter over a sharded DexYCB manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--passed-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--handflow-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--tapir-root", required=True)
    parser.add_argument("--tapir-checkpoint", required=True)
    parser.add_argument("--tapir-python", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--boundary-blend-frames", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def passed_dir_for(record: dict, root: Path) -> Path:
    stream_id = record["stream_id"]
    parts = stream_id.split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid stream_id: {stream_id}")
    return root.joinpath(*parts)


def completed_summary(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = payload.get("rts_json")
    return bool(result and Path(result).is_file())


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    script = Path(__file__).with_name(
        "run_segmented_foundationpose_filter.py"
    )
    manifest = Path(args.manifest).expanduser().resolve()
    passed_root = Path(args.passed_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    status_path = Path(args.status_json).expanduser().resolve()
    records = load_jsonl(manifest)[args.shard_index :: args.num_shards]
    if args.limit > 0:
        records = records[: args.limit]

    state = {
        "manifest": str(manifest),
        "split": args.split,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_requested": len(records),
        "completed": {},
        "failed": {},
    }
    for index, record in enumerate(records, start=1):
        stream_id = record["stream_id"]
        passed_dir = passed_dir_for(record, passed_root)
        summary = out_root / args.split / stream_id / "pipeline_summary.json"
        print(f"[{index}/{len(records)}] {stream_id}", flush=True)
        if not args.overwrite and completed_summary(summary):
            state["completed"][stream_id] = {
                "summary": str(summary),
                "cached": True,
            }
            print(f"  cached: {summary}", flush=True)
            write_json_atomic(status_path, state)
            continue
        if not passed_dir.is_dir():
            state["failed"][stream_id] = {
                "error": f"Missing passed directory: {passed_dir}"
            }
            print(f"  failed: missing {passed_dir}", flush=True)
            write_json_atomic(status_path, state)
            continue

        command = [
            sys.executable,
            "-u",
            str(script),
            "--passed-dir",
            str(passed_dir),
            "--split",
            args.split,
            "--manifest",
            str(manifest),
            "--handflow-root",
            str(Path(args.handflow_root).expanduser().resolve()),
            "--out-root",
            str(out_root),
            "--tapir-root",
            str(Path(args.tapir_root).expanduser().resolve()),
            "--tapir-checkpoint",
            str(Path(args.tapir_checkpoint).expanduser().resolve()),
            "--tapir-python",
            str(Path(args.tapir_python).expanduser().resolve()),
            "--device",
            args.device,
            "--boundary-blend-frames",
            str(args.boundary_blend_frames),
            "--num-points",
            str(args.num_points),
        ]
        if args.compact:
            command.append("--compact")
        if args.overwrite:
            command.append("--overwrite")
        log_path = summary.parent / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("command: " + " ".join(command) + "\n")
                log.flush()
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not completed_summary(summary):
                raise RuntimeError("Pipeline returned without a valid summary")
            state["completed"][stream_id] = {
                "summary": str(summary),
                "log": str(log_path),
                "cached": False,
            }
            print(f"  done: {summary}", flush=True)
        except Exception as error:
            state["failed"][stream_id] = {
                "error": f"{type(error).__name__}: {error}",
                "log": str(log_path),
            }
            print(f"  failed: {type(error).__name__}: {error}", flush=True)
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
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
