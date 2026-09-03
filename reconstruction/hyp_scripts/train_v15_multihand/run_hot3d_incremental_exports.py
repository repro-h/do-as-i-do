#!/usr/bin/env python3
"""Wait for Aria downloads and incrementally run the original V15 exporters.

Only the existing downloader writes raw downloads. This runner checks its
.download_status.json and required files before handing a sequence to the API.
"""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


SCRIPTS = Path(__file__).resolve().parent
FULL = Path("/data2/hyp/full_v15")
PROJECTS = Path("/home/mengxiangting/nas/mengxt/Projects")
REQUIRED_GROUPS = ("main_vrs", "ground_truth", "hand_data")
REQUIRED_FILES = (
    "recording.vrs", "metadata.json", "mano_hand_pose_trajectory.jsonl",
    "headset_trajectory.csv", "dynamic_objects.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Physical GPU IDs, e.g. 5,6")
    parser.add_argument("--hot3d-root", type=Path, default=Path("/data2/hyp/data/HOT3D"))
    parser.add_argument("--hot3d-code-root", type=Path, default=Path("/data2/hyp/data/tools/hot3d"))
    parser.add_argument("--out-root", type=Path, default=FULL / "hot3d")
    parser.add_argument("--processed-root", type=Path, help="Otherwise infer from existing manifest labels")
    parser.add_argument("--additional-list", type=Path)
    parser.add_argument("--url-json", type=Path)
    parser.add_argument("--old-split", type=Path)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--max-wait-hours", type=float, default=0, help="0 waits indefinitely")
    parser.add_argument("--mano-model-folder", type=Path, default=PROJECTS / "Pi3_WiLoR_Hand/mano_data")
    parser.add_argument("--visibility-python", type=Path, default=PROJECTS / "hand_visibility_detector/.venv/bin/python")
    parser.add_argument("--visibility-root", type=Path, default=PROJECTS / "hand_visibility_detector")
    parser.add_argument("--visibility-checkpoint", type=Path, default=Path(
        "/data2/hyp/test_v15/huggingface/hub/models--ryhara--hand-visibility-detector/"
        "snapshots/941b791bcba4a0bb381c325c225f56e0a80cf98f/best.pt"
    ))
    parser.add_argument("--hand-uni-root", type=Path, default=PROJECTS / "Pi3_WiLoR_Hand")
    parser.add_argument("--pi3-root", type=Path, default=PROJECTS / "Pi3")
    parser.add_argument("--pi3x-checkpoint", type=Path, default=Path("/mnt/nas/mengxt/Projects/Pi3/ckpts/model.safetensors"))
    parser.add_argument("--compact-cache-root", type=Path)
    parser.add_argument("--training-checkpoint", type=Path, default=FULL /
        "mixed_five_dataset_full_track_aligned_v1/checkpoints/"
        "v16_2_five_dataset_track_aligned_tail_cosine_v1/best.pt")
    parser.add_argument("--dry-run", action="store_true", help="Inspect split and paths; no export or wait")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_sequences(path):
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"Empty/duplicate sequence list: {path}")
    if any(not re.fullmatch(r"P\d{4}_[A-Za-z0-9]+", row) for row in rows):
        raise ValueError(f"Invalid sequence ID in {path}")
    return rows


def make_split(old, additional, val_count, seed):
    train = list(old["train_sequences"])
    val = list(old["val_sequences"])
    if len(train + val) != len(set(train + val)):
        raise ValueError("Existing train/val split has duplicates or overlap")
    total = sorted(set(train + val + additional))
    if not len(val) <= val_count < len(total):
        raise ValueError("Validation target cannot shrink old validation or include all sequences")
    candidates = sorted(set(total) - set(train + val))
    counts = Counter(item.split("_", 1)[0] for item in val)
    while len(val) < val_count:
        if not candidates:
            raise ValueError("Cannot increase validation without moving old training sequences")
        item = min(candidates, key=lambda item: (
            counts[item.split("_", 1)[0]],
            hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(),
        ))
        candidates.remove(item)
        val.append(item)
        counts[item.split("_", 1)[0]] += 1
    train.extend(candidates)
    return {"train_sequences": sorted(train), "val_sequences": sorted(val), "seed": seed}


def infer_processed_root(out_root, sequences):
    manifest = out_root / "manifests/train_windows.jsonl"
    with manifest.open() as handle:
        row = next(json.loads(line) for line in handle if line.strip())
    label = Path(row["label_paths"][0])
    for parent in label.parents:
        if parent.name in sequences:
            return parent.parent
    raise ValueError(f"Cannot infer processed-root from {label}; pass --processed-root")


def ready_signature(directory):
    try:
        status = read_json(directory / ".download_status.json")
        incomplete = [group for group in REQUIRED_GROUPS if status.get(group) is not True]
        if incomplete:
            return None, "download groups incomplete: " + ",".join(incomplete)
        signature = []
        for filename in REQUIRED_FILES:
            path = directory / filename
            stat = path.stat()
            if not path.is_file() or stat.st_size == 0:
                return None, "empty/non-file: " + filename
            signature.append((filename, stat.st_size, stat.st_mtime_ns))
        return tuple(signature), None
    except (OSError, ValueError, TypeError) as error:
        return None, str(error)


def prepared_is_current(directory, settings):
    try:
        summary = read_json(directory / "summary.json")
        if summary.get("schema_version") != "hot3d_aria_v15_export_v1":
            return False
        if summary.get("horizontal_mirror") is not False:
            return False
        if any(summary.get(key) != value for key, value in settings.items()):
            return False
        rows = [json.loads(line) for line in (directory / "train_windows.jsonl").read_text().splitlines() if line.strip()]
        if not rows or len(rows) != summary["windows"]:
            return False
        files = {path for row in rows for key in ("image_paths", "label_paths") for path in row[key]}
        return all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in files)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_stage(name, command, log_root, env=None):
    command = list(map(str, command))
    path = log_root / f"{name}.log"
    log(f"starting {name}; log={path}")
    with path.open("w") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=True)
    log(f"completed {name}")


def main():
    args = parse_args()
    gpu_ids = args.gpus.split(",")
    if not all(item.isdigit() for item in gpu_ids) or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpus must be distinct comma-separated numeric IDs")
    if args.poll_seconds <= 0 or args.stable_checks < 2 or args.max_wait_hours < 0:
        raise ValueError("Need positive poll interval, >=2 stable checks and nonnegative max wait")
    args.additional_list = args.additional_list or args.hot3d_root / "aria_train_additional_96.txt"
    args.url_json = args.url_json or args.hot3d_root / "download_urls/Hot3DAria_download_urls.json"
    args.old_split = args.old_split or args.out_root / "manifests/split.json"
    old = read_json(args.old_split)
    additional = read_sequences(args.additional_list)
    plan = make_split(old, additional, args.val_count, args.seed)
    sequences = sorted(plan["train_sequences"] + plan["val_sequences"])
    cdn = read_json(args.url_json)["sequences"]
    for sequence in sequences:
        if sequence not in cdn or any(not cdn[sequence].get(group, {}).get("download_url") for group in REQUIRED_GROUPS):
            raise ValueError(f"Missing required Aria/GT download URLs for {sequence}")
    processed = args.processed_root or infer_processed_root(args.out_root, set(sequences))
    saved = [read_json(processed / item / "summary.json")
             for item in old["train_sequences"] + old["val_sequences"]]
    keys = ("frame_stride", "window_size", "window_stride", "camera_stream_id")
    settings = {key: saved[0][key] for key in keys}
    if any(any(summary[key] != settings[key] for key in keys) for summary in saved):
        raise ValueError("Existing HOT3D preparation settings differ; audit before extending")
    if settings["window_size"] != 16 or settings["window_stride"] != 8:
        raise ValueError("This shared-cache pipeline expects the original ws16/s8 windows")
    for path in (args.visibility_python, args.visibility_checkpoint, args.pi3x_checkpoint,
                 args.mano_model_folder / "MANO_LEFT.pkl", args.mano_model_folder / "MANO_RIGHT.pkl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(args.visibility_python, os.X_OK):
        raise ValueError("Visibility Python is not executable")
    for path in (args.visibility_root, args.hand_uni_root, args.pi3_root, args.hot3d_code_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.compact_cache_root is None:
        import torch
        checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=False)
        cache_root = checkpoint["args"]["compact_cache_root"]
        if not isinstance(cache_root, str) or not cache_root.strip():
            raise ValueError("Training checkpoint has no compact_cache_root")
        args.compact_cache_root = Path(cache_root)
    if not args.compact_cache_root.is_dir():
        raise FileNotFoundError(args.compact_cache_root)
    log_root = args.out_root / "logs/incremental_exports"
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log(f"sequences={len(sequences)} train={len(plan['train_sequences'])} val={len(plan['val_sequences'])}")
        log(f"processed={processed}; compact={args.compact_cache_root}; settings={settings}")
        if args.dry_run:
            log("dry run complete; no files exported and no download launched")
            return
        completion = log_root / "completed.json"
        if completion.exists():
            completion.replace(log_root / "previous_completed.json")
        plan_path = log_root / "split_plan.json"
        write_json(plan_path, plan)
        all_list = log_root / "all_sequences.txt"
        all_list.write_text("".join(item + "\n" for item in sequences))
        pending = [item for item in sequences if not prepared_is_current(processed / item, settings)]
        log(f"preparation cached={len(sequences) - len(pending)} pending={len(pending)}")
        stable = {}
        started = time.monotonic()
        while pending:
            reasons = {}
            for sequence in list(pending):
                signature, reason = ready_signature(args.hot3d_root / "data" / sequence)
                if signature is None:
                    stable.pop(sequence, None)
                    reasons[sequence] = reason
                    continue
                previous, count = stable.get(sequence, (None, 0))
                count = count + 1 if signature == previous else 1
                stable[sequence] = (signature, count)
                if count < args.stable_checks:
                    reasons[sequence] = "waiting for stable required files"
                    continue
                command = [sys.executable, "-u", SCRIPTS / "prepare_hot3d_v15.py",
                    "--sequence-dir", args.hot3d_root / "data" / sequence,
                    "--hot3d-code-root", args.hot3d_code_root,
                    "--mano-model-folder", args.mano_model_folder,
                    "--out-dir", processed / sequence, "--split", "train",
                    "--stream-id", settings["camera_stream_id"],
                    "--frame-stride", settings["frame_stride"],
                    "--window-size", 16, "--window-stride", 8,
                    "--overlay-count", 0, "--overwrite"]
                # Only missing/incomplete preparation reaches --overwrite; valid output is untouched.
                run_stage("prepare_" + sequence, command, log_root)
                if not prepared_is_current(processed / sequence, settings):
                    raise RuntimeError(f"Preparation validation failed: {sequence}")
                pending.remove(sequence)
            write_json(log_root / "waiting.json", {"pending": pending, "reasons": reasons})
            if pending:
                log(f"waiting for {len(pending)} sequences; details={log_root / 'waiting.json'}")
                if args.max_wait_hours and time.monotonic() - started >= args.max_wait_hours * 3600:
                    raise TimeoutError("Download wait exceeded --max-wait-hours")
                time.sleep(args.poll_seconds)
        run_stage("split", [sys.executable, "-u", SCRIPTS / "build_hot3d_sample_split.py",
            "--processed-root", processed, "--sequence-list", all_list,
            "--out-dir", args.out_root / "manifests", "--fixed-split", plan_path,
            "--val-count", args.val_count, "--overwrite"], log_root)
        for split in ("train", "val"):
            run_stage("tracks_" + split, [sys.executable, "-u", SCRIPTS / "export_multihand_tracks.py",
                "--windows", args.out_root / f"manifests/{split}_windows.jsonl",
                "--out-root", args.out_root / "tracks" / split, "--max-hands", 2,
                "--status-json", log_root / f"tracks_{split}.json"], log_root)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED="1")
        for stage in ("visibility", "compact"):
            for split in ("train", "val"):
                name = f"{stage}_{split}"
                command = ["bash", SCRIPTS / "run_sharded_export.sh", "--log-dir", log_root / name,
                    "--num-shards", len(gpu_ids), "--"]
                if stage == "visibility":
                    command += [args.visibility_python, "-u", SCRIPTS / "export_hand_visibility.py",
                        "--detector-root", args.visibility_root, "--checkpoint", args.visibility_checkpoint,
                        "--out-root", args.out_root / "visibility" / split, "--backbone", "wilor"]
                else:
                    command += [sys.executable, "-u", SCRIPTS / "export_compact_pi3x_cache.py",
                        "--visibility-root", args.out_root / "visibility" / split,
                        "--hand-uni-root", args.hand_uni_root, "--pi3-root", args.pi3_root,
                        "--pi3x-checkpoint", args.pi3x_checkpoint, "--out-root", args.compact_cache_root,
                        "--pixel-limit", 180000, "--joint-patch-radius", 1,
                        "--global-grid-size", 4, "--feature-dtype", "float16"]
                command += ["--windows", args.out_root / f"manifests/{split}_windows.jsonl",
                    "--track-root", args.out_root / "tracks" / split, "--max-hands", 2, "--device", "cuda"]
                run_stage(name, command, log_root, env)
        write_json(log_root / "completed.json", {"sequences": len(sequences), "split": plan,
            "compact_cache_root": str(args.compact_cache_root), "completed_at": time.strftime("%FT%T")})
        log("pipeline complete; no training launched")


if __name__ == "__main__":
    main()
