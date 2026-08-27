#!/usr/bin/env python3
"""Export original-camera Pi3X caches from V15 DexYCB S0 windows."""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


REQUIRED_WINDOW_KEYS = {
    "frame_indices",
    "geometry_patch_features",
    "metric_window_features",
    "confidence",
    "intrinsics_resized",
    "horizontal_mirror",
    "coordinate_frame",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-script", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--reuse-root")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--status-json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path):
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_streams(rows):
    streams = defaultdict(
        lambda: {"frames": {}, "intrinsics": None, "object_label": None}
    )
    for row in rows:
        stream = streams[row["stream_id"]]
        intrinsics = np.asarray(row["intrinsics"], dtype=np.float32).reshape(3, 3)
        if stream["intrinsics"] is None:
            stream["intrinsics"] = intrinsics
        elif not np.allclose(stream["intrinsics"], intrinsics):
            raise ValueError(f"Intrinsics change within {row['stream_id']}")
        if "object_label" in row:
            label = int(row["object_label"])
            if stream["object_label"] is None:
                stream["object_label"] = label
            elif stream["object_label"] != label:
                raise ValueError(f"Object label changes within {row['stream_id']}")
        for frame, image, label in zip(
            row["frame_indices"], row["image_paths"], row["label_paths"]
        ):
            stream["frames"][int(frame)] = (str(image), str(label))
    return streams


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        try:
            return yaml.safe_load(handle) or {}
        except yaml.constructor.ConstructorError:
            handle.seek(0)
            return yaml.load(handle, Loader=yaml.UnsafeLoader) or {}


def object_label(image_path):
    meta_path = Path(image_path).expanduser().resolve().parent.parent / "meta.yml"
    metadata = load_yaml(meta_path)
    ycb_ids = list(metadata.get("ycb_ids", []) or [])
    grasp_index = int(metadata.get("ycb_grasp_ind", 0))
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(f"Invalid object metadata in {meta_path}")
    return int(ycb_ids[grasp_index])


def expected_ranges(count, size, stride):
    size = min(max(1, size), count)
    stride = max(1, min(stride, size))
    if count <= size:
        starts = [0]
    else:
        starts = list(range(0, count - size + 1, stride))
        if starts[-1] != count - size:
            starts.append(count - size)
    return [(start, min(count, start + size)) for start in starts], size, stride


def valid_cache(stream_dir, count, window_size, window_stride):
    try:
        summary = json.loads((stream_dir / "summary.json").read_text(encoding="utf-8"))
        ranges, size, stride = expected_ranges(count, window_size, window_stride)
        if summary.get("coordinate_frame") != "original_camera":
            return False
        if bool(summary.get("horizontal_mirror")):
            return False
        if int(summary.get("num_frames", -1)) != count:
            return False
        if int(summary.get("window_size", -1)) != size:
            return False
        if int(summary.get("window_stride", -1)) != stride:
            return False
        records = summary.get("windows", [])
        if [(int(row["start"]), int(row["end"])) for row in records] != ranges:
            return False
        for start, end in ranges:
            path = stream_dir / "windows" / f"window_{start:06d}_{end:06d}.npz"
            with np.load(path, allow_pickle=False) as data:
                if not REQUIRED_WINDOW_KEYS.issubset(data.files):
                    return False
                if bool(data["horizontal_mirror"].item()):
                    return False
                if str(data["coordinate_frame"].item()) != "original_camera":
                    return False
                features = np.asarray(data["geometry_patch_features"])
                if features.shape[0] != end - start or not np.isfinite(features).all():
                    return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_inputs(stream_id, stream, stream_out):
    frame_items = sorted(stream["frames"].items())
    frames = []
    for output_index, (frame, (image, label)) in enumerate(frame_items):
        frames.append({
            "output_index": output_index,
            "output_frame": f"{output_index:06d}",
            "original_frame": f"{frame:06d}",
            "image_path": str(Path(image).expanduser().resolve()),
            "label_path": str(Path(label).expanduser().resolve()),
        })
    frame_map = stream_out / "dexycb_s0_frame_map.json"
    frame_map.write_text(json.dumps({
        "source": "dexycb_s0_multihand_window_v1",
        "stream_id": stream_id,
        "num_frames": len(frames),
        "frames": frames,
    }, indent=2), encoding="utf-8")
    intrinsics = stream_out / "intrinsics.json"
    intrinsics.write_text(json.dumps({
        "intrinsics": np.asarray(stream["intrinsics"]).tolist()
    }, indent=2), encoding="utf-8")
    label = stream["object_label"]
    if label is None:
        label = object_label(frames[0]["image_path"])
    return frame_map, intrinsics, label


def main():
    args = parse_args()
    streams = group_streams(load_rows(args.windows))
    items = sorted(streams.items())
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    items = items[args.shard_index::args.num_shards]
    out_root = Path(args.out_root).expanduser().resolve()
    reuse_root = None if not args.reuse_root else Path(args.reuse_root).expanduser().resolve()
    export_script = Path(args.export_script).expanduser().resolve()
    completed, reused, failures = [], [], []

    for index, (stream_id, stream) in enumerate(items, 1):
        count = len(stream["frames"])
        stream_out = out_root / stream_id
        if args.overwrite and stream_out.is_symlink():
            # Replace only the target link; never write through into V13.
            stream_out.unlink()
        if not args.overwrite and valid_cache(stream_out, count, args.window_size, args.window_stride):
            print(f"[{index}/{len(items)}] cached {stream_id}", flush=True)
            completed.append(stream_id)
            continue
        reuse = None if reuse_root is None else reuse_root / stream_id
        if (
            not args.overwrite
            and not stream_out.exists()
            and reuse is not None
            and valid_cache(reuse, count, args.window_size, args.window_stride)
        ):
            stream_out.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(reuse, stream_out, target_is_directory=True)
            print(f"[{index}/{len(items)}] reused {stream_id}", flush=True)
            reused.append(stream_id)
            completed.append(stream_id)
            continue
        try:
            stream_out.mkdir(parents=True, exist_ok=True)
            frame_map, intrinsics, label = write_inputs(stream_id, stream, stream_out)
            command = [
                sys.executable, "-u", str(export_script),
                "--frame-map-json", str(frame_map),
                "--intrinsics-json", str(intrinsics),
                "--hand-uni-root", args.hand_uni_root,
                "--pi3-root", args.pi3_root,
                "--checkpoint", args.checkpoint,
                "--out-dir", str(stream_out),
                "--object-label", str(label),
                "--window-size", str(args.window_size),
                "--window-stride", str(args.window_stride),
                "--pixel-limit", str(args.pixel_limit),
                "--confidence-threshold", str(args.confidence_threshold),
                "--feature-dtype", args.feature_dtype,
                "--device", args.device,
                "--export-metric-features", "--v13-minimal-cache", "--overwrite",
            ]
            print(f"[{index}/{len(items)}] export {stream_id} frames={count}", flush=True)
            subprocess.run(command, check=True)
            if not valid_cache(stream_out, count, args.window_size, args.window_stride):
                raise RuntimeError("export completed but cache validation failed")
            completed.append(stream_id)
        except Exception as error:
            failures.append({"stream_id": stream_id, "error": repr(error)})
            print(f"FAILED {stream_id}: {error}", flush=True)

    status = {
        "windows": str(Path(args.windows).expanduser().resolve()),
        "coordinate_frame": "original_camera",
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "requested": len(items),
        "completed": len(completed),
        "reused": len(reused),
        "failed": len(failures),
        "completed_streams": completed,
        "reused_streams": reused,
        "failures": failures,
        "out_root": str(out_root),
    }
    if args.status_json:
        path = Path(args.status_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
