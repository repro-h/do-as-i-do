#!/usr/bin/env bash
set -euo pipefail

python_bin=""
full_v15_root=""
old_mixed_root=""
new_mixed_root=""
compact_cache_root=""
hand_uni_root=""
pi3_root=""
pi3x_checkpoint=""
visibility_train_status_dir=""
visibility_val_status_dir=""
visibility_shards=8
export_gpus="0,1,2,3,4,5,6,7"
train_gpus="1,2,3,4"
interval=60
epochs=30
batch_size=32
num_workers=8
train_windows_per_dataset=2048

usage() {
  cat <<'EOF'
Usage: wait_for_track_aligned_visibility_then_train.sh [options]

Required:
  --python FILE
  --full-v15-root DIR
  --old-mixed-root DIR
  --new-mixed-root DIR
  --compact-cache-root DIR
  --hand-uni-root DIR
  --pi3-root DIR
  --pi3x-checkpoint FILE

Optional:
  --visibility-train-status-dir DIR
  --visibility-val-status-dir DIR
  --visibility-shards N            Default: 8 for both train and val
  --export-gpus LIST               Default: 0,1,2,3,4,5,6,7
  --train-gpus LIST                Default: 1,2,3,4; first is DP primary
  --interval SEC                   Default: 60
  --epochs N                       Default: 30
  --batch-size N                   Default: 32
  --num-workers N                  Default: 8
  --train-windows-per-dataset N    Default: 2048

The script waits for track-aligned DexYCB train/val visibility exports,
validates their track IDs, rewrites the previous five-dataset config, rebuilds
the full manifests, fills only missing compact Pi3X caches, and starts a fresh
multi-GPU training run.
EOF
}

while (($#)); do
  case "$1" in
    --python) python_bin="$2"; shift 2 ;;
    --full-v15-root) full_v15_root="$2"; shift 2 ;;
    --old-mixed-root) old_mixed_root="$2"; shift 2 ;;
    --new-mixed-root) new_mixed_root="$2"; shift 2 ;;
    --compact-cache-root) compact_cache_root="$2"; shift 2 ;;
    --hand-uni-root) hand_uni_root="$2"; shift 2 ;;
    --pi3-root) pi3_root="$2"; shift 2 ;;
    --pi3x-checkpoint) pi3x_checkpoint="$2"; shift 2 ;;
    --visibility-train-status-dir)
      visibility_train_status_dir="$2"; shift 2 ;;
    --visibility-val-status-dir)
      visibility_val_status_dir="$2"; shift 2 ;;
    --visibility-shards) visibility_shards="$2"; shift 2 ;;
    --export-gpus) export_gpus="$2"; shift 2 ;;
    --train-gpus) train_gpus="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --num-workers) num_workers="$2"; shift 2 ;;
    --train-windows-per-dataset)
      train_windows_per_dataset="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$python_bin" || -z "$full_v15_root" \
      || -z "$old_mixed_root" || -z "$new_mixed_root" \
      || -z "$compact_cache_root" || -z "$hand_uni_root" \
      || -z "$pi3_root" || -z "$pi3x_checkpoint" ]]; then
  usage >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_script="$script_dir/build_mixed_smoke_manifests.py"
compact_export_script="$script_dir/export_compact_pi3x_cache.py"
run_sharded_script="$script_dir/run_sharded_export.sh"
train_script="$script_dir/train_v16_1_compact_pi3x.py"

if [[ -z "$visibility_train_status_dir" ]]; then
  visibility_train_status_dir="$full_v15_root/logs/visibility_track_aligned_train_8gpu"
fi
if [[ -z "$visibility_val_status_dir" ]]; then
  visibility_val_status_dir="$full_v15_root/logs/visibility_track_aligned_val_8gpu"
fi

dex_train_windows="$full_v15_root/manifests/train_windows.jsonl"
dex_val_windows="$full_v15_root/manifests/val_windows.jsonl"
dex_train_tracks="$full_v15_root/tracks/train"
dex_val_tracks="$full_v15_root/tracks/val"
dex_train_visibility="$full_v15_root/visibility_track_aligned/train"
dex_val_visibility="$full_v15_root/visibility_track_aligned/val"

source_report="$old_mixed_root/manifests/selection_report.json"
manifest_dir="$new_mixed_root/manifests"
updated_config="$manifest_dir/five_dataset_track_aligned_config.json"
train_windows="$manifest_dir/train_windows.jsonl"
val_windows="$manifest_dir/val_windows.jsonl"
compact_train_logs="$new_mixed_root/logs/compact_train"
compact_val_logs="$new_mixed_root/logs/compact_val"
checkpoint_dir="$new_mixed_root/checkpoints/v16_2_five_dataset_track_aligned_full_v1"

required_paths=(
  "$python_bin" "$build_script" "$compact_export_script"
  "$run_sharded_script" "$train_script" "$dex_train_windows"
  "$dex_val_windows" "$source_report" "$hand_uni_root" "$pi3_root"
  "$pi3x_checkpoint"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

IFS=',' read -r -a export_gpu_array <<<"$export_gpus"
compact_shards=${#export_gpu_array[@]}
if ((compact_shards < 1)); then
  echo "--export-gpus must contain at least one GPU" >&2
  exit 2
fi

mkdir -p "$manifest_dir" "$compact_train_logs" "$compact_val_logs" \
  "$checkpoint_dir"

check_visibility() {
  "$python_bin" - \
    "$dex_train_windows" "$visibility_train_status_dir" \
    "$dex_train_visibility" "$dex_train_tracks" \
    "$dex_val_windows" "$visibility_val_status_dir" \
    "$dex_val_visibility" "$dex_val_tracks" \
    "$visibility_shards" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np


def expected_streams(manifest):
    streams = set()
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                streams.add(str(json.loads(line)["stream_id"]))
    return streams


def check_status(name, manifest, status_dir, visibility_root, track_root, shards):
    expected = expected_streams(manifest)
    status_paths = [
        Path(status_dir) / f"status_{index}.json"
        for index in range(shards)
    ]
    present = [path for path in status_paths if path.is_file()]
    if len(present) != len(status_paths):
        return "waiting", (
            f"{name}: status files {len(present)}/{len(status_paths)}, "
            f"expected streams={len(expected)}"
        )

    streams = completed = cached = failed = 0
    for path in status_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            return "failed", f"{name}: cannot read {path}: {error}"
        streams += int(data.get("streams", -1))
        completed += int(data.get("completed", -1))
        cached += int(data.get("cached", -1))
        failed += int(data.get("failed", -1))
    summary = (
        f"{name}: streams={streams}, completed={completed}, cached={cached}, "
        f"failed={failed}, expected={len(expected)}"
    )
    if min(streams, completed, cached, failed) < 0:
        return "failed", summary + " (invalid status schema)"
    if failed:
        return "failed", summary
    if streams != len(expected) or completed != streams:
        return "failed", summary + " (status/manifest mismatch)"

    visibility_root = Path(visibility_root)
    track_root = Path(track_root)
    missing = []
    broken = []
    observed = valid_track_ids = 0
    for stream in sorted(expected):
        cache_path = visibility_root / stream / "visibility_cache.npz"
        track_path = track_root / stream / "tracks.npz"
        if not cache_path.is_file() or not track_path.is_file():
            missing.append(stream)
            continue
        try:
            if cache_path.stat().st_mtime_ns < track_path.stat().st_mtime_ns:
                broken.append(f"{stream}: visibility older than tracks")
                continue
            with np.load(cache_path, allow_pickle=False) as cache:
                valid = np.asarray(cache["visibility_valid"], dtype=bool)
                track_ids = np.asarray(cache["track_ids"], dtype=np.int64)
                version = str(cache["cache_version"].item())
            if version != "hand_visibility_detector_multihand_v2":
                broken.append(f"{stream}: cache version {version}")
                continue
            if valid.shape != track_ids.shape:
                broken.append(
                    f"{stream}: valid {valid.shape} != track_ids {track_ids.shape}"
                )
                continue
            observed += int(valid.sum())
            valid_track_ids += int((valid & (track_ids >= 0)).sum())
        except Exception as error:
            broken.append(f"{stream}: {error!r}")
    if missing or broken or observed != valid_track_ids:
        return "failed", (
            summary + f", missing={len(missing)}, broken={len(broken)}, "
            f"observed={observed}, valid_track_ids={valid_track_ids}"
        )
    return "ready", summary + f", aligned observed={observed}"


shards = int(sys.argv[9])
specs = (
    ("train", sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], shards),
    ("val", sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], shards),
)
states = []
for spec in specs:
    state, message = check_status(*spec)
    states.append(state)
    print(message, flush=True)
if "failed" in states:
    raise SystemExit(2)
if all(state == "ready" for state in states):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "[$(date '+%F %T')] waiting for track-aligned DexYCB visibility"
while true; do
  set +e
  check_visibility
  result=$?
  set -e
  case "$result" in
    0) break ;;
    1) sleep "$interval" ;;
    *) echo "Visibility validation failed; pipeline stopped" >&2; exit 1 ;;
  esac
done

echo "[$(date '+%F %T')] visibility ready; rewriting five-dataset config"
"$python_bin" - \
  "$source_report" "$updated_config" \
  "$dex_train_visibility" "$dex_val_visibility" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]).expanduser().resolve()
output_path = Path(sys.argv[2]).expanduser().resolve()
train_visibility = str(Path(sys.argv[3]).expanduser().resolve())
val_visibility = str(Path(sys.argv[4]).expanduser().resolve())
report = json.loads(report_path.read_text(encoding="utf-8"))
source_config = Path(report["config"]).expanduser().resolve()
config = json.loads(source_config.read_text(encoding="utf-8"))
entries = config.get("datasets", config)
matches = [entry for entry in entries if str(entry["name"]).lower() == "dexycb"]
if len(matches) != 1:
    raise RuntimeError(f"Expected one dexycb entry, found {len(matches)}")
matches[0]["visibility_train_root"] = train_visibility
matches[0]["visibility_val_root"] = val_visibility
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(f"source config: {source_config}")
print(f"updated config: {output_path}")
PY

echo "[$(date '+%F %T')] rebuilding full mixed manifests"
"$python_bin" -u "$build_script" \
  --config "$updated_config" \
  --out-dir "$manifest_dir" \
  --train-streams-per-dataset 0 \
  --val-streams-per-dataset 0 \
  --windows-per-stream 0 \
  --seed 42 \
  --overwrite

echo "[$(date '+%F %T')] filling train compact cache with $compact_shards GPUs"
CUDA_VISIBLE_DEVICES="$export_gpus" \
  "$run_sharded_script" \
  --log-dir "$compact_train_logs" \
  --num-shards "$compact_shards" \
  -- "$python_bin" -u "$compact_export_script" \
  --windows "$train_windows" \
  --hand-uni-root "$hand_uni_root" \
  --pi3-root "$pi3_root" \
  --pi3x-checkpoint "$pi3x_checkpoint" \
  --out-root "$compact_cache_root" \
  --max-hands 2 \
  --pixel-limit 180000 \
  --feature-dtype float16 \
  --joint-patch-radius 1 \
  --global-grid-size 4 \
  --device cuda

echo "[$(date '+%F %T')] filling val compact cache with $compact_shards GPUs"
CUDA_VISIBLE_DEVICES="$export_gpus" \
  "$run_sharded_script" \
  --log-dir "$compact_val_logs" \
  --num-shards "$compact_shards" \
  -- "$python_bin" -u "$compact_export_script" \
  --windows "$val_windows" \
  --hand-uni-root "$hand_uni_root" \
  --pi3-root "$pi3_root" \
  --pi3x-checkpoint "$pi3x_checkpoint" \
  --out-root "$compact_cache_root" \
  --max-hands 2 \
  --pixel-limit 180000 \
  --feature-dtype float16 \
  --joint-patch-radius 1 \
  --global-grid-size 4 \
  --device cuda

echo "[$(date '+%F %T')] compact cache ready; starting training"
echo "CUDA order: $train_gpus (primary physical GPU: ${train_gpus%%,*})"
exec env \
  CUDA_VISIBLE_DEVICES="$train_gpus" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 \
  "$python_bin" -u "$train_script" \
  --train-windows "$train_windows" \
  --val-windows "$val_windows" \
  --hand-uni-root "$hand_uni_root" \
  --pi3-root "$pi3_root" \
  --pi3x-checkpoint "$pi3x_checkpoint" \
  --compact-cache-root "$compact_cache_root" \
  --out-dir "$checkpoint_dir" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --num-workers "$num_workers" \
  --max-hands 2 \
  --train-windows-per-dataset "$train_windows_per_dataset" \
  --data-parallel \
  --device cuda
