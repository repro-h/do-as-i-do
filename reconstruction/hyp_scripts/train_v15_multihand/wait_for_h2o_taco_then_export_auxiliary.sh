#!/usr/bin/env bash
set -euo pipefail

pi3_python=""
visibility_python=""
geometry_root=""
full_v15_root=""
detector_root=""
detector_checkpoint=""
gpus="0,1,2,3,4,5,6,7"
shards=8
interval=60
h2o_pid=""
taco_pid=""

usage() {
  cat <<'EOF'
Usage: wait_for_h2o_taco_then_export_auxiliary.sh [options]

Required:
  --pi3-python FILE
  --visibility-python FILE
  --geometry-root DIR
  --full-v15-root DIR
  --detector-root DIR
  --detector-checkpoint FILE

Optional:
  --h2o-pid PID       Wait for this producer process before checking H2O status
  --taco-pid PID      Wait for this producer process before checking TACO status
  --gpus LIST         Default: 0,1,2,3,4,5,6,7
  --shards N          Default: 8
  --interval SEC      Default: 60

Waits for H2O and TACO preparation, creates the TACO T64 manifests, then
incrementally exports tracks and track-aligned visibility for H2O train and
TACO train/val. Existing compatible caches are reused.
EOF
}

while (($#)); do
  case "$1" in
    --pi3-python) pi3_python="$2"; shift 2 ;;
    --visibility-python) visibility_python="$2"; shift 2 ;;
    --geometry-root) geometry_root="$2"; shift 2 ;;
    --full-v15-root) full_v15_root="$2"; shift 2 ;;
    --detector-root) detector_root="$2"; shift 2 ;;
    --detector-checkpoint) detector_checkpoint="$2"; shift 2 ;;
    --h2o-pid) h2o_pid="$2"; shift 2 ;;
    --taco-pid) taco_pid="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --shards) shards="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in pi3_python visibility_python geometry_root full_v15_root \
  detector_root detector_checkpoint; do
  if [[ -z "${!value}" ]]; then
    echo "Missing --${value//_/-}" >&2
    usage >&2
    exit 2
  fi
done
if ((shards < 1 || interval < 1)); then
  echo "--shards and --interval must be positive" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rewindow="$script_dir/rewindow_manifest.py"
tracks="$script_dir/export_multihand_tracks.py"
visibility="$script_dir/export_hand_visibility.py"
run_sharded="$script_dir/run_sharded_export.sh"

h2o_root="$geometry_root/h2o"
taco_processed="$full_v15_root/taco/processed_v1"
taco_t64="$geometry_root/manifests/taco"
audit_root="$geometry_root/audits"
log_root="$geometry_root/logs/auxiliary"

wait_pid() {
  local name=$1
  local pid=$2
  if [[ -z "$pid" ]]; then
    echo "[$(date '+%F %T')] $name PID not supplied; validating status directly"
    return
  fi
  echo "[$(date '+%F %T')] waiting for $name pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$interval"
  done
  echo "[$(date '+%F %T')] $name producer exited"
}

validate_status() {
  "$pi3_python" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

name, path = sys.argv[1], Path(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"{name}: missing status file: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
if name == "H2O":
    requested = int(data["sequences"])
    completed = int(data["completed"])
    failed = len(data.get("failures", []))
else:
    requested = int(data["requested_sequences"])
    completed = int(data["completed_sequences"])
    failed = int(data["failed_sequences"])
print(f"{name}: requested={requested}, completed={completed}, failed={failed}")
if requested <= 0 or completed != requested or failed:
    raise SystemExit(f"{name}: preparation is incomplete")
PY
}

run_tracks() {
  local name=$1 manifest=$2 out_root=$3
  local logs="$log_root/${name}_tracks"
  mkdir -p "$logs"
  echo "[$(date '+%F %T')] tracks: $name"
  "$pi3_python" -u "$tracks" \
    --windows "$manifest" \
    --out-root "$out_root" \
    --max-hands 2 \
    --status-json "$logs/status.json"
}

run_visibility() {
  local name=$1 manifest=$2 track_root=$3 out_root=$4
  local logs="$log_root/${name}_visibility"
  mkdir -p "$logs"
  echo "[$(date '+%F %T')] visibility: $name on GPUs $gpus"
  CUDA_VISIBLE_DEVICES="$gpus" "$run_sharded" \
    --log-dir "$logs" \
    --num-shards "$shards" \
    -- "$visibility_python" -u "$visibility" \
      --windows "$manifest" \
      --detector-root "$detector_root" \
      --checkpoint "$detector_checkpoint" \
      --track-root "$track_root" \
      --out-root "$out_root" \
      --backbone wilor \
      --max-hands 2 \
      --device cuda
}

mkdir -p "$taco_t64" "$audit_root" "$log_root"
wait_pid H2O "$h2o_pid"
wait_pid TACO "$taco_pid"
validate_status H2O "$h2o_root/status.json"
validate_status TACO "$taco_processed/status.json"

echo "[$(date '+%F %T')] rebuilding TACO T64 manifests"
for split in train val; do
  "$pi3_python" -u "$rewindow" \
    --input "$taco_processed/manifests/${split}_windows.jsonl" \
    --output "$taco_t64/${split}_windows.jsonl" \
    --report "$audit_root/taco_${split}.json" \
    --window-size 64 \
    --window-stride 32 \
    --overwrite
done

h2o_train="$h2o_root/manifests/train_windows.jsonl"
taco_train="$taco_t64/train_windows.jsonl"
taco_val="$taco_t64/val_windows.jsonl"

run_tracks h2o_train "$h2o_train" "$full_v15_root/h2o/tracks/train"
run_tracks taco_train "$taco_train" "$full_v15_root/taco/tracks/train"
run_tracks taco_val "$taco_val" "$full_v15_root/taco/tracks/val"

run_visibility h2o_train "$h2o_train" \
  "$full_v15_root/h2o/tracks/train" "$full_v15_root/h2o/visibility/train"
run_visibility taco_train "$taco_train" \
  "$full_v15_root/taco/tracks/train" "$full_v15_root/taco/visibility/train"
run_visibility taco_val "$taco_val" \
  "$full_v15_root/taco/tracks/val" "$full_v15_root/taco/visibility/val"

echo "[$(date '+%F %T')] H2O/TACO T64 auxiliary pipeline complete"
