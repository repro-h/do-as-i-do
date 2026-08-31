#!/usr/bin/env bash
set -euo pipefail

python_bin=""
train_windows=""
val_windows=""
train_status_dir=""
val_status_dir=""
train_shards=0
val_shards=0
gpus=""
interval=60

usage() {
  cat <<'EOF'
Usage: wait_for_compact_cache_then_run.sh [options] -- command [args...]

Required options:
  --python FILE              Python used to validate exporter status files
  --train-windows FILE       Full training JSONL manifest
  --val-windows FILE         Full validation JSONL manifest
  --train-status-dir DIR     Directory containing train status_*.json files
  --val-status-dir DIR       Directory containing val status_*.json files
  --train-shards N           Expected number of train exporter shards
  --val-shards N             Expected number of val exporter shards
  --gpus LIST                CUDA order for the launched command, e.g. 2,1,3,4

Optional:
  --interval SEC             Poll interval (default: 60)

The first GPU in --gpus becomes logical cuda:0 and therefore the
torch.nn.DataParallel primary GPU. The command starts only after every status
file reports failed=0, requested=completed+cached, and the shard totals match
the corresponding manifest line counts.
EOF
}

while (($#)); do
  case "$1" in
    --python) python_bin="$2"; shift 2 ;;
    --train-windows) train_windows="$2"; shift 2 ;;
    --val-windows) val_windows="$2"; shift 2 ;;
    --train-status-dir) train_status_dir="$2"; shift 2 ;;
    --val-status-dir) val_status_dir="$2"; shift 2 ;;
    --train-shards) train_shards="$2"; shift 2 ;;
    --val-shards) val_shards="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$python_bin" || -z "$train_windows" || -z "$val_windows" \
      || -z "$train_status_dir" || -z "$val_status_dir" || -z "$gpus" \
      || "$train_shards" -le 0 || "$val_shards" -le 0 || "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi

for path in "$python_bin" "$train_windows" "$val_windows"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

check_statuses() {
  "$python_bin" - \
    "$train_windows" "$train_status_dir" "$train_shards" \
    "$val_windows" "$val_status_dir" "$val_shards" <<'PY'
import json
import sys
from pathlib import Path


def count_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def check_split(name, manifest, status_dir, expected_shards):
    expected_rows = count_rows(manifest)
    paths = [Path(status_dir) / f"status_{index}.json"
             for index in range(int(expected_shards))]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return "waiting", (
            f"{name}: status files {len(paths) - len(missing)}/{len(paths)}, "
            f"manifest windows={expected_rows}"
        )

    requested = completed = cached = failed = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            return "failed", f"{name}: cannot read {path}: {error}"
        shard_requested = int(data.get("unique_requested", data.get("requested", -1)))
        shard_completed = int(data.get("completed", -1))
        shard_cached = int(data.get("cached", -1))
        shard_failed = int(data.get("failed", -1))
        if min(shard_requested, shard_completed, shard_cached, shard_failed) < 0:
            return "failed", f"{name}: incomplete status schema in {path}"
        requested += shard_requested
        completed += shard_completed
        cached += shard_cached
        failed += shard_failed

    summary = (
        f"{name}: shards={len(paths)}, requested={requested}, "
        f"completed={completed}, cached={cached}, failed={failed}, "
        f"manifest windows={expected_rows}"
    )
    if failed:
        return "failed", summary
    if requested != expected_rows:
        return "failed", summary + " (requested/manifest mismatch)"
    if completed + cached != requested:
        return "failed", summary + " (unfinished windows)"
    return "ready", summary


states = []
for split in (
    ("train", sys.argv[1], sys.argv[2], sys.argv[3]),
    ("val", sys.argv[4], sys.argv[5], sys.argv[6]),
):
    state, message = check_split(*split)
    states.append(state)
    print(message, flush=True)

if "failed" in states:
    raise SystemExit(2)
if all(state == "ready" for state in states):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "[$(date '+%F %T')] waiting for compact train/val exports"
echo "CUDA launch order: $gpus (primary physical GPU: ${gpus%%,*})"

while true; do
  set +e
  check_statuses
  result=$?
  set -e
  case "$result" in
    0) break ;;
    1) sleep "$interval" ;;
    *) echo "Compact cache export failed validation; training not started" >&2; exit 1 ;;
  esac
done

echo "[$(date '+%F %T')] compact exports complete; launching training"
exec env CUDA_VISIBLE_DEVICES="$gpus" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 "$@"
