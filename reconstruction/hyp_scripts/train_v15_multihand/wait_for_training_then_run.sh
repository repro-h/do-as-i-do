#!/usr/bin/env bash
set -euo pipefail

wait_pid=""
python_bin=""
history=""
checkpoint=""
min_epoch=0
gpus=""
interval=60

usage() {
  cat <<'EOF'
Usage: wait_for_training_then_run.sh [options] -- command [args...]

Required:
  --wait-pid PID       Existing training process to wait for
  --python FILE        Python used to validate history.json
  --history FILE       Training history written by the existing run
  --checkpoint FILE    Checkpoint that must exist before continuation
  --min-epoch N        Required final completed epoch
  --gpus LIST          CUDA order for the continuation, e.g. 2,1,3,4

Optional:
  --interval SEC       Poll interval (default: 60)
EOF
}

while (($#)); do
  case "$1" in
    --wait-pid) wait_pid="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --history) history="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --min-epoch) min_epoch="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$wait_pid" || -z "$python_bin" || -z "$history" \
      || -z "$checkpoint" || -z "$gpus" || "$min_epoch" -le 0 \
      || "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$wait_pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid wait PID: $wait_pid" >&2
  exit 2
fi

printf '[%s] waiting for training pid=%s through epoch %s\n' \
  "$(date '+%F %T')" "$wait_pid" "$min_epoch"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep "$interval"
done

if [[ ! -f "$checkpoint" ]]; then
  echo "Training exited without checkpoint: $checkpoint" >&2
  exit 1
fi
if [[ ! -f "$history" ]]; then
  echo "Training exited without history: $history" >&2
  exit 1
fi

"$python_bin" - "$history" "$min_epoch" <<'PY'
import json
import sys
from pathlib import Path

history = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = int(sys.argv[2])
completed = max((int(row["epoch"]) for row in history), default=0)
print(f"completed epoch={completed}, required epoch={required}", flush=True)
if completed < required:
    raise SystemExit(1)
PY

printf '[%s] base training complete; launching continuation on GPUs %s\n' \
  "$(date '+%F %T')" "$gpus"
exec env CUDA_VISIBLE_DEVICES="$gpus" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 "$@"
