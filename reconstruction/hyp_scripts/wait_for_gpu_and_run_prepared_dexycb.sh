#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: wait_for_gpu_and_run_prepared_dexycb.sh --run-dir DIR [options]

Wait for a sufficiently empty GPU, then continue a prepared DexYCB run.

Options:
  --run-dir DIR        Prepared DexYCB run directory (required)
  --gpus LIST          Comma-separated physical GPU IDs (default: 0,1,2,3,4,5,6,7)
  --min-free-mib N     Required free memory in MiB (default: 20000)
  --poll-seconds N     Seconds between checks (default: 60)
  --                   Forward remaining arguments to run_prepared_dexycb.sh
EOF
}

RUN_DIR=""
GPU_LIST="0,1,2,3,4,5,6,7"
MIN_FREE_MIB=20000
POLL_SECONDS=60
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --) shift; FORWARD_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_DIR" ]] || { usage >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_prepared_dexycb.sh"
IFS=',' read -r -a ALLOWED_GPUS <<< "$GPU_LIST"

is_allowed_gpu() {
  local candidate="$1"
  local allowed
  for allowed in "${ALLOWED_GPUS[@]}"; do
    [[ "$candidate" == "$allowed" ]] && return 0
  done
  return 1
}

while true; do
  best_gpu=""
  best_free=-1

  while IFS=',' read -r index free_mib; do
    index="${index//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    is_allowed_gpu "$index" || continue

    if (( free_mib >= MIN_FREE_MIB && free_mib > best_free )); then
      best_gpu="$index"
      best_free="$free_mib"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ -n "$best_gpu" ]]; then
    echo "[$timestamp] selected physical GPU $best_gpu with ${best_free} MiB free"
    exec env CUDA_VISIBLE_DEVICES="$best_gpu" bash "$RUNNER" \
      --run-dir "$RUN_DIR" \
      --gpu "$best_gpu" \
      "${FORWARD_ARGS[@]}"
  fi

  echo "[$timestamp] no GPU in {$GPU_LIST} has ${MIN_FREE_MIB} MiB free; retrying in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done
