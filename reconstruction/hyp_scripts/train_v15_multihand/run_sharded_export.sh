#!/usr/bin/env bash
set -euo pipefail

log_dir=""
num_shards=0

usage() {
  cat <<'EOF'
Usage: run_sharded_export.sh --log-dir DIR --num-shards N -- command [args...]

CUDA_VISIBLE_DEVICES must contain at least N comma-separated physical GPU IDs.
The command receives --num-shards N, --shard-index I and --status-json PATH.
EOF
}

while (($#)); do
  case "$1" in
    --log-dir) log_dir="$2"; shift 2 ;;
    --num-shards) num_shards="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$log_dir" || "$num_shards" -lt 1 || "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES:-}"
if ((${#gpus[@]} < num_shards)); then
  echo "Need $num_shards GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

mkdir -p "$log_dir"
pids=()
for ((shard=0; shard<num_shards; shard++)); do
  log="$log_dir/shard_${shard}.log"
  status="$log_dir/status_${shard}.json"
  CUDA_VISIBLE_DEVICES="${gpus[$shard]}" \
    PYTHONUNBUFFERED=1 \
    "$@" \
    --num-shards "$num_shards" \
    --shard-index "$shard" \
    --status-json "$status" \
    >"$log" 2>&1 &
  pids+=("$!")
  printf 'shard=%d gpu=%s pid=%s log=%s\n' \
    "$shard" "${gpus[$shard]}" "$!" "$log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
