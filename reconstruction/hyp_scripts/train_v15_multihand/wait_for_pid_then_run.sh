#!/usr/bin/env bash
set -euo pipefail

wait_pid=""
wait_status=""
wait_status_key="failed"
gpu=""
log=""
status=""
interval=30

usage() {
  cat <<'EOF'
Usage: wait_for_pid_then_run.sh \
  --wait-pid PID --wait-status JSON --gpu ID \
  [--wait-status-key KEY] \
  --log FILE --status-json FILE [--interval SEC] -- command [args...]

Waits for an existing process to exit, verifies that its exporter status has
"failed": 0, then execs the next exporter on the requested GPU. The command
receives --status-json FILE automatically.
EOF
}

while (($#)); do
  case "$1" in
    --wait-pid) wait_pid="$2"; shift 2 ;;
    --wait-status) wait_status="$2"; shift 2 ;;
    --wait-status-key) wait_status_key="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --log) log="$2"; shift 2 ;;
    --status-json) status="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$wait_pid" || -z "$wait_status" || -z "$gpu" \
      || -z "$log" || -z "$status" || "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$wait_pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid wait PID: $wait_pid" >&2
  exit 2
fi

mkdir -p "$(dirname "$log")" "$(dirname "$status")"
printf '[%s] waiting for pid=%s before GPU %s launch\n' \
  "$(date '+%F %T')" "$wait_pid" "$gpu"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep "$interval"
done

if [[ ! -f "$wait_status" ]]; then
  echo "Previous process exited without status: $wait_status" >&2
  exit 1
fi
if ! grep -Eq \
  "\"${wait_status_key}\"[[:space:]]*:[[:space:]]*0([,}]|$)" \
  "$wait_status"; then
  echo "Previous shard did not complete cleanly: $wait_status" >&2
  cat "$wait_status" >&2
  exit 1
fi

printf '[%s] previous shard complete; launching on GPU %s\n' \
  "$(date '+%F %T')" "$gpu"
exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$@" --status-json "$status" >"$log" 2>&1
