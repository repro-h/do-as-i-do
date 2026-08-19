#!/usr/bin/env bash
set -euo pipefail

count=1
max_used_mib=2000
interval_seconds=60
stable_checks=3

usage() {
  cat <<'EOF'
Usage: wait_for_gpus_and_run.sh [options] -- command [args...]

Options:
  --count N             Number of GPUs required (default: 1)
  --max-used-mib N      Maximum used memory per available GPU (default: 2000)
  --interval N          Seconds between checks (default: 60)
  --stable-checks N     Consecutive successful checks before launch (default: 3)

The selected physical GPU indices are exported through CUDA_VISIBLE_DEVICES.
EOF
}

while (($#)); do
  case "$1" in
    --count)
      count="$2"
      shift 2
      ;;
    --max-used-mib)
      max_used_mib="$2"
      shift 2
      ;;
    --interval)
      interval_seconds="$2"
      shift 2
      ;;
    --stable-checks)
      stable_checks="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (($# == 0)); then
  echo "Missing command after --" >&2
  usage >&2
  exit 2
fi
if ((count < 1 || max_used_mib < 0 || interval_seconds < 1 || stable_checks < 1)); then
  echo "Invalid numeric option" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found" >&2
  exit 1
fi

successful_checks=0
previous_selection=""

while true; do
  mapfile -t rows < <(
    nvidia-smi \
      --query-gpu=index,memory.used \
      --format=csv,noheader,nounits \
      | tr -d ' '
  )

  available=()
  status=()
  for row in "${rows[@]}"; do
    IFS=',' read -r index used_mib <<<"$row"
    status+=("${index}:${used_mib}MiB")
    if ((used_mib <= max_used_mib)); then
      available+=("$index")
    fi
  done

  selection=""
  if ((${#available[@]} >= count)); then
    selection="$(IFS=,; echo "${available[*]:0:count}")"
  fi

  if [[ -n "$selection" && "$selection" == "$previous_selection" ]]; then
    ((successful_checks += 1))
  elif [[ -n "$selection" ]]; then
    successful_checks=1
  else
    successful_checks=0
  fi
  previous_selection="$selection"

  printf '[%s] GPU memory: %s; candidate=%s; stable=%d/%d\n' \
    "$(date '+%F %T')" \
    "${status[*]}" \
    "${selection:-none}" \
    "$successful_checks" \
    "$stable_checks"

  if [[ -n "$selection" ]] && ((successful_checks >= stable_checks)); then
    export CUDA_VISIBLE_DEVICES="$selection"
    echo "[$(date '+%F %T')] Launching on CUDA_VISIBLE_DEVICES=$selection"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
    exec "$@"
  fi

  sleep "$interval_seconds"
done
