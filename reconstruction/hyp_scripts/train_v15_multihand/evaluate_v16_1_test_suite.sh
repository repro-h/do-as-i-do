#!/usr/bin/env bash
set -euo pipefail

python_bin=""
eval_script=""
checkpoint=""
full_v15=""
full_root=""
out_root=""
gpus=""
batch_size=32
num_workers=8
dexycb_visibility_root=""

usage() {
  cat <<'EOF'
Usage: evaluate_v16_1_test_suite.sh [options]

Required:
  --python FILE
  --eval-script FILE
  --checkpoint FILE
  --full-v15 DIR
  --full-root DIR
  --out-root DIR
  --gpus LIST

Optional:
  --batch-size N    Default: 32
  --num-workers N   Default: 8
  --dexycb-visibility-root DIR
                    Override DexYCB visibility cache root

Evaluates DexYCB S0 test followed by TACO test_1 through test_4.
EOF
}

while (($#)); do
  case "$1" in
    --python) python_bin="$2"; shift 2 ;;
    --eval-script) eval_script="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --full-v15) full_v15="$2"; shift 2 ;;
    --full-root) full_root="$2"; shift 2 ;;
    --out-root) out_root="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --num-workers) num_workers="$2"; shift 2 ;;
    --dexycb-visibility-root) dexycb_visibility_root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$python_bin" || -z "$eval_script" || -z "$checkpoint" \
      || -z "$full_v15" || -z "$full_root" || -z "$out_root" \
      || -z "$gpus" ]]; then
  usage >&2
  exit 2
fi

for path in "$python_bin" "$eval_script" "$checkpoint"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done
mkdir -p "$out_root"
dexycb_visibility_root=${dexycb_visibility_root:-"$full_v15/visibility/test"}

run_eval() {
  local name=$1
  local windows=$2
  local visibility=$3
  local tracks=$4
  local cache=$5
  local output="$out_root/${name}.json"
  local log="$out_root/${name}.log"

  printf '[%s] evaluating %s\n' "$(date '+%F %T')" "$name"
  env CUDA_VISIBLE_DEVICES="$gpus" PYTHONUNBUFFERED=1 \
    "$python_bin" -u "$eval_script" \
      --windows "$windows" \
      --visibility-root "$visibility" \
      --track-root "$tracks" \
      --compact-cache-root "$cache" \
      --checkpoint "$checkpoint" \
      --out-json "$output" \
      --batch-size "$batch_size" \
      --num-workers "$num_workers" \
      --max-hands 2 \
      --data-parallel \
      --device cuda \
    >"$log" 2>&1
  printf '[%s] completed %s: %s\n' "$(date '+%F %T')" "$name" "$output"
}

run_eval \
  dexycb_s0 \
  "$full_v15/manifests/test_windows.jsonl" \
  "$dexycb_visibility_root" \
  "$full_v15/tracks/test" \
  "$full_root/test/dexycb/compact_cache"

for split in test_1 test_2 test_3 test_4; do
  run_eval \
    "taco_${split}" \
    "$full_v15/taco/processed_v1/manifests/${split}_windows.jsonl" \
    "$full_v15/taco/visibility/test_all" \
    "$full_v15/taco/tracks/test_all" \
    "$full_root/test/taco/compact_cache"
done

printf '[%s] test suite complete\n' "$(date '+%F %T')"
