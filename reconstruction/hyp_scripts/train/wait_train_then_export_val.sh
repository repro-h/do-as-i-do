#!/usr/bin/env bash
set -euo pipefail

# Wait until all train shard runners exit, then export validation on whichever
# allowed GPUs have enough free memory. Paths and thresholds are configurable
# through environment variables.

DO_AS_I_DO="${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}"
HANDFLOW="${HANDFLOW:-$DO_AS_I_DO/reconstruction/hyp_modules/HandFlow}"
HANDFLOW_PYTHON="${HANDFLOW_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/handflow/bin/python}"
HANDFLOW_CKPT="${HANDFLOW_CKPT:-$HANDFLOW/weights/handflow_denoiser.pt}"

HAMER_CKPT="${HAMER_CKPT:-/home/mengxiangting/nas/mengxt/Projects/Dyn-HaMR/_DATA/hamer_ckpts/checkpoints/hamer.ckpt}"
DETECTOR_CKPT="${DETECTOR_CKPT:-/home/mengxiangting/nas/mengxt/Projects/WiLoR/pretrained_models/detector.pt}"
MANO_ROOT="${MANO_ROOT:-/home/mengxiangting/nas/mengxt/Projects/Pi3_WiLoR_Hand/mano_data/mano}"
HANDFLOW_NORMALIZATION_STATS="${HANDFLOW_NORMALIZATION_STATS:-/home/mengxiangting/nas/mengxt/Projects/hand/HandFlow/weights/normalization_stats.npz}"

HYBRID_ROOT="${HYBRID_ROOT:-$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1}"
VAL_MANIFEST="${VAL_MANIFEST:-$HYBRID_ROOT/manifests/val.jsonl}"
TRAIN_RUNS="${TRAIN_RUNS:-$HYBRID_ROOT/handflow_cache/train_v1/runs_7gpu_keep_overlay}"
VAL_STREAMS="${VAL_STREAMS:-$HYBRID_ROOT/handflow_cache/val_v1/streams}"
VAL_RUNS="${VAL_RUNS:-$HYBRID_ROOT/handflow_cache/val_v1/runs_auto_gpu}"

ALLOWED_GPUS="${ALLOWED_GPUS:-0 2 3 4 5 6 7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
KEEP_VAL_OVERLAY="${KEEP_VAL_OVERLAY:-1}"

export PATH="/home/mengxiangting/nas/mengxt/anaconda3/envs/handflow/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH PYTHONNOUSERSITE
unset PYTORCH_CUDA_ALLOC_CONF

timestamp() {
  /bin/date '+%Y-%m-%d %H:%M:%S'
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[$(timestamp)] missing file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[$(timestamp)] missing directory: $1" >&2
    exit 2
  fi
}

count_live_train_runners() {
  local count=0
  local pid_file pid
  shopt -s nullglob
  for pid_file in "$TRAIN_RUNS"/shard_*/run.pid; do
    pid=$(/bin/cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" ]] && /bin/kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  shopt -u nullglob
  echo "$count"
}

find_free_gpus() {
  local allowed=" $ALLOWED_GPUS "
  local gpu free
  while IFS=, read -r gpu free; do
    gpu="${gpu//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    if [[ "$allowed" == *" $gpu "* ]] && [[ "$free" -ge "$MIN_FREE_MIB" ]]; then
      echo "$gpu"
    fi
  done < <(
    /usr/bin/nvidia-smi \
      --query-gpu=index,memory.free \
      --format=csv,noheader,nounits
  )
}

for path in \
  "$VAL_MANIFEST" \
  "$HANDFLOW_CKPT" \
  "$HAMER_CKPT" \
  "$DETECTOR_CKPT" \
  "$HANDFLOW_NORMALIZATION_STATS"
do
  require_file "$path"
done
require_dir "$HANDFLOW"
require_dir "$MANO_ROOT"
require_dir "$TRAIN_RUNS"

echo "[$(timestamp)] waiting for all train shard runners to exit"
echo "[$(timestamp)] train runs: $TRAIN_RUNS"

while true; do
  live_train=$(count_live_train_runners)
  echo "[$(timestamp)] live train shards: $live_train"
  if [[ "$live_train" -eq 0 ]]; then
    break
  fi
  /bin/sleep "$POLL_SECONDS"
done

echo "[$(timestamp)] all train shard runners have exited"
echo "[$(timestamp)] waiting for a GPU with at least $MIN_FREE_MIB MiB free"

while true; do
  FREE_GPUS=($(find_free_gpus))
  if [[ "${#FREE_GPUS[@]}" -gt 0 ]]; then
    break
  fi
  echo "[$(timestamp)] no eligible GPU is currently free"
  /bin/sleep "$POLL_SECONDS"
done

num_shards="${#FREE_GPUS[@]}"
/bin/mkdir -p "$VAL_STREAMS" "$VAL_RUNS"
cd "$DO_AS_I_DO"

echo "[$(timestamp)] launching val on GPUs: ${FREE_GPUS[*]}"
echo "[$(timestamp)] num val shards: $num_shards"

for shard_index in "${!FREE_GPUS[@]}"; do
  gpu_id="${FREE_GPUS[$shard_index]}"
  shard_run="$VAL_RUNS/shard_${shard_index}_gpu${gpu_id}"
  /bin/mkdir -p "$shard_run"

  command=(
    "$HANDFLOW_PYTHON" -u
    reconstruction/hyp_scripts/train/run_handflow_hybrid_jobs.py
    --manifest "$VAL_MANIFEST"
    --handflow-root "$HANDFLOW"
    --handflow-python "$HANDFLOW_PYTHON"
    --fm-ckpt "$HANDFLOW_CKPT"
    --out-root "$VAL_STREAMS"
    --status-json "$shard_run/status.json"
    --device cuda
    --num-shards "$num_shards"
    --shard-index "$shard_index"
  )
  if [[ "$KEEP_VAL_OVERLAY" == "1" ]]; then
    command+=(--keep-overlay)
  fi

  /usr/bin/nohup /usr/bin/env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    HAMER_CKPT="$HAMER_CKPT" \
    DETECTOR_CKPT="$DETECTOR_CKPT" \
    MANO_ROOT="$MANO_ROOT" \
    HANDFLOW_NORMALIZATION_STATS="$HANDFLOW_NORMALIZATION_STATS" \
    "${command[@]}" \
    > "$shard_run/run.log" 2>&1 < /dev/null &

  pid=$!
  printf '%s\n' "$pid" > "$shard_run/run.pid"
  echo "[$(timestamp)] val shard=$shard_index gpu=$gpu_id pid=$pid"
done

echo "[$(timestamp)] val export launched"
echo "[$(timestamp)] output: $VAL_STREAMS"
echo "[$(timestamp)] logs: $VAL_RUNS"
