#!/usr/bin/env bash
set -euo pipefail

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
V15_DIR=${V15_DIR:-$DO_AS_I_DO/reconstruction/hyp_scripts/train_v15_multihand}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$V15_DIR/train_v16_1_compact_pi3x.py}

T64_ROOT=${T64_ROOT:-/data2/hyp/geometry_v2_t64}
MIXED_ROOT=${MIXED_ROOT:-$T64_ROOT/manifests/mixed}
VAL_CACHE=${VAL_CACHE:-$T64_ROOT/compact_cache/val_gt_t64_v1}
VAL_LOG_ROOT=${VAL_LOG_ROOT:-$T64_ROOT/logs/compact_val_gt_t64_v1}
FULL_OUT=${FULL_OUT:-$T64_ROOT/checkpoints/v2_geometry_dynamic_t64_ddp_full_v1}

HAND_UNI_ROOT=${HAND_UNI_ROOT:-/home/mengxiangting/nas/mengxt/Projects/Pi3_WiLoR_Hand}
PI3_ROOT=${PI3_ROOT:-/home/mengxiangting/nas/mengxt/Projects/Pi3}
PI3X_CHECKPOINT=${PI3X_CHECKPOINT:-/mnt/nas/mengxt/Projects/Pi3/ckpts/model.safetensors}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
POLL_SECONDS=${POLL_SECONDS:-60}
EPOCHS=${EPOCHS:-30}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-320}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2e-4}
MIN_LR=${MIN_LR:-1e-5}
WANDB_PROJECT=${WANDB_PROJECT:-uni-hand-geometry-v2}
WANDB_NAME=${WANDB_NAME:-geometry-v2-t64-ddp-full-v1}

TRAIN_WINDOWS=$MIXED_ROOT/train_windows.jsonl
VAL_WINDOWS=$MIXED_ROOT/val_windows.jsonl
PIPELINE_LOG=$FULL_OUT/pipeline.log
TRAIN_LOG=$FULL_OUT/train.log

for path in \
  "$PI3_PYTHON" "$TRAIN_SCRIPT" "$TRAIN_WINDOWS" "$VAL_WINDOWS" \
  "$HAND_UNI_ROOT" "$PI3_ROOT" "$PI3X_CHECKPOINT" "$VAL_CACHE" "$VAL_LOG_ROOT"
do
  test -e "$path" || { echo "Missing required path: $path" >&2; exit 1; }
done

mkdir -p "$FULL_OUT"
exec >> "$PIPELINE_LOG" 2>&1

echo "[$(date '+%F %T')] waiting for $NUM_GPUS validation cache shards"
while true; do
  RESULT=$(
    "$PI3_PYTHON" - "$VAL_LOG_ROOT" "$VAL_WINDOWS" "$VAL_CACHE" "$NUM_GPUS" <<'PY'
import json
import sys
from pathlib import Path

log_root, manifest, cache_root, num_shards = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
expected = sum(1 for line in manifest.open() if line.strip())
statuses = sorted(log_root.glob("status_*.json"))
cached_files = sum(1 for _ in cache_root.rglob("window_*.npz"))
if len(statuses) < num_shards:
    print(f"WAIT statuses={len(statuses)}/{num_shards} cache={cached_files}/{expected}")
    raise SystemExit
requested = completed = cached = failed = 0
for path in statuses:
    data = json.loads(path.read_text())
    requested += int(data.get("requested", 0))
    completed += int(data.get("completed", 0))
    cached += int(data.get("cached", 0))
    failed += int(data.get("failed", 0))
if failed:
    print(f"FAILED requested={requested} completed={completed} cached={cached} failed={failed}")
elif requested == expected and completed + cached == expected and cached_files >= expected:
    print(f"COMPLETE requested={requested} completed={completed} cached={cached} files={cached_files}")
else:
    print(f"WAIT requested={requested}/{expected} completed={completed} cached={cached} files={cached_files}")
PY
  )
  echo "[$(date '+%F %T')] $RESULT"
  case "$RESULT" in
    COMPLETE*) break ;;
    FAILED*) exit 1 ;;
  esac
  if ! pgrep -f '[e]xport_compact_pi3x_cache.py' >/dev/null; then
    echo "Validation exporters exited before cache completion" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

if pgrep -f "[t]rain_v16_1_compact_pi3x.py.*$FULL_OUT" >/dev/null; then
  echo "Training already appears to be running for $FULL_OUT" >&2
  exit 1
fi
if [[ -e "$FULL_OUT/last.pt" ]]; then
  echo "Refusing to overwrite existing training output: $FULL_OUT/last.pt" >&2
  exit 1
fi

echo "[$(date '+%F %T')] validation cache complete; launching $NUM_GPUS-rank DDP training"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  PYTHONUNBUFFERED=1 \
  "$PI3_PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    "$TRAIN_SCRIPT" \
    --train-windows "$TRAIN_WINDOWS" \
    --val-windows "$VAL_WINDOWS" \
    --val-compact-cache-root "$VAL_CACHE" \
    --hand-uni-root "$HAND_UNI_ROOT" \
    --pi3-root "$PI3_ROOT" \
    --pi3x-checkpoint "$PI3X_CHECKPOINT" \
    --out-dir "$FULL_OUT" \
    --architecture geometry-temporal-v2 \
    --query-source gt \
    --supervision-source target \
    --dynamic-train-sampling \
    --dynamic-window-size 64 \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --dataset-weights dexycb=1,h2o=1,hot3d=1,oakink2=1,taco=1 \
    --max-window-size 64 \
    --max-hands 2 \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lr-scheduler cosine \
    --min-lr "$MIN_LR" \
    --geometry-warmup-epochs "$EPOCHS" \
    --w-geometry-depth 1 \
    --w-depth 0.5 \
    --w-relative 0.5 \
    --w-velocity 0 \
    --w-acceleration 0 \
    --w-reprojection 0.1 \
    --w-temporal-correction 0 \
    --global-noise-px 4 \
    --temporal-noise-px 0.5 \
    --joint-noise-px 2 \
    --outlier-probability 0.03 \
    --query-dropout 0.1 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$WANDB_NAME" \
    --device cuda \
    > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$FULL_OUT/train.pid"
echo "[$(date '+%F %T')] training pid=$TRAIN_PID log=$TRAIN_LOG"
