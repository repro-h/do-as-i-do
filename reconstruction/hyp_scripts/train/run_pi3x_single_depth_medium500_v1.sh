#!/usr/bin/env bash
set -euo pipefail

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
GPU=${GPU:-4}

HYBRID_ROOT=$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1
GLOBAL_V2_ROOT=$HYBRID_ROOT/stage1_global_hand_v2
TRAIN_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/train_pi3x_relative_depth_refiner.py
PI3X_ROOT=$GLOBAL_V2_ROOT/pi3x_geometry_features_compact_v1
TRAIN_WINDOWS=$GLOBAL_V2_ROOT/windows/pi3x_v3_medium_train_500.jsonl
VAL_WINDOWS=$GLOBAL_V2_ROOT/windows/val_inference.jsonl
TRAIN_OUT=$GLOBAL_V2_ROOT/checkpoints/pi3x_single_depth_medium500_v1
TRAIN_LOG=$TRAIN_OUT/train.log
TRAIN_PID=$TRAIN_OUT/train.pid

for path in "$PI3_PYTHON" "$TRAIN_SCRIPT" "$TRAIN_WINDOWS" "$VAL_WINDOWS"
do
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
done

mkdir -p "$TRAIN_OUT"
cd "$DO_AS_I_DO"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PI3_PYTHON" -u "$TRAIN_SCRIPT" \
  --train-windows "$TRAIN_WINDOWS" \
  --val-windows "$VAL_WINDOWS" \
  --pi3x-train-root "$PI3X_ROOT/train" \
  --pi3x-val-root "$PI3X_ROOT/val" \
  --out-dir "$TRAIN_OUT" \
  --model-variant single_depth \
  --objective ray_depth_only \
  --left-coordinate-mode normalized \
  --depth-mse-scale-mm 20 \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 4 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --hidden-dim 128 \
  --spatial-layers 2 \
  --temporal-layers 3 \
  --heads 8 \
  --dropout 0.1 \
  --max-correction-mm 120 \
  --max-motion-residual-mm 60 \
  --max-target-mm 120 \
  --w-depth 1 \
  --w-velocity 0.05 \
  --w-acceleration 0.02 \
  --seed 42 \
  --device cuda \
  --wandb \
  --wandb-project do-as-i-do-pi3x-depth \
  --wandb-name pi3x-single-depth-medium500-v1 \
  >"$TRAIN_LOG" 2>&1 </dev/null &

pid=$!
printf '%s\n' "$pid" >"$TRAIN_PID"
echo "single-depth gpu=$GPU pid=$pid"
echo "log=$TRAIN_LOG"
echo "checkpoint=$TRAIN_OUT/best.pt"
