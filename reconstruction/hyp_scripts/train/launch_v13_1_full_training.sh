#!/usr/bin/env bash
set -euo pipefail

DO_AS_I_DO="${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}"
PI3_PYTHON="${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}"
HYBRID_ROOT="$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1"
GLOBAL_V2_ROOT="$HYBRID_ROOT/stage1_global_hand_v2"
FEATURE_ROOT="${FEATURE_ROOT:-/data2/hyp/unihand-pi3x-feature/v13_pi3x_full_ws16_s8_fp16}"

TRAIN_FEATURE_ROOT="$FEATURE_ROOT/train"
VAL_FEATURE_ROOT="$FEATURE_ROOT/val"
TRAIN_SOURCE_WINDOWS="$GLOBAL_V2_ROOT/object_frame_hand_se3_supervision_v2/train_windows_full.jsonl"
VAL_SOURCE_WINDOWS="$GLOBAL_V2_ROOT/object_frame_hand_se3_supervision_v2/_runs/val/shard_0.jsonl"
TRAIN_WINDOWS="$FEATURE_ROOT/manifests/v13_train_windows.jsonl"
VAL_WINDOWS="$FEATURE_ROOT/manifests/v13_val_windows.jsonl"
GLOBAL_TRAIN_ROOT="$GLOBAL_V2_ROOT/supervision/train"
GLOBAL_VAL_ROOT="$GLOBAL_V2_ROOT/supervision/val"

MAKE_WINDOWS="$DO_AS_I_DO/reconstruction/hyp_scripts/train/make_v9_4_dense_windows.py"
TRAIN_SCRIPT="$DO_AS_I_DO/reconstruction/hyp_scripts/train/train_v13_pi3x_absolute_hand_trajectory.py"
OUT_DIR="${V13_OUT:-$FEATURE_ROOT/checkpoints/v13_1_absolute_hand_translation_full_v1}"
LOG="$OUT_DIR/train.log"
PID_FILE="$OUT_DIR/train.pid"

mkdir -p "$OUT_DIR" "$(dirname "$TRAIN_WINDOWS")"

"$PI3_PYTHON" -u "$MAKE_WINDOWS" \
  --source-windows "$TRAIN_SOURCE_WINDOWS" \
  --dense-root "$TRAIN_FEATURE_ROOT" \
  --out "$TRAIN_WINDOWS"

"$PI3_PYTHON" -u "$MAKE_WINDOWS" \
  --source-windows "$VAL_SOURCE_WINDOWS" \
  --dense-root "$VAL_FEATURE_ROOT" \
  --out "$VAL_WINDOWS"

wc -l "$TRAIN_WINDOWS" "$VAL_WINDOWS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
nohup "$PI3_PYTHON" -u "$TRAIN_SCRIPT" \
  --train-windows "$TRAIN_WINDOWS" \
  --val-windows "$VAL_WINDOWS" \
  --global-train-root "$GLOBAL_TRAIN_ROOT" \
  --global-val-root "$GLOBAL_VAL_ROOT" \
  --dense-train-root "$TRAIN_FEATURE_ROOT" \
  --dense-val-root "$VAL_FEATURE_ROOT" \
  --out-dir "$OUT_DIR" \
  --epochs 20 \
  --batch-size 8 \
  --num-workers 8 \
  --data-parallel \
  --device cuda \
  > "$LOG" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf 'PID=%s\nlog=%s\noutput=%s\n' "$PID" "$LOG" "$OUT_DIR"
