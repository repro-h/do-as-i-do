#!/usr/bin/env bash
set -euo pipefail

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
CONTROL_GPU=${CONTROL_GPU:-4}
FIXED_GPU=${FIXED_GPU:-5}

HYBRID_ROOT=$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1
GLOBAL_V2_ROOT=$HYBRID_ROOT/stage1_global_hand_v2
APPLY_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/apply_pi3x_relative_depth_refiner.py
AUDIT_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/audit_pi3x_ray_depth_by_side.py
VAL_WINDOWS=$GLOBAL_V2_ROOT/windows/val_inference.jsonl
PI3X_VAL_ROOT=$GLOBAL_V2_ROOT/pi3x_geometry_features_compact_v1/val
HANDFLOW_ROOT=$HYBRID_ROOT/handflow_cache/val_v1/streams
SUPERVISION_ROOT=$GLOBAL_V2_ROOT/supervision/val

CONTROL_CKPT=$GLOBAL_V2_ROOT/checkpoints/pi3x_ray_depth_only_left_index_ab_control_v1/best.pt
FIXED_CKPT=$GLOBAL_V2_ROOT/checkpoints/pi3x_ray_depth_only_left_index_ab_fixed_v1/best.pt
CONTROL_PRED=$GLOBAL_V2_ROOT/predictions/pi3x_ray_depth_only_left_index_ab_control_v1
FIXED_PRED=$GLOBAL_V2_ROOT/predictions/pi3x_ray_depth_only_left_index_ab_fixed_v1
ANALYSIS_ROOT=$GLOBAL_V2_ROOT/analysis/pi3x_ray_depth_only_left_index_ab_v1
CONTROL_AUDIT=$ANALYSIS_ROOT/control_by_side.json
FIXED_AUDIT=$ANALYSIS_ROOT/fixed_by_side.json

for path in \
  "$PI3_PYTHON" \
  "$APPLY_SCRIPT" \
  "$AUDIT_SCRIPT" \
  "$VAL_WINDOWS" \
  "$CONTROL_CKPT" \
  "$FIXED_CKPT"
do
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
done

mkdir -p "$CONTROL_PRED" "$FIXED_PRED" "$ANALYSIS_ROOT"
cd "$DO_AS_I_DO"

CUDA_VISIBLE_DEVICES="$CONTROL_GPU" "$PI3_PYTHON" -u "$APPLY_SCRIPT" \
  --windows "$VAL_WINDOWS" \
  --pi3x-root "$PI3X_VAL_ROOT" \
  --checkpoint "$CONTROL_CKPT" \
  --handflow-root "$HANDFLOW_ROOT" \
  --out-root "$CONTROL_PRED" \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda \
  --overwrite \
  >"$ANALYSIS_ROOT/control_apply.log" 2>&1 &
control_pid=$!

CUDA_VISIBLE_DEVICES="$FIXED_GPU" "$PI3_PYTHON" -u "$APPLY_SCRIPT" \
  --windows "$VAL_WINDOWS" \
  --pi3x-root "$PI3X_VAL_ROOT" \
  --checkpoint "$FIXED_CKPT" \
  --handflow-root "$HANDFLOW_ROOT" \
  --out-root "$FIXED_PRED" \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda \
  --overwrite \
  >"$ANALYSIS_ROOT/fixed_apply.log" 2>&1 &
fixed_pid=$!

echo "control apply pid=$control_pid"
echo "fixed apply pid=$fixed_pid"
wait "$control_pid"
wait "$fixed_pid"

"$PI3_PYTHON" -u "$AUDIT_SCRIPT" \
  --prediction-root "$CONTROL_PRED" \
  --supervision-root "$SUPERVISION_ROOT" \
  --out-json "$CONTROL_AUDIT" \
  --worst-k 20 \
  >"$ANALYSIS_ROOT/control_audit.log"

"$PI3_PYTHON" -u "$AUDIT_SCRIPT" \
  --prediction-root "$FIXED_PRED" \
  --supervision-root "$SUPERVISION_ROOT" \
  --out-json "$FIXED_AUDIT" \
  --worst-k 20 \
  >"$ANALYSIS_ROOT/fixed_audit.log"

"$PI3_PYTHON" -c '
import json
import sys

control = json.load(open(sys.argv[1]))["sides"]
fixed = json.load(open(sys.argv[2]))["sides"]
for side in ("left", "right"):
    print(f"\n===== {side} =====")
    for key in ("degraded_fraction", "wrong_sign_fraction"):
        print(key, round(control[side][key], 4), "->", round(fixed[side][key], 4))
    for name in ("5_15", "15_30", "30_inf"):
        before = control[side]["by_abs_target_mm"][name]
        after = fixed[side]["by_abs_target_mm"][name]
        print(
            name,
            "remaining",
            round(before["remaining_median_mm"], 3),
            "->",
            round(after["remaining_median_mm"], 3),
            "wrong_sign",
            round(before["wrong_sign_fraction"], 4),
            "->",
            round(after["wrong_sign_fraction"], 4),
        )
' "$CONTROL_AUDIT" "$FIXED_AUDIT"

echo "control audit=$CONTROL_AUDIT"
echo "fixed audit=$FIXED_AUDIT"

