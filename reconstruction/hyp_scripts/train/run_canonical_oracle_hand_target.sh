#!/usr/bin/env bash
set -euo pipefail

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
VIS_PYTHON=${VIS_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/sam3d-objects/bin/python}
SEQ_DIR=${SEQ_DIR:-$DO_AS_I_DO/reconstruction/data/dexycb/foundationpose_quality_filter_v2/val/passed/20200709-subject-01/20200709_143531/836212060125}
PORT=${PORT:-8098}
SYMMETRY_AXIS=${SYMMETRY_AXIS:-none}
SYMMETRY_STEP_DEG=${SYMMETRY_STEP_DEG:-15}
SYMMETRY_AXIS_FLIP=${SYMMETRY_AXIS_FLIP:-0}

subject=$(basename "$(dirname "$(dirname "$SEQ_DIR")")")
sequence=$(basename "$(dirname "$SEQ_DIR")")
camera=$(basename "$SEQ_DIR")
STREAM_ID=${STREAM_ID:-${subject}__${sequence}__${camera}}

HYBRID_ROOT=$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1
GLOBAL_V2_ROOT=$HYBRID_ROOT/stage1_global_hand_v2
VAL_MANIFEST=$HYBRID_ROOT/manifests/val.jsonl
HANDFLOW_ROOT=$HYBRID_ROOT/handflow_cache/val_v1/streams
PREDICTION_ROOT=${PREDICTION_ROOT:-$GLOBAL_V2_ROOT/predictions/pi3x_ray_depth_only_full_v1}
SUPERVISION=$GLOBAL_V2_ROOT/supervision/val/$STREAM_ID.npz
FILTERED_OBJECT_JSON=$HYBRID_ROOT/object_motion_filter_v2_compact/val/$STREAM_ID/segmented_ekf_rts/foundationpose_segmented_ekf_rts.json
CANONICAL_ROOT=${CANONICAL_ROOT:-$GLOBAL_V2_ROOT/canonical_alignment/sam_ycb_bank_v1}
VIS_ROOT=$GLOBAL_V2_ROOT/visualization/pi3x_ray_depth_only_full_v1_val
STREAM_VIS=$VIS_ROOT/$STREAM_ID
OUT_DIR=${OUT_DIR:-$GLOBAL_V2_ROOT/optim_seq/canonical_oracle_hand_target_v1/$STREAM_ID}

PREPARE_VIS=$DO_AS_I_DO/reconstruction/hyp_scripts/train/visualize_stage1_dexycb.py
PREPARE_TARGET=$DO_AS_I_DO/reconstruction/hyp_scripts/train/prepare_relative_hand_target_pilot.py
VIEW_TARGET=$DO_AS_I_DO/reconstruction/hyp_scripts/train/visualize_relative_hand_target_pilot.py
MANO_ROOT=${MANO_ROOT:-/home/mengxiangting/nas/mengxt/Projects/Pi3_WiLoR_Hand/mano_data/mano}
DEXYCB_MODELS=${DEXYCB_MODELS:-/mnt/nas/wuke/HumanData/DexYCB/models}

for path in \
  "$PI3_PYTHON" \
  "$VIS_PYTHON" \
  "$VAL_MANIFEST" \
  "$SUPERVISION" \
  "$FILTERED_OBJECT_JSON" \
  "$PREPARE_VIS" \
  "$PREPARE_TARGET" \
  "$VIEW_TARGET"
do
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
done

IFS=$'\t' read -r HAND_SIDE OBJECT_NAME OBJECT_MESH OBJECT_SCALE < <(
  "$PI3_PYTHON" -c '
import json, sys
manifest, stream_id = sys.argv[1:]
rows = [json.loads(line) for line in open(manifest) if line.strip()]
row = next(row for row in rows if row["stream_id"] == stream_id)
print(
    row["hand_side"],
    row["object_name"],
    row["sam3d_glb"],
    row["foundationpose_source_mesh_scale"],
    sep="\t",
)
' "$VAL_MANIFEST" "$STREAM_ID"
)

CANONICAL_ALIGNMENT=$CANONICAL_ROOT/$OBJECT_NAME/canonical_alignment.json
GT_YCB_OBJECT_MESH=$DEXYCB_MODELS/$OBJECT_NAME/textured_simple.obj
if [[ ! -f "$CANONICAL_ALIGNMENT" ]]; then
  echo "Missing canonical alignment: $CANONICAL_ALIGNMENT" >&2
  exit 1
fi
if [[ ! -f "$GT_YCB_OBJECT_MESH" ]]; then
  echo "Missing GT YCB object mesh: $GT_YCB_OBJECT_MESH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cd "$DO_AS_I_DO"

"$PI3_PYTHON" -u "$PREPARE_VIS" \
  --manifest "$VAL_MANIFEST" \
  --prediction-root "$PREDICTION_ROOT" \
  --handflow-root "$HANDFLOW_ROOT" \
  --mano-data-dir "$MANO_ROOT" \
  --object-model-root "$DEXYCB_MODELS" \
  --out-root "$VIS_ROOT" \
  --viewer-python "$VIS_PYTHON" \
  --stream-id "$STREAM_ID" \
  --foundationpose-json "$FILTERED_OBJECT_JSON" \
  --prepare-only

RAW_HAND_MESHES=$STREAM_VIS/original/all_hand_meshes_handflow.npz
V8_HAND_MESHES=$STREAM_VIS/stage1_corrected/all_hand_meshes_handflow.npz
GT_HAND_MESHES=$STREAM_VIS/gt/dexycb_gt_hand_meshes.npz
GT_OBJECT_LAYOUT=$STREAM_VIS/gt/dexycb_gt_object_layout_camera_frame.json
V8_PREDICTION=$PREDICTION_ROOT/$STREAM_ID/handflow_camera_result_pi3x_depth_refined.npz

for path in \
  "$RAW_HAND_MESHES" \
  "$V8_HAND_MESHES" \
  "$GT_HAND_MESHES" \
  "$GT_OBJECT_LAYOUT" \
  "$V8_PREDICTION"
do
  if [[ ! -f "$path" ]]; then
    echo "Missing prepared file: $path" >&2
    exit 1
  fi
done

symmetry_args=(
  --symmetry-axis "$SYMMETRY_AXIS"
  --symmetry-step-deg "$SYMMETRY_STEP_DEG"
)
if [[ "$SYMMETRY_AXIS_FLIP" == "1" ]]; then
  symmetry_args+=(--symmetry-axis-flip)
fi

"$PI3_PYTHON" -u "$PREPARE_TARGET" \
  --supervision-npz "$SUPERVISION" \
  --v8-prediction-npz "$V8_PREDICTION" \
  --raw-hand-meshes "$RAW_HAND_MESHES" \
  --v8-hand-meshes "$V8_HAND_MESHES" \
  --gt-hand-meshes "$GT_HAND_MESHES" \
  --filtered-object-json "$FILTERED_OBJECT_JSON" \
  --gt-object-json "$GT_OBJECT_LAYOUT" \
  --gt-ycb-object-mesh "$GT_YCB_OBJECT_MESH" \
  --canonical-alignment-json "$CANONICAL_ALIGNMENT" \
  --object-mesh "$OBJECT_MESH" \
  --object-mesh-scale "$OBJECT_SCALE" \
  --hand-side "$HAND_SIDE" \
  --transform-mode full_se3 \
  "${symmetry_args[@]}" \
  --out-dir "$OUT_DIR"

echo "Stream: $STREAM_ID"
echo "Object: $OBJECT_NAME"
echo "Canonical alignment: $CANONICAL_ALIGNMENT"
echo "Audit: $OUT_DIR/audit.json"
echo "Viewer: http://localhost:$PORT"
echo "Press Ctrl+C to stop."

"$VIS_PYTHON" -u "$VIEW_TARGET" \
  --audit-json "$OUT_DIR/audit.json" \
  --port "$PORT" \
  --fps 10
