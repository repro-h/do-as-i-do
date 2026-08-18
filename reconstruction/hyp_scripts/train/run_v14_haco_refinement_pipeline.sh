#!/usr/bin/env bash
set -euo pipefail

# Run one DexYCB stream through WiLoR query export, V14 placement, HACO,
# phase estimation, rigid Stage1, and object-normal local Stage2.

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
HACO_PYTHON=${HACO_PYTHON:-/home/mengxiangting/nas/mengxt/miniconda3/envs/haco/bin/python}
GPU=${GPU:-7}
FORCE=${FORCE:-0}
INPUT_SEQUENCE=${1:-}
SPLIT=${2:-}

if [[ -z "$INPUT_SEQUENCE" ]]; then
  cat >&2 <<'EOF'
Usage:
  run_v14_haco_refinement_pipeline.sh STREAM_ID_OR_SEQUENCE_DIR [train|val]

The train/val split is detected from the V13 manifests when omitted.

Optional environment variables:
  GPU=7 FORCE=1 V14_CKPT=/path/to/best.pt
EOF
  exit 2
fi

HYBRID_ROOT=${HYBRID_ROOT:-$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1}
GLOBAL_V2_ROOT=${GLOBAL_V2_ROOT:-$HYBRID_ROOT/stage1_global_hand_v2}
FEATURE_ROOT=${FEATURE_ROOT:-/data2/hyp/unihand-pi3x-feature/v13_pi3x_full_ws16_s8_fp16}

if [[ -d "$INPUT_SEQUENCE" || "$INPUT_SEQUENCE" == */* ]]; then
  SEQUENCE_DIR=${INPUT_SEQUENCE%/}
  CAMERA_ID=$(basename "$SEQUENCE_DIR")
  SEQUENCE_ID=$(basename "$(dirname "$SEQUENCE_DIR")")
  SUBJECT_ID=$(basename "$(dirname "$(dirname "$SEQUENCE_DIR")")")
  if [[ -z "$SUBJECT_ID" || -z "$SEQUENCE_ID" || -z "$CAMERA_ID" ]]; then
    echo "Cannot derive subject/sequence/camera from '$INPUT_SEQUENCE'" >&2
    exit 2
  fi
  STREAM_ID=${SUBJECT_ID}__${SEQUENCE_ID}__${CAMERA_ID}
else
  SEQUENCE_DIR=
  STREAM_ID=$INPUT_SEQUENCE
fi

if [[ -z "$SPLIT" ]]; then
  SPLIT=$(
    "$PI3_PYTHON" -c '
import json
import sys
from pathlib import Path

stream_id = sys.argv[1]
matches = []
for split, raw_path in zip(("train", "val"), sys.argv[2:]):
    path = Path(raw_path)
    if not path.is_file():
        continue
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            if str(json.loads(line).get("stream_id")) == stream_id:
                matches.append(split)
                break

if len(matches) == 1:
    print(matches[0])
elif not matches:
    raise SystemExit(
        f"Stream {stream_id} is absent from both V13 manifests. "
        "Export its Pi3X cache first or set FEATURE_ROOT correctly."
    )
else:
    raise SystemExit(
        f"Stream {stream_id} occurs in both train and val manifests: {matches}"
    )
' "$STREAM_ID" \
      "$FEATURE_ROOT/manifests/train.jsonl" \
      "$FEATURE_ROOT/manifests/val.jsonl"
  )
fi

if [[ "$SPLIT" != train && "$SPLIT" != val ]]; then
  echo "Invalid split '$SPLIT'; expected train or val" >&2
  exit 2
fi

TEST_ROOT=$DO_AS_I_DO/reconstruction/test_contact
OUT_DIR=$TEST_ROOT/$STREAM_ID
QUERY_ROOT=$OUT_DIR/wilor_query_v2

# Sequence-dependent paths are always rebuilt here. This prevents exported
# variables from an earlier run from silently redirecting a new sequence.
MANIFEST=$FEATURE_ROOT/manifests/$SPLIT.jsonl
WINDOWS=$FEATURE_ROOT/manifests/v13_${SPLIT}_windows.jsonl
DENSE_ROOT=$FEATURE_ROOT/$SPLIT
GLOBAL_ROOT=$GLOBAL_V2_ROOT/supervision/$SPLIT
SUPERVISION_NPZ=$GLOBAL_V2_ROOT/object_frame_hand_se3_supervision_v2/$SPLIT/$STREAM_ID.npz

V14_CKPT=${V14_CKPT:-$FEATURE_ROOT/checkpoints/v14_wilor_pi3x_absolute_full_v1/best.pt}
V14_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/apply_v14_wilor_pi3x_absolute_hand_trajectory.py
PHASE_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/audit_v14_haco_contact_phase.py
STAGE1_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/refine_v14_haco_sequence_contact_containment.py
STAGE2_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/refine_v14_haco_containment_pushout.py
HACO_VIS_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/visualize_haco_contact_cache.py
GT_EXPORT_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/prepare_dexycb_gt_visualization.py

WILOR_ROOT=${WILOR_ROOT:-$DO_AS_I_DO/reconstruction/hyp_modules/WiLoR}
WILOR_EXPORT=$WILOR_ROOT/scripts/export_dexycb_wilor_queries.py
WILOR_CKPT=${WILOR_CKPT:-$WILOR_ROOT/pretrained_models/wilor_final.ckpt}
WILOR_CONFIG=${WILOR_CONFIG:-$WILOR_ROOT/pretrained_models/model_config.yaml}
WILOR_DETECTOR=${WILOR_DETECTOR:-$WILOR_ROOT/pretrained_models/detector.pt}
MANO_DATA_DIR=${MANO_DATA_DIR:-/home/mengxiangting/nas/mengxt/Projects/Pi3_WiLoR_Hand/mano_data/mano}

HACO_ROOT=${HACO_ROOT:-$DO_AS_I_DO/reconstruction/hyp_modules/HACO_RELEASE}
HACO_EXPORT=$HACO_ROOT/export_dexycb_contact_sequence.py
HACO_CKPT=${HACO_CKPT:-$HACO_ROOT/release_checkpoint/haco_neurips_hamer_checkpoint.ckpt}

DEXYCB_MODELS=${DEXYCB_MODELS:-/mnt/nas/wuke/HumanData/DexYCB/models}
FRAME_MAP_JSON=$DENSE_ROOT/$STREAM_ID/dexycb_frame_map.json
GT_DIR=$OUT_DIR/gt
GT_HAND_NPZ=$GT_DIR/dexycb_gt_hand_meshes.npz

QUERY_NPZ=$QUERY_ROOT/$STREAM_ID/wilor_query_cache.npz
TRAJECTORY_NPZ=$OUT_DIR/v14_trajectory.npz
CONTACT_NPZ=$OUT_DIR/haco_contact_sequence.npz
HACO_VIS_DIR=$OUT_DIR/haco_contact_visualization
HACO_VIS_SUMMARY=$HACO_VIS_DIR/summary.json
PHASE_NPZ=$OUT_DIR/haco_contact_phase.npz
PHASE_JSON=$OUT_DIR/haco_contact_phase.json
STAGE1_NPZ=$OUT_DIR/haco_contact_containment_stage1_v3_t30.npz
STAGE1_JSON=$OUT_DIR/haco_contact_containment_stage1_v3_t30.json
STAGE2_NPZ=$OUT_DIR/haco_stage2_object_normal_pushout_joint16_v1.npz
STAGE2_JSON=$OUT_DIR/haco_stage2_object_normal_pushout_joint16_v1.json

timestamp() {
  /bin/date '+%Y-%m-%d %H:%M:%S'
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[$(timestamp)] missing file: $1" >&2
    exit 2
  fi
}

run_stage() {
  local name=$1
  local output=$2
  shift 2
  if [[ "$FORCE" != 1 && -s "$output" ]]; then
    echo "[$(timestamp)] $name: cached $output"
    return
  fi
  echo "[$(timestamp)] $name: start"
  "$@"
  require_file "$output"
  echo "[$(timestamp)] $name: done $output"
}

stage2_is_finite() {
  [[ -s "$STAGE2_NPZ" ]] || return 1
  "$PI3_PYTHON" -c '
import numpy as np
import sys

def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)

required = (
    "refined_hand_vertices_camera",
    "refined_hand_pose_canonical_right",
    "joint_rotation_delta_rotvec",
)
with np.load(sys.argv[1], allow_pickle=False) as data, \
     np.load(sys.argv[2], allow_pickle=False) as query, \
     np.load(sys.argv[3], allow_pickle=False) as trajectory:
    stage_ids = [text(value) for value in data["frame_ids"]]
    query_valid = {
        text(value): bool(valid)
        for value, valid in zip(query["frame_ids"], query["model_valid"])
    }
    trajectory_valid = {
        text(value): bool(valid)
        for value, valid in zip(
            trajectory["frame_ids"], trajectory["prediction_valid"]
        )
    }
    valid = np.asarray([
        query_valid.get(value, False)
        and trajectory_valid.get(value, False)
        for value in stage_ids
    ])
    if not valid.any():
        print("Stage2 validation found no valid frames", file=sys.stderr)
        raise SystemExit(1)
    for key in required:
        if key not in data.files:
            print(f"Stage2 output lacks {key}", file=sys.stderr)
            raise SystemExit(1)
        values = data[key]
        if len(values) != len(valid):
            print(f"Stage2 {key} frame count mismatch", file=sys.stderr)
            raise SystemExit(1)
        bad = int((~np.isfinite(values[valid])).sum())
        if bad:
            print(
                f"Stage2 {key} has {bad} non-finite values in valid frames",
                file=sys.stderr,
            )
            raise SystemExit(1)
' "$STAGE2_NPZ" "$QUERY_NPZ" "$TRAJECTORY_NPZ"
}

for path in \
  "$PI3_PYTHON" "$HACO_PYTHON" "$MANIFEST" "$WINDOWS" \
  "$SUPERVISION_NPZ" "$V14_CKPT" "$V14_SCRIPT" "$PHASE_SCRIPT" \
  "$STAGE1_SCRIPT" "$STAGE2_SCRIPT" "$HACO_VIS_SCRIPT" \
  "$GT_EXPORT_SCRIPT" "$FRAME_MAP_JSON" \
  "$WILOR_EXPORT" "$WILOR_CKPT" \
  "$WILOR_CONFIG" "$WILOR_DETECTOR" "$HACO_EXPORT" "$HACO_CKPT"
do
  require_file "$path"
done
for path in "$DENSE_ROOT" "$GLOBAL_ROOT" "$MANO_DATA_DIR"; do
  if [[ ! -d "$path" ]]; then
    echo "[$(timestamp)] missing directory: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR" "$QUERY_ROOT"

selection=$(
  "$PI3_PYTHON" -c '
import json, sys
manifest, stream_id = sys.argv[1:]
for line in open(manifest):
    if not line.strip():
        continue
    row = json.loads(line)
    if str(row.get("stream_id")) == stream_id:
        print(row.get("object_name", ""))
        break
else:
    raise SystemExit(f"Stream not found in manifest: {stream_id}")
' "$MANIFEST" "$STREAM_ID"
)
OBJECT_NAME=$selection
if [[ -z "$OBJECT_NAME" ]]; then
  echo "[$(timestamp)] manifest row lacks object_name" >&2
  exit 2
fi
OBJECT_MESH=$DEXYCB_MODELS/$OBJECT_NAME/textured_simple.obj
require_file "$OBJECT_MESH"

run_stage "DexYCB GT hand/object export" "$GT_HAND_NPZ" \
  "$PI3_PYTHON" -u "$GT_EXPORT_SCRIPT" \
    --frame-map-json "$FRAME_MAP_JSON" \
    --mano-data-dir "$MANO_DATA_DIR" \
    --object-model-root "$DEXYCB_MODELS" \
    --out-dir "$GT_DIR"

query_valid=0
if [[ -s "$QUERY_NPZ" ]]; then
  if "$PI3_PYTHON" -c '
import numpy as np, sys
required = {
    "mano_global_orient_canonical_right",
    "mano_hand_pose_canonical_right",
    "mano_betas",
    "vertices_3d_root_relative_original",
    "image_paths",
    "model_valid",
}
with np.load(sys.argv[1], allow_pickle=False) as data:
    missing = required - set(data.files)
    if missing:
        raise SystemExit(1)
' "$QUERY_NPZ"
  then
    query_valid=1
  fi
fi

if [[ "$FORCE" == 1 || "$query_valid" != 1 ]]; then
  echo "[$(timestamp)] WiLoR query v2: start"
  (
    cd "$WILOR_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PI3_PYTHON" -u "$WILOR_EXPORT" \
      --manifest "$MANIFEST" \
      --out-root "$QUERY_ROOT" \
      --checkpoint "$WILOR_CKPT" \
      --config "$WILOR_CONFIG" \
      --detector "$WILOR_DETECTOR" \
      --mano-data-dir "$MANO_DATA_DIR" \
      --stream-id "$STREAM_ID" \
      --status-json "$OUT_DIR/wilor_status.json" \
      --detector-batch-size 16 \
      --model-batch-size 16 \
      --num-workers 4 \
      --device cuda \
      --overwrite
  )
else
  echo "[$(timestamp)] WiLoR query v2: cached $QUERY_NPZ"
fi
require_file "$QUERY_NPZ"

run_stage "V14 trajectory" "$TRAJECTORY_NPZ" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PI3_PYTHON" -u "$V14_SCRIPT" \
    --windows "$WINDOWS" \
    --global-root "$GLOBAL_ROOT" \
    --dense-root "$DENSE_ROOT" \
    --query-root "$QUERY_ROOT" \
    --checkpoint "$V14_CKPT" \
    --stream-id "$STREAM_ID" \
    --out-npz "$TRAJECTORY_NPZ" \
    --device cuda

if [[ "$FORCE" != 1 && -s "$CONTACT_NPZ" ]]; then
  echo "[$(timestamp)] HACO contact: cached $CONTACT_NPZ"
else
  echo "[$(timestamp)] HACO contact: start"
  (
    cd "$HACO_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_OFFLINE=1 \
    HACO_TIMM_PRETRAINED=0 \
    "$HACO_PYTHON" -u "$HACO_EXPORT" \
      --query-npz "$QUERY_NPZ" \
      --checkpoint "$HACO_CKPT" \
      --backbone hamer \
      --batch-size 8 \
      --out-npz "$CONTACT_NPZ" \
      --device cuda \
      --overwrite
  )
  require_file "$CONTACT_NPZ"
  echo "[$(timestamp)] HACO contact: done $CONTACT_NPZ"
fi

run_stage "HACO contact visualization" "$HACO_VIS_SUMMARY" \
  env CUDA_VISIBLE_DEVICES="$GPU" PYOPENGL_PLATFORM=egl \
    "$HACO_PYTHON" -u "$HACO_VIS_SCRIPT" \
      --haco-root "$HACO_ROOT" \
      --query-npz "$QUERY_NPZ" \
      --contact-npz "$CONTACT_NPZ" \
      --out-dir "$HACO_VIS_DIR" \
      --backbone hamer \
      --stride 1 \
      --overwrite

gt_phase_args=(--gt-hand-npz "$GT_HAND_NPZ")
gt_stage_args=(--gt-hand-npz "$GT_HAND_NPZ")

run_stage "contact phase" "$PHASE_NPZ" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PI3_PYTHON" -u "$PHASE_SCRIPT" \
    --trajectory-npz "$TRAJECTORY_NPZ" \
    --query-npz "$QUERY_NPZ" \
    --contact-sequence-npz "$CONTACT_NPZ" \
    --supervision-npz "$SUPERVISION_NPZ" \
    --object-mesh "$OBJECT_MESH" \
    "${gt_phase_args[@]}" \
    --out-npz "$PHASE_NPZ" \
    --out-json "$PHASE_JSON" \
    --object-samples 8192

run_stage "Stage1 rigid contact/containment" "$STAGE1_NPZ" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PI3_PYTHON" -u "$STAGE1_SCRIPT" \
    --trajectory-npz "$TRAJECTORY_NPZ" \
    --query-npz "$QUERY_NPZ" \
    --contact-sequence-npz "$CONTACT_NPZ" \
    --phase-npz "$PHASE_NPZ" \
    --supervision-npz "$SUPERVISION_NPZ" \
    --object-mesh "$OBJECT_MESH" \
    "${gt_stage_args[@]}" \
    --out-npz "$STAGE1_NPZ" \
    --out-json "$STAGE1_JSON" \
    --contact-target-mm 6 \
    --correction-stop-mm 10 \
    --correction-full-mm 18 \
    --collision-margin-mm 0.5 \
    --containment-refresh 25 \
    --max-translation-mm 30 \
    --max-rotation-deg 5 \
    --steps 300 \
    --device cuda

stage2_force=$FORCE
if [[ -s "$STAGE2_NPZ" ]] && ! stage2_is_finite; then
  echo "[$(timestamp)] Stage2 cache is non-finite; recomputing $STAGE2_NPZ"
  FORCE=1
fi

run_stage "Stage2 object-normal local pose" "$STAGE2_NPZ" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PI3_PYTHON" -u "$STAGE2_SCRIPT" \
    --trajectory-npz "$TRAJECTORY_NPZ" \
    --query-npz "$QUERY_NPZ" \
    --stage1-npz "$STAGE1_NPZ" \
    --base-mode stage1 \
    --containment-npz "$STAGE1_NPZ" \
    --contact-sequence-npz "$CONTACT_NPZ" \
    --phase-npz "$PHASE_NPZ" \
    --supervision-npz "$SUPERVISION_NPZ" \
    --object-mesh "$OBJECT_MESH" \
    --wilor-root "$WILOR_ROOT" \
    --wilor-checkpoint "$WILOR_CKPT" \
    --wilor-config "$WILOR_CONFIG" \
    --mano-data-dir "$MANO_DATA_DIR" \
    "${gt_stage_args[@]}" \
    --out-npz "$STAGE2_NPZ" \
    --out-json "$STAGE2_JSON" \
    --adaptive-balance \
    --adaptive-refresh-steps 10 \
    --adaptive-reset-optimizer-on-refresh \
    --adaptive-inside-low-fraction 0.005 \
    --adaptive-inside-high-fraction 0.015 \
    --adaptive-contact-floor 0.2 \
    --adaptive-collision-min-scale 2 \
    --adaptive-collision-max-scale 4 \
    --adaptive-gate-ema 0.5 \
    --contact-target-mm 0 \
    --contact-activation-mm 12 \
    --collision-margin-mm 0.5 \
    --w-contact 1 \
    --w-collision 1 \
    --w-object-normal-pushout 5 \
    --w-tangential 2 \
    --w-vertex-anchor 2 \
    --w-pose-anchor 0.00075 \
    --w-pose-velocity 0.0015 \
    --w-pose-acceleration 0.003 \
    --max-joint-delta-deg 16 \
    --correspondence-topk 8 \
    --object-samples 2048 \
    --point-chunk 1024 \
    --frame-chunk 4 \
    --steps 500 \
    --lr 0.001 \
    --device cuda

FORCE=$stage2_force
if ! stage2_is_finite; then
  echo "[$(timestamp)] Stage2 output failed finite-value validation" >&2
  exit 1
fi

cat <<EOF

[$(timestamp)] pipeline complete
stream:      $STREAM_ID
split:       $SPLIT
object:      $OBJECT_NAME
query:       $QUERY_NPZ
trajectory:  $TRAJECTORY_NPZ
contact:     $CONTACT_NPZ
contact vis: $HACO_VIS_DIR
GT hand:     $GT_HAND_NPZ
phase:       $PHASE_NPZ
stage1:      $STAGE1_NPZ
stage2:      $STAGE2_NPZ
summary:     $STAGE2_JSON
EOF
