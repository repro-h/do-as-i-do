#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH PYTHONNOUSERSITE || true

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
HYBRID_ROOT=${HYBRID_ROOT:-$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1}
GLOBAL_V2_ROOT=${GLOBAL_V2_ROOT:-$HYBRID_ROOT/stage1_global_hand_v2}
MANIFEST=${MANIFEST:-$HYBRID_ROOT/manifests/val.jsonl}
DEXYCB_MODELS=${DEXYCB_MODELS:-/mnt/nas/wuke/HumanData/DexYCB/models}
OBJECT_NAME=${OBJECT_NAME:?Set OBJECT_NAME, for example 035_power_drill}
OBJECT_INDEX=${OBJECT_INDEX:-0}
PORT=${PORT:-8098}
FORCE_ALIGN=${FORCE_ALIGN:-0}

ALIGN_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/optim_seq/align_sam_mesh_to_dexycb_cad.py
VIEW_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/run_canonical_oracle_hand_target.sh

selection=$(
  "$PI3_PYTHON" -c '
import json, sys

manifest, object_name, object_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = sorted(
    [
        json.loads(line)
        for line in open(manifest)
        if line.strip() and json.loads(line)["object_name"] == object_name
    ],
    key=lambda row: (str(row.get("hand_side", "")), str(row["stream_id"])),
)
if not rows:
    raise SystemExit(f"Object not found: {object_name}")
if not 0 <= object_index < len(rows):
    raise SystemExit(f"OBJECT_INDEX must be in [0, {len(rows) - 1}]")
row = rows[object_index]
print(row["stream_id"], row["stream_dir"], sep="\t")
' "$MANIFEST" "$OBJECT_NAME" "$OBJECT_INDEX"
)

IFS=$'\t' read -r STREAM_ID SEQ_DIR <<< "$selection"
FILTERED_OBJECT_JSON=$HYBRID_ROOT/object_motion_filter_v2_compact/val/$STREAM_ID/segmented_ekf_rts/foundationpose_segmented_ekf_rts.json
ALIGN_OUT=$GLOBAL_V2_ROOT/optim_seq/pose_consensus_canonical_tests_v1/$OBJECT_NAME/$STREAM_ID
ALIGN_SUMMARY=$ALIGN_OUT/alignment_summary.json

for path in "$PI3_PYTHON" "$MANIFEST" "$ALIGN_SCRIPT" "$VIEW_SCRIPT" "$FILTERED_OBJECT_JSON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
done

mkdir -p "$ALIGN_OUT"
cd "$DO_AS_I_DO"

if [[ "$FORCE_ALIGN" == "1" || ! -f "$ALIGN_SUMMARY" ]]; then
  "$PI3_PYTHON" -u "$ALIGN_SCRIPT" \
    --sequence-dir "$SEQ_DIR" \
    --manifest "$MANIFEST" \
    --ycb-model-root "$DEXYCB_MODELS" \
    --filtered-object-json "$FILTERED_OBJECT_JSON" \
    --out-dir "$ALIGN_OUT" \
    --alignment-mode pose_consensus \
    --samples 12000 \
    --trim-fraction 0.7 \
    --seed 42
fi

export STREAM_ID SEQ_DIR PORT
export CANONICAL_ALIGNMENT=$ALIGN_SUMMARY
export OUT_DIR=$GLOBAL_V2_ROOT/optim_seq/pose_consensus_canonical_targets_v1/$OBJECT_NAME/$STREAM_ID
export SYMMETRY_AXIS=none
export SYMMETRY_STEP_DEG=15
export SYMMETRY_AXIS_FLIP=0
export SYMMETRY_SELECTION_MODE=sequence

echo "object=$OBJECT_NAME"
echo "stream=$STREAM_ID"
echo "sequence=$SEQ_DIR"
echo "canonical_alignment=$CANONICAL_ALIGNMENT"
echo "target_output=$OUT_DIR"

exec /bin/bash "$VIEW_SCRIPT"
