#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH PYTHONNOUSERSITE || true

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
SPLIT=${SPLIT:-val}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
LIMIT=${LIMIT:-0}
STREAM_ID=${STREAM_ID:-}
OVERWRITE=${OVERWRITE:-0}
DISABLE_POSE_GATE=${DISABLE_POSE_GATE:-0}

HYBRID_ROOT=${HYBRID_ROOT:-$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1}
GLOBAL_V2_ROOT=${GLOBAL_V2_ROOT:-$HYBRID_ROOT/stage1_global_hand_v2}
MANIFEST=${MANIFEST:-$HYBRID_ROOT/manifests/$SPLIT.jsonl}
GLOBAL_SUPERVISION_ROOT=${GLOBAL_SUPERVISION_ROOT:-$GLOBAL_V2_ROOT/supervision/$SPLIT}
CANONICAL_ROOT=${CANONICAL_ROOT:-$GLOBAL_V2_ROOT/canonical_alignment/sam_ycb_bank_v2}
OBJECT_PROFILE_JSON=${OBJECT_PROFILE_JSON:-$DO_AS_I_DO/reconstruction/hyp_scripts/train/object_frame_hand_se3_profiles_v2.yaml}
SE3_ROOT=${SE3_ROOT:-$GLOBAL_V2_ROOT/object_frame_hand_se3_supervision_v2}
OUT_ROOT=${OUT_ROOT:-$SE3_ROOT/$SPLIT}
RUN_ROOT=${RUN_ROOT:-$SE3_ROOT/_runs/$SPLIT}
WINDOW_JSONL=${WINDOW_JSONL:-$RUN_ROOT/shard_${SHARD_INDEX}.jsonl}
EXPORT_SCRIPT=$DO_AS_I_DO/reconstruction/hyp_scripts/train/prepare_object_frame_hand_se3_supervision.py

for path in "$PI3_PYTHON" "$MANIFEST" "$EXPORT_SCRIPT" "$OBJECT_PROFILE_JSON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
done
for path in "$GLOBAL_SUPERVISION_ROOT" "$CANONICAL_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT" "$RUN_ROOT"
cd "$DO_AS_I_DO"

command=(
  "$PI3_PYTHON" -u "$EXPORT_SCRIPT"
  --manifest "$MANIFEST"
  --split "$SPLIT"
  --global-supervision-root "$GLOBAL_SUPERVISION_ROOT"
  --canonical-root "$CANONICAL_ROOT"
  --object-profile-json "$OBJECT_PROFILE_JSON"
  --out-root "$OUT_ROOT"
  --window-jsonl "$WINDOW_JSONL"
  --window-size 16
  --window-stride 4
  --min-valid-frames 8
  --min-visible-hand-pixels 64
  --hand-presence-mode span
  --scale-warning-threshold 0.1
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --limit "$LIMIT"
)
if [[ -n "$STREAM_ID" ]]; then
  command+=(--stream-id "$STREAM_ID")
fi
if [[ "$DISABLE_POSE_GATE" == "1" ]]; then
  command+=(--disable-pose-gate)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  command+=(--overwrite)
fi

echo "split=$SPLIT shard=$SHARD_INDEX/$NUM_SHARDS stream=${STREAM_ID:-all}"
echo "output=$OUT_ROOT"
echo "windows=$WINDOW_JSONL"
"${command[@]}"
