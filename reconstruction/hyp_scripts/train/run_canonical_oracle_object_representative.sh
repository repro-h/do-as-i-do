#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH PYTHONNOUSERSITE || true

DO_AS_I_DO=${DO_AS_I_DO:-/home/mengxiangting/nas/mengxt/Projects/do-as-i-do}
PI3_PYTHON=${PI3_PYTHON:-/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python}
HYBRID_ROOT=${HYBRID_ROOT:-$DO_AS_I_DO/reconstruction/data/dexycb/hybrid_training_v1}
MANIFEST=${MANIFEST:-$HYBRID_ROOT/manifests/val.jsonl}
OBJECT_NAME=${OBJECT_NAME:-}
OBJECT_INDEX=${OBJECT_INDEX:-0}
LIST_ONLY=${LIST_ONLY:-0}
PORT=${PORT:-8098}
RUN_ORACLE=$DO_AS_I_DO/reconstruction/hyp_scripts/train/run_canonical_oracle_hand_target.sh
SYMMETRY_PROFILE=${SYMMETRY_PROFILE:-auto}

# In auto mode, reset every value so settings from the previous object cannot
# leak into the next audit. Set SYMMETRY_PROFILE=manual to override them.
if [[ "$SYMMETRY_PROFILE" == "auto" ]]; then
  case "$OBJECT_NAME" in
    002_master_chef_can|005_tomato_soup_can|040_large_marker)
      SYMMETRY_AXIS=y
      SYMMETRY_STEP_DEG=15
      SYMMETRY_AXIS_FLIP=1
      SYMMETRY_SELECTION_MODE=temporal
      ;;
    003_cracker_box|004_sugar_box|008_pudding_box|009_gelatin_box|010_potted_meat_can|036_wood_block|061_foam_brick)
      SYMMETRY_AXIS=y
      SYMMETRY_STEP_DEG=180
      SYMMETRY_AXIS_FLIP=1
      SYMMETRY_SELECTION_MODE=sequence
      ;;
    024_bowl)
      SYMMETRY_AXIS=y
      SYMMETRY_STEP_DEG=15
      SYMMETRY_AXIS_FLIP=0
      SYMMETRY_SELECTION_MODE=temporal
      ;;
    *)
      SYMMETRY_AXIS=none
      SYMMETRY_STEP_DEG=15
      SYMMETRY_AXIS_FLIP=0
      SYMMETRY_SELECTION_MODE=sequence
      ;;
  esac
elif [[ "$SYMMETRY_PROFILE" != "manual" ]]; then
  echo "SYMMETRY_PROFILE must be auto or manual" >&2
  exit 1
fi
SYMMETRY_YAW_TRANSITION=${SYMMETRY_YAW_TRANSITION:-0.2}
export SYMMETRY_AXIS SYMMETRY_STEP_DEG SYMMETRY_AXIS_FLIP
export SYMMETRY_SELECTION_MODE SYMMETRY_YAW_TRANSITION

for path in "$PI3_PYTHON" "$MANIFEST" "$RUN_ORACLE"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
done

list_representatives() {
  "$PI3_PYTHON" -c '
import json, sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
by_object = {}
for row in rows:
    by_object.setdefault(str(row["object_name"]), []).append(row)
for name in sorted(by_object):
    candidates = sorted(
        by_object[name],
        key=lambda row: (
            str(row.get("hand_side", "")),
            str(row["stream_id"]),
        ),
    )
    row = candidates[0]
    print(
        name,
        row.get("hand_side", "unknown"),
        row["stream_id"],
        row["stream_dir"],
        sep="\t",
    )
' "$MANIFEST"
}

if [[ "$OBJECT_NAME" == "--list" || -z "$OBJECT_NAME" ]]; then
  echo -e "object\thand_side\tstream_id\tsequence_dir"
  list_representatives
  exit 0
fi

if [[ "$LIST_ONLY" == "1" ]]; then
  "$PI3_PYTHON" -c '
import json, sys

manifest, object_name = sys.argv[1:]
rows = sorted(
    [
        json.loads(line)
        for line in open(manifest)
        if line.strip()
        and str(json.loads(line)["object_name"]) == object_name
    ],
    key=lambda row: (
        str(row.get("hand_side", "")),
        str(row["stream_id"]),
    ),
)
if not rows:
    raise SystemExit(f"Object not found in manifest: {object_name}")
print("index\thand_side\tstream_id\tsequence_dir")
for index, row in enumerate(rows):
    print(
        index,
        row.get("hand_side", "unknown"),
        row["stream_id"],
        row["stream_dir"],
        sep="\t",
    )
' "$MANIFEST" "$OBJECT_NAME"
  exit 0
fi

selection=$(
  "$PI3_PYTHON" -c '
import json, sys

manifest, object_name, object_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(manifest) if line.strip()]
candidates = sorted(
    [row for row in rows if str(row["object_name"]) == object_name],
    key=lambda row: (
        str(row.get("hand_side", "")),
        str(row["stream_id"]),
    ),
)
if not candidates:
    raise SystemExit(f"Object not found in manifest: {object_name}")
if not 0 <= object_index < len(candidates):
    raise SystemExit(
        f"OBJECT_INDEX={object_index} outside [0, {len(candidates) - 1}]"
    )
row = candidates[object_index]
print(row["stream_id"], row["stream_dir"], row.get("hand_side", "unknown"), sep="\t")
' "$MANIFEST" "$OBJECT_NAME" "$OBJECT_INDEX"
)

IFS=$'\t' read -r STREAM_ID SEQ_DIR HAND_SIDE <<< "$selection"
export STREAM_ID SEQ_DIR PORT

echo "object=$OBJECT_NAME"
echo "hand_side=$HAND_SIDE"
echo "stream_id=$STREAM_ID"
echo "sequence_dir=$SEQ_DIR"
echo "port=$PORT"
echo "symmetry_profile=$SYMMETRY_PROFILE"
echo "symmetry_axis=$SYMMETRY_AXIS"
echo "symmetry_step_deg=$SYMMETRY_STEP_DEG"
echo "symmetry_axis_flip=$SYMMETRY_AXIS_FLIP"
echo "symmetry_selection_mode=$SYMMETRY_SELECTION_MODE"

cd "$DO_AS_I_DO"
exec /bin/bash "$RUN_ORACLE"
