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
PORT=${PORT:-8098}
RUN_ORACLE=$DO_AS_I_DO/reconstruction/hyp_scripts/train/run_canonical_oracle_hand_target.sh

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

cd "$DO_AS_I_DO"
exec /bin/bash "$RUN_ORACLE"
