#!/usr/bin/env bash
# Resume the original OakInk2 export locations without changing the split.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=/home/mengxiangting/nas/mengxt/anaconda3/envs/pi3/bin/python
visibility_root=/home/mengxiangting/nas/mengxt/Projects/hand_visibility_detector
visibility_python="$visibility_root/.venv/bin/python"
visibility_checkpoint=/data2/hyp/test_v15/huggingface/hub/models--ryhara--hand-visibility-detector/snapshots/941b791bcba4a0bb381c325c225f56e0a80cf98f/best.pt
oak_root=/data2/hyp/data/OakInk2
out_root=/data2/hyp/full_v15/oakink2
sequence_list=""
mano_folder=/home/mengxiangting/nas/mengxt/Projects/Pi3_WiLoR_Hand/mano_data
hand_uni_root=/home/mengxiangting/nas/mengxt/Projects/Pi3_WiLoR_Hand
pi3_root=/home/mengxiangting/nas/mengxt/Projects/Pi3
pi3x_checkpoint=/mnt/nas/mengxt/Projects/Pi3/ckpts/model.safetensors
training_checkpoint=/data2/hyp/full_v15/mixed_five_dataset_full_track_aligned_v1/checkpoints/v16_2_five_dataset_track_aligned_tail_cosine_v1/best.pt
compact_root=""
gpus=""
wait_pid=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: bash run_oakink2_incremental_exports.sh --gpus 5,6 [options]
  --wait-pid PID              Wait for an already running preparation process
  --sequence-list FILE        Default: OakInk2/sequence_sample_128_clean.txt
  --oak-root DIR              Original extracted OakInk2 root
  --out-root DIR              Original processed root, including status.json
  --compact-cache-root DIR    Existing shared compact cache (or read checkpoint)
  --training-checkpoint FILE  Trusted training checkpoint providing cache root
  --pi3x-checkpoint FILE      Pi3X weights used for the existing cache
  --python FILE --visibility-python FILE
  --visibility-root DIR --visibility-checkpoint FILE
  --mano-model-folder DIR --hand-uni-root DIR --pi3-root DIR
  --dry-run                   Validate paths and print commands, without exporting

Order: prepare -> tracks train/val -> visibility train/val -> compact train/val.
Existing caches are checked by their exporters; no --overwrite is supplied.
Uses original A001/O001 validation subjects, ego, ws16/s8, max-hands=2.
EOF
}

while (($#)); do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --wait-pid) wait_pid="$2"; shift 2 ;;
    --sequence-list) sequence_list="$2"; shift 2 ;;
    --oak-root) oak_root="$2"; shift 2 ;;
    --out-root) out_root="$2"; shift 2 ;;
    --compact-cache-root) compact_root="$2"; shift 2 ;;
    --training-checkpoint) training_checkpoint="$2"; shift 2 ;;
    --pi3x-checkpoint) pi3x_checkpoint="$2"; shift 2 ;;
    --python) python="$2"; shift 2 ;;
    --visibility-python) visibility_python="$2"; shift 2 ;;
    --visibility-root) visibility_root="$2"; shift 2 ;;
    --visibility-checkpoint) visibility_checkpoint="$2"; shift 2 ;;
    --mano-model-folder) mano_folder="$2"; shift 2 ;;
    --hand-uni-root) hand_uni_root="$2"; shift 2 ;;
    --pi3-root) pi3_root="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Specify --gpus, e.g. 5,6' >&2; exit 2; }
IFS=',' read -r -a gpu_ids <<<"$gpus"
seen_gpus=","
for gpu in "${gpu_ids[@]}"; do
  [[ "$seen_gpus" != *",$gpu,"* ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }
  seen_gpus+="$gpu,"
done
[[ -z "$wait_pid" || "$wait_pid" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid wait PID' >&2; exit 2; }
sequence_list=${sequence_list:-"$oak_root/sequence_sample_128_clean.txt"}
for executable in "$python" "$visibility_python"; do
  [[ -x "$executable" ]] || { echo "Python not executable: $executable" >&2; exit 1; }
done
for file in "$sequence_list" "$out_root/status.json" "$visibility_checkpoint" "$pi3x_checkpoint"; do
  [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done
for directory in "$oak_root" "$visibility_root" "$mano_folder" "$hand_uni_root" "$pi3_root"; do
  [[ -d "$directory" ]] || { echo "Missing directory: $directory" >&2; exit 1; }
done
if [[ -z "$compact_root" ]]; then
  [[ -f "$training_checkpoint" ]] || { echo "Missing trusted training checkpoint: $training_checkpoint" >&2; exit 1; }
  compact_root=$("$python" - "$training_checkpoint" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
root = checkpoint["args"]["compact_cache_root"]
if not isinstance(root, str) or not root.strip():
    raise ValueError("Checkpoint has no compact_cache_root")
print(root)
PY
  )
fi
[[ -d "$compact_root" ]] || { echo "Existing compact cache not found: $compact_root" >&2; exit 1; }

log_root="$out_root/logs/incremental_exports"
mkdir -p "$log_root"
exec 9>"$log_root/pipeline.lock"
flock -n 9 || { echo 'Another incremental export pipeline is running' >&2; exit 1; }
trap 'echo "[$(date +%FT%T)] FAILED near line $LINENO; inspect stage logs in $log_root" >&2' ERR
echo "GPUs=$gpus processed=$out_root compact=$compact_root"

if [[ -n "$wait_pid" && "$dry_run" -eq 0 ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do
    echo "[$(date +%FT%T)] waiting for preparation pid=$wait_pid"
    sleep 30
  done
fi

stage() {
  local name="$1"
  shift
  echo "[$(date +%FT%T)] starting $name; log=$log_root/$name.log"
  if ((dry_run)); then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@" >"$log_root/$name.log" 2>&1
    echo "[$(date +%FT%T)] completed $name"
  fi
}

stage prepare "$python" -u "$script_dir/prepare_oakink2_sample_v15.py" \
  --oakink2-root "$oak_root" --sequence-list "$sequence_list" \
  --mano-model-folder "$mano_folder" --out-root "$out_root" \
  --cameras egocentric --val-subjects A001 O001 \
  --frame-stride 1 --window-size 16 --window-stride 8 --overlay-count 0

for split in train val; do
  stage "tracks_$split" "$python" -u "$script_dir/export_multihand_tracks.py" \
    --windows "$out_root/manifests/${split}_windows.jsonl" \
    --out-root "$out_root/tracks/$split" --max-hands 2 \
    --status-json "$log_root/tracks_${split}.json"
done

for split in train val; do
  stage "visibility_$split" env CUDA_VISIBLE_DEVICES="$gpus" \
    bash "$script_dir/run_sharded_export.sh" \
    --log-dir "$log_root/visibility_$split" --num-shards "${#gpu_ids[@]}" -- \
    "$visibility_python" -u "$script_dir/export_hand_visibility.py" \
    --windows "$out_root/manifests/${split}_windows.jsonl" \
    --track-root "$out_root/tracks/$split" --out-root "$out_root/visibility/$split" \
    --detector-root "$visibility_root" --checkpoint "$visibility_checkpoint" \
    --backbone wilor --max-hands 2 --device cuda
done

for split in train val; do
  stage "compact_$split" env CUDA_VISIBLE_DEVICES="$gpus" \
    bash "$script_dir/run_sharded_export.sh" \
    --log-dir "$log_root/compact_$split" --num-shards "${#gpu_ids[@]}" -- \
    "$python" -u "$script_dir/export_compact_pi3x_cache.py" \
    --windows "$out_root/manifests/${split}_windows.jsonl" \
    --track-root "$out_root/tracks/$split" --visibility-root "$out_root/visibility/$split" \
    --hand-uni-root "$hand_uni_root" --pi3-root "$pi3_root" \
    --pi3x-checkpoint "$pi3x_checkpoint" --out-root "$compact_root" \
    --max-hands 2 --pixel-limit 180000 --joint-patch-radius 1 \
    --global-grid-size 4 --feature-dtype float16 --device cuda
done
echo "[$(date +%FT%T)] pipeline complete (dry_run=$dry_run); no training launched"
