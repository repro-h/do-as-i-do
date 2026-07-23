#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_prepared_dexycb.sh --run-dir DIR [options]

Continue the do-as-i-do reconstruction pipeline from a DexYCB stream prepared
by prepare_dexycb_sequence.py and prepare_existing_sam3d_shape.py.

Options:
  --run-dir DIR             Prepared stream directory (required)
  --gpu ID                  CUDA_VISIBLE_DEVICES value (default: 0)
  --sam3d-env NAME          SAM3D/Fast-SAM3D conda env (default: sam3d-objects)
  --hawor-env NAME          HaWoR conda env (default: hawor)
  --tapnet-env NAME         TAPIR conda env (default: tapnet)
  --pose-samples N          Fast-SAM3D pose samples (default: 8)
  --euler-steps N           Fast-SAM3D Euler steps (default: 12)
  --force                   Recompute stages even when their final output exists
  --skip-hawor              Skip HaWoR and final hand-anchored scale optimization
EOF
}

RUN_DIR=""
GPU=0
ENV_SAM3D=sam3d-objects
ENV_HAWOR=hawor
ENV_TAPNET=tapnet
POSE_SAMPLES=8
EULER_STEPS=12
FORCE=false
SKIP_HAWOR=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --sam3d-env) ENV_SAM3D="$2"; shift 2 ;;
    --hawor-env) ENV_HAWOR="$2"; shift 2 ;;
    --tapnet-env) ENV_TAPNET="$2"; shift 2 ;;
    --pose-samples) POSE_SAMPLES="$2"; shift 2 ;;
    --euler-steps) EULER_STEPS="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --skip-hawor) SKIP_HAWOR=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  usage >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECON_ROOT="$(cd "$HERE/.." && pwd)"
SCRIPTS_DIR="$RECON_ROOT/scripts"
SAM3D_DIR="$RECON_ROOT/modules/sam-3d-objects"
FASTSAM3D_DIR="$RECON_ROOT/modules/Fast-SAM3D"
HAWOR_DIR="$RECON_ROOT/modules/HaWoR"
WEIGHTS_DIR="$RECON_ROOT/weights"
TAPNET_CKPT="$WEIGHTS_DIR/tapnet/bootstapir_checkpoint_v2.pt"

RUN_DIR="$(realpath "$RUN_DIR")"
FRAME_MAP="$RUN_DIR/dexycb_frame_map.json"
CONFIG_JSON="$RUN_DIR/config.json"
VIDEO_PATH="$RUN_DIR/dexycb_sequence.mp4"
VIDEO_MASKS_DIR="$RUN_DIR/video_segmentation/masks"

for path in "$FRAME_MAP" "$CONFIG_JSON" "$VIDEO_PATH" "$TAPNET_CKPT"; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 1; }
done

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

activate_env() {
  # Some conda activate.d scripts read toolchain variables before defining
  # them, which is incompatible with bash nounset mode.
  set +u
  conda activate "$1"
  set -u
}

read_json() {
  "$CONDA_BASE/bin/python" - "$1" "$2" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
value = data
for part in sys.argv[2].split("."):
    value = value[int(part)] if isinstance(value, list) else value[part]
print(value)
PY
}

INIT_FRAME="$(read_json "$CONFIG_JSON" frame_number)"
OBJECT_ID="$(read_json "$CONFIG_JSON" object_names.0)"
ANCHOR_HAND="$(read_json "$CONFIG_JSON" anchor_hand)"
FRAME_PATH="$(printf '%s/%04d.png' "$RUN_DIR" "$INIT_FRAME")"
POINTMAP_PATH="$(printf '%s/%04d_pointmap.npy' "$RUN_DIR" "$INIT_FRAME")"
INTRINSICS_PATH="$(printf '%s/%04d_intrinsics.txt' "$RUN_DIR" "$INIT_FRAME")"
MASKS_DIR="$(printf '%s/frame_%06d_masks' "$VIDEO_MASKS_DIR" "$INIT_FRAME")"
OBJECT_MESH="$MASKS_DIR/$OBJECT_ID/$OBJECT_ID.obj"
MOTION_JSON="$RUN_DIR/perframe_tracking_$OBJECT_ID/motion_stats.json"
TRACK_ROOT="$RUN_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization"
LAYOUT_JSON="$TRACK_ROOT/layout.json"
LAYOUT_CF="$TRACK_ROOT/layout_camera_frame.json"
LAYOUT_OPT="$TRACK_ROOT/layout_camera_frame_optimized.json"
HAND_MESHES="$RUN_DIR/dexycb_sequence/all_hand_meshes.npz"

for path in "$FRAME_PATH" "$MASKS_DIR/layout.json" "$OBJECT_MESH"; do
  [[ -e "$path" ]] || { echo "Missing prepared shape asset: $path" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="$GPU"
echo "run_dir=$RUN_DIR"
echo "object=$OBJECT_ID init_frame=$INIT_FRAME hand=$ANCHOR_HAND gpu=$GPU"

activate_env "$ENV_SAM3D"
if [[ "$FORCE" == true || ! -f "$POINTMAP_PATH" || ! -f "$INTRINSICS_PATH" ]]; then
  echo "=== Reference pointmap ==="
  cd "$SCRIPTS_DIR"
  python get_pointmap_dir.py --image "$FRAME_PATH" --output "$POINTMAP_PATH"
else
  echo "=== Reference pointmap: cached ==="
fi

if [[ "$SKIP_HAWOR" == false ]]; then
  if [[ "$FORCE" == true || ! -f "$HAND_MESHES" ]]; then
    echo "=== HaWoR ==="
    activate_env "$ENV_HAWOR"
    cd "$HAWOR_DIR"
    IMG_FOCAL="$(head -n 1 "$INTRINSICS_PATH")"
    python demo.py --video_path "$VIDEO_PATH" --vis_mode cam --img_focal "$IMG_FOCAL" --static_camera
  else
    echo "=== HaWoR: cached ==="
  fi
fi

activate_env "$ENV_SAM3D"
FIRST_POINTMAP="$RUN_DIR/all_frames/000000_pointmap.npy"
if [[ "$FORCE" == true || ! -f "$FIRST_POINTMAP" ]]; then
  echo "=== All-frame pointmaps ==="
  cd "$SCRIPTS_DIR"
  python get_pointmap_dir.py --image_dir "$RUN_DIR/all_frames"
else
  echo "=== All-frame pointmaps: cached ==="
fi

if [[ "$FORCE" == true || ! -f "$RUN_DIR/gravity.json" ]]; then
  echo "=== Gravity ==="
  cd "$SCRIPTS_DIR"
  python predict_video_gravity.py "$RUN_DIR/all_frames" --output_path "$RUN_DIR/gravity.json"
else
  echo "=== Gravity: cached ==="
fi

if [[ "$FORCE" == true || ! -f "$MOTION_JSON" ]]; then
  echo "=== TAPIR ==="
  activate_env "$ENV_TAPNET"
  cd "$SCRIPTS_DIR"
  python tapir_velocity_tracking.py \
    --video "$VIDEO_PATH" \
    --mask-dir "$VIDEO_MASKS_DIR" \
    --object "$OBJECT_ID" \
    --checkpoint "$TAPNET_CKPT" \
    --fps 30
else
  echo "=== TAPIR: cached ==="
fi

if [[ "$FORCE" == true || ! -f "$LAYOUT_JSON" ]]; then
  echo "=== Fast-SAM3D tracking ==="
  activate_env "$ENV_SAM3D"
  cd "$FASTSAM3D_DIR"
  PYTHONPATH="$SAM3D_DIR${PYTHONPATH:+:$PYTHONPATH}" python track_object.py \
    --config checkpoints/hf/pipeline.yaml \
    --vid_dir "$RUN_DIR" \
    --masks_root "$VIDEO_MASKS_DIR" \
    --object_name "$OBJECT_ID" \
    --init_frame "$INIT_FRAME" \
    --output_dir "$RUN_DIR/obj_tracking_out/$OBJECT_ID" \
    --guidance_strength 1 \
    --save_layout \
    --fix_scale_to_init_frame \
    --pose_guidance_strength 0.5 \
    --num_pose_samples "$POSE_SAMPLES" \
    --scoring_metric render_iou \
    --pose_selection cluster \
    --cluster_dist_thresh 0.3 \
    --cluster_min_size 2 \
    --cluster_w_rot 1.5 \
    --chain_poses \
    --post_optimize \
    --no-enable_shape_icp \
    --chain_on_diffusion \
    --enable_ss_cache \
    --euler_steps "$EULER_STEPS" \
    --rotvel_json "$MOTION_JSON"
else
  echo "=== Fast-SAM3D tracking: cached ==="
fi

activate_env "$ENV_SAM3D"
cd "$SCRIPTS_DIR"
if [[ "$FORCE" == true || ! -f "$TRACK_ROOT/projected/video.mp4" ]]; then
  echo "=== Project tracked mesh ==="
  python run_project_mesh_combined.py \
    --video "$VIDEO_PATH" \
    --mesh "$OBJECT_MESH" \
    --json "$LAYOUT_JSON" \
    --output-base "$TRACK_ROOT/projected"
fi

if [[ "$FORCE" == true || ! -f "$LAYOUT_CF" ]]; then
  echo "=== Convert layout to camera frame ==="
  python convert_layout_to_camera_frame.py --input "$LAYOUT_JSON" --output "$LAYOUT_CF"
fi

if [[ "$SKIP_HAWOR" == false ]]; then
  if [[ "$FORCE" == true || ! -f "$LAYOUT_OPT" ]]; then
    echo "=== Translation/scale optimization ==="
    python optimize_translation_scale.py \
      --video-dir "$RUN_DIR" \
      --layout-json "$LAYOUT_CF" \
      --hand-meshes "$HAND_MESHES" \
      --anchor-hand "$ANCHOR_HAND" \
      --ref-frame "$INIT_FRAME"
  fi
fi

echo "=== Done ==="
echo "layout=$LAYOUT_JSON"
echo "camera_layout=$LAYOUT_CF"
[[ -f "$LAYOUT_OPT" ]] && echo "optimized_layout=$LAYOUT_OPT"
