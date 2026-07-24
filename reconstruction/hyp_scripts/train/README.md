# Hybrid hand-object refinement training

This directory contains the experimental training pipeline that combines:

- the fixed per-object SAM3D shape bank;
- QA-approved FoundationPose object tracks;
- HandFlow hand initialization;
- DexYCB camera-space ground truth.

The pipeline is intentionally staged:

1. Build and audit a compact stream manifest.
2. Cache HandFlow initialization for approved streams.
3. Export temporal windows without duplicating RGB-D data.
4. Train the rigid object/wrist temporal residual refiner.
5. Train the local MANO contact refiner.

The first stage does not require SAM3D-to-YCB canonical alignment. Object
supervision is defined using camera-space rendering, depth, masks, and surface
geometry.

## Local data layout

Machine-local inputs and generated artifacts can be grouped under:

```text
reconstruction/data/dexycb/
├── foundationpose_quality_filter_v2 -> external QA results
├── objects -> external SAM3D object shape bank
└── hybrid_training_v1/
    ├── manifests/
    ├── handflow_cache/
    ├── audits/
    └── windows/
```

The complete `reconstruction/data/dexycb/` directory is gitignored because it
contains machine-specific absolute symlinks and generated data.

## Build a manifest

Pass each QA-approved pose index explicitly so stale shards are not discovered
accidentally:

```bash
python reconstruction/hyp_scripts/train/build_dexycb_hybrid_manifest.py \
  --index train=/path/to/train_passed_pose_index.json \
  --index val=/path/to/val_passed_pose_index.json \
  --index test=/path/to/test_passed_pose_index.json \
  --dexycb-root /path/to/DexYCB \
  --shape-bank-root /path/to/object_shape_bank_v2/objects \
  --out-dir /path/to/hybrid_training_v1 \
  --exclude-object 007_tuna_fish_can
```

This writes one JSONL per split, an all-splits JSONL, and an audit summary. It
does not copy images, depth maps, labels, poses, or meshes.

## Select and run a pilot

Use `select_dexycb_hybrid_pilot.py` to select a deterministic pilot with equal
left/right coverage and as many distinct objects as possible. Then run
`run_handflow_hybrid_jobs.py`.

Left-hand streams are mirrored before right-hand HandFlow inference, then their
camera-space vertices are mirrored back and face winding is corrected. Raw
right-MANO parameters from mirrored inference are retained under
`handflow_raw_*` keys but must not be interpreted as left-MANO parameters.

Run `audit_hand_object_initialization.py` before scaling up. It reports
HandFlow-to-GT hand error, temporal speed/acceleration, FoundationPose
translation motion, relative hand-object motion, and unsigned hand-to-SAM3D
surface distances. It intentionally does not report signed penetration until a
reliable collision representation is selected.

Hand/contact temporal metrics are gated by both HandFlow validity and DexYCB GT
hand validity. This prevents off-screen HandFlow hallucinations from becoming
contact supervision. The future training dataset must carry the same explicit
visibility mask.

Full HandFlow export supports deterministic `--num-shards/--shard-index`
partitioning. By default it removes raw HandFlow NPZ files and rendered videos
after producing `handflow_camera_result.npz`; use `--keep-raw` or
`--keep-videos` only for small debugging runs.

`wait_train_then_export_val.sh` waits for all configured train shard PIDs to
exit, detects sufficiently idle GPUs from an allowlist, and launches validation
with a matching dynamic shard count. It intentionally does not require every
train manifest record to have succeeded; failed or missing train exports are
audited separately.

## Stage-1 rigid temporal refinement

After train and validation HandFlow caches finish, run
`prepare_stage1_rigid_supervision.py` for each split. The compact per-stream
files contain predicted/GT hand centers, predicted/GT object surface centers,
FoundationPose rotation context, and explicit hand/object/relative validity
masks. SAM3D and DexYCB CAD centroids are transformed independently, so their
different canonical origins are never treated as the same translation target.

`train_stage1_rigid_refiner.py` trains a temporal Transformer that predicts
small hand and object translation residuals. The initial stage intentionally
does not supervise object rotation across incompatible canonical frames and
does not optimize local MANO articulation or contact. Those belong to the
second-stage geometric/contact refiner.

Use `--mode object_only` to train the object stabilization stage first. In
this mode the output head predicts only object translation, hand residuals are
identically zero, and velocity/acceleration losses supervise the object
trajectory itself. `--mode hand_only` is reserved for the subsequent
object-frozen wrist/relative-position stage.

`--w-delta-velocity` and `--w-delta-acceleration` directly regularize temporal
changes in the predicted correction. They are separate from trajectory losses:
the latter match GT motion, while delta smoothness prevents the refiner itself
from injecting visible jitter.

`stabilize_foundationpose_object.py` merges an object-only translation
prediction into its source FoundationPose track and applies motion-aware local
SO(3) smoothing to each contiguous rotation segment. The output preserves the
FoundationPose JSON schema and includes before/after angular speed and
acceleration audits.

`fit_isolated_object_sequence.py` performs CHOIR-style per-sequence 6DoF
fitting directly from a FoundationPose initialization. Its silhouette
repulsion ignores dilated hand-occlusion pixels, attraction uses visible
DexYCB object pixels, and depth fitting uses valid visible RGB-D object pixels.
It does not read DexYCB object poses.

Run `audit_stage1_rigid_supervision.py` before training. It reports hand,
object, and relative residual distributions, 2D projection error, left/right
breakdowns, and streams exceeding the configured residual range.

Use `apply_stage1_rigid_refiner.py` with `best.pt` to aggregate overlapping
window predictions into one hand/object translation residual per frame. The
output keeps prediction counts and corrected centers for evaluation and later
mesh/pose visualization adapters.

Use `visualize_stage1_dexycb.py` to prepare GT overlays and launch paired
before/after Viser viewers. Without `--stream-id` it selects a representative
validation stream whose relative median error improved; pass an exact stream
ID to inspect a specific prediction. Prepared assets are cached per stream.

Use `render_stage1_dexycb_comparison.py` for deterministic offline rendering.
It writes every RGB frame to the output timeline and produces original,
Stage-1-corrected, and side-by-side MP4 files without depending on Viser's
browser update rate.
