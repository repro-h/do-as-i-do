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
