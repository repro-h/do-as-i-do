# DexYCB reconstruction adapters

Local adapters for running the `do-as-i-do` reconstruction pipeline on DexYCB.
The upstream reconstruction scripts remain unchanged.

## Prepare one stream

`prepare_dexycb_sequence.py` skips SAM3 and converts DexYCB RGB frames and
ground-truth segmentation into the layout expected by TAPIR and Fast-SAM3D:

```text
<out_dir>/
  dexycb_sequence.mp4
  dexycb_frame_map.json
  config.json
  all_frames/000000.png
  video_segmentation/masks/frame_000000_masks/<object_name>.png
  video_segmentation/masks/frame_000000_masks/<left|right>_hand_0.png
```

The target object is read from `meta.yml` using
`ycb_ids[ycb_grasp_ind]`. Output frame IDs are contiguous and the mapping back
to original DexYCB frame IDs is stored in `dexycb_frame_map.json`.

```bash
python reconstruction/hyp_scripts/prepare_dexycb_sequence.py \
  --stream_dir /path/to/DexYCB/<subject>/<sequence>/<camera> \
  --object_model_root /path/to/DexYCB/models \
  --out_dir /path/to/output \
  --fps 30 \
  --overwrite
```

## Reuse an existing SAM3D shape

`prepare_existing_sam3d_shape.py` converts the canonical GLB/NPZ already
produced by hand-uni into the OBJ directory and `layout.json` written by
`generate_mesh_sam3d.py`. It preserves the original network rotation,
translation and scale and links the source GLB/NPZ for provenance.

This conversion does not load a SAM3D model or use CUDA. It only requires
NumPy, SciPy and trimesh.

```bash
python reconstruction/hyp_scripts/prepare_existing_sam3d_shape.py \
  --run_dir /path/to/prepared/stream \
  --shape_bank_root /path/to/object_shape_bank_v2 \
  --overwrite
```

## Continue the reconstruction pipeline

After RGB/masks and the existing SAM3D shape are prepared, continue from
pointmaps through tracking without re-running frame extraction, SAM3, or SAM3D
shape generation:

```bash
bash reconstruction/hyp_scripts/run_prepared_dexycb.sh \
  --run-dir /path/to/prepared/stream \
  --gpu 0
```

The default Fast-SAM3D settings use 8 pose samples and 12 Euler steps to fit a
24 GB GPU. Pass `--force` to recompute cached stages.
