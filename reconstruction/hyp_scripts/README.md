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
