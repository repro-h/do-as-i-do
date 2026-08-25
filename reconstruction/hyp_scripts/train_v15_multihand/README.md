# V15 side-free multi-hand trajectory

V15 predicts one absolute camera-frame translation per hand instance. Pi3X is
encoded once per scene frame; every padded hand slot queries the shared feature
grid with its own 21 two-dimensional joints. Handedness is metadata only and is
never passed to the model.

The first DexYCB experiment uses native 2D joints with train-only synthetic
noise. This is a controlled test of Pi3X depth observability, not the final
inference query provider. A later detector/WiLoR provider can emit the same
`[T, H, 21, ...]` schema.

## Coordinate contract

- Images, 2D joints, intrinsics, Pi3X cache and 3D targets use original camera
  coordinates.
- Do not export Pi3X with `--canonical-right`.
- `horizontal_mirror=True` caches are rejected by the dataset.
- DexYCB currently fills hand slot zero; remaining slots are padding. The model
  already supports multiple hands and multiple hands of the same side.

## Build S0 manifests

```bash
python build_dexycb_s0_windows.py \
  --dexycb-root /mnt/nas/wuke/HumanData/DexYCB \
  --out-dir /data2/hyp/unihand-v15/manifests \
  --window-size 16 \
  --window-stride 8 \
  --overwrite
```

S0 follows the existing `hand-uni` implementation: training uses sequence
indices whose index modulo five is not four; validation uses the held-out
sequences from subjects 1-2; test uses held-out sequences from subjects 3-10.

## Audit after original-camera Pi3X export

```bash
python train.py \
  --train-windows /data2/hyp/unihand-v15/manifests/train_windows.jsonl \
  --val-windows /data2/hyp/unihand-v15/manifests/val_windows.jsonl \
  --pi3x-train-root /data2/hyp/unihand-v15/pi3x/train \
  --pi3x-val-root /data2/hyp/unihand-v15/pi3x/val \
  --out-dir /tmp/v15-audit \
  --audit-only
```

