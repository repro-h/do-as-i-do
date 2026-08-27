# V15 side-free multi-hand trajectory

V15 predicts one absolute camera-frame translation per hand instance. Pi3X is
encoded once per scene frame; every padded hand slot queries the shared feature
grid with its own 21 two-dimensional joints. Handedness is metadata only and is
never passed to the model.

The first DexYCB experiment uses native 2D joints with train-only synthetic
noise. This is a controlled test of Pi3X depth observability, not the final
inference query provider. A later detector/WiLoR provider can emit the same
`[T, H, 21, ...]` schema.

Joint visibility comes from the external `hand_visibility_detector` as 21
sigmoid probabilities. Export it once per stream with
`export_hand_visibility.py`; training never runs that network online. Missing
detections mask all GT joint queries and use a temporally propagated wrist ray
anchor, so an unavailable GT location cannot leak into the model input. Their
3D targets retain weight 0.5 within four frames of an observed anchor and 0.2
within eight frames; targets farther from an anchor receive zero loss. These
distances and weights are configurable with `--near-anchor-frames`,
`--max-anchor-frames`, `--near-missing-weight`, and `--far-missing-weight`.
`mask` and `ones` remain available only as ablations through
`--visibility-source`.

Joint validity remains per joint. A hand that is partly outside the image keeps
its in-frame joint queries while out-of-frame joints become missing tokens.
Each missing token retains its joint identity. For valid joints, visibility
softly blends the observed and missing representations and scales the local
Pi3X attention bias, so low-visibility joints rely less on the occluder's local
feature without introducing a separate gating head.

## Stable hand slots

`export_multihand_tracks.py` converts frame-level `[N,21,...]` annotations to
stable `[T,H,21,...]` slots over the complete stream. Association uses side
metadata when available, constant-velocity 2D joint prediction and one-to-one
matching. It also separates `track_valid`, `observation_valid` and
`target_valid`. Single-hand labels naturally occupy slot zero, so old DexYCB
experiments remain compatible.

```bash
python export_multihand_tracks.py \
  --windows /data2/hyp/full_v15/manifests/train_windows.jsonl \
  --out-root /data2/hyp/full_v15/tracks/train \
  --max-hands 4
```

For a real multi-hand visibility cache, pass the track root so all detections
are matched one-to-one to stable slots:

```bash
python export_hand_visibility.py \
  --windows /data2/hyp/full_v15/manifests/train_windows.jsonl \
  --track-root /data2/hyp/full_v15/tracks/train \
  --max-hands 4 \
  --detector-root "$HAND_VISIBILITY_ROOT" \
  --checkpoint "$VISIBILITY_CHECKPOINT" \
  --out-root /data2/hyp/full_v15/visibility/train
```

Training consumes the caches through `--track-train-root` and
`--track-val-root`. Without these flags, label order is used as a backward-
compatible fallback; it is suitable for DexYCB single-hand data only.

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
  --visibility-train-root /data2/hyp/unihand-v15/visibility/train \
  --visibility-val-root /data2/hyp/unihand-v15/visibility/val \
  --visibility-source detector \
  --out-dir /tmp/v15-audit \
  --audit-only
```

## Sharded cache export

Both exporters support `--num-shards`, `--shard-index` and strict cache
validation. `run_sharded_export.sh` launches one shard per selected GPU, while
the existing `wait_for_gpus_and_run.sh` waits until enough GPUs are free.

Pi3X S0 caches must use original camera coordinates. A prior V13 cache may be
passed through `--reuse-root`: valid right-hand streams are symlinked, while
canonical-right/mirrored left-hand streams are rejected and recomputed.

```bash
wait_for_gpus_and_run.sh --count 4 --max-used-mib 2000 -- \
  run_sharded_export.sh \
  --log-dir /data2/hyp/test_v15/logs/pi3x_train \
  --num-shards 4 -- \
  "$PI3_PYTHON" -u export_pi3x_s0.py \
  --windows /data2/hyp/test_v15/manifests/train_windows.jsonl \
  --hand-uni-root "$HAND_UNI_ROOT" \
  --pi3-root "$PI3_ROOT" \
  --checkpoint "$PI3X_CHECKPOINT" \
  --export-script "$EXPORT_PI3X_SCRIPT" \
  --out-root /data2/hyp/test_v15/pi3x/train \
  --reuse-root /data2/hyp/unihand-pi3x-feature/v13_pi3x_full_ws16_s8_fp16/train \
  --window-size 16 --window-stride 8 --device cuda
```

Visibility export uses the same launcher and the visibility environment:

```bash
wait_for_gpus_and_run.sh --count 4 --max-used-mib 2000 -- \
  run_sharded_export.sh \
  --log-dir /data2/hyp/test_v15/logs/visibility_train \
  --num-shards 4 -- \
  "$VISIBILITY_PYTHON" -u export_hand_visibility.py \
  --windows /data2/hyp/test_v15/manifests/train_windows.jsonl \
  --detector-root "$HAND_VISIBILITY_ROOT" \
  --out-root /data2/hyp/test_v15/visibility/train \
  --backbone wilor --device cuda
```

The default Pi3X regime is 16 frames with stride 8. A 74-frame stream therefore
produces starts `0,8,...,56,58` and nine independently inferred overlapping
windows. Larger windows are supported by the exporter. Training and inference
must use the same context regime; for windows above 64 frames, also pass a
sufficient `--max-window-size` to `train.py`.

## H2O two-hand smoke

`build_h2o_sequence_windows.py` adapts one H2O camera sequence to the same
`[T,H,21,...]` label and window contract. H2O stores left then right hand in
camera coordinates and meters. The adapter projects both hands with the native
camera intrinsics and keeps their original image coordinates. It writes a zero
segmentation only to carry image dimensions; use detector visibility and the
minimal Pi3X cache, so this placeholder never supplies a hand mask.

```bash
python build_h2o_sequence_windows.py \
  --sequence-dir /data2/hyp/data/H2O/subject1_ego/h1/0/cam4 \
  --out-dir /data2/hyp/test_v15/h2o_subject1_h1_0_cam4 \
  --split val --window-size 16 --window-stride 8 --overwrite
```

## V16 online Pi3X clips

`train_v16_online_pi3x.py` is independent of the disk-cache training path. It
runs frozen Pi3X exactly once for every unique manifest clip, keeps the decoder
features in host RAM, releases Pi3X GPU memory, and then trains the trajectory
head for all epochs. Existing Pi3X exports are neither read nor modified.

Clip length is controlled by the manifest builder. For example, use
`--window-size 100 --window-stride 50` and set `--max-window-size 128` in V16.
This mode is intended for a sequence smoke or a host with enough RAM. The
feature footprint grows approximately linearly with frames and clips; use the
existing disk-cache V15 path for full-dataset training when the complete cache
does not fit in host RAM.

## V16.1 compact Pi3X candidates

`train_v16_1_compact_pi3x.py` keeps Pi3X frozen and runs it once per unique
manifest clip. Before transferring anything to host RAM, it gathers a fixed
local candidate patch around every clean 2D joint and adaptive global context
tokens, then releases the dense decoder grid. The local attention, visibility
and missing-token fusion, temporal transformer, and translation heads remain
trainable. No Pi3X feature file is written to disk.

The clip length still comes from the manifest. `--joint-patch-radius 1` keeps a
3x3 candidate neighborhood and `--global-grid-size 4` keeps 16 global tokens.
Set `--max-hands` to the actual dataset maximum because compact memory scales
linearly with hand slots (DexYCB: 1, this H2O smoke: 2).
