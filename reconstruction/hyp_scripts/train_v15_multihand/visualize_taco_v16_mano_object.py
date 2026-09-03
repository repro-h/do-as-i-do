#!/usr/bin/env python3
"""Viser viewer for TACO GT/predicted MANO hands and GT object meshes."""

import argparse
import json
import pickle
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
import viser

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_dataset import CompactWindowDataset  # noqa: E402
from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from online_pi3x import DiskCompactFeatureProvider, DummyDenseProvider  # noqa: E402
from prepare_taco_v15 import (  # noqa: E402
    SMPLX_MANO_TO_WRIST_FIRST,
    create_mano_models,
)
from train import move  # noqa: E402
from visualize_taco_translation_predictions import make_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--triplet", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--taco-code-root", help="Use the same TACO manopth backend as preprocessing")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--visibility-root", required=True)
    parser.add_argument("--track-root", required=True)
    parser.add_argument("--compact-cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object-cache", required=True)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def transform_points(transform, points):
    points = np.asarray(points, dtype=np.float32)
    return points @ np.asarray(transform[:3, :3]).T + np.asarray(transform[:3, 3])


def mano_faces(model, vertex_count):
    faces = getattr(model, "faces", None)
    if faces is None:
        faces = getattr(model, "th_faces", None)
    if faces is None:
        raise ValueError("MANO model has neither faces nor th_faces")
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Invalid MANO faces shape: {faces.shape}")
    if faces.min() < 0 or faces.max() >= vertex_count:
        raise ValueError("MANO faces index outside vertex array")
    return faces.astype(np.uint32)


def mano_local_geometry(backend, model, side, pose, betas):
    pose = np.asarray(pose, dtype=np.float32).reshape(1, 48)
    betas = np.asarray(betas, dtype=np.float32).reshape(1, 10)
    import torch

    pose_tensor = torch.from_numpy(pose)
    betas_tensor = torch.from_numpy(betas)
    with torch.no_grad():
        if backend == "manopth":
            output = model(pose_tensor, betas_tensor)
            vertices = output[0][0].detach().cpu().numpy().astype(np.float32) / 1000.0
            joints = output[1][0].detach().cpu().numpy().astype(np.float32) / 1000.0
        else:
            output = model(
                global_orient=pose_tensor[:, :3],
                hand_pose=pose_tensor[:, 3:],
                betas=betas_tensor,
                return_verts=True,
            )
            vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
            joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
    wrist = joints[0:1].copy()
    if backend != "manopth":
        if len(joints) == 16:
            tips = [745, 317, 444 if side == "right" else 445, 556, 673]
            joints = np.concatenate((joints, vertices[tips]), axis=0)
        joints = joints[:21][SMPLX_MANO_TO_WRIST_FIRST]
    if joints.shape != (21, 3):
        raise ValueError(f"Unexpected MANO joint shape: {joints.shape}")
    return vertices - wrist, joints - wrist


def load_predictions(args, rows, stream_id):
    import torch
    from torch.utils.data import DataLoader

    selected_rows = [row for row in rows if row["stream_id"] == stream_id]
    if not selected_rows:
        raise RuntimeError(f"Sequence is absent from manifest: {stream_id}")
    filtered = Path(args.object_cache).with_name("selected_windows.jsonl")
    filtered.write_text(
        "".join(json_line(row) for row in selected_rows), encoding="utf-8"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("args", {})
    metadata = DexYCBMultiHandWindowDataset(
        filtered,
        None,
        max_hands=args.max_hands,
        training=False,
        noise=QueryNoise(),
        visibility_source="detector",
        visibility_root=args.visibility_root,
        track_root=args.track_root,
        near_anchor_frames=config.get("near_anchor_frames", 4),
        max_anchor_frames=config.get("max_anchor_frames", 8),
        near_missing_weight=config.get("near_missing_weight", 0.5),
        far_missing_weight=config.get("far_missing_weight", 0.2),
        dense_provider=DummyDenseProvider(),
    )
    provider = DiskCompactFeatureProvider(args.compact_cache_root)
    dataset = CompactWindowDataset(metadata, provider)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    sample = dataset[0]
    device = torch.device(args.device)
    model = make_model(checkpoint, sample, device)
    predictions = defaultdict(list)
    with torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            output, _ = model(batch)
            valid = (
                (batch["supervision_weight"] > 0)
                & batch["target_valid"]
                & batch["hand_slot_valid"]
            ).detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            target = batch["target_t"].detach().cpu().numpy()
            frames = batch["frame_index"].detach().cpu().numpy()
            tracks = batch["track_id"].detach().cpu().numpy()
            time_length = output.shape[1]
            weights = np.maximum(
                1.0 - np.abs(np.linspace(-1.0, 1.0, time_length)), 0.1
            )
            for batch_index in range(output.shape[0]):
                for local in range(time_length):
                    for hand in range(output.shape[2]):
                        if not valid[batch_index, local, hand]:
                            continue
                        key = (int(frames[batch_index, local]), int(tracks[batch_index, local, hand]))
                        predictions[key].append(
                            (output[batch_index, local, hand], target[batch_index, local, hand], weights[local])
                        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fused = {}
    for key, values in predictions.items():
        weights = np.asarray([value[2] for value in values], dtype=np.float64)
        fused[key] = {
            "prediction": np.average(
                np.stack([value[0] for value in values]), axis=0, weights=weights
            ).astype(np.float32),
            "target": values[0][1].astype(np.float32),
        }
    return fused


def json_line(row):
    return json.dumps(row, separators=(",", ":")) + "\n"


def update_mesh(server, handles, path, vertices, faces, color, visible, opacity):
    valid = vertices is not None and np.isfinite(vertices).all()
    if not visible or not valid:
        if path in handles:
            handles[path].visible = False
        return
    handles[path] = server.scene.add_mesh_simple(
        path, vertices=np.asarray(vertices, dtype=np.float32),
        faces=faces, color=color, opacity=opacity, side="double",
    )
    handles[path].visible = True


def error_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "median_mm": float(np.median(values)) if values.size else None,
        "max_mm": float(values.max()) if values.size else None,
    }


def main():
    args = parse_args()

    stream_id = f"taco__{args.sequence}"
    root = Path(args.taco_root).expanduser().resolve()
    object_cache_path = Path(args.object_cache).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in Path(args.windows).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predictions = load_predictions(args, rows, stream_id)

    with np.load(object_cache_path, allow_pickle=False) as data:
        object_data = {key: data[key].copy() for key in data.files}
    tool_mesh = trimesh.load(str(object_data["tool_mesh"]), process=False)
    target_mesh = trimesh.load(str(object_data["target_mesh"]), process=False)
    tool_vertices = np.asarray(tool_mesh.vertices, dtype=np.float32) * float(object_data["mesh_scale"])
    target_vertices = np.asarray(target_mesh.vertices, dtype=np.float32) * float(object_data["mesh_scale"])
    faces = {
        "tool": np.asarray(tool_mesh.faces, dtype=np.uint32),
        "target": np.asarray(target_mesh.faces, dtype=np.uint32),
    }

    track_path = Path(args.track_root).expanduser().resolve() / stream_id / "tracks.npz"
    with np.load(track_path, allow_pickle=False) as track:
        track_frames = track["frame_indices"].copy()
        track_ids = track["track_ids"].copy()
        track_sides = track["hand_side"].copy()
        track_xyz = track["joint_xyz"].copy()
    frame_to_track = {}
    reference_joints = {}
    for frame_index, frame in enumerate(track_frames):
        for slot in range(track_ids.shape[1]):
            track_id = int(track_ids[frame_index, slot])
            side = int(track_sides[frame_index, slot])
            if track_id >= 0 and side in (0, 1):
                side_name = "left" if side == 0 else "right"
                frame_to_track[(int(frame), track_id)] = side_name
                side_key = (int(frame), side_name)
                if side_key in reference_joints:
                    raise ValueError(f"Ambiguous TACO hand-side mapping: {side_key}")
                reference_joints[side_key] = track_xyz[frame_index, slot]

    hand_root = root / "Hand_Poses" / args.triplet / args.sequence
    hand_pose = {side: load_pickle(hand_root / f"{side}_hand.pkl") for side in ("left", "right")}
    hand_betas = {
        side: load_pickle(hand_root / f"{side}_hand_shape.pkl")["hand_shape"]
        for side in ("left", "right")
    }
    extrinsics = np.asarray(object_data["extrinsics_world_to_camera"], dtype=np.float32)
    frame_ids = np.asarray(object_data["frame_indices"], dtype=np.int64)
    if not len(frame_ids) or len(set(frame_ids.tolist())) != len(frame_ids):
        raise ValueError("Object cache needs nonempty, unique frame_indices")
    if extrinsics.shape != (len(frame_ids), 4, 4):
        raise ValueError("Object extrinsics and frame_indices disagree")
    for name in ("tool", "target"):
        if object_data[f"{name}_pose_object_to_world"].shape != extrinsics.shape:
            raise ValueError(f"Object pose frame count mismatch: {name}")
    mano_backend, mano_models = create_mano_models(
        args.mano_model_folder, args.taco_code_root
    )
    pose_keys = {side: sorted(hand_pose[side], key=str) for side in hand_pose}
    annotation_keys = {}
    for row in rows:
        if row["stream_id"] != stream_id:
            continue
        for frame, label in zip(row["frame_indices"], row["label_paths"]):
            if int(frame) in annotation_keys:
                continue
            with np.load(label, allow_pickle=False) as data:
                if "source_frame_id" in data and int(data["source_frame_id"]) != int(frame):
                    raise ValueError(f"TACO source/cache frame mismatch: {label}")
                if "taco_annotation_key" in data:
                    annotation_keys[int(frame)] = dict(zip(
                        data["hand_sides"].astype(str), data["taco_annotation_key"].astype(str)
                    ))
    local_camera = {"left": [], "right": []}
    gt_wrist_camera = {"left": [], "right": []}
    audits = {side: {"wrist": [], "joint": []} for side in hand_pose}
    hand_faces = {}
    for offset, frame in enumerate(frame_ids):
        for side in ("left", "right"):
            if frame < 0 or frame >= len(pose_keys[side]):
                raise ValueError(f"Annotation missing frame {frame}: {side}")
            key = annotation_keys.get(int(frame), {}).get(side, pose_keys[side][frame])
            item = hand_pose[side][key]
            local_world, local_joints = mano_local_geometry(
                mano_backend, mano_models[side], side, item["hand_pose"], hand_betas[side]
            )
            ext = extrinsics[offset]
            wrist = transform_points(
                ext, np.asarray(item["hand_trans"], dtype=np.float32).reshape(1, 3)
            )[0]
            local_camera[side].append(local_world @ ext[:3, :3].T)
            gt_wrist_camera[side].append(wrist)
            if side not in hand_faces:
                hand_faces[side] = mano_faces(mano_models[side], len(local_world))
            reference = reference_joints.get((int(frame), side))
            if reference is not None:
                reconstructed = local_joints @ ext[:3, :3].T + wrist
                errors = np.linalg.norm(reconstructed - reference, axis=-1) * 1000
                audits[side]["wrist"].append(errors[0])
                audits[side]["joint"].extend(errors)

    prediction_by_side = {}
    for key, result in predictions.items():
        side = frame_to_track.get(key)
        if side is None:
            raise ValueError(f"Prediction has no known hand side: {key}")
        prediction_by_side[(key[0], side)] = result["prediction"]
    audit_report = {
        "mano_backend": mano_backend,
        "prediction_geometry": "GT MANO pose/shape plus predicted camera wrist translation",
        "coordinate_frame": "camera_x_right_y_down_z_forward_meters",
        "frames": len(frame_ids),
        "prediction_hand_frames": len(prediction_by_side),
        "gt_vs_tracks": {
            side: {name: error_summary(values) for name, values in errors.items()}
            for side, errors in audits.items()
        },
    }
    audit_path = object_cache_path.with_name("viser_mano_audit.json")
    audit_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(json.dumps(audit_report, indent=2), flush=True)
    if any(error_summary(audits[side]["joint"])["median_mm"] is not None
           and error_summary(audits[side]["joint"])["median_mm"] > 5
           for side in audits):
        print(
            "WARNING: MANO reconstruction differs from track annotations by >5 mm median. "
            "Check MANO models and --taco-code-root against preprocessing.", flush=True,
        )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/camera_axes", axes_length=0.1, axes_radius=0.002)
    frame_slider = server.gui.add_slider(
        "Frame Index", min=0, max=max(1, len(frame_ids) - 1), step=1, initial_value=0,
    )
    play = server.gui.add_checkbox("Play", initial_value=False)
    fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=15)
    show_gt = server.gui.add_checkbox("Show GT", initial_value=True)
    show_pred = server.gui.add_checkbox("Show Prediction", initial_value=True)
    show_objects = server.gui.add_checkbox("Show Objects", initial_value=True)
    gt_opacity = server.gui.add_slider(
        "GT Opacity", min=0.05, max=1.0, step=0.05, initial_value=0.3,
    )
    handles = {}
    dirty = threading.Event()
    dirty.set()

    def update(offset):
        frame = int(frame_ids[offset])
        ext = extrinsics[offset]
        for name, vertices, color in (
            ("tool", tool_vertices, (255, 170, 20)),
            ("target", target_vertices, (35, 120, 255)),
        ):
            pose = object_data[f"{name}_pose_object_to_world"][offset]
            camera = transform_points(ext, transform_points(pose, vertices))
            update_mesh(
                server, handles, f"/objects/{name}", camera, faces[name],
                color, show_objects.value, 1.0,
            )
        for side in ("left", "right"):
            local = np.asarray(local_camera[side][offset])
            gt = local + gt_wrist_camera[side][offset]
            wrist = prediction_by_side.get((frame, side))
            pred = None if wrist is None else local + wrist
            update_mesh(
                server, handles, f"/gt/hand/{side}", gt, hand_faces[side],
                (80, 235, 100), show_gt.value, gt_opacity.value,
            )
            update_mesh(
                server, handles, f"/prediction/hand/{side}", pred, hand_faces[side],
                (255, 150, 40), show_pred.value, 1.0,
            )

    # Viser callbacks may run on worker threads; only the main loop updates meshes.
    def request_update(_):
        dirty.set()

    for control in (frame_slider, show_gt, show_pred, show_objects, gt_opacity):
        control.on_update(request_update)

    print(f"Viewer running at http://localhost:{args.port}", flush=True)
    print("Use SSH port forwarding if the server is remote.")
    next_tick = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if play.value and now >= next_tick:
                frame_slider.value = (int(frame_slider.value) + 1) % len(frame_ids)
                next_tick = now + 1.0 / fps.value
                dirty.set()
            if dirty.is_set():
                dirty.clear()
                update(min(int(frame_slider.value), len(frame_ids) - 1))
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
