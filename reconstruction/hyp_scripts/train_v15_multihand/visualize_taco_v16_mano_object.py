#!/usr/bin/env python3
"""Viser viewer for TACO GT/predicted MANO hands and GT object meshes."""

import argparse
import pickle
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import trimesh
import viser

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_dataset import CompactWindowDataset  # noqa: E402
from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from online_pi3x import DiskCompactFeatureProvider, DummyDenseProvider  # noqa: E402
from prepare_taco_v15 import create_mano_models  # noqa: E402
from train import move  # noqa: E402
from train_v16_1_compact_pi3x import load_model_state  # noqa: E402
from visualize_taco_translation_predictions import make_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--triplet", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--visibility-root", required=True)
    parser.add_argument("--track-root", required=True)
    parser.add_argument("--compact-cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object-cache", required=True)
    parser.add_argument("--out-frame-dir", default=None)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=17893)
    return parser.parse_args()


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def transform_points(transform, points):
    points = np.asarray(points, dtype=np.float32)
    return points @ np.asarray(transform[:3, :3]).T + np.asarray(transform[:3, 3])


def mano_local_vertices(backend, model, pose, betas):
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
    return vertices - joints[0:1]


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
    metadata = DexYCBMultiHandWindowDataset(
        filtered,
        None,
        max_hands=args.max_hands,
        training=False,
        noise=QueryNoise(),
        visibility_source="detector",
        visibility_root=args.visibility_root,
        track_root=args.track_root,
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    return fused, metadata


def json_line(row):
    import json
    return json.dumps(row, separators=(",", ":")) + "\n"


def main():
    args = parse_args()
    import json

    stream_id = f"taco__{args.sequence}"
    root = Path(args.taco_root).expanduser().resolve()
    object_cache_path = Path(args.object_cache).expanduser().resolve()
    with object_cache_path.open("rb"):
        pass
    rows = [
        json.loads(line)
        for line in Path(args.windows).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predictions, metadata = load_predictions(args, rows, stream_id)

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
    frame_to_track = {}
    for frame_index, frame in enumerate(track_frames):
        for slot in range(track_ids.shape[1]):
            track_id = int(track_ids[frame_index, slot])
            side = int(track_sides[frame_index, slot])
            if track_id >= 0 and side in (0, 1):
                frame_to_track[(int(frame), track_id)] = ("left" if side == 0 else "right")

    hand_root = root / "Hand_Poses" / args.triplet / args.sequence
    hand_pose = {side: load_pickle(hand_root / f"{side}_hand.pkl") for side in ("left", "right")}
    hand_betas = {
        side: load_pickle(hand_root / f"{side}_hand_shape.pkl")["hand_shape"]
        for side in ("left", "right")
    }
    camera_root = root / "Egocentric_Camera_Parameters" / args.triplet / args.sequence
    intrinsics = np.loadtxt(camera_root / "egocentric_intrinsic.txt").astype(np.float32)
    extrinsics = np.asarray(object_data["extrinsics_world_to_camera"], dtype=np.float32)
    mano_backend, mano_models = create_mano_models(args.mano_model_folder)
    local_camera = {"left": [], "right": []}
    gt_wrist_camera = {"left": [], "right": []}
    for frame in range(len(extrinsics)):
        for side in ("left", "right"):
            keys = sorted(hand_pose[side], key=str)
            item = hand_pose[side][keys[frame]]
            local_world = mano_local_vertices(
                mano_backend, mano_models[side], item["hand_pose"], hand_betas[side]
            )
            local_camera[side].append(local_world @ extrinsics[frame, :3, :3].T)
            gt_wrist_camera[side].append(
                transform_points(extrinsics[frame], np.asarray(item["hand_trans"], dtype=np.float32).reshape(1, 3))[0]
            )

    capture = cv2.VideoCapture(
        str(root / "Egocentric_RGB_Videos" / args.triplet / args.sequence / "color.mp4")
    )
    first_ok, first = capture.read()
    if not first_ok:
        raise RuntimeError("Could not read TACO video")
    height, width = first.shape[:2]
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_frame_dir = Path(args.out_frame_dir).expanduser().resolve() if args.out_frame_dir else None
    if out_frame_dir:
        out_frame_dir.mkdir(parents=True, exist_ok=True)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    frame_slider = server.gui.add_slider("Frame", min=0, max=len(extrinsics) - 1, step=1, initial_value=0)
    play_button = server.gui.add_button("Play")
    show_gt = server.gui.add_checkbox("Show GT", initial_value=True)
    show_pred = server.gui.add_checkbox("Show Prediction", initial_value=True)
    show_objects = server.gui.add_checkbox("Show Objects", initial_value=True)
    playing = False
    lock = threading.Lock()
    fov_y = 2.0 * np.arctan(height / (2.0 * intrinsics[1, 1]))

    def read_frame(frame_index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = capture.read()
        return None if not ok else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def add_hand(path, vertices, color, visible, opacity=1.0):
        handle = server.scene.add_mesh_simple(
            path, vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(mano_models["right"].faces, dtype=np.uint32),
            color=color, opacity=opacity, side="double",
        )
        handle.visible = visible

    def update(frame_index):
        frame_index = int(frame_index)
        image = read_frame(frame_index)
        if image is None:
            return
        server.scene.add_camera_frustum(
            "/camera", fov=float(fov_y), aspect=width / height, scale=0.35,
            image=image, format="jpeg", jpeg_quality=90,
        )
        if out_frame_dir:
            cv2.imwrite(str(out_frame_dir / f"{frame_index:06d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        ext = extrinsics[frame_index]
        if show_objects.value:
            for name, vertices, pose, color in (
                ("tool", tool_vertices, object_data["tool_pose_object_to_world"][frame_index], (255, 170, 20)),
                ("target", target_vertices, object_data["target_pose_object_to_world"][frame_index], (35, 120, 255)),
            ):
                world = vertices @ pose[:3, :3].T + pose[:3, 3]
                camera = world @ ext[:3, :3].T + ext[:3, 3]
                handle = server.scene.add_mesh_simple(
                    f"/gt/object/{name}", vertices=camera.astype(np.float32),
                    faces=faces[name], color=color, opacity=0.65, side="double",
                )
                handle.visible = True

        for side in ("left", "right"):
            gt = np.asarray(local_camera[side][frame_index]) + gt_wrist_camera[side][frame_index]
            add_hand(f"/gt/hand/{side}", gt, (80, 235, 100), show_gt.value, 0.5)
            for (frame, track_id), result in predictions.items():
                if frame != frame_index or frame_to_track.get((frame, track_id)) != side:
                    continue
                pred = np.asarray(local_camera[side][frame_index]) + result["prediction"]
                add_hand(f"/prediction/hand/{side}", pred, (255, 150, 40), show_pred.value, 0.8)

    update(0)

    @frame_slider.on_update
    def on_frame(_):
        update(int(frame_slider.value))

    @show_gt.on_update
    def on_gt(_):
        update(int(frame_slider.value))

    @show_pred.on_update
    def on_pred(_):
        update(int(frame_slider.value))

    @show_objects.on_update
    def on_objects(_):
        update(int(frame_slider.value))

    @play_button.on_click
    def on_play(_):
        nonlocal playing
        with lock:
            playing = not playing
            play_button.name = "Pause" if playing else "Play"

    def loop():
        nonlocal playing
        while True:
            if playing:
                next_frame = int(frame_slider.value) + 1
                frame_slider.value = next_frame if next_frame < len(extrinsics) else 0
                update(int(frame_slider.value))
                time.sleep(1.0 / 30.0)
            else:
                time.sleep(0.05)

    threading.Thread(target=loop, daemon=True).start()
    print(f"Viewer running at http://localhost:{args.port}")
    print("Use SSH port forwarding if the server is remote.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
