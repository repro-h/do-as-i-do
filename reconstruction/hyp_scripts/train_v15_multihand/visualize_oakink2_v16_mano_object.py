#!/usr/bin/env python3
"""OakInk2 camera-space GT/predicted MANO and GT objects, without RGB decoding."""

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np


def frame_records(rows):
    records = {}
    for row in rows:
        frames, labels = row["frame_indices"], row["label_paths"]
        if len(frames) != len(labels):
            raise ValueError("Window frame/label count mismatch")
        sources = row.get("source_frame_ids")
        if sources is not None and len(sources) != len(frames):
            raise ValueError("Window source-frame count mismatch")
        for offset, (frame, label) in enumerate(zip(frames, labels)):
            frame = int(frame)
            if frame in records:
                if records[frame]["label"] != str(label):
                    raise ValueError(f"Conflicting labels for frame {frame}")
                if sources is not None and records[frame]["source"] != int(sources[offset]):
                    raise ValueError(f"Conflicting source frame {frame}")
                continue
            with np.load(label, allow_pickle=False) as data:
                source = int(data["source_frame_id"])
                if sources is not None and source != int(sources[offset]):
                    raise ValueError(f"Label/manifest source-frame mismatch: {label}")
                sides = data["hand_sides"].astype(str).tolist()
                joints = np.asarray(data["joint_3d"], dtype=np.float32)
                if len(set(sides)) != len(sides) or joints.shape != (len(sides), 21, 3):
                    raise ValueError(f"Invalid hand labels: {label}")
                records[frame] = {
                    "label": str(label), "source": source,
                    "joints": dict(zip(sides, joints)),
                    "extrinsics": np.asarray(data["extrinsics"], dtype=np.float32),
                    "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32),
                    "image_wh": np.asarray(data["image_wh"], dtype=np.int64),
                }
    return records


def object_mesh_paths(root, object_ids):
    result = {}
    for object_id in object_ids:
        if Path(object_id).name != object_id or object_id in (".", ".."):
            raise ValueError(f"Invalid object ID: {object_id}")
        path = Path(root) / object_id / "model.obj"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[object_id] = path
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oakink2-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--camera", default="egocentric", choices=("egocentric", "allocentric_top", "allocentric_left", "allocentric_right"))
    parser.add_argument("--object-model-root", required=True, help="object_repair/align_ds directory")
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--compact-cache-root", help="Defaults to checkpoint args.compact_cache_root")
    parser.add_argument("--visibility-root")
    parser.add_argument("--track-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gt-only", action="store_true")
    parser.add_argument("--max-windows", type=int, default=32, help="0 uses all matching windows")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    args.max_hands = 2
    if args.max_windows < 0:
        parser.error("--max-windows must be nonnegative")
    if not args.gt_only and not all((args.checkpoint, args.visibility_root, args.track_root)):
        parser.error("Predictions require --checkpoint, --visibility-root, --track-root")
    return args


def main():
    args = parse_args()
    import torch
    import trimesh
    import viser
    from prepare_oakink2_v15 import MANO_FINGERTIP_VERTICES, SMPLX_MANO_TO_WRIST_FIRST, quaternion_to_axis_angle, tensor
    from prepare_taco_v15 import create_mano_models
    from taco_mano_diagnostics import camera_rigidity
    from visualize_taco_v16_mano_object import error_summary, load_pickle, load_predictions, mano_faces, transform_points, update_mesh

    root = Path(args.oakink2_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stream = f"{args.sequence}__oakink2_{args.camera}"
    with Path(args.windows).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = sorted((row for row in rows if row["stream_id"] == stream), key=lambda row: row["start"])
    if args.max_windows:
        rows = rows[:args.max_windows]
    if not rows:
        raise ValueError(f"No windows for {stream} in {args.windows}")
    records = frame_records(rows)
    frames = sorted(records)
    annotation = load_pickle(root / "anno_preview" / f"{args.sequence}.pkl")
    objects = object_mesh_paths(args.object_model_root, annotation["obj_list"])
    meshes = {}
    object_report = {}
    for object_id, path in objects.items():
        mesh = trimesh.load(str(path), process=False, skip_materials=True, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"Not a finite triangle mesh: {path}")
        meshes[object_id] = mesh
        object_report[object_id] = {"path": str(path), "extent_m": mesh.extents.tolist(), "vertices": len(mesh.vertices)}

    extrinsics = np.stack([records[frame]["extrinsics"] for frame in frames])
    rigid = camera_rigidity(extrinsics)
    if not rigid["proper_rigid_within_1e_4"]:
        raise ValueError(f"Nonrigid camera transforms: {rigid}")
    _, models = create_mano_models(args.mano_model_folder)
    geometry, faces, errors, object_poses = {}, {}, {"left": [], "right": []}, {}
    print(f"Reconstructing {len(frames)} frames and {len(meshes)} objects...", flush=True)
    for offset, frame in enumerate(frames):
        record = records[frame]
        source, ext = record["source"], record["extrinsics"]
        if not np.allclose(ext, annotation["cam_extr"][args.camera][source], atol=1e-5, rtol=0):
            raise ValueError(f"Stale/different camera annotation at frame {frame}, source {source}")
        raw = annotation["raw_mano"][source]
        for side, prefix in (("left", "lh"), ("right", "rh")):
            quaternion = tensor(raw[f"{prefix}__pose_coeffs"])
            if quaternion.shape[-2:] != (16, 4) or not torch.isfinite(quaternion).all() or torch.any(torch.linalg.vector_norm(quaternion, dim=-1) < 1e-8):
                raise ValueError(f"Invalid MANO quaternions: {source}/{side}")
            rotation = quaternion_to_axis_angle(quaternion).reshape(1, 16, 3)
            with torch.no_grad():
                result = models[side](
                    global_orient=rotation[:, 0], hand_pose=rotation[:, 1:].reshape(1, 45),
                    betas=tensor(raw[f"{prefix}__betas"]).reshape(1, 10), return_verts=True,
                )
            vertices = result.vertices[0].cpu().numpy()
            joints = result.joints[0].cpu().numpy()
            wrist = joints[:1].copy()
            if len(joints) == 16:
                joints = np.concatenate((joints, vertices[MANO_FINGERTIP_VERTICES]))
            joints = (joints[:21] - wrist)[SMPLX_MANO_TO_WRIST_FIRST]
            translation = tensor(raw[f"{prefix}__tsl"]).reshape(1, 3).numpy()
            gt_wrist = transform_points(ext, translation)[0]
            local = (vertices - wrist) @ ext[:3, :3].T
            geometry[(frame, side)] = (local, gt_wrist)
            faces[side] = mano_faces(models[side], len(vertices))
            reference = record["joints"].get(side)
            if reference is not None:
                reconstructed = joints @ ext[:3, :3].T + gt_wrist
                errors[side].extend(np.linalg.norm(reconstructed - reference, axis=-1) * 1000)
        for object_id in objects:
            pose = annotation["obj_transf"][object_id].get(source)
            if pose is None:
                object_poses[(frame, object_id)] = None
            else:
                pose = np.asarray(pose, dtype=np.float32)
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError(f"Invalid object transform: {source}/{object_id}")
                object_poses[(frame, object_id)] = ext @ pose
        if offset % 50 == 0:
            print(f"Geometry {offset + 1}/{len(frames)}", flush=True)

    audit = {side: error_summary(values) for side, values in errors.items()}
    if any(value["max_mm"] is None or value["max_mm"] > 1 for value in audit.values()):
        raise ValueError(f"GT MANO does not reproduce the processed labels: {audit}")
    predictions = {}
    if not args.gt_only:
        if not args.compact_cache_root:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            args.compact_cache_root = checkpoint.get("args", {}).get("compact_cache_root")
            del checkpoint
        if not args.compact_cache_root or not Path(args.compact_cache_root).is_dir():
            raise FileNotFoundError(f"Pass the existing --compact-cache-root: {args.compact_cache_root}")
        print(f"Inferring {len(rows)} windows; compact cache={args.compact_cache_root}", flush=True)
        fused = load_predictions(args, rows, stream, filtered_path=out / "selected_windows.jsonl")
        with np.load(Path(args.track_root) / stream / "tracks.npz", allow_pickle=False) as tracks:
            sides_by_track = {}
            for i, frame in enumerate(tracks["frame_indices"]):
                for slot, track in enumerate(tracks["track_ids"][i]):
                    side = int(tracks["hand_side"][i, slot])
                    if int(track) >= 0 and side in (0, 1):
                        sides_by_track[(int(frame), int(track))] = "left" if side == 0 else "right"
        for (frame, track), value in fused.items():
            side = sides_by_track.get((frame, track))
            if side is None or (frame, side) in predictions:
                raise ValueError(f"Missing/ambiguous track side: {frame}/{track}")
            predictions[(frame, side)] = value["prediction"]
        if not predictions:
            raise ValueError("No valid predictions; inspect visibility/track alignment")

    report = {
        "stream_id": stream, "camera": args.camera, "frames": len(frames), "windows": len(rows),
        "first_source_frame": records[frames[0]]["source"], "last_source_frame": records[frames[-1]]["source"],
        "gt_vs_labels_mm": audit, "camera_rigidity": rigid, "objects": object_report,
        "object_scale": 1.0, "missing_object_poses": sum(value is None for value in object_poses.values()),
        "prediction_hand_frames": len(predictions), "compact_cache_root": args.compact_cache_root,
        "prediction_geometry": "GT MANO pose/shape with predicted camera wrist translation",
        "coordinate_frame": "camera_x_right_y_down_z_forward_meters",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/camera_axes", axes_length=0.1, axes_radius=0.002)
    first = records[frames[0]]
    fov = float(2 * np.arctan(first["image_wh"][1] / (2 * first["intrinsics"][1, 1])))

    @server.on_client_connect
    def initialize_camera(client):
        client.camera.position = (0.0, 0.0, 0.0)
        client.camera.look_at = (0.0, 0.0, 1.0)
        client.camera.up_direction = (0.0, -1.0, 0.0)
        client.camera.fov = fov

    slider = server.gui.add_slider("Frame Index", min=0, max=max(1, len(frames) - 1), step=1, initial_value=0)
    play = server.gui.add_checkbox("Play", initial_value=False)
    fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=10)
    show_gt = server.gui.add_checkbox("Show GT", initial_value=True)
    show_pred = server.gui.add_checkbox("Show Prediction", initial_value=not args.gt_only)
    show_objects = server.gui.add_checkbox("Show Objects", initial_value=True)
    opacity = server.gui.add_slider("GT Opacity", min=0.05, max=1.0, step=0.05, initial_value=0.3)
    hands_visible = {side: server.gui.add_checkbox(f"Show {side}", initial_value=True) for side in ("left", "right")}
    handles = {}
    dirty = threading.Event()
    dirty.set()

    def request_update(_):
        dirty.set()

    for control in (slider, show_gt, show_pred, show_objects, opacity, *hands_visible.values()):
        control.on_update(request_update)

    def update(index):
        frame = frames[index]
        for i, (object_id, mesh) in enumerate(meshes.items()):
            pose = object_poses[(frame, object_id)]
            vertices = None if pose is None else transform_points(pose, mesh.vertices)
            color = ((160, 175, 190), (70, 155, 215), (225, 185, 75))[i % 3]
            update_mesh(server, handles, f"/objects/{object_id}", vertices, mesh.faces, color, show_objects.value, 1.0)
        for side in ("left", "right"):
            local, wrist = geometry[(frame, side)]
            pred = predictions.get((frame, side))
            visible = hands_visible[side].value
            update_mesh(server, handles, f"/gt/{side}", local + wrist, faces[side], (80, 235, 100), show_gt.value and visible, opacity.value)
            update_mesh(server, handles, f"/pred/{side}", None if pred is None else local + pred, faces[side], (255, 150, 40), show_pred.value and visible, 1.0)

    print(f"Viewer running at http://localhost:{args.port} (3D only, no RGB)", flush=True)
    tick = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if play.value and now >= tick:
                slider.value = (int(slider.value) + 1) % len(frames)
                tick = now + 1 / fps.value
                dirty.set()
            if dirty.is_set():
                dirty.clear()
                update(min(int(slider.value), len(frames) - 1))
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
