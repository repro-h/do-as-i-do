#!/usr/bin/env python3
"""Convert one extracted OakInk2 sequence to V15 multi-hand windows."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


for _name, _value in {
    "float_": np.float64,
    "complex_": np.complex128,
    "int_": np.int64,
    "object_": np.dtype("O").type,
    "str_": np.dtype("U").type,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)


# smplx MANO output before conversion to the common wrist-first layout:
# wrist, index/middle/pinky/ring/thumb x3, then five fingertips.
SMPLX_MANO_TO_WRIST_FIRST = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
     10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)

# MANO v1.2 fingertip vertices in thumb/index/middle/ring/pinky order.
MANO_FINGERTIP_VERTICES = np.asarray([744, 320, 443, 555, 672], dtype=np.int64)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--annotation-pkl", required=True)
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--camera",
        choices=("egocentric", "allocentric_top", "allocentric_left", "allocentric_right"),
        default="egocentric",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument("--overlay-count", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tensor(value):
    if torch.is_tensor(value):
        return value.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(value, dtype=torch.float32)


def quaternion_to_axis_angle(quaternion):
    quaternion = tensor(quaternion)
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    cos_theta = quaternion[..., 0]
    vector = quaternion[..., 1:]
    sin_theta = torch.linalg.vector_norm(vector, dim=-1)
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )
    scale = torch.where(
        sin_theta > 1e-8,
        two_theta / sin_theta.clamp_min(1e-8),
        torch.full_like(sin_theta, 2.0),
    )
    return vector * scale[..., None]


def create_mano_models(model_folder):
    import smplx

    model_folder = Path(model_folder).expanduser().resolve()
    for filename in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        if not (model_folder / filename).is_file():
            raise FileNotFoundError(model_folder / filename)
    common = {
        "model_path": str(model_folder),
        "model_type": "mano",
        "use_pca": False,
        "flat_hand_mean": False,
        "num_betas": 10,
        "batch_size": 1,
    }
    return {
        "left": smplx.create(is_rhand=False, **common).eval(),
        "right": smplx.create(is_rhand=True, **common).eval(),
    }


def mano_joints_world(model, pose_quaternion, betas, translation):
    rotation = quaternion_to_axis_angle(pose_quaternion).reshape(1, 16, 3)
    with torch.no_grad():
        output = model(
            global_orient=rotation[:, 0],
            hand_pose=rotation[:, 1:].reshape(1, 45),
            betas=tensor(betas).reshape(1, 10),
            transl=tensor(translation).reshape(1, 3),
            return_verts=True,
        )
    joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
    if joints.shape[0] == 16:
        vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
        fingertips = vertices[MANO_FINGERTIP_VERTICES]
        joints = np.concatenate([joints, fingertips], axis=0)
    if joints.shape[0] < 21:
        raise ValueError(f"MANO returned {joints.shape}; expected at least 21 joints")
    return joints[:21][SMPLX_MANO_TO_WRIST_FIRST]


def transform_points(transform, points):
    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=points.dtype)], axis=-1
    )
    return (np.asarray(transform) @ points_h.T).T[:, :3].astype(np.float32)


def project_points(intrinsics, points):
    projected = (np.asarray(intrinsics) @ points.T).T
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1) & (points[:, 2] > 1e-8)
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    return uv


def window_ranges(length, size, stride):
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return [(start, start + size) for start in starts]


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"median": None, "p10": None, "p90": None}
    return {
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def write_overlay(image_path, label_path, output_path):
    colors = {"left": (0, 220, 255), "right": (255, 80, 120)}
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    with np.load(label_path, allow_pickle=False) as data:
        joints = np.asarray(data["joint_2d"], dtype=np.float32)
        in_frame = np.asarray(data["joint_in_frame"], dtype=bool)
        sides = np.asarray(data["hand_sides"]).astype(str)
    for hand, side in enumerate(sides):
        color = colors[side]
        for start, end in HAND_CONNECTIONS:
            if in_frame[hand, start] and in_frame[hand, end]:
                draw.line(
                    [tuple(joints[hand, start]), tuple(joints[hand, end])],
                    fill=color,
                    width=3,
                )
        for joint, point in enumerate(joints[hand]):
            if not in_frame[hand, joint]:
                continue
            radius = 6 if joint == 0 else 3
            x, y = map(float, point)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main():
    args = parse_args()
    if args.frame_stride <= 0 or args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("frame/window strides and window size must be positive")

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    annotation_path = Path(args.annotation_pkl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = out_dir / f"{args.split}_windows.jsonl"
    summary_path = out_dir / "summary.json"
    if not args.overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError(f"Output exists; pass --overwrite: {out_dir}")
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    with annotation_path.open("rb") as handle:
        annotation = pickle.load(handle)
    serial = next(
        key for key, name in annotation["cam_def"].items() if name == args.camera
    )
    image_dir = sequence_dir / serial
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    models = create_mano_models(args.mano_model_folder)
    if args.start_index < 0:
        raise ValueError("start index must be non-negative")
    frames = annotation["frame_id_list"][args.start_index::args.frame_stride]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    label_dir = out_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    records = []
    side_counts = {"left": 0, "right": 0}
    in_frame_joint_count = 0
    positive_joint_count = 0
    joint_count = 0
    observed_hand_count = 0
    wrist_depths = []
    for output_index, frame_id in enumerate(frames):
        image_path = image_dir / f"{int(frame_id):06d}.png"
        mano = annotation["raw_mano"].get(frame_id)
        intrinsics = annotation["cam_intr"][args.camera].get(frame_id)
        extrinsics = annotation["cam_extr"][args.camera].get(frame_id)
        if not image_path.is_file() or mano is None or intrinsics is None or extrinsics is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        joint_2d = []
        joint_3d = []
        joint_in_frame = []
        observation_valid = []
        sides = []
        for side, prefix in (("left", "lh"), ("right", "rh")):
            joints_world = mano_joints_world(
                models[side],
                mano[f"{prefix}__pose_coeffs"],
                mano[f"{prefix}__betas"],
                mano[f"{prefix}__tsl"],
            )
            joints_camera = transform_points(extrinsics, joints_world)
            uv = project_points(intrinsics, joints_camera)
            positive = np.isfinite(joints_camera).all(axis=-1) & (joints_camera[:, 2] > 0)
            visible = (
                positive
                & np.isfinite(uv).all(axis=-1)
                & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            )
            observed = bool(visible.any())
            joint_2d.append(uv)
            joint_3d.append(joints_camera)
            joint_in_frame.append(visible)
            observation_valid.append(observed)
            sides.append(side)
            side_counts[side] += 1
            observed_hand_count += int(observed)
            joint_count += len(visible)
            positive_joint_count += int(positive.sum())
            in_frame_joint_count += int(visible.sum())
            if np.isfinite(joints_camera[0]).all():
                wrist_depths.append(float(joints_camera[0, 2]))

        label_path = label_dir / f"{output_index:06d}.npz"
        np.savez_compressed(
            label_path,
            joint_2d=np.asarray(joint_2d, dtype=np.float32),
            joint_3d=np.asarray(joint_3d, dtype=np.float32),
            joint_in_frame=np.asarray(joint_in_frame, dtype=bool),
            observation_valid=np.asarray(observation_valid, dtype=bool),
            hand_sides=np.asarray(sides, dtype="U5"),
            source_frame_id=np.int64(frame_id),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            image_wh=np.asarray([width, height], dtype=np.int32),
        )
        records.append({
            "frame_index": output_index,
            "source_frame_id": int(frame_id),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "valid_hands": int(sum(observation_valid)),
        })

    stream_id = f"{sequence_dir.name}__oakink2_{args.camera}"
    rows = []
    for start, end in window_ranges(len(records), args.window_size, args.window_stride):
        window = records[start:end]
        if sum(record["valid_hands"] > 0 for record in window) < args.min_valid_frames:
            continue
        rows.append({
            "schema_version": "oakink2_multihand_window_v1",
            "split": args.split,
            "stream_id": stream_id,
            "sequence": sequence_dir.name,
            "camera_name": args.camera,
            "camera_serial": serial,
            "view_type": "egocentric" if args.camera == "egocentric" else "allocentric",
            "hand_side_metadata_only": "multi",
            "hand_sides_metadata_only": ["left", "right"],
            "start": start,
            "end": end,
            "frame_indices": [record["frame_index"] for record in window],
            "source_frame_ids": [record["source_frame_id"] for record in window],
            "image_paths": [record["image_path"] for record in window],
            "label_paths": [record["label_path"] for record in window],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    observable_records = [record for record in records if record["valid_hands"] > 0]
    if args.overlay_count > 0 and observable_records:
        indices = np.linspace(
            0,
            len(observable_records) - 1,
            min(args.overlay_count, len(observable_records)),
            dtype=np.int64,
        )
        for overlay_index, record_index in enumerate(np.unique(indices)):
            record = observable_records[int(record_index)]
            write_overlay(
                record["image_path"],
                record["label_path"],
                out_dir / "overlay" / (
                    f"{overlay_index:03d}_source_{record['source_frame_id']:06d}.jpg"
                ),
            )

    summary = {
        "schema_version": "oakink2_v15_export_v1",
        "sequence": sequence_dir.name,
        "source": str(sequence_dir),
        "annotation": str(annotation_path),
        "stream_id": stream_id,
        "camera_name": args.camera,
        "camera_serial": serial,
        "view_type": "egocentric" if args.camera == "egocentric" else "allocentric",
        "coordinate_frame": f"oakink2_{args.camera}_camera",
        "extrinsics_convention": "world_to_camera_T_c_w",
        "joint_order": "wrist_thumb_index_middle_ring_pinky_21",
        "horizontal_mirror": False,
        "source_frames": len(frames),
        "exported_frames": len(records),
        "hand_instances": side_counts,
        "observed_hand_instances": observed_hand_count,
        "positive_depth_joint_fraction": (
            float(positive_joint_count / joint_count) if joint_count else 0.0
        ),
        "joint_in_frame_fraction": (
            float(in_frame_joint_count / joint_count) if joint_count else 0.0
        ),
        "wrist_depth_m": distribution(wrist_depths),
        "windows": len(rows),
        "overlay_count": len(list((out_dir / "overlay").glob("*.jpg"))),
        "frame_stride": args.frame_stride,
        "start_index": args.start_index,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
