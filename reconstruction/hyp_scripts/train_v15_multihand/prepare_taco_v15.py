#!/usr/bin/env python3
"""Convert one aligned TACO egocentric sequence to V15 windows."""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
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


SMPLX_MANO_TO_WRIST_FIRST = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
     10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--triplet", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--taco-code-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument("--overlay-count", type=int, default=20)
    parser.add_argument(
        "--overlay-require",
        choices=("any", "both", "left", "right"),
        default="both",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def create_mano_models(model_folder, taco_code_root=None):
    model_folder = Path(model_folder).expanduser().resolve()
    for filename in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        if not (model_folder / filename).is_file():
            raise FileNotFoundError(model_folder / filename)
    if taco_code_root:
        dataset_utils = (
            Path(taco_code_root).expanduser().resolve() / "dataset_utils"
        )
        if not dataset_utils.is_dir():
            raise FileNotFoundError(dataset_utils)
        sys.path.insert(0, str(dataset_utils / "manopth"))
        sys.path.insert(0, str(dataset_utils))
        from manopth.manopth.manolayer import ManoLayer

        return "manopth", {
            side: ManoLayer(
                mano_root=str(model_folder),
                use_pca=False,
                ncomps=45,
                side=side,
                center_idx=0,
                flat_hand_mean=True,
            ).eval()
            for side in ("left", "right")
        }

    import smplx

    common = {
        "model_type": "mano",
        "use_pca": False,
        "flat_hand_mean": True,
        "num_betas": 10,
        "batch_size": 1,
    }
    return "smplx", {
        side: smplx.create(
            model_path=str(model_folder / f"MANO_{side.upper()}.pkl"),
            is_rhand=(side == "right"),
            **common,
        ).eval()
        for side in ("left", "right")
    }


def mano_joints_world(backend, model, side, pose, betas, translation):
    pose = torch.from_numpy(as_numpy(pose).reshape(1, 48))
    betas = torch.from_numpy(as_numpy(betas).reshape(1, 10))
    if backend == "manopth":
        with torch.no_grad():
            result = model(pose, betas)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise TypeError(
                "TACO ManoLayer must return at least (vertices, joints)"
            )
        joints = result[1]
        joints = joints[0].detach().cpu().numpy().astype(np.float32) / 1000.0
        joints += as_numpy(translation).reshape(1, 3)
        return joints

    with torch.no_grad():
        output = model(
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:],
            betas=betas,
            return_verts=True,
        )
    joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
    if joints.shape[0] == 16:
        vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
        tip_ids = {
            "right": np.asarray([745, 317, 444, 556, 673]),
            "left": np.asarray([745, 317, 445, 556, 673]),
        }[side]
        joints = np.concatenate([joints, vertices[tip_ids]], axis=0)
    if joints.shape[0] < 21:
        raise ValueError(f"MANO returned {joints.shape}; expected 21 joints")
    # TACO's official manopth loader uses center_idx=0 and adds hand_trans.
    joints = joints[:21] - joints[0:1]
    joints += as_numpy(translation).reshape(1, 3)
    return joints[SMPLX_MANO_TO_WRIST_FIRST]


def transform_points(transform, points):
    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=-1
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
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window size and stride must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    root = Path(args.taco_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    video_path = (
        root / "Egocentric_RGB_Videos" / args.triplet
        / args.sequence / "color.mp4"
    )
    hand_root = root / "Hand_Poses" / args.triplet / args.sequence
    camera_root = (
        root / "Egocentric_Camera_Parameters" / args.triplet / args.sequence
    )
    generated = (out_dir / "frames", out_dir / "labels", out_dir / "overlay")
    manifest_path = out_dir / f"{args.split}_windows.jsonl"
    summary_path = out_dir / "summary.json"
    if not args.overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError(f"Output exists; pass --overwrite: {out_dir}")
    if args.overwrite:
        for path in generated:
            if path.exists():
                shutil.rmtree(path)
        manifest_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    intrinsics = np.loadtxt(camera_root / "egocentric_intrinsic.txt")
    extrinsics = np.load(camera_root / "egocentric_frame_extrinsic.npy")
    poses = {
        side: load_pickle(hand_root / f"{side}_hand.pkl")
        for side in ("left", "right")
    }
    betas = {
        side: load_pickle(hand_root / f"{side}_hand_shape.pkl")["hand_shape"]
        for side in ("left", "right")
    }
    pose_keys = {side: sorted(poses[side], key=str) for side in poses}
    annotation_count = len(extrinsics)
    counts = {
        "extrinsics": annotation_count,
        "left_pose": len(pose_keys["left"]),
        "right_pose": len(pose_keys["right"]),
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"TACO annotation frame mismatch: {counts}")

    mano_backend, models = create_mano_models(
        args.mano_model_folder, args.taco_code_root
    )
    for path in generated[:2]:
        path.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    records = []
    wrist_depths = []
    in_frame_joints = 0
    positive_joints = 0
    total_joints = 0
    observed_hands = 0
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if decoded >= annotation_count:
            raise RuntimeError("Video has more frames than TACO annotations")
        if args.max_frames > 0 and decoded >= args.max_frames:
            break
        height, width = image.shape[:2]
        image_path = out_dir / "frames" / f"{decoded:06d}.jpg"
        if not cv2.imwrite(
            str(image_path), image,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        ):
            raise RuntimeError(f"Could not write {image_path}")

        frame_uv = []
        frame_xyz = []
        frame_in_frame = []
        frame_observed = []
        for side in ("left", "right"):
            item = poses[side][pose_keys[side][decoded]]
            joints_world = mano_joints_world(
                mano_backend, models[side], side,
                item["hand_pose"], betas[side], item["hand_trans"]
            )
            joints_camera = transform_points(extrinsics[decoded], joints_world)
            uv = project_points(intrinsics, joints_camera)
            positive = (
                np.isfinite(joints_camera).all(axis=-1)
                & (joints_camera[:, 2] > 1e-8)
            )
            visible = (
                positive & np.isfinite(uv).all(axis=-1)
                & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            )
            observed = bool(visible.any())
            frame_uv.append(uv)
            frame_xyz.append(joints_camera)
            frame_in_frame.append(visible)
            frame_observed.append(observed)
            total_joints += len(visible)
            positive_joints += int(positive.sum())
            in_frame_joints += int(visible.sum())
            observed_hands += int(observed)
            if np.isfinite(joints_camera[0]).all():
                wrist_depths.append(float(joints_camera[0, 2]))

        label_path = out_dir / "labels" / f"{decoded:06d}.npz"
        np.savez_compressed(
            label_path,
            seg=np.zeros((height, width), dtype=np.uint8),
            joint_2d=np.asarray(frame_uv, dtype=np.float32),
            joint_3d=np.asarray(frame_xyz, dtype=np.float32),
            joint_in_frame=np.asarray(frame_in_frame, dtype=bool),
            observation_valid=np.asarray(frame_observed, dtype=bool),
            hand_sides=np.asarray(["left", "right"], dtype="U5"),
            source_frame_id=np.int64(decoded),
            taco_annotation_key=np.asarray(
                [pose_keys["left"][decoded], pose_keys["right"][decoded]],
                dtype="U16",
            ),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
            extrinsics=np.asarray(extrinsics[decoded], dtype=np.float32),
            image_wh=np.asarray([width, height], dtype=np.int32),
        )
        records.append({
            "frame_index": decoded,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "valid_hands": int(sum(frame_observed)),
            "left_observed": bool(frame_observed[0]),
            "right_observed": bool(frame_observed[1]),
        })
        decoded += 1
    capture.release()

    if args.max_frames <= 0 and decoded != annotation_count:
        raise RuntimeError(
            f"RGB/annotation mismatch: decoded={decoded}, annotation={annotation_count}"
        )

    stream_id = f"taco__{args.sequence}"
    rows = []
    for start, end in window_ranges(
        len(records), args.window_size, args.window_stride
    ):
        window = records[start:end]
        if sum(item["valid_hands"] > 0 for item in window) < args.min_valid_frames:
            continue
        rows.append({
            "schema_version": "taco_multihand_window_v1",
            "split": args.split,
            "stream_id": stream_id,
            "sequence": args.sequence,
            "triplet": args.triplet,
            "view_type": "egocentric",
            "hand_side_metadata_only": "multi",
            "hand_sides_metadata_only": ["left", "right"],
            "start": start,
            "end": end,
            "frame_indices": [item["frame_index"] for item in window],
            "image_paths": [item["image_path"] for item in window],
            "label_paths": [item["label_path"] for item in window],
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    if args.overlay_require == "both":
        candidates = [
            item for item in records
            if item["left_observed"] and item["right_observed"]
        ]
    elif args.overlay_require == "left":
        candidates = [item for item in records if item["left_observed"]]
    elif args.overlay_require == "right":
        candidates = [item for item in records if item["right_observed"]]
    else:
        candidates = [item for item in records if item["valid_hands"] > 0]
    if args.overlay_count > 0 and candidates:
        indices = np.linspace(
            0, len(candidates) - 1,
            min(args.overlay_count, len(candidates)), dtype=np.int64,
        )
        for output_index, candidate_index in enumerate(np.unique(indices)):
            item = candidates[int(candidate_index)]
            write_overlay(
                item["image_path"], item["label_path"],
                out_dir / "overlay" / f"{output_index:03d}_frame_{item['frame_index']:06d}.jpg",
            )

    summary = {
        "schema_version": "taco_v15_export_v1",
        "triplet": args.triplet,
        "sequence": args.sequence,
        "stream_id": stream_id,
        "video": str(video_path),
        "coordinate_frame": "taco_egocentric_camera",
        "extrinsics_convention": "world_to_camera_T_c_w",
        "joint_order": "wrist_thumb_index_middle_ring_pinky_21",
        "horizontal_mirror": False,
        "mano_backend": mano_backend,
        "annotation_frames": annotation_count,
        "exported_frames": len(records),
        "observed_hand_instances": observed_hands,
        "positive_depth_joint_fraction": (
            float(positive_joints / total_joints) if total_joints else 0.0
        ),
        "joint_in_frame_fraction": (
            float(in_frame_joints / total_joints) if total_joints else 0.0
        ),
        "wrist_depth_m": distribution(wrist_depths),
        "windows": len(rows),
        "overlay_require": args.overlay_require,
        "overlay_candidate_frames": len(candidates),
        "overlay_count": len(list((out_dir / "overlay").glob("*.jpg"))),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
