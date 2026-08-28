#!/usr/bin/env python3
"""Convert one HOT3D Aria sequence to the V15 multi-hand window schema.

The official HOT3D and Project Aria providers perform VRS decoding, timestamp
lookup, camera calibration, and MANO forward kinematics. This adapter exports a
fixed-pinhole RGB sequence plus camera-frame 21-joint supervision. It does not
mirror left hands.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# smplx MANO joints before HOT3D's display-oriented joint mapper:
# wrist, index/middle/pinky/ring/thumb x3, then thumb/index/... fingertips.
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
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument(
        "--hot3d-code-root",
        required=True,
        help="HOT3D checkout or its inner hot3d/ directory containing dataset_api.py",
    )
    parser.add_argument(
        "--mano-model-folder",
        required=True,
        help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--stream-id", default="214-1", help="Aria RGB stream")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-time-delta-ms", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overlay-count", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hot3d_python_root(path):
    path = Path(path).expanduser().resolve()
    candidates = [path, path / "hot3d"]
    candidates.extend(parent for parent in path.glob("*/hot3d"))
    for candidate in candidates:
        if (candidate / "dataset_api.py").is_file():
            return candidate
    matches = list(path.rglob("dataset_api.py"))
    if len(matches) == 1:
        return matches[0].parent
    raise FileNotFoundError(f"Cannot locate HOT3D dataset_api.py under {path}")


def window_ranges(length, size, stride):
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if not starts or starts[-1] != final:
        starts.append(final)
    return [(start, start + size) for start in starts]


def write_rgb(path, image, quality):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, image, quality=int(quality))
    except ImportError:
        from PIL import Image

        Image.fromarray(image).save(path, quality=int(quality))


def intrinsics_matrix(calibration):
    fx, fy = [float(value) for value in calibration.get_focal_lengths()]
    cx, cy = [float(value) for value in calibration.get_principal_point()]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def transform_points(transform, points):
    matrix = np.asarray(transform.to_matrix(), dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    return (matrix @ homogeneous.T).T[:, :3].astype(np.float32)


def project_points(calibration, points):
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    for index, point in enumerate(points):
        if not np.isfinite(point).all() or point[2] <= 1e-6:
            continue
        projected = calibration.project(point.astype(np.float64))
        if projected is not None:
            uv[index] = np.asarray(projected, dtype=np.float32)
    return uv


def write_overlay(image_path, label_path, output_path):
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    with np.load(label_path, allow_pickle=False) as data:
        joints = np.asarray(data["joint_2d"], dtype=np.float32)
        sides = np.asarray(data["hand_sides"]).reshape(-1)
    colors = {"left": (0, 220, 255), "right": (255, 70, 180)}
    for hand, uv in enumerate(joints):
        side = str(sides[hand]) if hand < len(sides) else "unknown"
        color = colors.get(side, (255, 220, 0))
        valid = (
            np.isfinite(uv).all(axis=-1)
            & (uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        )
        for first, second in HAND_CONNECTIONS:
            if valid[first] and valid[second]:
                draw.line(
                    [tuple(uv[first]), tuple(uv[second])],
                    fill=color,
                    width=3,
                )
        for joint, point in enumerate(uv):
            if not valid[joint]:
                continue
            radius = 5 if joint == 0 else 3
            x, y = float(point[0]), float(point[1])
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(0, 0, 0),
            )
        visible = uv[valid]
        if len(visible):
            x, y = visible.min(axis=0)
            draw.text((float(x), max(0.0, float(y) - 14)), side, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def metadata_has_gt(metadata):
    value = metadata.get("gt_available_status", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {
        "false", "0", "none", "not_available", "unavailable",
    }


def main():
    args = parse_args()
    if args.frame_stride <= 0 or args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("frame/window strides and window size must be positive")

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = out_dir / f"{args.split}_windows.jsonl"
    summary_path = out_dir / "summary.json"
    if not args.overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError(f"Output exists; pass --overwrite: {out_dir}")
    if not (sequence_dir / "recording.vrs").is_file():
        raise FileNotFoundError(sequence_dir / "recording.vrs")

    code_root = hot3d_python_root(args.hot3d_code_root)
    sys.path.insert(0, str(code_root))

    # HOT3D imports its Quest provider eagerly, even for Aria sequences.  The
    # Quest-only pyvrs2 package is not published for every Python environment,
    # so provide an import placeholder that fails only if Quest is instantiated.
    try:
        import pyvrs2  # noqa: F401
    except ImportError:
        import types

        class _UnavailableQuestReader:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "pyvrs2 is required for Quest sequences; this exporter "
                    "currently supports HOT3D Aria recording.vrs sequences"
                )

        pyvrs2_stub = types.ModuleType("pyvrs2")
        pyvrs2_stub.SyncVRSReader = _UnavailableQuestReader
        sys.modules["pyvrs2"] = pyvrs2_stub

    from dataset_api import Hot3dDataProvider
    from data_loaders.mano_layer import MANOHandModel
    from projectaria_tools.core.calibration import (
        FISHEYE624,
        LINEAR,
        distort_by_calibration,
    )
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    mano_folder = Path(args.mano_model_folder).expanduser().resolve()
    for filename in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        if not (mano_folder / filename).is_file():
            raise FileNotFoundError(mano_folder / filename)

    # Disable HOT3D's visualization joint mapper and export a common 21-joint
    # wrist-first layout shared with DexYCB and the V15 query interface.
    mano_model = MANOHandModel(str(mano_folder), joint_mapper=None)
    provider = Hot3dDataProvider(
        sequence_folder=str(sequence_dir),
        object_library=None,
        mano_hand_model=mano_model,
    )
    metadata = provider.get_sequence_metadata()
    if not metadata_has_gt(metadata):
        raise RuntimeError(f"Sequence does not expose training GT: {sequence_dir.name}")
    hand_provider = provider.mano_hand_data_provider
    if hand_provider is None:
        raise RuntimeError("MANO hand provider is unavailable")

    device = provider.device_data_provider
    pose_provider = provider.device_pose_data_provider
    stream_id = StreamId(args.stream_id)
    labels = {str(item): device.get_image_stream_label(item) for item in device.get_image_stream_ids()}
    if str(stream_id) not in labels:
        raise KeyError(f"Stream {stream_id} not found; available={labels}")
    if not labels[str(stream_id)].startswith("camera-rgb"):
        raise ValueError(f"Stream {stream_id} is not RGB: {labels[str(stream_id)]}")

    timestamps = list(device.get_sequence_timestamps(stream_id))[::args.frame_stride]
    if args.max_frames > 0:
        timestamps = timestamps[:args.max_frames]
    if not timestamps:
        raise RuntimeError("No RGB timestamps")

    T_device_camera, native_calibration = device.get_camera_calibration(
        stream_id, camera_model=FISHEYE624
    )
    _, linear_calibration = device.get_camera_calibration(
        stream_id, camera_model=LINEAR
    )
    K = intrinsics_matrix(linear_calibration)
    max_delta_ns = int(round(args.max_time_delta_ms * 1e6))
    image_dir = out_dir / "rgb"
    label_dir = out_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    records = []
    side_counts = {"left": 0, "right": 0}
    skipped_images = 0
    hand_delta_ms = []
    device_delta_ms = []
    joint_count = 0
    joint_in_frame_count = 0
    wrist_depths_m = []
    for frame_index, timestamp_ns in enumerate(timestamps):
        raw_image = device.get_image(timestamp_ns, stream_id)
        if raw_image is None:
            skipped_images += 1
            continue
        image = distort_by_calibration(
            np.asarray(raw_image), linear_calibration, native_calibration
        )
        height, width = image.shape[:2]

        device_pose = pose_provider.get_pose_at_timestamp(
            timestamp_ns=int(timestamp_ns),
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=max_delta_ns,
        )
        hand_poses = hand_provider.get_pose_at_timestamp(
            timestamp_ns=int(timestamp_ns),
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=max_delta_ns,
        )

        joint_2d = []
        joint_3d = []
        hand_sides = []
        hand_pose_dt_ns = None
        device_pose_dt_ns = None
        if device_pose is not None:
            device_pose_dt_ns = int(device_pose.time_delta_ns)
            device_delta_ms.append(abs(device_pose_dt_ns) / 1e6)
        if hand_poses is not None:
            hand_pose_dt_ns = int(hand_poses.time_delta_ns)
            hand_delta_ms.append(abs(hand_pose_dt_ns) / 1e6)
        if device_pose is not None and hand_poses is not None:
            T_world_camera = device_pose.pose3d.T_world_device @ T_device_camera
            T_camera_world = T_world_camera.inverse()
            poses = sorted(
                hand_poses.pose3d_collection.poses.values(),
                key=lambda pose: 0 if pose.is_left_hand() else 1,
            )
            for pose in poses:
                landmarks_world = hand_provider.get_hand_landmarks(pose)
                if landmarks_world is None:
                    continue
                landmarks_world = np.asarray(
                    landmarks_world.detach().cpu(), dtype=np.float32
                ).reshape(-1, 3)
                if landmarks_world.shape != (21, 3):
                    raise ValueError(
                        f"Expected raw MANO 21 joints, got {landmarks_world.shape}"
                    )
                landmarks_world = landmarks_world[SMPLX_MANO_TO_WRIST_FIRST]
                landmarks_camera = transform_points(T_camera_world, landmarks_world)
                if np.isfinite(landmarks_camera[0]).all():
                    wrist_depths_m.append(float(landmarks_camera[0, 2]))
                uv = project_points(linear_calibration, landmarks_camera)
                finite = np.isfinite(uv).all(axis=-1)
                joint_count += int(finite.sum())
                joint_in_frame_count += int((
                    finite
                    & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < height)
                ).sum())
                side = pose.handedness_label()
                joint_2d.append(uv)
                joint_3d.append(landmarks_camera)
                hand_sides.append(side)
                side_counts[side] += 1

        joint_2d = np.asarray(joint_2d, dtype=np.float32).reshape(-1, 21, 2)
        joint_3d = np.asarray(joint_3d, dtype=np.float32).reshape(-1, 21, 3)
        image_path = image_dir / f"{frame_index:06d}.jpg"
        label_path = label_dir / f"{frame_index:06d}.npz"
        write_rgb(image_path, image, args.jpeg_quality)
        np.savez_compressed(
            label_path,
            seg=np.zeros((height, width), dtype=np.uint8),
            joint_2d=joint_2d,
            joint_3d=joint_3d,
            hand_sides=np.asarray(hand_sides, dtype="U5"),
            timestamp_ns=np.int64(timestamp_ns),
            hand_pose_dt_ns=np.int64(
                -1 if hand_pose_dt_ns is None else hand_pose_dt_ns
            ),
            device_pose_dt_ns=np.int64(
                -1 if device_pose_dt_ns is None else device_pose_dt_ns
            ),
            intrinsics=K,
            image_wh=np.asarray([width, height], dtype=np.int32),
        )
        records.append({
            "frame_index": frame_index,
            "timestamp_ns": int(timestamp_ns),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "valid_hands": int(len(joint_3d)),
        })

    stream_name = f"{sequence_dir.name}__aria_rgb_{str(stream_id).replace('-', '_')}"
    rows = []
    for start, end in window_ranges(
        len(records), args.window_size, args.window_stride
    ):
        window = records[start:end]
        if sum(record["valid_hands"] > 0 for record in window) < args.min_valid_frames:
            continue
        rows.append({
            "schema_version": "hot3d_aria_multihand_window_v1",
            "split": args.split,
            "stream_id": stream_name,
            "sequence": sequence_dir.name,
            "camera_stream_id": str(stream_id),
            "hand_side_metadata_only": "multi",
            "hand_sides_metadata_only": ["left", "right"],
            "start": start,
            "end": end,
            "frame_indices": [record["frame_index"] for record in window],
            "timestamps_ns": [record["timestamp_ns"] for record in window],
            "image_paths": [record["image_path"] for record in window],
            "label_paths": [record["label_path"] for record in window],
            "intrinsics": K.tolist(),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    overlay_records = [record for record in records if record["valid_hands"] > 0]
    if args.overlay_count > 0 and overlay_records:
        indices = np.linspace(
            0, len(overlay_records) - 1,
            min(args.overlay_count, len(overlay_records)),
            dtype=np.int64,
        )
        for output_index, record_index in enumerate(np.unique(indices)):
            record = overlay_records[int(record_index)]
            write_overlay(
                record["image_path"],
                record["label_path"],
                out_dir / "overlay" / f"{output_index:03d}_frame_{record['frame_index']:06d}.jpg",
            )
    hand_delta = np.asarray(hand_delta_ms, dtype=np.float64)
    device_delta = np.asarray(device_delta_ms, dtype=np.float64)
    wrist_depth = np.asarray(wrist_depths_m, dtype=np.float64)
    summary = {
        "schema_version": "hot3d_aria_v15_export_v1",
        "sequence": sequence_dir.name,
        "source": str(sequence_dir),
        "stream_id": stream_name,
        "camera_stream_id": str(stream_id),
        "coordinate_frame": "undistorted_rgb_camera",
        "joint_order": "wrist_thumb_index_middle_ring_pinky_21",
        "horizontal_mirror": False,
        "source_timestamps": len(timestamps),
        "exported_frames": len(records),
        "skipped_images": skipped_images,
        "frames_with_hands": sum(record["valid_hands"] > 0 for record in records),
        "hand_instances": side_counts,
        "joint_in_frame_fraction": (
            float(joint_in_frame_count / joint_count) if joint_count else 0.0
        ),
        "wrist_depth_m": {
            "median": float(np.median(wrist_depth)) if len(wrist_depth) else None,
            "p10": float(np.percentile(wrist_depth, 10)) if len(wrist_depth) else None,
            "p90": float(np.percentile(wrist_depth, 90)) if len(wrist_depth) else None,
        },
        "hand_pose_delta_ms": {
            "median": float(np.median(hand_delta)) if len(hand_delta) else None,
            "p90": float(np.percentile(hand_delta, 90)) if len(hand_delta) else None,
            "max": float(hand_delta.max()) if len(hand_delta) else None,
        },
        "device_pose_delta_ms": {
            "median": float(np.median(device_delta)) if len(device_delta) else None,
            "p90": float(np.percentile(device_delta, 90)) if len(device_delta) else None,
            "max": float(device_delta.max()) if len(device_delta) else None,
        },
        "overlay_count": len(list((out_dir / "overlay").glob("*.jpg"))),
        "frame_stride": args.frame_stride,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "windows": len(rows),
        "intrinsics": K.tolist(),
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
