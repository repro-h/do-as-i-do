#!/usr/bin/env python3
"""Render stitched base/tail TACO translation predictions over RGB frames."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_dataset import CompactWindowDataset  # noqa: E402
from compact_model import CompactMultiHandPi3XTrajectoryModel  # noqa: E402
from dataset import DexYCBMultiHandWindowDataset, QueryNoise  # noqa: E402
from online_pi3x import DiskCompactFeatureProvider, DummyDenseProvider  # noqa: E402
from prepare_taco_v15 import HAND_CONNECTIONS  # noqa: E402
from train import move  # noqa: E402
from train_v16_1_compact_pi3x import load_model_state  # noqa: E402


COLORS = {
    "gt": (40, 230, 90),
    "base": (255, 165, 40),
    "tail": (40, 180, 255),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--visibility-root", required=True)
    parser.add_argument("--track-root", required=True)
    parser.add_argument("--compact-cache-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--tail-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--object-cache", default=None)
    parser.add_argument("--object-point-stride", type=int, default=20)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def value(config, name, default):
    return config[name] if name in config else default


def make_model(checkpoint, sample, device):
    config = dict(checkpoint.get("args", {}))
    model = CompactMultiHandPi3XTrajectoryModel(
        point_dim=sample["joint_patch_features"].shape[-1],
        metric_dim=sample["metric_window_features"].shape[-1],
        token_dim=value(config, "token_dim", 128),
        hidden_dim=value(config, "hidden_dim", 192),
        heads=value(config, "heads", 4),
        temporal_layers=value(config, "temporal_layers", 2),
        dropout=value(config, "dropout", 0.1),
        max_window_size=value(config, "max_window_size", 128),
        translation_parameterization=value(
            config, "translation_parameterization", "ray_depth_uv"
        ),
        max_image_offset_fraction=value(
            config, "max_image_offset_fraction", 0.15
        ),
    ).to(device)
    load_model_state(model, checkpoint["model"])
    model.eval()
    return model


def infer(checkpoint_path, sample, loader, device):
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    model = make_model(checkpoint, sample, device)
    predictions = defaultdict(list)
    with torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            prediction, _ = model(batch)
            supervised = (
                (batch["supervision_weight"] > 0)
                & batch["target_valid"]
                & batch["hand_slot_valid"]
            )
            prediction = prediction.detach().cpu().numpy()
            target = batch["target_t"].detach().cpu().numpy()
            valid = supervised.detach().cpu().numpy().astype(bool)
            streams = batch["stream_index"].detach().cpu().numpy()
            frames = batch["frame_index"].detach().cpu().numpy()
            tracks = batch["track_id"].detach().cpu().numpy()
            time = prediction.shape[1]
            weights = 1.0 - np.abs(
                np.linspace(-1.0, 1.0, time, dtype=np.float32)
            )
            weights = np.maximum(weights, 0.1)
            for batch_index in range(prediction.shape[0]):
                for local in range(time):
                    for hand in range(prediction.shape[2]):
                        if not valid[batch_index, local, hand]:
                            continue
                        key = (
                            int(streams[batch_index]),
                            int(frames[batch_index, local]),
                            int(tracks[batch_index, local, hand]),
                        )
                        predictions[key].append((
                            prediction[batch_index, local, hand],
                            target[batch_index, local, hand],
                            float(weights[local]),
                        ))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stitched = {}
    for key, samples in predictions.items():
        weights = np.asarray([item[2] for item in samples], dtype=np.float64)
        values = np.stack([item[0] for item in samples]).astype(np.float64)
        prediction = (values * weights[:, None]).sum(axis=0) / weights.sum()
        stitched[key] = {
            "prediction": prediction.astype(np.float32),
            "target": np.asarray(samples[0][1], dtype=np.float32),
            "windows": len(samples),
        }
    return stitched, int(checkpoint.get("epoch", -1))


def track_joints(track_root, stream, frame, track_id, cache):
    if stream not in cache:
        path = track_root / stream / "tracks.npz"
        with np.load(path, allow_pickle=False) as data:
            cache[stream] = {
                key: np.asarray(data[key]).copy()
                for key in ("frame_indices", "track_ids", "joint_xyz")
            }
        cache[stream]["frame_lookup"] = {
            int(value): index
            for index, value in enumerate(cache[stream]["frame_indices"])
        }
    data = cache[stream]
    frame_offset = data["frame_lookup"].get(int(frame))
    if frame_offset is None:
        return None
    slots = np.flatnonzero(data["track_ids"][frame_offset] == int(track_id))
    if not len(slots):
        return None
    return np.asarray(data["joint_xyz"][frame_offset, slots[0]], dtype=np.float32)


def project(points, intrinsics):
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=-1) & (points[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected = (intrinsics @ points[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def draw_skeleton(draw, pixels, valid, color, width):
    for start, end in HAND_CONNECTIONS:
        if valid[start] and valid[end]:
            draw.line(
                [tuple(pixels[start]), tuple(pixels[end])],
                fill=color, width=width,
            )
    for joint, point in enumerate(pixels):
        if not valid[joint]:
            continue
        radius = 5 if joint == 0 else 2
        x, y = map(float, point)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def load_object_cache(path):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        required = {
            "frame_indices",
            "tool_pose_object_to_world",
            "target_pose_object_to_world",
            "extrinsics_world_to_camera",
            "intrinsics",
            "tool_mesh",
            "target_mesh",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"Object cache missing keys: {missing}")
        cache = {key: data[key].copy() for key in required}
    cache["frame_lookup"] = {
        int(frame): index
        for index, frame in enumerate(cache["frame_indices"])
    }
    cache["tool_vertices"], _ = object_mesh(cache["tool_mesh"])
    cache["target_vertices"], _ = object_mesh(cache["target_mesh"])
    return cache


def object_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces)


def project_object(vertices, object_pose, extrinsic, intrinsics):
    world = vertices @ object_pose[:3, :3].T + object_pose[:3, 3]
    camera = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    return project(camera, intrinsics)


def draw_object(draw, vertices, object_pose, extrinsic, intrinsics, color, stride):
    pixels, valid = project_object(
        vertices[::max(1, stride)], object_pose, extrinsic, intrinsics
    )
    for point in pixels[valid]:
        x, y = map(float, point)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)


def contact_sheet(paths, output, columns=4):
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "black")
    for index, image in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main():
    args = parse_args()
    windows_path = Path(args.windows).expanduser().resolve()
    selected_streams = {f"taco__{value}" for value in args.sequence}
    rows = [
        json.loads(line)
        for line in windows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["stream_id"] in selected_streams]
    if not rows:
        raise RuntimeError("None of the selected sequences occur in --windows")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    filtered_manifest = out_root / "selected_windows.jsonl"
    filtered_manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    metadata = DexYCBMultiHandWindowDataset(
        filtered_manifest,
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
    sample = dataset[0]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device(args.device)
    object_cache = load_object_cache(args.object_cache)
    base, base_epoch = infer(Path(args.base_checkpoint), sample, loader, device)
    tail, tail_epoch = infer(Path(args.tail_checkpoint), sample, loader, device)

    stream_names = {index: name for name, index in metadata.stream_indices.items()}
    frame_records = {}
    for row in rows:
        for frame, image, label in zip(
            row["frame_indices"], row["image_paths"], row["label_paths"]
        ):
            frame_records[(row["stream_id"], int(frame))] = (Path(image), Path(label))

    common = sorted(set(base) & set(tail))
    by_sequence = defaultdict(list)
    for key in common:
        stream = stream_names[key[0]].split("::", 1)[-1]
        base_error = float(np.linalg.norm(base[key]["prediction"] - base[key]["target"]))
        tail_error = float(np.linalg.norm(tail[key]["prediction"] - tail[key]["target"]))
        by_sequence[stream].append((key, base_error, tail_error))

    track_cache = {}
    report = {
        "base_epoch": base_epoch,
        "tail_epoch": tail_epoch,
        "sequences": {},
    }
    track_root = Path(args.track_root).expanduser().resolve()
    for stream in sorted(selected_streams):
        candidates = by_sequence.get(stream, [])
        if not candidates:
            continue
        frames = sorted({item[0][1] for item in candidates})
        frame_scores = []
        for frame in frames:
            samples = [item for item in candidates if item[0][1] == frame]
            base_error = max(item[1] for item in samples)
            tail_error = max(item[2] for item in samples)
            frame_scores.append((frame, base_error, tail_error, tail_error - base_error))

        count = min(args.frames_per_sequence, len(frame_scores))
        bucket = max(count // 3, 1)
        selected = {
            item[0] for item in sorted(frame_scores, key=lambda x: x[3], reverse=True)[:bucket]
        }
        selected.update(
            item[0] for item in sorted(frame_scores, key=lambda x: x[2], reverse=True)[:bucket]
        )
        remaining = max(count - len(selected), 0)
        if remaining:
            indices = np.linspace(0, len(frame_scores) - 1, remaining, dtype=np.int64)
            selected.update(frame_scores[int(index)][0] for index in indices)
        if len(selected) < count:
            for item in frame_scores:
                selected.add(item[0])
                if len(selected) == count:
                    break

        sequence_dir = out_root / stream
        overlay_dir = sequence_dir / "overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        rendered = []
        frame_report = []
        for output_index, frame in enumerate(sorted(selected)):
            image_path, label_path = frame_records[(stream, frame)]
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            with np.load(label_path, allow_pickle=False) as label:
                intrinsics = np.asarray(label["intrinsics"], dtype=np.float64)
            text = [
                f"frame={frame} GT=green base{base_epoch}=orange tail{tail_epoch}=blue"
            ]
            if object_cache is not None:
                object_index = object_cache["frame_lookup"].get(int(frame))
                if object_index is not None:
                    draw_object(
                        draw,
                        object_cache["tool_vertices"],
                        object_cache["tool_pose_object_to_world"][object_index],
                        object_cache["extrinsics_world_to_camera"][object_index],
                        intrinsics,
                        (255, 175, 20),
                        args.object_point_stride,
                    )
                    draw_object(
                        draw,
                        object_cache["target_vertices"],
                        object_cache["target_pose_object_to_world"][object_index],
                        object_cache["extrinsics_world_to_camera"][object_index],
                        intrinsics,
                        (35, 120, 255),
                        args.object_point_stride,
                    )
                    text.append("objects: tool=orange target=blue")
            samples = [item for item in candidates if item[0][1] == frame]
            for key, base_error, tail_error in samples:
                joints = track_joints(
                    track_root, stream, frame, key[2], track_cache
                )
                if joints is None:
                    continue
                target = base[key]["target"]
                local = joints - target[None]
                variants = {
                    "gt": joints,
                    "base": local + base[key]["prediction"][None],
                    "tail": local + tail[key]["prediction"][None],
                }
                for name, points in variants.items():
                    pixels, valid = project(points, intrinsics)
                    valid &= (
                        (pixels[:, 0] >= 0) & (pixels[:, 0] < image.width)
                        & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.height)
                    )
                    draw_skeleton(
                        draw, pixels, valid, COLORS[name], 4 if name == "gt" else 2
                    )
                base_depth = float(abs(base[key]["prediction"][2] - target[2]) * 1000)
                tail_depth = float(abs(tail[key]["prediction"][2] - target[2]) * 1000)
                text.append(
                    f"track={key[2]} base={base_error * 1000:.1f}mm "
                    f"tail={tail_error * 1000:.1f}mm "
                    f"z={base_depth:.1f}->{tail_depth:.1f}mm"
                )
                frame_report.append({
                    "frame": frame,
                    "track_id": key[2],
                    "base_error_mm": base_error * 1000,
                    "tail_error_mm": tail_error * 1000,
                    "delta_mm": (tail_error - base_error) * 1000,
                    "base_depth_error_mm": base_depth,
                    "tail_depth_error_mm": tail_depth,
                })
            header_height = 18 * len(text) + 8
            header = Image.new("RGB", (image.width, image.height + header_height), "black")
            header.paste(image, (0, header_height))
            header_draw = ImageDraw.Draw(header)
            for line_index, line in enumerate(text):
                header_draw.text((6, 4 + line_index * 18), line, fill="white")
            output_path = overlay_dir / f"{output_index:03d}_frame_{frame:06d}.jpg"
            header.save(output_path, quality=94)
            rendered.append(output_path)

        contact_sheet(rendered, sequence_dir / "contact_sheet.jpg")
        report["sequences"][stream] = {
            "candidate_frames": len(frame_scores),
            "rendered_frames": len(rendered),
            "frames": frame_report,
            "contact_sheet": str(sequence_dir / "contact_sheet.jpg"),
        }
        print(stream, sequence_dir / "contact_sheet.jpg", flush=True)

    (out_root / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
