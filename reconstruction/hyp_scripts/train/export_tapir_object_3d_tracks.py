#!/usr/bin/env python3
"""Export adjacent-frame TAPIR object tracks and metric 3D rigid motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tapnet.torch import tapir_model
from tapnet.utils import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument(
        "--mask-dir",
        help="Optional Do-As-I-Do frame_N_masks directory; defaults to DexYCB labels.",
    )
    parser.add_argument("--object", required=True)
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--intrinsics-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--preview-dir")
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--resize", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--mask-erosion-px", type=int, default=5)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--min-3d-points", type=int, default=8)
    parser.add_argument("--ransac-trials", type=int, default=128)
    parser.add_argument("--ransac-threshold-mm", type=float, default=12.0)
    parser.add_argument("--pnp-threshold-px", type=float, default=4.0)
    parser.add_argument("--min-pnp-points", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def numeric_frames(path: Path) -> list[Path]:
    result = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        result.extend(path.glob(extension))
    return sorted(result, key=lambda item: int(item.stem.split("_")[-1]))


def load_intrinsics(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if all(key in payload for key in ("fx", "fy", "cx", "cy")):
        return np.asarray(
            [
                [payload["fx"], 0.0, payload["cx"]],
                [0.0, payload["fy"], payload["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    for key in ("K", "intrinsics", "camera_intrinsics"):
        if key in payload:
            value = np.asarray(payload[key], dtype=np.float64)
            if value.size == 9:
                return value.reshape(3, 3)
    raise KeyError(f"No camera intrinsics in {path}")


def resolve_depth(stream_dir: Path, frame_id: str) -> Path:
    candidates = (
        stream_dir / f"aligned_depth_to_color_{frame_id}.png",
        stream_dir / f"depth_{frame_id}.png",
        stream_dir / f"aligned_depth_{frame_id}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No aligned depth for frame {frame_id} in {stream_dir}"
    )


def target_object_id(stream_dir: Path) -> int:
    meta_path = stream_dir.parent / "meta.yml"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    ycb_ids = list(metadata.get("ycb_ids", []) or [])
    grasp_index = int(metadata.get("ycb_grasp_ind", 0))
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(f"Invalid ycb_grasp_ind={grasp_index} in {meta_path}")
    return int(ycb_ids[grasp_index])


def load_mask(
    mask_dir: Path | None,
    frame_index: int,
    object_name: str,
    frame_row: dict,
    object_id: int | None,
) -> np.ndarray:
    if mask_dir is not None:
        path = (
            mask_dir
            / f"frame_{frame_index:06d}_masks"
            / f"{object_name}.png"
        )
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        return mask > 127
    if object_id is None:
        raise ValueError("object_id is required when reading DexYCB labels")
    label_path = Path(frame_row["label_path"])
    with np.load(label_path) as payload:
        segmentation = np.squeeze(np.asarray(payload["seg"]))
    return segmentation == object_id


def sample_mask_points(
    mask: np.ndarray,
    count: int,
    erosion_px: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if erosion_px > 0:
        size = 2 * erosion_px + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        if eroded.sum() >= max(8, count // 4):
            mask = eroded
    ys, xs = np.where(mask)
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)

    cell_size = max(4, int(np.sqrt(len(xs) / max(count, 1))))
    cells: dict[tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(zip(xs, ys)):
        cells.setdefault((int(y // cell_size), int(x // cell_size)), []).append(index)
    selected = [
        indices[int(rng.integers(0, len(indices)))]
        for indices in cells.values()
    ]
    if len(selected) > count:
        selected = rng.choice(selected, size=count, replace=False).tolist()
    elif len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(xs)), np.asarray(selected))
        extra_count = min(count - len(selected), len(remaining))
        if extra_count:
            selected.extend(
                rng.choice(remaining, size=extra_count, replace=False).tolist()
            )
    return np.stack([xs[selected], ys[selected]], axis=1).astype(np.float32)


def local_depth(depth: np.ndarray, points: np.ndarray, radius: int = 2) -> np.ndarray:
    height, width = depth.shape
    values = np.full(len(points), np.nan, dtype=np.float32)
    for index, (x_value, y_value) in enumerate(points):
        x = int(np.clip(round(float(x_value)), 0, width - 1))
        y = int(np.clip(round(float(y_value)), 0, height - 1))
        patch = depth[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
        if len(valid):
            values[index] = float(np.median(valid))
    return values


def unproject(points: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    result = np.full((len(points), 3), np.nan, dtype=np.float32)
    valid = np.isfinite(depth)
    result[valid, 2] = depth[valid]
    result[valid, 0] = (
        (points[valid, 0] - intrinsic[0, 2])
        * depth[valid]
        / intrinsic[0, 0]
    )
    result[valid, 1] = (
        (points[valid, 1] - intrinsic[1, 2])
        * depth[valid]
        / intrinsic[1, 1]
    )
    return result


def weighted_rigid(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weights = np.maximum(weights.astype(np.float64), 1e-8)
    weights /= weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * weights[:, None]).T @ target_zero
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_error(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    prediction = source @ transform[:3, :3].T + transform[:3, 3]
    return np.linalg.norm(prediction - target, axis=1)


def ransac_rigid(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    trials: int,
    threshold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    best_inliers = np.zeros(len(source), dtype=bool)
    best_score = -1.0
    if len(source) < 3:
        return np.eye(4), best_inliers
    probability = np.maximum(weights, 1e-8)
    probability /= probability.sum()
    for _ in range(trials):
        indices = rng.choice(
            len(source), size=3, replace=False, p=probability
        )
        candidate = weighted_rigid(
            source[indices], target[indices], weights[indices]
        )
        inliers = transform_error(candidate, source, target) < threshold
        score = float(weights[inliers].sum())
        if inliers.sum() >= 3 and score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers.sum() < 3:
        return np.eye(4), best_inliers
    transform = weighted_rigid(
        source[best_inliers],
        target[best_inliers],
        weights[best_inliers],
    )
    return transform, best_inliers


def solve_relative_pnp(
    source_points: np.ndarray,
    target_pixels: np.ndarray,
    valid: np.ndarray,
    intrinsic: np.ndarray,
    threshold_px: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    transform = np.eye(4, dtype=np.float64)
    inlier_mask = np.zeros(len(source_points), dtype=bool)
    indices = np.flatnonzero(
        valid
        & np.isfinite(source_points).all(axis=1)
        & np.isfinite(target_pixels).all(axis=1)
    )
    if len(indices) < min_points:
        return transform, inlier_mask, "too_few_pnp_points"
    success, rotation_vector, translation, local_inliers = cv2.solvePnPRansac(
        source_points[indices].astype(np.float64),
        target_pixels[indices].astype(np.float64),
        intrinsic.astype(np.float64),
        None,
        iterationsCount=256,
        reprojectionError=float(threshold_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or local_inliers is None:
        return transform, inlier_mask, "pnp_failed"
    selected = indices[np.asarray(local_inliers).reshape(-1)]
    if len(selected) < min_points:
        return transform, inlier_mask, "too_few_pnp_inliers"
    rotation_vector, translation = cv2.solvePnPRefineLM(
        source_points[selected].astype(np.float64),
        target_pixels[selected].astype(np.float64),
        intrinsic.astype(np.float64),
        None,
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation).reshape(3)
    inlier_mask[selected] = True
    return transform, inlier_mask, "ok"


def make_preview(
    first: np.ndarray,
    second: np.ndarray,
    tracks: np.ndarray,
    valid: np.ndarray,
    inliers: np.ndarray,
    path: Path,
) -> None:
    left = first.copy()
    right = second.copy()
    for index, pair in enumerate(tracks):
        color = (0, 220, 0) if inliers[index] else ((0, 180, 255) if valid[index] else (80, 80, 80))
        first_point = tuple(np.round(pair[0]).astype(int))
        second_point = tuple(np.round(pair[1]).astype(int))
        cv2.circle(left, first_point, 2, color, -1)
        cv2.circle(right, second_point, 2, color, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate([left, right], axis=1))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    frame_paths = numeric_frames(Path(args.frames_dir).expanduser().resolve())
    frame_map = json.loads(
        Path(args.frame_map_json).expanduser().resolve().read_text(encoding="utf-8")
    )
    frame_rows = frame_map["frames"]
    if len(frame_paths) != len(frame_rows):
        raise ValueError(
            f"Frame count mismatch: images={len(frame_paths)} map={len(frame_rows)}"
        )
    stream_dir = Path(frame_map["stream_dir"])
    intrinsic = load_intrinsics(
        Path(args.intrinsics_json).expanduser().resolve()
    )
    mask_dir = (
        Path(args.mask_dir).expanduser().resolve()
        if args.mask_dir
        else None
    )
    object_id = None if mask_dir is not None else target_object_id(stream_dir)
    preview_dir = (
        Path(args.preview_dir).expanduser().resolve()
        if args.preview_dir
        else None
    )

    images = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    height, width = images[0].shape[:2]
    resize_height, resize_width = args.resize
    resized = np.stack(
        [
            cv2.resize(image[..., ::-1], (resize_width, resize_height))
            for image in images
        ]
    )
    frames_tensor = (
        torch.as_tensor(resized, dtype=torch.float32, device=device) / 255.0 * 2.0 - 1.0
    )
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )
    model = model.to(device).eval()

    pair_count = len(images) - 1
    point_count = args.num_points
    tracks_all = np.full((pair_count, point_count, 2, 2), np.nan, np.float32)
    confidence_all = np.zeros((pair_count, point_count, 2), np.float32)
    depth_all = np.full((pair_count, point_count, 2), np.nan, np.float32)
    points_all = np.full((pair_count, point_count, 2, 3), np.nan, np.float32)
    valid_2d_all = np.zeros((pair_count, point_count), bool)
    valid_3d_all = np.zeros((pair_count, point_count), bool)
    inliers_all = np.zeros((pair_count, point_count), bool)
    transforms_all = np.repeat(np.eye(4)[None], pair_count, axis=0)
    pnp_inliers_all = np.zeros((pair_count, point_count), bool)
    pnp_transforms_all = np.repeat(np.eye(4)[None], pair_count, axis=0)
    statuses = []
    pnp_statuses = []

    for frame_index in range(pair_count):
        mask_first = load_mask(
            mask_dir,
            frame_index,
            args.object,
            frame_rows[frame_index],
            object_id,
        )
        mask_second = load_mask(
            mask_dir,
            frame_index + 1,
            args.object,
            frame_rows[frame_index + 1],
            object_id,
        )
        query_xy = sample_mask_points(
            mask_first,
            point_count,
            args.mask_erosion_px,
            rng,
        )
        if len(query_xy) < 3:
            statuses.append("too_few_mask_points")
            pnp_statuses.append("too_few_mask_points")
            continue
        query_tyx = np.concatenate(
            [
                np.zeros((len(query_xy), 1), dtype=np.float32),
                query_xy[:, [1, 0]],
            ],
            axis=1,
        )
        query_resized = transforms.convert_grid_coordinates(
            query_tyx,
            (1, height, width),
            (1, resize_height, resize_width),
            coordinate_format="tyx",
        )
        query_tensor = torch.as_tensor(
            query_resized[None], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            outputs = model(
                frames_tensor[frame_index : frame_index + 2][None],
                query_tensor,
            )
        tracks = outputs["tracks"][0]
        confidence = (
            (1.0 - torch.sigmoid(outputs["occlusion"][0]))
            * (1.0 - torch.sigmoid(outputs["expected_dist"][0]))
        )
        tracks = transforms.convert_grid_coordinates(
            tracks.detach().cpu().numpy(),
            (resize_width, resize_height),
            (width, height),
        )
        confidence = confidence.detach().cpu().numpy()
        count = min(len(tracks), point_count)
        tracks = tracks[:count]
        confidence = confidence[:count]

        rounded = np.round(tracks).astype(int)
        on_masks = np.ones((count, 2), dtype=bool)
        for time_index, mask in enumerate((mask_first, mask_second)):
            x = np.clip(rounded[:, time_index, 0], 0, width - 1)
            y = np.clip(rounded[:, time_index, 1], 0, height - 1)
            on_masks[:, time_index] = mask[y, x]
        valid_2d = (
            (confidence[:, 0] >= args.visibility_threshold)
            & (confidence[:, 1] >= args.visibility_threshold)
            & on_masks[:, 0]
            & on_masks[:, 1]
        )

        depth_pair = []
        points_pair = []
        for time_index in range(2):
            original_frame = str(
                frame_rows[frame_index + time_index]["original_frame"]
            ).zfill(6)
            depth_image = cv2.imread(
                str(resolve_depth(stream_dir, original_frame)),
                cv2.IMREAD_UNCHANGED,
            )
            if depth_image is None:
                raise FileNotFoundError(
                    resolve_depth(stream_dir, original_frame)
                )
            depth_image = depth_image.astype(np.float32) / 1000.0
            depth_values = local_depth(depth_image, tracks[:, time_index])
            depth_pair.append(depth_values)
            points_pair.append(
                unproject(tracks[:, time_index], depth_values, intrinsic)
            )
        depth_pair = np.stack(depth_pair, axis=1)
        points_pair = np.stack(points_pair, axis=1)
        valid_3d = valid_2d & np.isfinite(points_pair).all(axis=(1, 2))
        transform = np.eye(4)
        inliers = np.zeros(count, dtype=bool)
        if valid_3d.sum() >= args.min_3d_points:
            valid_indices = np.flatnonzero(valid_3d)
            weights = np.sqrt(
                confidence[valid_indices, 0]
                * confidence[valid_indices, 1]
            )
            transform, local_inliers = ransac_rigid(
                points_pair[valid_indices, 0],
                points_pair[valid_indices, 1],
                weights,
                args.ransac_trials,
                args.ransac_threshold_mm / 1000.0,
                rng,
            )
            inliers[valid_indices] = local_inliers
            status = "ok" if local_inliers.sum() >= args.min_3d_points else "too_few_inliers"
        else:
            status = "too_few_3d_points"
        pnp_transform, pnp_inliers, pnp_status = solve_relative_pnp(
            points_pair[:, 0],
            tracks[:, 1],
            valid_2d & np.isfinite(points_pair[:, 0]).all(axis=1),
            intrinsic,
            args.pnp_threshold_px,
            args.min_pnp_points,
        )

        tracks_all[frame_index, :count] = tracks
        confidence_all[frame_index, :count] = confidence
        depth_all[frame_index, :count] = depth_pair
        points_all[frame_index, :count] = points_pair
        valid_2d_all[frame_index, :count] = valid_2d
        valid_3d_all[frame_index, :count] = valid_3d
        inliers_all[frame_index, :count] = inliers
        transforms_all[frame_index] = transform
        pnp_inliers_all[frame_index, :count] = pnp_inliers
        pnp_transforms_all[frame_index] = pnp_transform
        statuses.append(status)
        pnp_statuses.append(pnp_status)
        if preview_dir is not None:
            make_preview(
                images[frame_index],
                images[frame_index + 1],
                tracks,
                valid_3d,
                pnp_inliers,
                preview_dir / f"{frame_index:06d}_{frame_index + 1:06d}.jpg",
            )
        translation_mm = np.linalg.norm(transform[:3, 3]) * 1000.0
        angle_deg = np.degrees(
            np.arccos(
                np.clip((np.trace(transform[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
            )
        )
        pnp_translation_mm = np.linalg.norm(pnp_transform[:3, 3]) * 1000.0
        pnp_angle_deg = np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(pnp_transform[:3, :3]) - 1.0) * 0.5,
                    -1.0,
                    1.0,
                )
            )
        )
        print(
            f"[{frame_index + 1}/{pair_count}] 3d={status} pnp={pnp_status} "
            f"2d={valid_2d.sum()} 3d={valid_3d.sum()} "
            f"inliers={inliers.sum()} pnp_inliers={pnp_inliers.sum()} "
            f"3d_t={translation_mm:.2f}mm "
            f"3d_R={angle_deg:.2f}deg "
            f"pnp_t={pnp_translation_mm:.2f}mm "
            f"pnp_R={pnp_angle_deg:.2f}deg",
            flush=True,
        )

    out_path = Path(args.out_npz).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_pairs=np.stack(
            [np.arange(pair_count), np.arange(1, pair_count + 1)], axis=1
        ),
        tracks_uv=tracks_all,
        confidence=confidence_all,
        depth_m=depth_all,
        points_camera_m=points_all,
        valid_2d=valid_2d_all,
        valid_3d=valid_3d_all,
        inlier_mask=inliers_all,
        relative_transform=transforms_all,
        pnp_inlier_mask=pnp_inliers_all,
        relative_transform_pnp=pnp_transforms_all,
        status=np.asarray(statuses),
        pnp_status=np.asarray(pnp_statuses),
        intrinsics=intrinsic,
    )
    valid_transforms = np.asarray(statuses) == "ok"
    translation = (
        np.linalg.norm(transforms_all[:, :3, 3], axis=1) * 1000.0
    )
    angle = np.degrees(
        np.arccos(
            np.clip(
                (np.trace(transforms_all[:, :3, :3], axis1=1, axis2=2) - 1.0)
                * 0.5,
                -1.0,
                1.0,
            )
        )
    )
    valid_pnp = np.asarray(pnp_statuses) == "ok"
    pnp_translation = (
        np.linalg.norm(pnp_transforms_all[:, :3, 3], axis=1) * 1000.0
    )
    pnp_angle = np.degrees(
        np.arccos(
            np.clip(
                (
                    np.trace(
                        pnp_transforms_all[:, :3, :3],
                        axis1=1,
                        axis2=2,
                    )
                    - 1.0
                )
                * 0.5,
                -1.0,
                1.0,
            )
        )
    )
    summary = {
        "settings": vars(args),
        "num_frames": len(images),
        "num_pairs": pair_count,
        "status_counts": {
            name: statuses.count(name) for name in sorted(set(statuses))
        },
        "pnp_status_counts": {
            name: pnp_statuses.count(name)
            for name in sorted(set(pnp_statuses))
        },
        "valid_2d_median": float(np.median(valid_2d_all.sum(axis=1))),
        "valid_3d_median": float(np.median(valid_3d_all.sum(axis=1))),
        "inlier_median": float(np.median(inliers_all.sum(axis=1))),
        "pnp_inlier_median": float(np.median(pnp_inliers_all.sum(axis=1))),
        "translation_mm": {
            "median": float(np.median(translation[valid_transforms]))
            if valid_transforms.any()
            else None,
            "p90": float(np.quantile(translation[valid_transforms], 0.9))
            if valid_transforms.any()
            else None,
            "max": float(np.max(translation[valid_transforms]))
            if valid_transforms.any()
            else None,
        },
        "rotation_deg": {
            "median": float(np.median(angle[valid_transforms]))
            if valid_transforms.any()
            else None,
            "p90": float(np.quantile(angle[valid_transforms], 0.9))
            if valid_transforms.any()
            else None,
            "max": float(np.max(angle[valid_transforms]))
            if valid_transforms.any()
            else None,
        },
        "pnp_translation_mm": {
            "median": float(np.median(pnp_translation[valid_pnp]))
            if valid_pnp.any()
            else None,
            "p90": float(np.quantile(pnp_translation[valid_pnp], 0.9))
            if valid_pnp.any()
            else None,
            "max": float(np.max(pnp_translation[valid_pnp]))
            if valid_pnp.any()
            else None,
        },
        "pnp_rotation_deg": {
            "median": float(np.median(pnp_angle[valid_pnp]))
            if valid_pnp.any()
            else None,
            "p90": float(np.quantile(pnp_angle[valid_pnp], 0.9))
            if valid_pnp.any()
            else None,
            "max": float(np.max(pnp_angle[valid_pnp]))
            if valid_pnp.any()
            else None,
        },
        "out_npz": str(out_path),
    }
    summary_path = Path(args.out_summary).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
