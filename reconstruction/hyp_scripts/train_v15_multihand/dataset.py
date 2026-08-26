"""DexYCB and Pi3X inputs for side-free multi-hand trajectory training."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_float(value):
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def dense_path(row, root):
    if row.get("dense_pi3x_npz"):
        return Path(row["dense_pi3x_npz"]).expanduser().resolve()
    return (
        Path(root) / row["stream_id"] / "windows"
        / f"window_{int(row['start']):06d}_{int(row['end']):06d}.npz"
    ).resolve()


def mask_visibility(mask, uv, radius):
    """Local visible-hand occupancy; this is a proxy, not occlusion GT."""
    height, width = mask.shape
    result = np.zeros(len(uv), dtype=np.float32)
    for index, point in enumerate(uv):
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 < x1 and y0 < y1:
            result[index] = float(mask[y0:y1, x0:x1].mean())
    return result


class QueryNoise:
    def __init__(
        self,
        global_sigma_px=4.0,
        temporal_sigma_px=0.5,
        joint_sigma_px=2.0,
        outlier_probability=0.03,
        outlier_sigma_px=12.0,
        dropout_probability=0.1,
    ):
        self.global_sigma_px = float(global_sigma_px)
        self.temporal_sigma_px = float(temporal_sigma_px)
        self.joint_sigma_px = float(joint_sigma_px)
        self.outlier_probability = float(outlier_probability)
        self.outlier_sigma_px = float(outlier_sigma_px)
        self.dropout_probability = float(dropout_probability)

    def __call__(self, uv, valid):
        time, hands, joints, _ = uv.shape
        output = uv.copy()
        global_offset = np.random.normal(0.0, self.global_sigma_px, (1, hands, 1, 2))
        drift = np.random.normal(0.0, self.temporal_sigma_px, (time, hands, 1, 2))
        drift = np.cumsum(drift, axis=0)
        drift -= drift.mean(axis=0, keepdims=True)
        local = np.random.normal(0.0, self.joint_sigma_px, output.shape)
        outliers = np.random.random(valid.shape) < self.outlier_probability
        local += outliers[..., None] * np.random.normal(
            0.0, self.outlier_sigma_px, output.shape
        )
        output += global_offset + drift + local
        dropped = np.random.random(valid.shape) < self.dropout_probability
        return output.astype(np.float32), valid & ~dropped


class DexYCBMultiHandWindowDataset(Dataset):
    """V15 schema: [time, hand_slot, joint, ...], with DexYCB hand_slot=1."""

    def __init__(
        self,
        windows,
        pi3x_root,
        max_hands=4,
        training=False,
        noise=None,
        visibility_radius_px=5,
        require_original_camera=True,
        visibility_source="detector",
        visibility_root=None,
        track_root=None,
    ):
        self.rows = load_jsonl(windows)
        if not self.rows:
            raise RuntimeError(f"No windows in {windows}")
        self.pi3x_root = Path(pi3x_root).expanduser().resolve()
        self.max_hands = int(max_hands)
        self.training = bool(training)
        self.noise = noise if noise is not None else QueryNoise()
        self.visibility_radius_px = int(visibility_radius_px)
        self.require_original_camera = bool(require_original_camera)
        self.visibility_source = str(visibility_source)
        self.visibility_root = (
            None if visibility_root is None
            else Path(visibility_root).expanduser().resolve()
        )
        self.track_root = (
            None if track_root is None
            else Path(track_root).expanduser().resolve()
        )
        if self.visibility_source not in ("detector", "mask", "ones"):
            raise ValueError(f"Unknown visibility source: {self.visibility_source}")
        if self.visibility_source == "detector" and self.visibility_root is None:
            raise ValueError("visibility_root is required for detector visibility")
        streams = sorted({row["stream_id"] for row in self.rows})
        self.stream_indices = {stream: index for index, stream in enumerate(streams)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        labels = row["label_paths"]
        time = len(labels)
        hands, joints = self.max_hands, 21
        query_uv_px = np.zeros((time, hands, joints, 2), dtype=np.float32)
        query_valid = np.zeros((time, hands, joints), dtype=bool)
        visibility = np.zeros((time, hands, joints), dtype=np.float32)
        target = np.zeros((time, hands, 3), dtype=np.float32)
        target_valid = np.zeros((time, hands), dtype=bool)
        hand_slot_valid = np.zeros((time, hands), dtype=bool)
        observation_valid = np.zeros((time, hands), dtype=bool)
        track_ids = np.full((time, hands), -1, dtype=np.int64)
        track_data = None
        track_index = {}
        if self.track_root is not None:
            track_file = self.track_root / row["stream_id"] / "tracks.npz"
            track_data = np.load(str(track_file), allow_pickle=False)
            track_frames = np.asarray(track_data["frame_indices"], dtype=np.int64)
            track_index = {
                int(frame): offset for offset, frame in enumerate(track_frames)
            }
        detector_visibility = {}
        if self.visibility_source == "detector":
            visibility_file = (
                self.visibility_root / row["stream_id"] / "visibility_cache.npz"
            )
            with np.load(str(visibility_file), allow_pickle=False) as cache:
                cache_frames = np.asarray(cache["frame_indices"], dtype=np.int64)
                cache_values = np.asarray(cache["joint_visibility"], dtype=np.float32)
                cache_valid = np.asarray(cache["visibility_valid"], dtype=bool)
            if cache_values.ndim == 2:
                cache_values = cache_values[:, None]
            if cache_valid.ndim == 1:
                cache_valid = cache_valid[:, None]
            detector_visibility = {
                int(frame): (cache_values[offset], cache_valid[offset])
                for offset, frame in enumerate(cache_frames)
            }

        for frame, label_path in enumerate(labels):
            with np.load(label_path, allow_pickle=False) as data:
                seg = np.asarray(data["seg"])
            height, width = seg.shape[:2]
            source_frame = int(row["frame_indices"][frame])
            if track_data is None:
                with np.load(label_path, allow_pickle=False) as data:
                    uv = np.asarray(data["joint_2d"], dtype=np.float32)
                    xyz = np.asarray(data["joint_3d"], dtype=np.float32)
                if uv.ndim == 2:
                    uv = uv[None]
                if xyz.ndim == 2:
                    xyz = xyz[None]
                count = min(hands, len(uv), len(xyz))
                for hand in range(count):
                    valid = (
                        np.isfinite(uv[hand]).all(axis=-1)
                        & (uv[hand, :, 0] >= 0) & (uv[hand, :, 0] < width)
                        & (uv[hand, :, 1] >= 0) & (uv[hand, :, 1] < height)
                    )
                    query_uv_px[frame, hand] = finite_float(uv[hand])
                    query_valid[frame, hand] = valid
                    target[frame, hand] = finite_float(xyz[hand, 0])
                    target_valid[frame, hand] = bool(
                        np.isfinite(xyz[hand, 0]).all() and xyz[hand, 0, 2] > 0
                    )
                    hand_slot_valid[frame, hand] = True
                    observation_valid[frame, hand] = bool(valid.any())
                    track_ids[frame, hand] = hand
            else:
                if source_frame not in track_index:
                    continue
                source = track_index[source_frame]
                cached_uv = np.asarray(track_data["joint_uv"][source], dtype=np.float32)
                cached_xyz = np.asarray(track_data["joint_xyz"][source], dtype=np.float32)
                count = min(hands, len(cached_uv))
                query_uv_px[frame, :count] = finite_float(cached_uv[:count])
                valid = np.asarray(track_data["joint_valid"][source, :count], dtype=bool)
                valid &= (
                    (cached_uv[:count, :, 0] >= 0)
                    & (cached_uv[:count, :, 0] < width)
                    & (cached_uv[:count, :, 1] >= 0)
                    & (cached_uv[:count, :, 1] < height)
                )
                query_valid[frame, :count] = valid
                target[frame, :count] = finite_float(cached_xyz[:count, 0])
                target_valid[frame, :count] = np.asarray(
                    track_data["target_valid"][source, :count], dtype=bool
                )
                hand_slot_valid[frame, :count] = np.asarray(
                    track_data["track_valid"][source, :count], dtype=bool
                )
                observation_valid[frame, :count] = np.asarray(
                    track_data["observation_valid"][source, :count], dtype=bool
                )
                track_ids[frame, :count] = np.asarray(
                    track_data["track_ids"][source, :count], dtype=np.int64
                )
            if self.visibility_source == "mask":
                for hand in np.flatnonzero(hand_slot_valid[frame]):
                    visibility[frame, hand] = mask_visibility(
                        seg == 255, query_uv_px[frame, hand],
                        self.visibility_radius_px,
                    )
            elif self.visibility_source == "ones":
                visibility[frame, hand_slot_valid[frame]] = 1.0
            else:
                values, values_valid = detector_visibility.get(
                    source_frame,
                    (
                        np.full((hands, joints), 0.5, dtype=np.float32),
                        np.zeros(hands, dtype=bool),
                    ),
                )
                count = min(hands, len(values))
                visibility[frame, :count] = np.where(
                    values_valid[:count, None], values[:count], 0.5
                )

        if track_data is not None:
            track_data.close()

        clean_query_uv_px = query_uv_px.copy()
        clean_query_valid = query_valid.copy()
        if self.training:
            query_uv_px, query_valid = self.noise(query_uv_px, query_valid)

        dense_file = dense_path(row, self.pi3x_root)
        with np.load(str(dense_file), allow_pickle=False) as dense:
            mirrored = bool(np.asarray(dense.get("horizontal_mirror", False)).item())
            if self.require_original_camera and mirrored:
                raise ValueError(f"V15 requires original-camera Pi3X cache: {dense_file}")
            point_features = finite_float(dense["geometry_patch_features"])
            grid_hw = tuple(int(x) for x in np.asarray(dense["geometry_feature_grid_hw"]).reshape(2))
            resized_wh = np.asarray(dense["resized_wh"], dtype=np.float32).reshape(2)
            intrinsics = np.asarray(dense["intrinsics_resized"], dtype=np.float32)
            metric = np.asarray(dense.get("metric_window_features", []), dtype=np.float32)
            if metric.size == 0:
                raise KeyError(f"{dense_file} lacks metric_window_features")
            metric = finite_float(metric.reshape(-1, metric.shape[-1]).mean(axis=0))
            confidence = np.asarray(dense.get("confidence", []), dtype=np.float32)

        if point_features.shape[:3] != (time, grid_hw[0], grid_hw[1]):
            raise ValueError(f"Pi3X/window shape mismatch: {dense_file}")
        original_wh = np.array([seg.shape[1], seg.shape[0]], dtype=np.float32)
        scale = resized_wh / original_wh
        query_uv_px *= scale.reshape(1, 1, 1, 2)
        clean_query_uv_px *= scale.reshape(1, 1, 1, 2)
        query_uv01 = query_uv_px / np.maximum(resized_wh - 1.0, 1.0).reshape(1, 1, 1, 2)
        clean_query_uv01 = clean_query_uv_px / np.maximum(resized_wh - 1.0, 1.0).reshape(1, 1, 1, 2)
        query_valid &= np.isfinite(query_uv01).all(axis=-1)
        query_valid &= (query_uv01 >= 0).all(axis=-1) & (query_uv01 <= 1).all(axis=-1)

        grid_y, grid_x = np.meshgrid(
            np.linspace(-1.0, 1.0, grid_hw[0], dtype=np.float32),
            np.linspace(-1.0, 1.0, grid_hw[1], dtype=np.float32), indexing="ij",
        )
        grid_uv = np.stack((grid_x, grid_y), axis=-1)
        grid_uv = np.broadcast_to(grid_uv[None], (time, *grid_uv.shape)).copy()
        grid_confidence = np.ones((time, *grid_hw), dtype=np.float32)
        if confidence.shape[:3] == (time, int(original_wh[1]), int(original_wh[0])):
            ys = np.linspace(0, confidence.shape[1] - 1, grid_hw[0]).round().astype(int)
            xs = np.linspace(0, confidence.shape[2] - 1, grid_hw[1]).round().astype(int)
            grid_confidence = confidence[:, ys][:, :, xs]
        grid_valid = np.isfinite(point_features).all(axis=-1) & np.isfinite(grid_confidence)
        if intrinsics.ndim == 2:
            intrinsics = np.broadcast_to(intrinsics[None], (time, 3, 3)).copy()

        return {
            "point_features": torch.from_numpy(point_features),
            "grid_uv": torch.from_numpy(grid_uv),
            "grid_confidence": torch.from_numpy(finite_float(grid_confidence)),
            "grid_valid": torch.from_numpy(grid_valid),
            "metric_window_features": torch.from_numpy(metric),
            "joint_uv": torch.from_numpy(finite_float(query_uv01 * 2.0 - 1.0)),
            "joint_query_valid": torch.from_numpy(query_valid),
            "joint_visibility": torch.from_numpy(visibility),
            "clean_joint_uv": torch.from_numpy(finite_float(clean_query_uv01 * 2.0 - 1.0)),
            "clean_joint_valid": torch.from_numpy(clean_query_valid),
            "hand_slot_valid": torch.from_numpy(hand_slot_valid),
            "observation_valid": torch.from_numpy(observation_valid),
            "target_t": torch.from_numpy(target),
            "target_valid": torch.from_numpy(target_valid),
            "intrinsics": torch.from_numpy(finite_float(intrinsics)),
            "image_wh": torch.from_numpy(
                np.broadcast_to(resized_wh[None], (time, 2)).copy()
            ),
            "stream_index": torch.tensor(self.stream_indices[row["stream_id"]]),
            "frame_index": torch.tensor(row["frame_indices"], dtype=torch.long),
            "track_id": torch.from_numpy(track_ids),
        }
