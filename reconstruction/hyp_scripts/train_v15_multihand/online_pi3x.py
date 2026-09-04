"""Frozen Pi3X inference and RAM/disk feature providers."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def final_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        values = list(value.values())
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        raise TypeError(f"Decoder output contains no tensor: {type(value)!r}")
    for item in reversed(values):
        try:
            return final_tensor(item)
        except TypeError:
            pass
    raise TypeError(f"Decoder output contains no tensor: {type(value)!r}")


def row_key(row):
    return (
        str(row["stream_id"]),
        int(row["start"]),
        int(row["end"]),
    )


def row_intrinsics(row):
    """Return one camera matrix per frame from a row or its label files."""
    time = len(row["frame_indices"])
    if "intrinsics" in row:
        intrinsics = np.asarray(row["intrinsics"], dtype=np.float32)
        if intrinsics.shape == (3, 3):
            return np.broadcast_to(intrinsics[None], (time, 3, 3)).copy()
        if intrinsics.shape == (time, 3, 3):
            return intrinsics.copy()
        raise ValueError(
            f"Unexpected manifest intrinsics shape {intrinsics.shape} for "
            f"{row_key(row)}"
        )

    matrices = []
    for label_path in row["label_paths"]:
        path = Path(label_path).expanduser().resolve()
        with np.load(str(path), allow_pickle=False) as data:
            if "intrinsics" not in data.files:
                raise KeyError(
                    f"Neither manifest nor label provides intrinsics: {label_path}"
                )
            matrix = np.asarray(data["intrinsics"], dtype=np.float32).reshape(3, 3)
        matrices.append(matrix)
    if len(matrices) != time:
        raise ValueError(
            f"Loaded {len(matrices)} intrinsics for {time} frames in {row_key(row)}"
        )
    return np.stack(matrices)


def resized_row_intrinsics(row, original_wh, resized_wh):
    from pi3_wilor_hand.geometry import resize_intrinsics

    return np.stack([
        resize_intrinsics(matrix, original_wh, resized_wh)
        for matrix in row_intrinsics(row)
    ]).astype(np.float32)


def compact_cache_path(root, row):
    return (
        Path(root).expanduser().resolve()
        / str(row["stream_id"])
        / "windows"
        / f"window_{int(row['start']):06d}_{int(row['end']):06d}.npz"
    )


COMPACT_CACHE_KEYS = (
    "joint_patch_features",
    "joint_patch_uv",
    "joint_patch_confidence",
    "joint_patch_valid",
    "global_features",
    "global_uv",
    "global_confidence",
    "source_grid_hw",
    "metric_window_features",
    "intrinsics_resized",
    "resized_wh",
    "horizontal_mirror",
    "coordinate_frame",
)


def valid_compact_cache(
    path, row, patch_radius=None, global_grid_size=None, query_source=None,
):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if not set(COMPACT_CACHE_KEYS).issubset(data.files):
                return False
            if str(data["stream_id"].item()) != str(row["stream_id"]):
                return False
            if int(data["start"].item()) != int(row["start"]):
                return False
            if int(data["end"].item()) != int(row["end"]):
                return False
            if not np.array_equal(
                np.asarray(data["frame_indices"], dtype=np.int64),
                np.asarray(row["frame_indices"], dtype=np.int64),
            ):
                return False
            if patch_radius is not None and int(
                data["joint_patch_radius"].item()
            ) != int(patch_radius):
                return False
            if global_grid_size is not None and int(
                data["global_grid_size"].item()
            ) != int(global_grid_size):
                return False
            if query_source is not None:
                cached_source = (
                    str(data["query_source"].item())
                    if "query_source" in data.files else "gt"
                )
                if cached_source != str(query_source):
                    return False
            return True
    except (OSError, KeyError, ValueError):
        return False


def write_compact_cache(
    path, row, payload, joint_uv, patch_radius, global_grid_size,
    query_source="gt",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    content = {key: np.asarray(payload[key]) for key in COMPACT_CACHE_KEYS}
    content.update({
        "cache_version": np.asarray("compact_pi3x_window_v1"),
        "stream_id": np.asarray(str(row["stream_id"])),
        "start": np.int64(row["start"]),
        "end": np.int64(row["end"]),
        "frame_indices": np.asarray(row["frame_indices"], dtype=np.int64),
        "joint_uv_clean": np.asarray(joint_uv, dtype=np.float32),
        "joint_patch_radius": np.int32(patch_radius),
        "global_grid_size": np.int32(global_grid_size),
        "query_source": np.asarray(str(query_source)),
    })
    try:
        np.savez(str(temporary), **content)
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


class Pi3XWindowMaterializer:
    def __init__(
        self,
        hand_uni_root,
        pi3_root,
        checkpoint,
        device="cuda",
        pixel_limit=180000,
        feature_dtype="float16",
    ):
        hand_uni_root = Path(hand_uni_root).expanduser().resolve()
        if str(hand_uni_root) not in sys.path:
            sys.path.insert(0, str(hand_uni_root))
        from pi3_wilor_hand.factory import load_pi3x
        from pi3_wilor_hand.handsh_pipeline import Pi3XReconstructionBranch

        self.device = torch.device(device)
        self.pixel_limit = int(pixel_limit)
        self.feature_dtype = (
            np.float16 if feature_dtype == "float16" else np.float32
        )
        self.model = load_pi3x(
            pi3_root=str(Path(pi3_root).expanduser().resolve()),
            ckpt=str(Path(checkpoint).expanduser().resolve()),
            device=str(self.device),
        )
        self.reconstruction = Pi3XReconstructionBranch(
            self.model,
            freeze_pi3=True,
            use_intrinsics=True,
        ).to(self.device).eval()
        self.captured = {}
        self.hooks = [
            self.model.point_decoder.register_forward_hook(self._capture_point),
            self.model.metric_decoder.register_forward_hook(self._capture_metric),
        ]

    def _capture_point(self, _module, _inputs, output):
        self.captured["ret_point"] = final_tensor(output).detach()

    def _capture_metric(self, _module, _inputs, output):
        self.captured["ret_metric"] = final_tensor(output).detach()

    def __call__(self, row):
        from pi3_wilor_hand.pi3_runner import load_images_for_pi3

        image_paths = [
            str(Path(path).expanduser().resolve())
            for path in row["image_paths"]
        ]
        images, resized_wh, original_wh = load_images_for_pi3(
            image_paths,
            pixel_limit=self.pixel_limit,
        )
        resized_intrinsics = resized_row_intrinsics(
            row, original_wh, resized_wh
        )
        image_batch = images[None].to(self.device)
        intrinsics_batch = (
            torch.from_numpy(resized_intrinsics)
            .to(device=self.device, dtype=torch.float32)[None]
        )
        self.captured.clear()
        with torch.inference_mode():
            outputs = self.reconstruction(
                image_batch,
                intrinsics=intrinsics_batch,
            )
        ret_point = self.captured.get("ret_point")
        ret_metric = self.captured.get("ret_metric")
        if ret_point is None or ret_metric is None:
            raise RuntimeError("Pi3X decoder hooks did not capture both outputs")
        point_tokens = ret_point[:, int(self.model.patch_start_idx):]
        resized_w, resized_h = (int(value) for value in resized_wh)
        patch_h = resized_h // int(self.model.patch_size)
        patch_w = resized_w // int(self.model.patch_size)
        expected = len(image_paths) * patch_h * patch_w
        if point_tokens.numel() // point_tokens.shape[-1] != expected:
            raise RuntimeError(
                f"Unexpected point token shape {tuple(point_tokens.shape)} "
                f"for {len(image_paths)}x{patch_h}x{patch_w}"
            )
        point = point_tokens.reshape(
            len(image_paths), patch_h, patch_w, point_tokens.shape[-1]
        )
        confidence = torch.sigmoid(outputs["conf"][0, ..., 0]).float().cpu().numpy()
        confidence = np.stack([
            cv2.resize(
                frame,
                (patch_w, patch_h),
                interpolation=cv2.INTER_AREA,
            )
            for frame in confidence
        ])
        return {
            "geometry_patch_features": point.float().cpu().numpy().astype(
                self.feature_dtype
            ),
            "geometry_feature_grid_hw": np.asarray(
                [patch_h, patch_w], dtype=np.int32
            ),
            "metric_window_features": ret_metric.float().cpu().numpy().astype(
                self.feature_dtype
            ),
            "confidence": confidence.astype(np.float16),
            "intrinsics_resized": resized_intrinsics.astype(np.float32),
            "resized_wh": np.asarray(resized_wh, dtype=np.int32),
            "horizontal_mirror": np.asarray(False),
            "coordinate_frame": np.asarray("original_camera"),
        }

    def compact(self, row, joint_uv, patch_radius=1, global_grid_size=4):
        """Run Pi3X once and transfer only local/global query candidates."""
        from pi3_wilor_hand.pi3_runner import load_images_for_pi3

        image_paths = [
            str(Path(path).expanduser().resolve())
            for path in row["image_paths"]
        ]
        images, resized_wh, original_wh = load_images_for_pi3(
            image_paths,
            pixel_limit=self.pixel_limit,
        )
        resized_intrinsics = resized_row_intrinsics(
            row, original_wh, resized_wh
        )
        image_batch = images[None].to(self.device)
        intrinsics_batch = (
            torch.from_numpy(resized_intrinsics)
            .to(device=self.device, dtype=torch.float32)[None]
        )
        self.captured.clear()
        with torch.inference_mode():
            outputs = self.reconstruction(image_batch, intrinsics=intrinsics_batch)
        ret_point = self.captured.get("ret_point")
        ret_metric = self.captured.get("ret_metric")
        if ret_point is None or ret_metric is None:
            raise RuntimeError("Pi3X decoder hooks did not capture both outputs")

        resized_w, resized_h = (int(value) for value in resized_wh)
        patch_h = resized_h // int(self.model.patch_size)
        patch_w = resized_w // int(self.model.patch_size)
        point_tokens = ret_point[:, int(self.model.patch_start_idx):]
        expected = len(image_paths) * patch_h * patch_w
        if point_tokens.numel() // point_tokens.shape[-1] != expected:
            raise RuntimeError(
                f"Unexpected point token shape {tuple(point_tokens.shape)} "
                f"for {len(image_paths)}x{patch_h}x{patch_w}"
            )
        point = point_tokens.reshape(
            len(image_paths), patch_h, patch_w, point_tokens.shape[-1]
        )
        query = torch.as_tensor(
            joint_uv, dtype=torch.float32, device=self.device
        )
        if query.shape[0] != len(image_paths) or query.shape[-1] != 2:
            raise ValueError(
                f"Joint query shape {tuple(query.shape)} does not match "
                f"{len(image_paths)} frames"
            )

        radius = int(patch_radius)
        if radius < 0:
            raise ValueError("patch_radius must be non-negative")
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-radius, radius + 1, device=self.device),
            torch.arange(-radius, radius + 1, device=self.device),
            indexing="ij",
        ), dim=-1).reshape(-1, 2)
        center_x = torch.round((query[..., 0] + 1.0) * 0.5 * (patch_w - 1)).long()
        center_y = torch.round((query[..., 1] + 1.0) * 0.5 * (patch_h - 1)).long()
        sample_y = center_y[..., None] + offsets[:, 0]
        sample_x = center_x[..., None] + offsets[:, 1]
        sample_valid = (
            (sample_x >= 0) & (sample_x < patch_w)
            & (sample_y >= 0) & (sample_y < patch_h)
        )
        sample_x = sample_x.clamp(0, patch_w - 1)
        sample_y = sample_y.clamp(0, patch_h - 1)
        frame_index = torch.arange(
            len(image_paths), device=self.device
        ).view(-1, *([1] * (sample_x.ndim - 1))).expand_as(sample_x)
        local_feature = point[frame_index, sample_y, sample_x]
        local_uv = torch.stack((
            sample_x.to(torch.float32) / max(patch_w - 1, 1) * 2.0 - 1.0,
            sample_y.to(torch.float32) / max(patch_h - 1, 1) * 2.0 - 1.0,
        ), dim=-1)

        confidence = torch.sigmoid(outputs["conf"][0, ..., 0]).float()
        confidence = F.interpolate(
            confidence[:, None], size=(patch_h, patch_w),
            mode="area",
        )[:, 0]
        local_confidence = confidence[frame_index, sample_y, sample_x]

        global_size = int(global_grid_size)
        if global_size <= 0:
            raise ValueError("global_grid_size must be positive")
        point_chw = point.permute(0, 3, 1, 2).float()
        global_feature = F.adaptive_avg_pool2d(
            point_chw, (global_size, global_size)
        ).permute(0, 2, 3, 1).reshape(len(image_paths), -1, point.shape[-1])
        global_confidence = F.adaptive_avg_pool2d(
            confidence[:, None], (global_size, global_size)
        )[:, 0].reshape(len(image_paths), -1)
        axis = torch.linspace(-1.0, 1.0, global_size, device=self.device)
        global_y, global_x = torch.meshgrid(axis, axis, indexing="ij")
        global_uv = torch.stack((global_x, global_y), dim=-1).reshape(-1, 2)

        result = {
            "joint_patch_features": local_feature.float().cpu().numpy().astype(
                self.feature_dtype
            ),
            "joint_patch_uv": local_uv.float().cpu().numpy().astype(np.float16),
            "joint_patch_confidence": local_confidence.float().cpu().numpy().astype(
                np.float16
            ),
            "joint_patch_valid": sample_valid.cpu().numpy(),
            "global_features": global_feature.cpu().numpy().astype(self.feature_dtype),
            "global_uv": global_uv.cpu().numpy().astype(np.float16),
            "global_confidence": global_confidence.cpu().numpy().astype(np.float16),
            "source_grid_hw": np.asarray([patch_h, patch_w], dtype=np.int32),
            "metric_window_features": ret_metric.float().cpu().numpy().astype(
                self.feature_dtype
            ),
            "intrinsics_resized": resized_intrinsics.astype(np.float32),
            "resized_wh": np.asarray(resized_wh, dtype=np.int32),
            "horizontal_mirror": np.asarray(False),
            "coordinate_frame": np.asarray("original_camera"),
        }
        del outputs, point, point_tokens, local_feature, global_feature
        return result

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        del self.reconstruction
        del self.model
        torch.cuda.empty_cache()


class RamFeatureProvider:
    def __init__(self, payloads):
        self.payloads = payloads

    def __call__(self, row, joint_uv=None):
        return self.payloads[row_key(row)]

    @property
    def nbytes(self):
        return sum(
            value.nbytes
            for payload in self.payloads.values()
            for value in payload.values()
            if isinstance(value, np.ndarray)
        )


class DummyDenseProvider:
    """Supply geometry metadata so labels/tracks can be decoded before Pi3X."""

    def __call__(self, row):
        with np.load(row["label_paths"][0], allow_pickle=False) as label:
            height, width = np.asarray(label["seg"]).shape[:2]
        time = len(row["frame_indices"])
        return {
            "geometry_patch_features": np.zeros((time, 1, 1, 1), np.float32),
            "geometry_feature_grid_hw": np.asarray([1, 1], np.int32),
            "metric_window_features": np.zeros((1, 1), np.float32),
            "intrinsics_resized": row_intrinsics(row),
            "resized_wh": np.asarray([width, height], np.int32),
            "horizontal_mirror": np.asarray(False),
            "coordinate_frame": np.asarray("original_camera"),
        }


class CompactFeatureProvider(RamFeatureProvider):
    pass


class OnlineCompactFeatureProvider:
    """Run frozen Pi3X and sample around this sample's actual joint query."""

    def __init__(self, materializer, patch_radius=1, global_grid_size=4):
        self.materializer = materializer
        self.patch_radius = int(patch_radius)
        self.global_grid_size = int(global_grid_size)

    def __call__(self, row, joint_uv=None):
        if joint_uv is None:
            raise ValueError("Online compact sampling requires joint_uv")
        return self.materializer.compact(
            row,
            joint_uv,
            patch_radius=self.patch_radius,
            global_grid_size=self.global_grid_size,
        )

    def close(self):
        self.materializer.close()


class DiskCompactFeatureProvider:
    def __init__(
        self, root, patch_radius=None, global_grid_size=None, query_source=None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.patch_radius = patch_radius
        self.global_grid_size = global_grid_size
        self.query_source = query_source

    def path(self, row):
        return compact_cache_path(self.root, row)

    def __call__(self, row, joint_uv=None):
        path = self.path(row)
        if not valid_compact_cache(
            path, row, self.patch_radius, self.global_grid_size,
            query_source=self.query_source,
        ):
            raise FileNotFoundError(
                f"Missing or incompatible compact Pi3X cache: {path}"
            )
        with np.load(str(path), allow_pickle=False) as data:
            return {key: np.asarray(data[key]).copy() for key in COMPACT_CACHE_KEYS}
