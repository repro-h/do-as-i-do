"""Frozen Pi3X inference materialized once per manifest window in host RAM."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


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
        from pi3_wilor_hand.geometry import resize_intrinsics
        from pi3_wilor_hand.pi3_runner import load_images_for_pi3

        image_paths = [
            str(Path(path).expanduser().resolve())
            for path in row["image_paths"]
        ]
        images, resized_wh, original_wh = load_images_for_pi3(
            image_paths,
            pixel_limit=self.pixel_limit,
        )
        intrinsics = np.asarray(row["intrinsics"], dtype=np.float32).reshape(3, 3)
        resized_intrinsics = resize_intrinsics(
            intrinsics,
            original_wh,
            resized_wh,
        )
        image_batch = images[None].to(self.device)
        intrinsics_batch = (
            torch.from_numpy(resized_intrinsics)
            .to(device=self.device, dtype=torch.float32)[None, None]
            .repeat(1, len(image_paths), 1, 1)
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

    def __call__(self, row):
        return self.payloads[row_key(row)]

    @property
    def nbytes(self):
        return sum(
            value.nbytes
            for payload in self.payloads.values()
            for value in payload.values()
            if isinstance(value, np.ndarray)
        )
