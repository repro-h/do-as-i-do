#!/usr/bin/env python3
"""Export Pi3X geometry-head patch features for one DexYCB sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--object-label", type=int, required=True)
    parser.add_argument("--hand-label", type=int, default=255)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload.get("frame_map", payload))
    if isinstance(rows, dict):
        rows = list(rows.values())
    return sorted(rows, key=lambda row: int(row["output_index"]))


def load_segmentation(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("seg", "segmentation", "label"):
            if key in payload:
                segmentation = np.asarray(payload[key])
                break
        else:
            raise KeyError(f"No segmentation found in {path}")
    return np.squeeze(segmentation)


def patch_coverage(
    segmentation: np.ndarray,
    label: int,
    resized_wh: tuple[int, int],
    patch_hw: tuple[int, int],
) -> np.ndarray:
    resized_w, resized_h = resized_wh
    patch_h, patch_w = patch_hw
    binary = (segmentation == label).astype(np.float32)
    resized = cv2.resize(
        binary,
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.resize(
        resized,
        (patch_w, patch_h),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float16)


def verify_pi3x_checkpoint(path: Path) -> None:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "pi3" in payload:
            payload = payload["pi3"]
        keys = list(payload.keys()) if isinstance(payload, dict) else []
    if not any(
        key.startswith("depth_encoder.")
        or key.startswith("metric_decoder.")
        for key in keys
    ):
        raise ValueError(f"Checkpoint is not Pi3X: {path}")


def window_starts(count: int, size: int, stride: int) -> list[int]:
    if count <= size:
        return [0]
    starts = list(range(0, count - size + 1, stride))
    final = count - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def main() -> None:
    args = parse_args()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    hand_uni_root = Path(args.hand_uni_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    if str(hand_uni_root) not in sys.path:
        sys.path.insert(0, str(hand_uni_root))
    from pi3_wilor_hand.factory import load_pi3x
    from pi3_wilor_hand.geometry import resize_intrinsics
    from pi3_wilor_hand.handsh_pipeline import Pi3XReconstructionBranch
    from pi3_wilor_hand.pi3_runner import load_images_for_pi3

    rows = load_rows(frame_map_path)
    frame_start = max(0, args.frame_start)
    frame_end = (
        len(rows) if args.frame_end < 0 else min(len(rows), args.frame_end)
    )
    if not frame_start < frame_end:
        raise ValueError(f"Invalid frame range [{frame_start}, {frame_end})")
    selected_rows = rows[frame_start:frame_end]

    with np.load(hand_path, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]
    intrinsics = intrinsics.reshape(3, 3)

    verify_pi3x_checkpoint(checkpoint)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    model = load_pi3x(
        pi3_root=args.pi3_root,
        ckpt=str(checkpoint),
        device=str(device),
    )
    reconstruction = Pi3XReconstructionBranch(
        model,
        freeze_pi3=True,
        use_intrinsics=True,
    ).to(device).eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_point_decoder(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["ret_point"] = output.detach()

    hook = model.point_decoder.register_forward_hook(capture_point_decoder)
    size = min(max(1, args.window_size), len(selected_rows))
    stride = max(1, min(args.window_stride, size))
    starts = window_starts(len(selected_rows), size, stride)
    feature_dtype = (
        np.float16 if args.feature_dtype == "float16" else np.float32
    )
    records: list[dict] = []

    try:
        for window_number, local_start in enumerate(starts):
            local_end = min(len(selected_rows), local_start + size)
            global_start = frame_start + local_start
            global_end = frame_start + local_end
            output_path = (
                windows_dir
                / f"window_{global_start:06d}_{global_end:06d}.npz"
            )
            if output_path.is_file() and not args.overwrite:
                print(
                    f"[{window_number + 1}/{len(starts)}] cached "
                    f"{global_start}:{global_end}",
                    flush=True,
                )
                records.append(
                    {
                        "start": global_start,
                        "end": global_end,
                        "path": str(output_path),
                    }
                )
                continue

            window_rows = selected_rows[local_start:local_end]
            image_paths = [
                str(Path(row["image_path"]).expanduser().resolve())
                for row in window_rows
            ]
            images, resized_wh, original_wh = load_images_for_pi3(
                image_paths,
                pixel_limit=args.pixel_limit,
            )
            K_resized = resize_intrinsics(
                intrinsics,
                original_wh,
                resized_wh,
            )
            images_device = images[None].to(device)
            intrinsics_device = (
                torch.from_numpy(K_resized)
                .to(device=device, dtype=torch.float32)[None, None]
                .repeat(1, len(window_rows), 1, 1)
            )

            print(
                f"[{window_number + 1}/{len(starts)}] Pi3X features "
                f"{global_start}:{global_end}",
                flush=True,
            )
            captured.clear()
            with torch.no_grad():
                outputs = reconstruction(
                    images_device,
                    intrinsics=intrinsics_device,
                )
            if "ret_point" not in captured:
                raise RuntimeError("Pi3X point_decoder hook was not called")

            ret_point = captured["ret_point"]
            point_tokens = ret_point[:, int(model.patch_start_idx):]
            resized_w, resized_h = resized_wh
            patch_h = int(resized_h) // int(model.patch_size)
            patch_w = int(resized_w) // int(model.patch_size)
            if point_tokens.shape[1] != patch_h * patch_w:
                raise RuntimeError(
                    f"Unexpected geometry token count {point_tokens.shape[1]} "
                    f"for grid {patch_h}x{patch_w}"
                )
            geometry_features = point_tokens.reshape(
                len(window_rows),
                patch_h,
                patch_w,
                point_tokens.shape[-1],
            )

            hand_coverage = []
            object_coverage = []
            for row in window_rows:
                segmentation = load_segmentation(
                    Path(row["label_path"]).expanduser().resolve()
                )
                hand_coverage.append(
                    patch_coverage(
                        segmentation,
                        args.hand_label,
                        resized_wh,
                        (patch_h, patch_w),
                    )
                )
                object_coverage.append(
                    patch_coverage(
                        segmentation,
                        args.object_label,
                        resized_wh,
                        (patch_h, patch_w),
                    )
                )

            confidence = torch.sigmoid(outputs["conf"][0, ..., 0])
            local_points = outputs["local_points"][0]
            metric = outputs.get("metric")
            np.savez_compressed(
                output_path,
                start=np.int32(global_start),
                end=np.int32(global_end),
                frame_indices=np.arange(
                    global_start,
                    global_end,
                    dtype=np.int32,
                ),
                geometry_patch_features=geometry_features
                .float()
                .cpu()
                .numpy()
                .astype(feature_dtype),
                geometry_feature_layer=np.asarray(
                    "point_decoder.final_output"
                ),
                geometry_feature_grid_hw=np.asarray(
                    [patch_h, patch_w], dtype=np.int32
                ),
                geometry_feature_dim=np.int32(
                    geometry_features.shape[-1]
                ),
                geometry_feature_dtype=np.asarray(args.feature_dtype),
                patch_size=np.int32(model.patch_size),
                patch_start_idx=np.int32(model.patch_start_idx),
                hand_patch_coverage=np.stack(hand_coverage),
                object_patch_coverage=np.stack(object_coverage),
                hand_patch_mask=(
                    np.stack(hand_coverage) >= 0.5
                ).astype(np.uint8),
                object_patch_mask=(
                    np.stack(object_coverage) >= 0.5
                ).astype(np.uint8),
                local_points=local_points
                .float()
                .cpu()
                .numpy()
                .astype(np.float16),
                confidence=confidence
                .float()
                .cpu()
                .numpy()
                .astype(np.float16),
                valid_mask=(
                    confidence >= args.confidence_threshold
                ).cpu().numpy().astype(np.uint8),
                camera_poses=outputs["camera_poses"][0]
                .float()
                .cpu()
                .numpy()
                .astype(np.float32),
                intrinsics_resized=K_resized.astype(np.float32),
                resized_wh=np.asarray(resized_wh, dtype=np.int32),
                original_wh=np.asarray(original_wh, dtype=np.int32),
                metric=(
                    np.asarray([], dtype=np.float32)
                    if metric is None
                    else metric.float().cpu().numpy().astype(np.float32)
                ),
                gt_intrinsics_conditioned=np.asarray(True),
                object_label=np.int32(args.object_label),
                hand_label=np.int32(args.hand_label),
            )
            records.append(
                {
                    "start": global_start,
                    "end": global_end,
                    "path": str(output_path),
                    "feature_shape": list(geometry_features.shape),
                    "size_bytes": output_path.stat().st_size,
                }
            )
    finally:
        hook.remove()

    summary = {
        "model": "Pi3X",
        "checkpoint": str(checkpoint),
        "frame_map_json": str(frame_map_path),
        "hand_npz": str(hand_path),
        "uses_gt_intrinsics": True,
        "feature_layer": "point_decoder.final_output",
        "feature_dtype": args.feature_dtype,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "num_frames": len(selected_rows),
        "window_size": size,
        "window_stride": stride,
        "object_label": args.object_label,
        "hand_label": args.hand_label,
        "windows": records,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: {summary_path}")


if __name__ == "__main__":
    main()
