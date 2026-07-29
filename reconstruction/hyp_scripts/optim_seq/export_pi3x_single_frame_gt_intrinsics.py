#!/usr/bin/env python3
"""Export one Pi3X pointmap conditioned on the dataset camera intrinsics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-map-json", required=True)
    parser.add_argument("--hand-npz", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--hand-uni-root", required=True)
    parser.add_argument("--pi3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pixel-limit", type=int, default=180000)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload.get("frame_map", payload))
    if isinstance(rows, dict):
        rows = list(rows.values())
    return sorted(rows, key=lambda row: int(row["output_index"]))


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
        raise ValueError(
            f"Checkpoint is not Pi3X (missing multimodal/metric keys): {path}"
        )


def main() -> None:
    args = parse_args()
    frame_map_path = Path(args.frame_map_json).expanduser().resolve()
    hand_path = Path(args.hand_npz).expanduser().resolve()
    hand_uni_root = Path(args.hand_uni_root).expanduser().resolve()
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
    if not 0 <= args.frame < len(rows):
        raise IndexError(f"Frame {args.frame} outside [0, {len(rows) - 1}]")
    image_path = str(
        Path(rows[args.frame]["image_path"]).expanduser().resolve()
    )
    with np.load(hand_path, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[min(args.frame, len(intrinsics) - 1)]
    intrinsics = intrinsics.reshape(3, 3)

    images, resized_wh, original_wh = load_images_for_pi3(
        [image_path], pixel_limit=args.pixel_limit
    )
    K_resized = resize_intrinsics(intrinsics, original_wh, resized_wh)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    verify_pi3x_checkpoint(checkpoint)
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
    point_decoder_output: dict[str, torch.Tensor] = {}

    def capture_point_decoder(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        point_decoder_output["tokens"] = output.detach()

    point_decoder_hook = model.point_decoder.register_forward_hook(
        capture_point_decoder
    )
    images_device = images[None].to(device)
    intrinsics_device = (
        torch.from_numpy(K_resized)
        .to(device=device, dtype=torch.float32)[None, None]
    )

    output_path = (
        windows_dir
        / f"window_{args.frame:06d}_{args.frame + 1:06d}.npz"
    )
    feature_summary: dict[str, object]
    if output_path.is_file() and not args.overwrite:
        point_decoder_hook.remove()
        with np.load(output_path, allow_pickle=False) as cached:
            if "geometry_patch_features" not in cached:
                raise RuntimeError(
                    f"Cached file has no geometry features: {output_path}. "
                    "Run again with --overwrite."
                )
            cached_features = cached["geometry_patch_features"]
            feature_summary = {
                "key": "geometry_patch_features",
                "layer": str(cached["geometry_feature_layer"]),
                "shape": list(cached_features.shape),
                "dtype": str(cached_features.dtype),
                "patch_size": int(model.patch_size),
                "patch_start_idx": int(model.patch_start_idx),
            }
        print(f"Cached: {output_path}")
    else:
        print(
            f"Running Pi3X frame {args.frame:06d} with GT intrinsics",
            flush=True,
        )
        with torch.no_grad():
            outputs = reconstruction(
                images_device,
                intrinsics=intrinsics_device,
            )
        point_decoder_hook.remove()
        if "tokens" not in point_decoder_output:
            raise RuntimeError("Pi3X point_decoder feature hook was not called")
        point_tokens = point_decoder_output["tokens"]
        patch_start_idx = int(model.patch_start_idx)
        point_tokens = point_tokens[:, patch_start_idx:]
        resized_w, resized_h = resized_wh
        patch_h = int(resized_h) // int(model.patch_size)
        patch_w = int(resized_w) // int(model.patch_size)
        expected_tokens = patch_h * patch_w
        if point_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected Pi3X geometry token count: "
                f"{point_tokens.shape[1]} != {patch_h}*{patch_w}"
            )
        geometry_features = point_tokens.reshape(
            1,
            patch_h,
            patch_w,
            point_tokens.shape[-1],
        )
        feature_dtype = (
            np.float16 if args.feature_dtype == "float16" else np.float32
        )
        feature_summary = {
            "key": "geometry_patch_features",
            "layer": "point_decoder.final_output",
            "shape": list(geometry_features.shape),
            "dtype": args.feature_dtype,
            "patch_size": int(model.patch_size),
            "patch_start_idx": int(model.patch_start_idx),
        }
        confidence = torch.sigmoid(outputs["conf"][0, 0, ..., 0])
        local_points = outputs["local_points"][0, 0]
        camera_pose = outputs["camera_poses"][0, 0]
        metric = outputs.get("metric")
        np.savez_compressed(
            output_path,
            start=np.int32(args.frame),
            end=np.int32(args.frame + 1),
            frame_indices=np.asarray([args.frame], dtype=np.int32),
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
            geometry_feature_dim=np.int32(geometry_features.shape[-1]),
            geometry_feature_dtype=np.asarray(args.feature_dtype),
            local_points=local_points[None]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float16),
            confidence=confidence[None]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float16),
            valid_mask=(
                confidence[None] >= args.confidence_threshold
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8),
            camera_poses=camera_pose[None]
            .detach()
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
                else metric.detach().float().cpu().numpy().astype(np.float32)
            ),
            gt_intrinsics_conditioned=np.asarray(True),
        )

    summary = {
        "model": "Pi3X",
        "checkpoint": str(checkpoint),
        "frame": args.frame,
        "image_path": image_path,
        "uses_gt_intrinsics": True,
        "K_original": intrinsics.tolist(),
        "K_resized": K_resized.tolist(),
        "resized_wh": list(resized_wh),
        "geometry_feature": feature_summary,
        "pointmap": str(output_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
