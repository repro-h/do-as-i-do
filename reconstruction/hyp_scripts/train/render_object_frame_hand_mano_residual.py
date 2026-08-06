#!/usr/bin/env python3
"""Render object-frame hand residual predictions through MANO.

The input prediction NPZ contains object-frame hand root poses, while the
cached HandFlow vertices are already posed in camera coordinates.  This tool
rebuilds a zero-global-orientation MANO mesh from HandFlow's raw hand pose and
then applies the predicted object-frame root pose exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--mano-data-dir", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-handflow",
        action="store_true",
        help="Compare a raw MANO reconstruction with cached HandFlow verts.",
    )
    return parser.parse_args()


def rotation_matrices(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec)
    num_rotations = rotvec.shape[-1] // 3
    if rotvec.shape[-1] != num_rotations * 3:
        raise ValueError(f"Rotation vector width must be divisible by 3: {rotvec.shape}")
    matrices = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    return matrices.reshape(rotvec.shape[:-1] + (num_rotations, 3, 3))


def load_mano(mano_dir: Path, device: torch.device):
    # HandFlow stores raw parameters from the right-hand model for mirrored
    # left-hand inputs.  Left normalization is applied after rendering.
    layer = smplx.MANOLayer(
        model_path=str(mano_dir),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
    )
    layer = layer.to(device).eval()
    return layer


def mano_local_vertices(
    layer,
    pose: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    count = len(pose)
    hand_rot = rotation_matrices(pose[:, 3:]).astype(np.float32)
    zero_global = np.broadcast_to(np.eye(3, dtype=np.float32), (count, 1, 3, 3))
    beta_batch = np.broadcast_to(
        betas.astype(np.float32)[None], (count, betas.shape[-1])
    )
    vertices = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            output = layer(
                global_orient=torch.from_numpy(zero_global[start:end]).to(device),
                hand_pose=torch.from_numpy(hand_rot[start:end]).to(device),
                betas=torch.from_numpy(beta_batch[start:end]).to(device),
                pose2rot=False,
            )
            vertices.append(output.vertices.detach().cpu().numpy())
    return np.concatenate(vertices, axis=0).astype(np.float32)


def apply_object_pose(
    local_vertices: np.ndarray,
    object_pose: np.ndarray,
    hand_translation_object: np.ndarray,
    hand_rotation_object: np.ndarray,
) -> np.ndarray:
    # Row-vector equivalent of T_camera_object @ T_object_hand.
    vertices_object = (
        local_vertices @ hand_rotation_object.transpose(0, 2, 1)
        + hand_translation_object[:, None]
    )
    return (
        vertices_object @ object_pose[:, :3, :3].transpose(0, 2, 1)
        + object_pose[:, None, :3, 3]
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()
    output_path = Path(args.out_npz).expanduser().resolve()
    if output_path.is_file() and not args.overwrite:
        raise FileExistsError(output_path)

    with np.load(prediction_path, allow_pickle=False) as prediction:
        payload = {key: np.asarray(prediction[key]) for key in prediction.files}
    with np.load(supervision_path, allow_pickle=False) as supervision:
        object_pose = np.asarray(supervision["object_pose"], dtype=np.float32)
        normalized_left = bool(np.asarray(supervision["normalized_left"]).item())

    raw_path = Path(str(payload["handflow_camera_result"].item()))
    with np.load(raw_path, allow_pickle=False) as raw:
        raw_pose = np.asarray(raw["handflow_raw_pose"], dtype=np.float32)
        raw_betas = np.asarray(raw["handflow_raw_betas"], dtype=np.float32)
        raw_vertices = np.asarray(raw["verts_cam"], dtype=np.float32)

    count = min(
        len(raw_pose),
        len(object_pose),
        len(payload["predicted_translation_object"]),
    )
    raw_pose = raw_pose[:count]
    raw_betas = raw_betas[:count]
    object_pose = object_pose[:count]
    raw_vertices = raw_vertices[:count]

    device = torch.device(args.device)
    layer = load_mano(Path(args.mano_data_dir).expanduser().resolve(), device)
    local_vertices = mano_local_vertices(
        layer,
        raw_pose,
        raw_betas,
        device,
        args.batch_size,
    )

    # Raw HandFlow left inputs were mirrored in camera coordinates.  Apply the
    # same reflection to the reconstructed normalized-left mesh and its faces
    # are handled by the visualization adapter.
    if normalized_left:
        local_vertices[..., 0] *= -1.0

    predicted_t = np.asarray(
        payload["predicted_translation_object"][:count], dtype=np.float32
    )
    predicted_r = np.asarray(
        payload["predicted_rotation_object"][:count], dtype=np.float32
    )
    rendered = apply_object_pose(
        local_vertices,
        object_pose,
        predicted_t,
        predicted_r,
    )

    valid = np.asarray(
        payload.get("camera_mesh_correction_valid", np.ones(count, dtype=bool)),
        dtype=bool,
    )[:count]
    valid &= np.isfinite(rendered).all(axis=(1, 2))
    output_vertices = raw_vertices.copy()
    output_vertices[valid] = rendered[valid]

    if args.validate_handflow:
        raw_global = rotation_matrices(raw_pose[:, :3])[:, 0]
        raw_translation = np.asarray(
            np.load(raw_path, allow_pickle=False)["handflow_raw_trans"],
            dtype=np.float32,
        )[:count]
        if normalized_left:
            mirror = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
            raw_global = mirror[None] @ raw_global @ mirror[None]
            raw_translation = raw_translation @ mirror.T
        raw_rebuilt = (
            local_vertices @ raw_global.transpose(0, 2, 1)
            + raw_translation[:, None]
        )
        reconstruction_error = np.linalg.norm(
            raw_rebuilt - raw_vertices, axis=-1
        )
        payload["mano_raw_reconstruction_rmse_mm"] = np.asarray(
            np.sqrt(np.mean(reconstruction_error**2, axis=1)) * 1000.0,
            dtype=np.float32,
        )

    payload["verts_cam"] = output_vertices.astype(np.float32)
    payload["mano_render_valid"] = valid
    payload["mano_render_source"] = np.asarray("raw_pose_betas_mano")
    payload["mano_render_supervision"] = np.asarray(str(supervision_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    summary = {
        "prediction": str(prediction_path),
        "supervision": str(supervision_path),
        "output": str(output_path),
        "frames": int(count),
        "normalized_left": normalized_left,
        "render_valid": int(valid.sum()),
    }
    if "mano_raw_reconstruction_rmse_mm" in payload:
        values = payload["mano_raw_reconstruction_rmse_mm"]
        finite = values[np.isfinite(values)]
        summary["raw_reconstruction_rmse_mm"] = {
            "median": float(np.median(finite)) if len(finite) else None,
            "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
            "max": float(np.max(finite)) if len(finite) else None,
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
