#!/usr/bin/env python3
"""Apply object-frame hand pose residuals to HandFlow camera meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--supervision-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def object_delta_to_camera(
    object_pose: np.ndarray,
    initial_t: np.ndarray,
    initial_r: np.ndarray,
    predicted_t: np.ndarray,
    predicted_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_co = object_pose[:, :3, :3]
    t_co = object_pose[:, :3, 3]

    r_delta_o = predicted_r @ np.swapaxes(initial_r, -1, -2)
    t_delta_o = predicted_t - np.einsum("nij,nj->ni", r_delta_o, initial_t)

    r_delta_c = r_co @ r_delta_o @ np.swapaxes(r_co, -1, -2)
    t_delta_c = (
        t_co
        - np.einsum("nij,nj->ni", r_delta_c, t_co)
        + np.einsum("nij,nj->ni", r_co, t_delta_o)
    )
    return r_delta_c, t_delta_c


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction_npz).expanduser().resolve()
    supervision_path = Path(args.supervision_npz).expanduser().resolve()
    output_path = Path(args.out_npz).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)

    prediction = load_npz(prediction_path)
    supervision = load_npz(supervision_path)
    raw_path = Path(str(prediction["handflow_camera_result"].item()))
    raw = load_npz(raw_path)

    raw_vertices = np.asarray(raw["verts_cam"], dtype=np.float32)
    object_pose = np.asarray(supervision["object_pose"], dtype=np.float32)
    initial_t = np.asarray(prediction["initial_translation_object"], dtype=np.float32)
    initial_r = np.asarray(prediction["initial_rotation_object"], dtype=np.float32)
    predicted_t = np.asarray(prediction["predicted_translation_object"], dtype=np.float32)
    predicted_r = np.asarray(prediction["predicted_rotation_object"], dtype=np.float32)

    count = min(
        len(raw_vertices), len(object_pose), len(initial_t), len(predicted_t)
    )
    r_delta, t_delta = object_delta_to_camera(
        object_pose[:count],
        initial_t[:count],
        initial_r[:count],
        predicted_t[:count],
        predicted_r[:count],
    )
    corrected = np.asarray(raw_vertices[:count]).copy()
    corrected = np.einsum("nvi,nji->nvj", corrected, r_delta)
    corrected += t_delta[:, None]

    valid = np.asarray(
        prediction.get("camera_mesh_correction_valid", np.ones(count, dtype=bool)),
        dtype=bool,
    )[:count]
    valid &= np.isfinite(corrected).all(axis=(1, 2))

    output = dict(prediction)
    output["verts_cam"] = raw_vertices[:count].copy()
    output["verts_cam"][valid] = corrected[valid]
    output["camera_mesh_correction_valid"] = valid
    output["camera_mesh_correction_source"] = np.asarray(
        "handflow_camera_mesh_object_frame_delta"
    )
    output["camera_mesh_delta_rotation"] = r_delta.astype(np.float32)
    output["camera_mesh_delta_translation"] = t_delta.astype(np.float32)
    output["camera_mesh_left_mirror_applied"] = np.asarray(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print("prediction:", prediction_path)
    print("supervision:", supervision_path)
    print("output:", output_path)
    print("frames:", count)
    print("valid:", int(valid.sum()), "/", len(valid))
    print("left mirror applied: False")


if __name__ == "__main__":
    main()
