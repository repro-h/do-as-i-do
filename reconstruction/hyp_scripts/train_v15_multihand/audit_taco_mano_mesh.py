#!/usr/bin/env python3
"""Compare viewer geometry against TACO's manopth decoder, without inference."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np


def distance_summary(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.shape[-1] != 3:
        raise ValueError(f"Geometry shapes disagree: {first.shape}, {second.shape}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite geometry")
    errors = np.linalg.norm(first - second, axis=-1) * 1000.0
    return {
        "count": int(errors.size),
        "median_mm": float(np.median(errors)),
        "p90_mm": float(np.percentile(errors, 90)),
        "max_mm": float(errors.max()),
    }


def compare_faces(first, second):
    def canonical(faces, oriented):
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Invalid faces shape: {faces.shape}")
        if oriented:
            return sorted(min(tuple(np.roll(row, i)) for i in range(3)) for row in faces)
        return sorted(tuple(sorted(row)) for row in faces)

    return {
        "exact": bool(np.array_equal(first, second)),
        "same_triangles": canonical(first, False) == canonical(second, False),
        "same_winding": canonical(first, True) == canonical(second, True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taco-root", required=True)
    parser.add_argument("--taco-code-root", required=True)
    parser.add_argument("--mano-model-folder", required=True)
    parser.add_argument("--triplet", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    # Import the actual viewer implementation so this audit checks its decoding.
    from prepare_taco_v15 import as_numpy, create_mano_models, load_pickle
    from visualize_taco_v16_mano_object import mano_faces, mano_local_geometry
    import torch
    import trimesh

    code_root = Path(args.taco_code_root).expanduser().resolve()
    root = Path(args.taco_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    print("Loading viewer SMPL-X and TACO manopth models...", flush=True)
    _, viewer_models = create_mano_models(args.mano_model_folder)
    _, official_models = create_mano_models(args.mano_model_folder, code_root)
    implementation = Path(inspect.getfile(type(official_models["right"]))).resolve()
    if code_root not in implementation.parents:
        raise RuntimeError(f"manopth was imported outside TACO code root: {implementation}")
    out.mkdir(parents=True, exist_ok=True)

    def official_geometry(model, pose, betas):
        with torch.no_grad():
            result = model(
                torch.from_numpy(as_numpy(pose).reshape(1, 48)),
                torch.from_numpy(as_numpy(betas).reshape(1, 10)),
            )
        vertices, joints = (as_numpy(item[0]) / 1000.0 for item in result[:2])
        if vertices.shape != (778, 3) or joints.shape != (21, 3):
            raise ValueError(f"Unexpected official output: {vertices.shape}, {joints.shape}")
        # Official center_idx=0 already centers both arrays at the wrist.
        return vertices, joints

    report = {
        "sequence": args.sequence,
        "triplet": args.triplet,
        "mano_model_folder": str(Path(args.mano_model_folder).resolve()),
        "official_implementation": str(implementation),
        "coordinate_frame": "wrist_centered_annotation_axes_meters",
        "settings": {"use_pca": False, "flat_hand_mean": True, "center_idx": 0},
        "sides": {},
    }
    for side in ("left", "right"):
        hand_root = root / "Hand_Poses" / args.triplet / args.sequence
        poses = load_pickle(hand_root / f"{side}_hand.pkl")
        betas = as_numpy(load_pickle(hand_root / f"{side}_hand_shape.pkl")["hand_shape"])
        if not poses:
            raise ValueError(f"Empty hand poses: {side}")
        viewer = viewer_models[side]
        official = official_models[side]
        viewer_faces = mano_faces(viewer, 778)
        official_faces = mano_faces(official, 778)
        face_report = compare_faces(viewer_faces, official_faces)
        collected = {name: [] for name in ("viewer_vertices", "official_vertices", "viewer_joints", "official_joints")}
        per_frame = []
        raw_poses = []
        pose_shapes = set()
        for index, key in enumerate(sorted(poses, key=str)):
            pose = as_numpy(poses[key]["hand_pose"])
            if pose.size != 48 or not np.isfinite(pose).all():
                raise ValueError(f"Invalid hand_pose at {side}/{key}: {pose.shape}")
            pose_shapes.add(tuple(pose.shape))
            raw_poses.append(pose.reshape(48))
            vv, vj = mano_local_geometry("smplx", viewer, side, pose, betas)
            ov, oj = official_geometry(official, pose, betas)
            for name, value in zip(collected, (vv, ov, vj, oj)):
                collected[name].append(value)
            per_frame.append({
                "frame_index": index, "annotation_key": str(key),
                "vertices": distance_summary(vv, ov),
                "joints": distance_summary(vj, oj),
            })
            if index % 25 == 0 or index + 1 == len(poses):
                print(f"{side}: {index + 1}/{len(poses)} frames compared", flush=True)
        worst = max(per_frame, key=lambda row: row["vertices"]["max_mm"])
        worst_index = worst["frame_index"]
        for backend, faces in (("viewer", viewer_faces), ("official", official_faces)):
            trimesh.Trimesh(
                vertices=collected[f"{backend}_vertices"][worst_index], faces=faces, process=False,
            ).export(out / f"{side}_{backend}_worst.obj")

        controls = {}
        for name, shape in (("zero_pose_dataset_shape", betas), ("zero_pose_zero_shape", np.zeros(10, dtype=np.float32))):
            pose = np.zeros(48, dtype=np.float32)
            vv, vj = mano_local_geometry("smplx", viewer, side, pose, shape)
            ov, oj = official_geometry(official, pose, shape)
            controls[name] = {"vertices": distance_summary(vv, ov), "joints": distance_summary(vj, oj)}
            for backend, vertices, faces in (("viewer", vv, viewer_faces), ("official", ov, official_faces)):
                trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
                    out / f"{side}_{backend}_{name}.obj"
                )
        report["sides"][side] = {
            "frames": len(poses), "faces": face_report,
            "raw_hand_pose": {
                "shapes": sorted(pose_shapes),
                "min": float(np.min(raw_poses)),
                "max": float(np.max(raw_poses)),
                "first_frame_values": raw_poses[0].tolist(),
                "interpretation": "3 global + 45 local axis-angle values; PCA disabled as in TACO loader",
            },
            "vertices": distance_summary(collected["viewer_vertices"], collected["official_vertices"]),
            "joints": distance_summary(collected["viewer_joints"], collected["official_joints"]),
            "worst_frame": worst, "controls": controls,
        }
        (out / f"{side}_per_frame.json").write_text(json.dumps(per_frame, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Mesh comparison saved to {out}", flush=True)


if __name__ == "__main__":
    main()
