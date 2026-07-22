#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


P3D_TO_ISAAC = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
R_ZUP_TO_YUP = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
R_YUP_TO_ZUP = R_ZUP_TO_YUP.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt an existing hand-uni SAM3D GLB/NPZ to do-as-i-do initialization files."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--shape_bank_root", required=True)
    parser.add_argument("--object_name", default=None)
    parser.add_argument("--source_glb", default=None)
    parser.add_argument("--source_npz", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_sources(args: argparse.Namespace, object_name: str) -> tuple[Path, Path]:
    if (args.source_glb is None) != (args.source_npz is None):
        raise ValueError("Pass both --source_glb and --source_npz, or neither")
    if args.source_glb is not None:
        return (
            Path(args.source_glb).expanduser().resolve(),
            Path(args.source_npz).expanduser().resolve(),
        )
    sam_dir = Path(args.shape_bank_root).expanduser().resolve() / "objects" / object_name / "sam3d"
    return sam_dir / "object_canonical_sam.glb", sam_dir / "sam3d_object_output.npz"


def load_canonical_mesh(glb_path: Path, payload: np.lib.npyio.NpzFile) -> trimesh.Trimesh:
    if "vertices" in payload.files and "faces" in payload.files:
        return trimesh.Trimesh(
            vertices=np.asarray(payload["vertices"], dtype=np.float32),
            faces=np.asarray(payload["faces"], dtype=np.int64),
            process=False,
        )
    loaded = trimesh.load(glb_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No triangle mesh in {glb_path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported GLB geometry: {type(loaded).__name__}")
    return loaded


def vector(payload: np.lib.npyio.NpzFile, key: str, size: int) -> np.ndarray:
    if key not in payload.files:
        raise KeyError(f"Missing {key} in SAM3D NPZ; available keys={payload.files}")
    value = np.asarray(payload[key], dtype=np.float32).reshape(-1)
    if value.size == 1 and size == 3:
        value = np.repeat(value, 3)
    if value.size != size:
        raise ValueError(f"Expected {key} to contain {size} values, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite {key}: {value}")
    return value


def compute_layout(mesh: trimesh.Trimesh, rotation: np.ndarray, translation: np.ndarray, scale: np.ndarray) -> dict:
    # SAM3D stores quaternions as (w, x, y, z). PyTorch3D Transform3d uses
    # row vectors; scale().rotate().translate() therefore evaluates as
    # p' = p @ (diag(scale) @ R) + translation.
    quat_xyzw = np.asarray([rotation[1], rotation[2], rotation[3], rotation[0]])
    rotation_matrix = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    linear = np.diag(scale.astype(np.float32)) @ rotation_matrix

    raw_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertices_zup = raw_vertices @ R_YUP_TO_ZUP
    transformed = vertices_zup @ linear + translation.reshape(1, 3)
    transformed_isaac = transformed @ P3D_TO_ISAAC
    matrix, _, _ = trimesh.registration.procrustes(
        raw_vertices,
        transformed_isaac,
        reflection=False,
        return_cost=True,
    )
    new_quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    new_quat = np.asarray(
        [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]],
        dtype=np.float32,
    )
    matrix_4x4 = np.eye(4, dtype=np.float32)
    matrix_4x4[:3, :3] = linear
    matrix_4x4[3, :3] = translation
    return {
        "translation": translation.tolist(),
        "scale": scale.tolist(),
        "quat_wxyz": rotation.tolist(),
        "new_quat": new_quat.tolist(),
        "quat_xyzw": quat_xyzw.tolist(),
        "matrix_4x4_row_major": matrix_4x4.reshape(-1).astype(float).tolist(),
    }


def replace_symlink(link_path: Path, target: Path, overwrite: bool) -> None:
    if link_path.is_symlink() or link_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {link_path}; pass --overwrite")
        link_path.unlink()
    link_path.symlink_to(target)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    frame_map_path = run_dir / "dexycb_frame_map.json"
    if not frame_map_path.is_file():
        raise FileNotFoundError(f"Missing prepared DexYCB frame map: {frame_map_path}")
    frame_map = json.loads(frame_map_path.read_text(encoding="utf-8"))
    object_name = str(args.object_name or frame_map["object_name"])
    init_index = int(frame_map["init_frame"]["output_index"])
    masks_dir = run_dir / "video_segmentation" / "masks" / f"frame_{init_index:06d}_masks"
    object_dir = masks_dir / object_name
    object_dir.mkdir(parents=True, exist_ok=True)

    glb_path, npz_path = resolve_sources(args, object_name)
    for source in (glb_path, npz_path):
        if not source.is_file():
            raise FileNotFoundError(f"Missing SAM3D source: {source}")
    with np.load(npz_path, allow_pickle=True) as payload:
        mesh = load_canonical_mesh(glb_path, payload)
        rotation = vector(payload, "rotation", 4)
        translation = vector(payload, "translation", 3)
        scale = vector(payload, "scale", 3)

    obj_path = object_dir / f"{object_name}.obj"
    layout_path = masks_dir / "layout.json"
    if not args.overwrite:
        for output in (obj_path, layout_path):
            if output.exists():
                raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    mesh.export(obj_path)
    replace_symlink(object_dir / "sam3d_object_output.npz", npz_path, args.overwrite)
    replace_symlink(object_dir / "object_canonical_sam.glb", glb_path, args.overwrite)

    layout = {
        "frame": "sam3d_scene",
        "note": "Adapted from an existing hand-uni SAM3D output; transform convention matches generate_mesh_sam3d.py.",
        "objects": [
            {
                "index": 0,
                "mesh_obj": f"{object_name}.obj",
                "local_to_scene": compute_layout(mesh, rotation, translation, scale),
            }
        ],
    }
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    summary = {
        "object_name": object_name,
        "init_output_frame": f"{init_index:06d}",
        "source_glb": str(glb_path),
        "source_npz": str(npz_path),
        "canonical_obj": str(obj_path),
        "layout_json": str(layout_path),
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "scale": scale.tolist(),
    }
    (object_dir / "shape_adapter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
