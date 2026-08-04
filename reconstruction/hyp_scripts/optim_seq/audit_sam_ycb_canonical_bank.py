#!/usr/bin/env python3
"""Audit fixed canonical alignments between a SAM3D and YCB object bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from align_sam_mesh_to_dexycb_cad import (
    alignment_score,
    load_mesh,
    orientation_candidates,
    run_icp,
    sample_surface,
    transform,
)


YCB_MESH_CANDIDATES = (
    "textured_simple.obj",
    "textured.obj",
    "nontextured.ply",
    "model.obj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-root", required=True)
    parser.add_argument("--ycb-model-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Optional JSONL manifest; repeat for train/val.",
    )
    parser.add_argument(
        "--exclude-object",
        action="append",
        default=[],
        help="Object directory name to skip; repeat as needed.",
    )
    parser.add_argument(
        "--include-unreferenced",
        action="store_true",
        help="Also audit bank objects not referenced by the supplied manifests.",
    )
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--trim-fraction", type=float, default=0.7)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--preview-points", type=int, default=16000)
    parser.add_argument("--max-rmse-mm", type=float, default=10.0)
    parser.add_argument("--max-scale-relative-error", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_ycb_mesh(object_dir: Path) -> Path | None:
    for name in YCB_MESH_CANDIDATES:
        path = object_dir / name
        if path.is_file():
            return path
    return None


def discover_ycb_meshes(root: Path) -> dict[str, Path]:
    output = {}
    for object_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        mesh = find_ycb_mesh(object_dir)
        if mesh is not None:
            output[object_dir.name] = mesh
    return output


def discover_sam_meshes(
    root: Path, object_names: set[str]
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    candidates: dict[str, list[Path]] = {}
    for path in root.rglob("object_canonical_sam.glb"):
        matches = [part for part in path.parts if part in object_names]
        if not matches:
            continue
        candidates.setdefault(matches[-1], []).append(path.resolve())

    selected = {}
    duplicates = {}
    for name, paths in candidates.items():
        ordered = sorted(set(paths), key=lambda path: (len(path.parts), str(path)))
        selected[name] = ordered[0]
        if len(ordered) > 1:
            duplicates[name] = [str(path) for path in ordered]
    return selected, duplicates


def load_production_scales(paths: list[Path]) -> dict[str, dict]:
    values: dict[str, list[float]] = {}
    sources: dict[str, set[str]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                name = row.get("object_name")
                scale = row.get("foundationpose_source_mesh_scale")
                if name is None or scale is None:
                    continue
                scale = float(scale)
                if not np.isfinite(scale) or scale <= 0:
                    continue
                values.setdefault(str(name), []).append(scale)
                sources.setdefault(str(name), set()).add(str(path))

    output = {}
    for name, rows in values.items():
        array = np.asarray(rows, dtype=np.float64)
        output[name] = {
            "count": int(len(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "sources": sorted(sources[name]),
        }
    return output


def align_meshes(
    sam_mesh: trimesh.Trimesh,
    ycb_mesh: trimesh.Trimesh,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray]:
    source = sample_surface(sam_mesh, args.samples, rng)
    target = sample_surface(ycb_mesh, args.samples, rng)
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    source_radius = np.sqrt(np.mean(np.sum(source_centered**2, axis=1)))
    target_radius = np.sqrt(np.mean(np.sum(target_centered**2, axis=1)))
    initial_scale = float(target_radius / max(source_radius, 1e-12))

    candidates = []
    rotations = orientation_candidates(source, target)[: args.max_candidates]
    for initial_rotation in rotations:
        scale, rotation, translation = run_icp(
            source,
            target,
            initial_rotation,
            args.icp_iterations,
            args.trim_fraction,
            initial_scale,
        )
        score, metrics = alignment_score(
            source,
            target,
            scale,
            rotation,
            translation,
            args.trim_fraction,
        )
        candidates.append(
            {
                "score": float(score),
                "scale": float(scale),
                "rotation": rotation,
                "translation": translation,
                "metrics": metrics,
            }
        )
    if not candidates:
        raise RuntimeError("No valid orientation candidates")
    candidates.sort(key=lambda row: row["score"])
    best = candidates[0]
    best["top_candidates"] = [
        {
            "rank": index + 1,
            "scale": row["scale"],
            **row["metrics"],
        }
        for index, row in enumerate(candidates[:5])
    ]
    return best, source, target


def export_preview(
    path: Path,
    source: np.ndarray,
    target: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> None:
    source = transform(source, scale, rotation, translation)
    if len(source) > count:
        source = source[rng.choice(len(source), count, replace=False)]
    if len(target) > count:
        target = target[rng.choice(len(target), count, replace=False)]
    points = np.concatenate([source, target], axis=0)
    colors = np.empty((len(points), 4), dtype=np.uint8)
    colors[: len(source)] = np.asarray([255, 40, 180, 255], dtype=np.uint8)
    colors[len(source) :] = np.asarray([20, 220, 230, 255], dtype=np.uint8)
    trimesh.points.PointCloud(points, colors=colors).export(path)


def matrix(rotation: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value.tolist()


def main() -> None:
    args = parse_args()
    sam_root = Path(args.sam_root).expanduser().resolve()
    ycb_root = Path(args.ycb_model_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ycb_meshes = discover_ycb_meshes(ycb_root)
    sam_meshes, duplicates = discover_sam_meshes(
        sam_root, set(ycb_meshes)
    )
    production_scales = load_production_scales(
        [Path(path).expanduser().resolve() for path in args.manifest]
    )
    excluded = set(args.exclude_object)
    matched_names = set(sam_meshes) & set(ycb_meshes)
    if args.manifest and not args.include_unreferenced:
        matched_names &= set(production_scales)
    object_names = sorted(matched_names - excluded)
    rng = np.random.default_rng(args.seed)
    rows = []
    failures = []

    for index, name in enumerate(object_names, start=1):
        object_out = out_root / name
        summary_path = object_out / "canonical_alignment.json"
        if summary_path.is_file() and not args.overwrite:
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"[{index}/{len(object_names)}] cached {name}", flush=True)
            continue
        object_out.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(object_names)}] aligning {name}", flush=True)
        try:
            sam_mesh = load_mesh(sam_meshes[name])
            ycb_mesh = load_mesh(ycb_meshes[name])
            best, source, target = align_meshes(
                sam_mesh, ycb_mesh, args, rng
            )
            fitted_scale = float(best["scale"])
            rotation = np.asarray(best["rotation"], dtype=np.float64)
            translation = np.asarray(best["translation"], dtype=np.float64)
            scale_record = production_scales.get(name)
            production_scale = (
                float(scale_record["median"]) if scale_record else None
            )
            residual_scale = (
                fitted_scale / production_scale
                if production_scale is not None
                else None
            )
            scale_ok = (
                abs(residual_scale - 1.0)
                <= args.max_scale_relative_error
                if residual_scale is not None
                else None
            )
            rmse = float(best["metrics"]["trimmed_symmetric_rmse_mm"])
            geometry_ok = rmse <= args.max_rmse_mm
            valid = bool(geometry_ok and scale_ok is not False)

            aligned = sam_mesh.copy()
            aligned.vertices = transform(
                np.asarray(aligned.vertices, dtype=np.float64),
                fitted_scale,
                rotation,
                translation,
            )
            aligned.export(object_out / "sam_aligned_to_ycb.obj")
            export_preview(
                object_out / "canonical_overlay.ply",
                source,
                target,
                fitted_scale,
                rotation,
                translation,
                args.preview_points,
                rng,
            )

            row = {
                "object_name": name,
                "valid": valid,
                "quality": {
                    "geometry_ok": geometry_ok,
                    "production_scale_ok": scale_ok,
                    "max_rmse_mm": args.max_rmse_mm,
                    "max_scale_relative_error": args.max_scale_relative_error,
                },
                "sam_mesh": str(sam_meshes[name]),
                "ycb_mesh": str(ycb_meshes[name]),
                "raw_sam_to_ycb_similarity": {
                    "scale": fitted_scale,
                    "rotation": rotation.tolist(),
                    "translation_m": translation.tolist(),
                    "rigid_matrix_without_scale": matrix(
                        rotation, translation
                    ),
                },
                "production_sam_to_ycb_similarity": {
                    "production_sam_scale": production_scale,
                    "residual_scale": residual_scale,
                    "rotation": rotation.tolist(),
                    "translation_m": translation.tolist(),
                    "scale_observations": scale_record,
                },
                "surface_alignment": best["metrics"],
                "sam_raw_extents": np.asarray(sam_mesh.extents).tolist(),
                "sam_fitted_metric_extents": (
                    np.asarray(sam_mesh.extents) * fitted_scale
                ).tolist(),
                "ycb_metric_extents": np.asarray(ycb_mesh.extents).tolist(),
                "top_candidates": best["top_candidates"],
                "aligned_mesh": str(object_out / "sam_aligned_to_ycb.obj"),
                "overlay_ply": str(object_out / "canonical_overlay.ply"),
            }
            summary_path.write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            rows.append(row)
            print(
                f"  rmse={rmse:.3f}mm scale={fitted_scale:.6f} "
                f"residual_scale={residual_scale} valid={valid}",
                flush=True,
            )
        except Exception as error:
            failure = {"object_name": name, "error": repr(error)}
            failures.append(failure)
            print(f"  failed: {error!r}", flush=True)

    status_counts = {
        "valid": sum(bool(row.get("valid")) for row in rows),
        "invalid": sum(not bool(row.get("valid")) for row in rows),
        "failed": len(failures),
    }
    summary = {
        "sam_root": str(sam_root),
        "ycb_model_root": str(ycb_root),
        "manifests": [str(Path(path).expanduser().resolve()) for path in args.manifest],
        "num_ycb_objects": len(ycb_meshes),
        "num_sam_objects": len(sam_meshes),
        "num_matched_objects": len(object_names),
        "excluded_objects": sorted(excluded),
        "missing_sam_objects": sorted(set(ycb_meshes) - set(sam_meshes)),
        "sam_mesh_duplicates": duplicates,
        "status_counts": status_counts,
        "objects": rows,
        "failures": failures,
        "settings": vars(args),
    }
    summary_path = out_root / "canonical_bank_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**status_counts, "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
