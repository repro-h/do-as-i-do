#!/usr/bin/env python3
"""Create a canonical bank revision with approved pose-consensus overrides."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Approved override in OBJECT_NAME=/path/alignment_summary.json form.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_overrides(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid override: {value}")
        name, path = value.split("=", 1)
        output[name] = Path(path).expanduser().resolve()
    return output


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    profile_path = Path(args.profile_json).expanduser().resolve()
    profiles = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    excluded = set(profiles.get("excluded_objects", []))
    overrides = parse_overrides(args.override)

    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    missing_overrides = [
        path for path in overrides.values() if not path.is_file()
    ]
    if missing_overrides:
        raise FileNotFoundError(
            f"Missing override files: {missing_overrides}"
        )

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    elif out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(
            f"Output is not empty; pass --overwrite: {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        name = source_dir.name
        if name in excluded:
            continue
        source_json = source_dir / "canonical_alignment.json"
        if not source_json.is_file():
            continue
        target_dir = out_root / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        payload = json.loads(source_json.read_text(encoding="utf-8"))
        override = overrides.get(name)
        if override is not None:
            approved = json.loads(override.read_text(encoding="utf-8"))
            matrix = np.asarray(
                approved["sam_to_ycb_rigid"], dtype=np.float64
            ).reshape(4, 4)
            rotation = matrix[:3, :3].tolist()
            translation = matrix[:3, 3].tolist()
            payload["raw_sam_to_ycb_similarity"]["rotation"] = rotation
            payload["raw_sam_to_ycb_similarity"]["translation_m"] = translation
            payload["raw_sam_to_ycb_similarity"][
                "rigid_matrix_without_scale"
            ] = matrix.tolist()
            production = payload.get("production_sam_to_ycb_similarity") or {}
            production["rotation"] = rotation
            production["translation_m"] = translation
            payload["production_sam_to_ycb_similarity"] = production
            payload["mapping_source"] = "approved_pose_consensus"
            payload["mapping_override"] = str(override)
            payload["pose_consensus"] = approved.get("pose_consensus")
        else:
            payload["mapping_source"] = "surface_bank_v1"
            payload["mapping_override"] = None
        payload["bank_version"] = "sam_ycb_bank_v2"
        payload["object_profile"] = profiles.get("objects", {}).get(name, {})
        (target_dir / "canonical_alignment.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        rows.append({
            "object_name": name,
            "mapping_source": payload["mapping_source"],
            "mapping_override": payload["mapping_override"],
        })

    missing = sorted(set(overrides) - {row["object_name"] for row in rows})
    if missing:
        raise KeyError(f"Override objects missing from source bank: {missing}")
    shutil.copy2(profile_path, out_root / "object_profiles.yaml")
    summary = {
        "version": "sam_ycb_bank_v2",
        "source_root": str(source_root),
        "profile_json": str(profile_path),
        "excluded_objects": sorted(excluded),
        "num_objects": len(rows),
        "objects": rows,
    }
    summary_path = out_root / "canonical_bank_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
