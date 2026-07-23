#!/usr/bin/env python3
"""Build a compact manifest for QA-approved DexYCB hybrid refinement data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        action="append",
        required=True,
        metavar="SPLIT=PATH",
        help="QA-approved stream_id -> FoundationPose JSON mapping.",
    )
    parser.add_argument("--dexycb-root", required=True)
    parser.add_argument("--shape-bank-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-object", action="append", default=[])
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write valid records even when some indexed streams fail validation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_index_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--index must use SPLIT=PATH, got: {spec}")
    split, raw_path = spec.split("=", 1)
    split = split.strip()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split in --index: {split}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return split, path


def parse_stream_id(stream_id: str) -> tuple[str, str, str]:
    parts = stream_id.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"Expected subject__sequence__camera stream ID, got: {stream_id}"
        )
    return parts[0], parts[1], parts[2]


def load_object_names(model_root: Path) -> dict[int, str]:
    numbered = []
    for path in sorted(model_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            numbered.append((int(path.name.split("_", 1)[0]), path.name))
        except ValueError:
            continue
    numbered.sort()
    if not numbered:
        raise RuntimeError(f"No numbered YCB model directories in {model_root}")
    return {class_id: name for class_id, (_, name) in enumerate(numbered, start=1)}


def resolve_pose_rows(payload: dict[str, Any]) -> tuple[int, list[str]]:
    for key in ("frames", "poses", "results"):
        value = payload.get(key)
        if isinstance(value, dict):
            keys = sorted(str(item) for item in value)
            return len(keys), keys
        if isinstance(value, list):
            frame_ids = []
            for index, row in enumerate(value):
                if isinstance(row, dict):
                    frame_ids.append(
                        str(
                            row.get(
                                "frame",
                                row.get("frame_id", row.get("source_frame", index)),
                            )
                        )
                    )
                else:
                    frame_ids.append(str(index))
            return len(value), frame_ids
    return 0, []


def resolve_shape_paths(shape_bank_root: Path, object_name: str) -> dict[str, str | None]:
    sam_dir = shape_bank_root / object_name / "sam3d"
    candidates = {
        "sam3d_glb": sam_dir / "object_canonical_sam.glb",
        "sam3d_npz": sam_dir / "sam3d_object_output.npz",
        "sam3d_metadata": sam_dir / "sam3d_object_output_metadata.json",
    }
    return {
        key: str(path.resolve()) if path.is_file() else None
        for key, path in candidates.items()
    }


def build_record(
    split: str,
    stream_id: str,
    pose_path: Path,
    dexycb_root: Path,
    shape_bank_root: Path,
    object_names: dict[int, str],
) -> dict[str, Any]:
    subject, sequence, camera = parse_stream_id(stream_id)
    stream_dir = dexycb_root / subject / sequence / camera
    meta_path = stream_dir.parent / "meta.yml"
    if not stream_dir.is_dir():
        raise FileNotFoundError(f"Missing DexYCB stream: {stream_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing DexYCB metadata: {meta_path}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing FoundationPose JSON: {pose_path}")

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    ycb_ids = list(meta.get("ycb_ids", []) or [])
    grasp_index = int(meta.get("ycb_grasp_ind", 0))
    if not 0 <= grasp_index < len(ycb_ids):
        raise ValueError(f"Invalid ycb_grasp_ind={grasp_index} in {meta_path}")
    object_class_id = int(ycb_ids[grasp_index])
    if object_class_id not in object_names:
        raise KeyError(f"Unknown DexYCB object class ID: {object_class_id}")
    object_name = object_names[object_class_id]
    hand_side = str((meta.get("mano_sides") or ["right"])[0]).lower()
    if hand_side not in {"left", "right"}:
        raise ValueError(f"Unexpected hand side in {meta_path}: {hand_side}")

    pose_payload = load_json(pose_path)
    num_pose_frames, pose_frame_ids = resolve_pose_rows(pose_payload)
    if num_pose_frames == 0:
        raise ValueError(f"FoundationPose JSON has no pose rows: {pose_path}")
    source_mesh_scale = pose_payload.get(
        "source_mesh_scale", pose_payload.get("final_global_scale")
    )
    if source_mesh_scale is None:
        raise KeyError(f"FoundationPose JSON has no mesh scale: {pose_path}")
    source_mesh_scale = float(source_mesh_scale)
    if not source_mesh_scale > 0:
        raise ValueError(f"Invalid source mesh scale: {source_mesh_scale}")

    shape_paths = resolve_shape_paths(shape_bank_root, object_name)
    if shape_paths["sam3d_glb"] is None:
        raise FileNotFoundError(
            shape_bank_root / object_name / "sam3d" / "object_canonical_sam.glb"
        )

    color_paths = sorted(stream_dir.glob("color_*.jpg"))
    if not color_paths:
        color_paths = sorted(stream_dir.glob("color_*.png"))
    label_paths = sorted(stream_dir.glob("labels_*.npz"))
    if not color_paths or not label_paths:
        raise RuntimeError(f"Stream has no RGB or labels: {stream_dir}")

    return {
        "split": split,
        "stream_id": stream_id,
        "subject": subject,
        "sequence": sequence,
        "camera": camera,
        "stream_dir": str(stream_dir.resolve()),
        "meta_path": str(meta_path.resolve()),
        "hand_side": hand_side,
        "object_class_id": object_class_id,
        "object_name": object_name,
        "foundationpose_json": str(pose_path.resolve()),
        "foundationpose_source_mesh_scale": source_mesh_scale,
        "foundationpose_num_frames": num_pose_frames,
        "foundationpose_first_frame": pose_frame_ids[0],
        "foundationpose_last_frame": pose_frame_ids[-1],
        "num_rgb_frames": len(color_paths),
        "num_label_frames": len(label_paths),
        **shape_paths,
    }


def main() -> None:
    args = parse_args()
    dexycb_root = Path(args.dexycb_root).expanduser().resolve()
    shape_bank_root = Path(args.shape_bank_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    object_names = load_object_names(dexycb_root / "models")
    excluded = set(args.exclude_object)

    specs = [parse_index_spec(spec) for spec in args.index]
    seen: dict[str, tuple[str, str]] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    failures = []
    excluded_rows = []

    for split, index_path in specs:
        payload = load_json(index_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Passed pose index must be a JSON object: {index_path}")
        for stream_id, raw_pose_path in sorted(payload.items()):
            if stream_id in seen:
                old_split, old_path = seen[stream_id]
                raise ValueError(
                    f"Duplicate stream {stream_id}: {old_split}={old_path}, "
                    f"{split}={index_path}"
                )
            seen[stream_id] = (split, str(index_path))
            pose_path = Path(str(raw_pose_path)).expanduser().resolve()
            try:
                record = build_record(
                    split,
                    stream_id,
                    pose_path,
                    dexycb_root,
                    shape_bank_root,
                    object_names,
                )
                if record["object_name"] in excluded:
                    excluded_rows.append(
                        {
                            "split": split,
                            "stream_id": stream_id,
                            "object_name": record["object_name"],
                        }
                    )
                    continue
                records_by_split[split].append(record)
            except Exception as error:
                failures.append(
                    {
                        "split": split,
                        "stream_id": stream_id,
                        "pose_path": str(pose_path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    all_records = []
    for split, records in records_by_split.items():
        records.sort(key=lambda row: row["stream_id"])
        all_records.extend(records)
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    all_path = out_dir / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")

    summary = {
        "dexycb_root": str(dexycb_root),
        "shape_bank_root": str(shape_bank_root),
        "out_dir": str(out_dir),
        "indexes": [
            {"split": split, "path": str(path)} for split, path in specs
        ],
        "excluded_objects": sorted(excluded),
        "counts": {
            split: len(records) for split, records in records_by_split.items()
        },
        "num_records": len(all_records),
        "num_excluded": len(excluded_rows),
        "num_failures": len(failures),
        "hand_side_counts": dict(
            Counter(record["hand_side"] for record in all_records)
        ),
        "object_counts": dict(
            Counter(record["object_name"] for record in all_records)
        ),
        "excluded": excluded_rows,
        "failures": failures,
    }
    write_json(out_dir / "manifest_summary.json", summary)
    print(json.dumps(summary, indent=2))

    if failures and not args.allow_missing:
        raise RuntimeError(
            f"{len(failures)} streams failed validation; inspect "
            f"{out_dir / 'manifest_summary.json'} or pass --allow-missing"
        )


if __name__ == "__main__":
    main()
