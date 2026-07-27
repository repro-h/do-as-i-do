#!/usr/bin/env python3
"""Audit contact and penetration changes frame by frame."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from optimize_hand_object_contact_sequence import (
    deterministic_surface_samples,
    load_mesh,
    nearest_surface,
    transform_surface,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-regression-tolerance-mm", type=float, default=1.0)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    settings = audit["settings"]

    with np.load(audit["hand_npz"], allow_pickle=False) as raw:
        initial_payload = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(audit["output_npz"], allow_pickle=False) as raw:
        final_payload = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(audit["supervision_npz"], allow_pickle=False) as raw:
        supervision = {key: np.asarray(raw[key]) for key in raw.files}

    initial_vertices = np.asarray(initial_payload["verts_cam"], dtype=np.float32)
    final_vertices = np.asarray(final_payload["verts_cam"], dtype=np.float32)
    sample_indices = np.asarray(
        final_payload["optim_seq_contact_sample_indices"], dtype=np.int64
    )
    contact_mask = np.asarray(
        final_payload["optim_seq_contact_mask"], dtype=bool
    )
    object_pose = np.asarray(supervision["object_pose"], dtype=np.float32)
    valid = (
        np.asarray(initial_payload["pred_valid"]).astype(bool)
        & np.asarray(supervision["object_valid"]).astype(bool)
    )
    count = min(
        len(initial_vertices),
        len(final_vertices),
        len(object_pose),
        len(valid),
        len(contact_mask),
    )
    initial_vertices = initial_vertices[:count, sample_indices]
    final_vertices = final_vertices[:count, sample_indices]
    object_pose = object_pose[:count]
    valid = valid[:count]
    contact_mask = contact_mask[:count]

    mesh = load_mesh(Path(audit["object_mesh"]), float(settings["mesh_scale"]))
    local_points, local_normals = deterministic_surface_samples(
        mesh, int(settings["object_samples"])
    )
    device = torch.device(args.device)
    poses = torch.from_numpy(object_pose).to(device)
    object_points, object_normals = transform_surface(
        torch.from_numpy(local_points).to(device),
        torch.from_numpy(local_normals).to(device),
        poses,
    )
    with torch.no_grad():
        initial_distance, _, _, initial_inside = nearest_surface(
            torch.from_numpy(initial_vertices).to(device),
            object_points,
            object_normals,
        )
        final_distance, _, _, final_inside = nearest_surface(
            torch.from_numpy(final_vertices).to(device),
            object_points,
            object_normals,
        )

    initial_distance = initial_distance.cpu().numpy() * 1000.0
    final_distance = final_distance.cpu().numpy() * 1000.0
    tolerance = float(settings["penetration_tolerance_mm"])
    initial_depth = np.maximum(initial_inside.cpu().numpy() * 1000.0 - tolerance, 0.0)
    final_depth = np.maximum(final_inside.cpu().numpy() * 1000.0 - tolerance, 0.0)

    rows = []
    for frame in range(count):
        initial_pen = distribution(initial_depth[frame][initial_depth[frame] > 0])
        final_pen = distribution(final_depth[frame][final_depth[frame] > 0])
        mask = contact_mask[frame]
        initial_contact = distribution(initial_distance[frame][mask])
        final_contact = distribution(final_distance[frame][mask])
        row = {
            "frame": frame,
            "valid": bool(valid[frame]),
            "initial_penetrating": initial_pen["count"],
            "final_penetrating": final_pen["count"],
            "penetrating_delta": final_pen["count"] - initial_pen["count"],
            "initial_penetration_median_mm": initial_pen["median"],
            "final_penetration_median_mm": final_pen["median"],
            "penetration_median_delta_mm": (
                final_pen["median"] - initial_pen["median"]
            ),
            "initial_penetration_p90_mm": initial_pen["p90"],
            "final_penetration_p90_mm": final_pen["p90"],
            "penetration_p90_delta_mm": final_pen["p90"] - initial_pen["p90"],
            "initial_penetration_max_mm": initial_pen["max"],
            "final_penetration_max_mm": final_pen["max"],
            "penetration_max_delta_mm": final_pen["max"] - initial_pen["max"],
            "num_contact_candidates": int(mask.sum()),
            "initial_contact_median_mm": initial_contact["median"],
            "final_contact_median_mm": final_contact["median"],
            "contact_median_delta_mm": (
                final_contact["median"] - initial_contact["median"]
            ),
        }
        row["regressed"] = bool(
            row["penetrating_delta"] > 0
            or row["penetration_max_delta_mm"]
            > args.max_regression_tolerance_mm
            or row["contact_median_delta_mm"]
            > args.max_regression_tolerance_mm
        )
        # Prioritize severe penetration, then newly penetrating samples.
        row["regression_score"] = float(
            max(row["penetration_max_delta_mm"], 0.0)
            + max(row["penetration_p90_delta_mm"], 0.0)
            + 0.25 * max(row["penetrating_delta"], 0)
            + max(row["contact_median_delta_mm"], 0.0)
        )
        rows.append(row)

    ranked = sorted(
        (row for row in rows if row["valid"] and row["regressed"]),
        key=lambda row: row["regression_score"],
        reverse=True,
    )
    summary = {
        "source_audit": str(audit_path),
        "num_frames": count,
        "num_valid_frames": int(valid.sum()),
        "num_regressed_frames": len(ranked),
        "regression_definition": {
            "penetrating_count_increased": True,
            "max_or_contact_tolerance_mm": args.max_regression_tolerance_mm,
        },
        "worst_frames": ranked,
        "frames": rows,
    }
    (out_dir / "frame_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "frame_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"valid frames: {int(valid.sum())}")
    print(f"regressed frames: {len(ranked)}")
    print("\nworst frames:")
    for row in ranked[:20]:
        print(
            f"{row['frame']:06d}",
            f"count {row['initial_penetrating']}"
            f"->{row['final_penetrating']}",
            f"p90 {row['initial_penetration_p90_mm']:.2f}"
            f"->{row['final_penetration_p90_mm']:.2f} mm",
            f"max {row['initial_penetration_max_mm']:.2f}"
            f"->{row['final_penetration_max_mm']:.2f} mm",
            f"contact {row['initial_contact_median_mm']:.2f}"
            f"->{row['final_contact_median_mm']:.2f} mm",
        )
    print(f"\nJSON: {out_dir / 'frame_audit.json'}")
    print(f"CSV: {out_dir / 'frame_audit.csv'}")


if __name__ == "__main__":
    main()
