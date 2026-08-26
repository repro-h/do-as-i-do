#!/usr/bin/env python3
"""Build stable per-stream hand slots from frame-level joint annotations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--max-match-distance-px", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--status-json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_streams(path):
    streams = defaultdict(dict)
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sides = row.get(
                "hand_sides_metadata_only",
                row.get("hand_side_metadata_only", "unknown"),
            )
            for frame, label in zip(row["frame_indices"], row["label_paths"]):
                streams[row["stream_id"]][int(frame)] = (str(label), sides)
    return streams


def side_code(value):
    value = str(value).lower()
    if value == "left":
        return 0
    if value == "right":
        return 1
    return -1


def instances(label_path, side_metadata):
    with np.load(label_path, allow_pickle=False) as data:
        uv = np.asarray(data["joint_2d"], dtype=np.float32)
        xyz = np.asarray(data["joint_3d"], dtype=np.float32)
        explicit_sides = None
        for key in ("hand_sides", "mano_sides", "is_right"):
            if key in data.files:
                explicit_sides = np.asarray(data[key]).reshape(-1)
                break
    if uv.ndim == 2:
        uv = uv[None]
    if xyz.ndim == 2:
        xyz = xyz[None]
    count = min(len(uv), len(xyz))
    if uv.shape[1:] != (21, 2) or xyz.shape[1:] != (21, 3):
        raise ValueError(f"Unexpected joint arrays in {label_path}: {uv.shape}, {xyz.shape}")
    if explicit_sides is not None and len(explicit_sides) >= count:
        if explicit_sides.dtype == np.bool_ or np.issubdtype(explicit_sides.dtype, np.number):
            sides = np.where(explicit_sides[:count].astype(bool), 1, 0).astype(np.int8)
        else:
            sides = np.asarray([side_code(value) for value in explicit_sides[:count]], dtype=np.int8)
    else:
        values = side_metadata if isinstance(side_metadata, list) else [side_metadata]
        sides = np.full(count, -1, dtype=np.int8)
        for index, value in enumerate(values[:count]):
            sides[index] = side_code(value)
    return uv[:count], xyz[:count], sides


def joint_distance(first, second):
    valid = np.isfinite(first).all(axis=-1) & np.isfinite(second).all(axis=-1)
    if not valid.any():
        return float("inf")
    return float(np.median(np.linalg.norm(first[valid] - second[valid], axis=-1)))


def valid_cache(path, stream_id, frames, max_hands):
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["stream_id"].item()) != stream_id:
                return False
            if not np.array_equal(data["frame_indices"], frames):
                return False
            return (
                data["joint_uv"].shape == (len(frames), max_hands, 21, 2)
                and data["joint_xyz"].shape == (len(frames), max_hands, 21, 3)
                and data["track_ids"].shape == (len(frames), max_hands)
            )
    except (OSError, KeyError, ValueError):
        return False


def build_track_cache(stream_id, records, max_hands, max_gap, max_distance):
    frames = np.asarray(sorted(records), dtype=np.int64)
    count = len(frames)
    joint_uv = np.full((count, max_hands, 21, 2), np.nan, dtype=np.float32)
    joint_xyz = np.full((count, max_hands, 21, 3), np.nan, dtype=np.float32)
    joint_valid = np.zeros((count, max_hands, 21), dtype=bool)
    track_valid = np.zeros((count, max_hands), dtype=bool)
    observation_valid = np.zeros((count, max_hands), dtype=bool)
    target_valid = np.zeros((count, max_hands), dtype=bool)
    track_ids = np.full((count, max_hands), -1, dtype=np.int64)
    hand_side = np.full((count, max_hands), -1, dtype=np.int8)
    states = [None] * max_hands
    next_track_id = 0

    for offset, frame in enumerate(frames):
        uv, xyz, sides = instances(*records[int(frame)])
        assigned_slots = set()
        assigned_instances = set()
        slot_to_instance = {}
        candidates = []
        for slot, state in enumerate(states):
            if state is None or offset - state["offset"] > max_gap + 1:
                continue
            gap = offset - state["offset"]
            predicted_uv = state["uv"] + state["velocity"] * gap
            for instance in range(len(uv)):
                if (
                    state["side"] >= 0 and sides[instance] >= 0
                    and state["side"] != sides[instance]
                ):
                    continue
                distance = joint_distance(predicted_uv, uv[instance])
                candidates.append((distance, slot, instance))
        for distance, slot, instance in sorted(candidates):
            if distance > max_distance:
                break
            if slot in assigned_slots or instance in assigned_instances:
                continue
            assigned_slots.add(slot)
            assigned_instances.add(instance)
            slot_to_instance[slot] = instance

        free_slots = [
            slot for slot, state in enumerate(states)
            if slot not in assigned_slots
            and (state is None or offset - state["offset"] > max_gap + 1)
        ]
        for instance in range(len(uv)):
            if instance in assigned_instances or not free_slots:
                continue
            slot = free_slots.pop(0)
            assigned_slots.add(slot)
            assigned_instances.add(instance)
            slot_to_instance[slot] = instance
            states[slot] = {
                "track_id": next_track_id,
                "offset": -10_000,
                "uv": uv[instance],
                "velocity": np.zeros_like(uv[instance]),
                "side": int(sides[instance]),
            }
            next_track_id += 1

        for slot, instance in slot_to_instance.items():
            if states[slot] is None:
                states[slot] = {"track_id": next_track_id}
                next_track_id += 1
            state = states[slot]
            joint_uv[offset, slot] = uv[instance]
            joint_xyz[offset, slot] = xyz[instance]
            joint_valid[offset, slot] = np.isfinite(uv[instance]).all(axis=-1)
            track_valid[offset, slot] = True
            observation_valid[offset, slot] = bool(joint_valid[offset, slot].any())
            target_valid[offset, slot] = bool(
                np.isfinite(xyz[instance, 0]).all() and xyz[instance, 0, 2] > 0
            )
            track_ids[offset, slot] = int(state["track_id"])
            hand_side[offset, slot] = int(sides[instance])
            previous_offset = state.get("offset", offset)
            gap = max(1, offset - previous_offset)
            previous_uv = state.get("uv", uv[instance])
            velocity = (uv[instance] - previous_uv) / gap
            velocity[~np.isfinite(velocity)] = 0.0
            state.update({
                "offset": offset,
                "uv": uv[instance].copy(),
                "velocity": velocity,
                "side": int(sides[instance]),
            })

    # Preserve a slot through short annotation/detection gaps.
    for slot in range(max_hands):
        ids = np.unique(track_ids[:, slot])
        for track_id in ids[ids >= 0]:
            locations = np.flatnonzero(track_ids[:, slot] == track_id)
            for first, second in zip(locations[:-1], locations[1:]):
                if second - first <= max_gap + 1:
                    track_valid[first:second + 1, slot] = True
                    track_ids[first:second + 1, slot] = track_id
                    hand_side[first:second + 1, slot] = hand_side[first, slot]

    return {
        "cache_version": np.asarray("multihand_tracks_v1"),
        "stream_id": np.asarray(stream_id),
        "frame_indices": frames,
        "joint_uv": joint_uv,
        "joint_xyz": joint_xyz,
        "joint_valid": joint_valid,
        "track_valid": track_valid,
        "observation_valid": observation_valid,
        "target_valid": target_valid,
        "track_ids": track_ids,
        "hand_side": hand_side,
        "num_tracks": np.int32(next_track_id),
    }


def main():
    args = parse_args()
    streams = load_streams(args.windows)
    items = sorted(streams.items())
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    items = items[args.shard_index::args.num_shards]
    out_root = Path(args.out_root).expanduser().resolve()
    completed, cached, failures = [], [], []
    for stream_id, records in tqdm(items, desc="tracks"):
        output = out_root / stream_id / "tracks.npz"
        frames = np.asarray(sorted(records), dtype=np.int64)
        if not args.overwrite and valid_cache(output, stream_id, frames, args.max_hands):
            completed.append(stream_id)
            cached.append(stream_id)
            continue
        try:
            payload = build_track_cache(
                stream_id, records, args.max_hands,
                args.max_gap, args.max_match_distance_px,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **payload)
            completed.append(stream_id)
        except Exception as error:
            failures.append({"stream_id": stream_id, "error": repr(error)})
            print(f"FAILED {stream_id}: {error}", flush=True)
    status = {
        "windows": str(Path(args.windows).expanduser().resolve()),
        "max_hands": args.max_hands,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "requested": len(items),
        "completed": len(completed),
        "cached": len(cached),
        "failed": len(failures),
        "failures": failures,
        "out_root": str(out_root),
    }
    if args.status_json:
        path = Path(args.status_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
