#!/usr/bin/env python3
"""Summarize V15 manifests, caches, and exporter status files."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


FAILURE_KEYS = ("failed", "failed_sequences", "num_failed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def dataset_name(root, path):
    relative = path.relative_to(root)
    first = relative.parts[0] if relative.parts else "root"
    if first.startswith(("manifest", "pi3x", "visibility", "log", "track")):
        return "dexycb"
    return first


def read_manifest(path):
    windows = 0
    streams = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            windows += 1
            row = json.loads(line)
            if "stream_id" in row:
                streams.add(str(row["stream_id"]))
    return windows, len(streams)


def status_failures(data):
    result = {}
    for key in FAILURE_KEYS:
        if key in data and isinstance(data[key], (int, float)):
            result[key] = int(data[key])
    if isinstance(data.get("failures"), list):
        result["failures"] = len(data["failures"])
    return result


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    manifests = defaultdict(list)
    for path in root.rglob("*.jsonl"):
        if "manifest" not in str(path.parent) and "windows" not in path.name:
            continue
        try:
            windows, streams = read_manifest(path)
        except (OSError, ValueError, KeyError) as error:
            manifests[dataset_name(root, path)].append(
                (path, None, None, repr(error))
            )
        else:
            manifests[dataset_name(root, path)].append(
                (path, windows, streams, None)
            )

    statuses = defaultdict(list)
    for path in root.rglob("*.json"):
        if "status" not in path.name and path.name != "audit.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            statuses[dataset_name(root, path)].append(
                (path, None, repr(error))
            )
            continue
        failures = status_failures(data)
        requested = data.get("requested_sequences", data.get("num_requested"))
        completed = data.get("completed_sequences", data.get("num_completed"))
        statuses[dataset_name(root, path)].append(
            (path, {"requested": requested, "completed": completed,
                    "failures": failures}, None)
        )

    counters = defaultdict(lambda: defaultdict(int))
    for path in root.rglob("visibility_cache.npz"):
        counters[dataset_name(root, path)]["visibility_caches"] += 1
    for path in root.rglob("summary.json"):
        name = dataset_name(root, path)
        if "pi3x" in str(path).lower():
            counters[name]["pi3x_streams"] += 1
        elif "streams" in path.parts:
            counters[name]["processed_streams"] += 1

    names = sorted(set(manifests) | set(statuses) | set(counters))
    for name in names:
        print(f"\n===== {name} =====")
        for path, windows, streams, error in sorted(
            manifests[name], key=lambda item: str(item[0])
        ):
            relative = path.relative_to(root)
            if error:
                print(f"MANIFEST BROKEN {relative}: {error}")
            else:
                print(
                    f"MANIFEST {relative}: windows={windows} streams={streams}"
                )
        for path, summary, error in sorted(
            statuses[name], key=lambda item: str(item[0])
        ):
            relative = path.relative_to(root)
            if error:
                print(f"STATUS BROKEN {relative}: {error}")
            else:
                print(f"STATUS {relative}: {summary}")
        if counters[name]:
            values = " ".join(
                f"{key}={value}" for key, value in sorted(counters[name].items())
            )
            print(f"CACHES {values}")


if __name__ == "__main__":
    main()
