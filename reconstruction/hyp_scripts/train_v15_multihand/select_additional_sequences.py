#!/usr/bin/env python3
"""Select deterministic, group-balanced additions to a sequence sample."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


HOT3D_PATTERN = re.compile(r"P\d{4}_[0-9A-Za-z]+")
OAKINK2_PATTERN = re.compile(r"scene_\d+__[A-Za-z0-9]+\+\+seq__[A-Za-z0-9_-]+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hot3d", "oakink2"), required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--existing-list", action="append", default=[])
    parser.add_argument("--existing-root", action="append", default=[])
    parser.add_argument("--additional-count", type=int, required=True)
    parser.add_argument("--allowed-group", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-additional", required=True)
    parser.add_argument("--out-combined", required=True)
    parser.add_argument("--report-json")
    return parser.parse_args()


def stable_key(seed, *values):
    payload = "|".join([str(seed), *map(str, values)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_sequence(dataset, value):
    name = Path(str(value)).name
    if name.endswith(".tar"):
        name = name[:-4]
    elif name.endswith(".pkl"):
        name = name[:-4]
    pattern = HOT3D_PATTERN if dataset == "hot3d" else OAKINK2_PATTERN
    # OakInk2 completion markers and partial downloads are not sequence IDs.
    match = pattern.fullmatch(name) if dataset == "oakink2" else pattern.search(name)
    return match.group(0) if match else None


def collect_json_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from collect_json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from collect_json_strings(item)
    elif isinstance(value, str):
        yield value


def load_candidates(dataset, source):
    path = Path(source).expanduser().resolve()
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = collect_json_strings(raw)
    else:
        values = path.read_text(encoding="utf-8").splitlines()
    return sorted({
        sequence
        for value in values
        if (sequence := normalized_sequence(dataset, value)) is not None
    })


def load_existing(dataset, list_paths, roots):
    result = []
    for raw_path in list_paths:
        path = Path(raw_path).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            sequence = normalized_sequence(dataset, line.strip())
            if sequence is not None:
                result.append(sequence)
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Existing root is not a directory: {root}")
        for path in root.iterdir():
            sequence = normalized_sequence(dataset, path.name)
            if sequence is not None:
                result.append(sequence)
    return sorted(set(result))


def group_for(dataset, sequence):
    if dataset == "hot3d":
        return sequence.split("_", 1)[0]
    match = re.search(r"__(.+?)\+\+seq__", sequence)
    if match is None:
        raise ValueError(f"Cannot parse OakInk2 subject from {sequence}")
    return match.group(1)


def balanced_select(candidates, existing, count, dataset, seed):
    by_group = defaultdict(list)
    for sequence in candidates:
        by_group[group_for(dataset, sequence)].append(sequence)
    for group, values in by_group.items():
        values.sort(key=lambda item: stable_key(seed, group, item))

    totals = Counter(group_for(dataset, sequence) for sequence in existing)
    selected = []
    while len(selected) < count:
        available = [group for group, values in by_group.items() if values]
        if not available:
            break
        group = min(
            available,
            key=lambda item: (totals[item], stable_key(seed, "group", item)),
        )
        selected.append(by_group[group].pop(0))
        totals[group] += 1
    return selected


def write_lines(path, values):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main():
    args = parse_args()
    if args.additional_count <= 0:
        raise ValueError("--additional-count must be positive")

    candidates = load_candidates(args.dataset, args.candidate_source)
    existing = load_existing(
        args.dataset, args.existing_list, args.existing_root
    )
    allowed = set(args.allowed_group)
    if allowed:
        candidates = [
            item for item in candidates if group_for(args.dataset, item) in allowed
        ]
    existing_set = set(existing)
    remaining = [item for item in candidates if item not in existing_set]
    selected = balanced_select(
        remaining, existing, args.additional_count, args.dataset, args.seed
    )
    if len(selected) != args.additional_count:
        raise RuntimeError(
            f"Requested {args.additional_count} additions, but only selected "
            f"{len(selected)} from {len(remaining)} eligible candidates"
        )

    combined = existing + selected
    if len(combined) != len(set(combined)):
        raise RuntimeError("Combined selection contains duplicate sequences")
    write_lines(args.out_additional, selected)
    write_lines(args.out_combined, combined)

    report = {
        "dataset": args.dataset,
        "seed": args.seed,
        "candidate_sequences": len(candidates),
        "existing_sequences": len(existing),
        "eligible_new_sequences": len(remaining),
        "additional_sequences": len(selected),
        "combined_sequences": len(combined),
        "existing_by_group": dict(sorted(Counter(
            group_for(args.dataset, item) for item in existing
        ).items())),
        "additional_by_group": dict(sorted(Counter(
            group_for(args.dataset, item) for item in selected
        ).items())),
        "combined_by_group": dict(sorted(Counter(
            group_for(args.dataset, item) for item in combined
        ).items())),
        "out_additional": str(Path(args.out_additional).resolve()),
        "out_combined": str(Path(args.out_combined).resolve()),
    }
    report_path = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else Path(args.out_combined).expanduser().resolve().with_suffix(".json")
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
