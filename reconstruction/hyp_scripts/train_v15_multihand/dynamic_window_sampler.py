"""Dynamic, dataset- and sequence-balanced batches from window manifests."""

import random
from collections import defaultdict


STATIC_LIST_KEYS = {
    "intrinsics",
    "hand_sides",
    "hand_sides_metadata_only",
}


def row_dataset(row):
    if row.get("dataset"):
        return str(row["dataset"])
    return str(row.get("schema_version", "unknown")).split("_", 1)[0]


def consecutive_runs(positions):
    positions = sorted(positions)
    if not positions:
        return
    run = [positions[0]]
    for position in positions[1:]:
        if position != run[-1] + 1:
            yield run
            run = []
        run.append(position)
    yield run


def _distribution(values):
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def parse_dataset_weights(value, datasets):
    """Parse ``dataset=weight,...``; omitted datasets retain weight one."""
    weights = {name: 1.0 for name in datasets}
    if not value:
        return weights
    for item in value.split(","):
        try:
            name, raw_weight = item.split("=", 1)
        except ValueError as error:
            raise ValueError(
                "--dataset-weights must use dataset=weight comma pairs"
            ) from error
        name = name.strip()
        if name not in weights:
            raise ValueError(f"Unknown dataset in --dataset-weights: {name}")
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("Dataset weights cannot be negative")
        weights[name] = weight
    if sum(weights.values()) <= 0:
        raise ValueError("At least one dataset weight must be positive")
    return weights


class DynamicSequenceBatchSampler:
    """Yield same-dataset batches with random contiguous windows.

    Existing overlapping rows are used only to reconstruct each stream's frame
    records. Sequence choices are balanced within a dataset; temporal starts are
    sampled independently and may repeat.
    """

    def __init__(
        self,
        rows,
        batch_size,
        steps_per_epoch,
        window_size=64,
        dataset_weights=None,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if min(self.batch_size, self.steps_per_epoch, self.window_size) <= 0:
            raise ValueError("Batch size, steps, and window size must be positive")

        grouped = defaultdict(list)
        for row in rows:
            grouped[(row_dataset(row), str(row["stream_id"]))].append(row)
        self.sequences = defaultdict(dict)
        self.reconstruction = {}
        for (dataset, stream), stream_rows in sorted(grouped.items()):
            sequence, stats = self._reconstruct(stream_rows)
            self.reconstruction[f"{dataset}::{stream}"] = stats
            if sequence is not None:
                self.sequences[dataset][stream] = sequence
        self.sequences = {
            dataset: streams for dataset, streams in self.sequences.items()
            if streams
        }
        if not self.sequences:
            raise RuntimeError(
                f"No streams contain a contiguous T={self.window_size} run"
            )
        supplied = dataset_weights or {}
        self.dataset_weights = {
            dataset: float(supplied.get(dataset, 1.0))
            for dataset in self.sequences
        }
        if any(weight < 0 for weight in self.dataset_weights.values()):
            raise ValueError("Dataset weights cannot be negative")
        if sum(self.dataset_weights.values()) <= 0:
            raise ValueError("At least one eligible dataset weight must be positive")
        unknown = set(supplied) - set(self.sequences)
        if unknown:
            raise ValueError(f"Weights given for ineligible datasets: {sorted(unknown)}")
        self.epoch = 0
        self.last_audit = None

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _reconstruct(self, rows):
        rows = sorted(rows, key=lambda row: (int(row["start"]), int(row["end"])))
        frame_records = {}
        templates = {}
        conflicts = set()
        for row in rows:
            frame_count = len(row["frame_indices"])
            frame_keys = {
                key for key, value in row.items()
                if (
                    key not in STATIC_LIST_KEYS
                    and isinstance(value, list)
                    and len(value) == frame_count
                )
            }
            for offset in range(frame_count):
                position = int(row["start"]) + offset
                record = {key: row[key][offset] for key in frame_keys}
                if position in frame_records and frame_records[position] != record:
                    conflicts.add(position)
                    continue
                frame_records[position] = record
                templates[position] = row
        for position in conflicts:
            frame_records.pop(position, None)
            templates.pop(position, None)
        runs = [
            run for run in consecutive_runs(frame_records)
            if len(run) >= self.window_size
        ]
        stats = {
            "source_rows": len(rows),
            "recoverable_frames": len(frame_records),
            "conflicting_frames": len(conflicts),
            "eligible_runs": len(runs),
            "possible_starts": sum(len(run) - self.window_size + 1 for run in runs),
        }
        if not runs:
            return None, stats
        return {
            "frame_records": frame_records,
            "templates": templates,
            "runs": runs,
            "possible_starts": stats["possible_starts"],
        }, stats

    def _batch_datasets(self, rng):
        names = sorted(self.sequences)
        total = sum(self.dataset_weights[name] for name in names)
        raw = {
            name: self.steps_per_epoch * self.dataset_weights[name] / total
            for name in names
        }
        counts = {name: int(raw[name]) for name in names}
        remaining = self.steps_per_epoch - sum(counts.values())
        tie_break = {name: rng.random() for name in names}
        order = sorted(
            names, key=lambda name: (raw[name] - counts[name], tie_break[name]),
            reverse=True,
        )
        for name in order[:remaining]:
            counts[name] += 1
        schedule = [name for name in names for _ in range(counts[name])]
        rng.shuffle(schedule)
        return schedule, counts

    def _choose_stream(self, dataset, counts, used_in_batch, rng):
        names = sorted(self.sequences[dataset])
        minimum = min(counts[dataset][name] for name in names)
        candidates = [
            name for name in names
            if counts[dataset][name] == minimum and name not in used_in_batch
        ]
        if not candidates:
            candidates = [name for name in names if counts[dataset][name] == minimum]
        if not candidates:
            candidates = names
        stream = rng.choice(candidates)
        counts[dataset][stream] += 1
        return stream

    def _sample_row(self, dataset, stream, rng):
        sequence = self.sequences[dataset][stream]
        offset = rng.randrange(sequence["possible_starts"])
        selected_run = None
        for run in sequence["runs"]:
            count = len(run) - self.window_size + 1
            if offset < count:
                selected_run = run
                break
            offset -= count
        positions = selected_run[offset:offset + self.window_size]
        template = sequence["templates"][positions[0]]
        row = {
            key: value for key, value in template.items()
            if (
                (not isinstance(value, list) or key in STATIC_LIST_KEYS)
                and key not in ("start", "end")
            )
        }
        records = sequence["frame_records"]
        common_keys = set.intersection(*(
            set(records[position]) for position in positions
        ))
        for key in sorted(common_keys):
            row[key] = [records[position][key] for position in positions]
        row["dataset"] = dataset
        row["stream_id"] = stream
        row["start"] = positions[0]
        row["end"] = positions[-1] + 1
        return row

    def __iter__(self):
        epoch_seed = self.seed + self.epoch * self.world_size + self.rank
        rng = random.Random(epoch_seed)
        schedule_rng = random.Random(self.seed + self.epoch)
        schedule, dataset_batch_counts = self._batch_datasets(schedule_rng)
        sequence_counts = {
            dataset: {stream: 0 for stream in streams}
            for dataset, streams in self.sequences.items()
        }
        starts = defaultdict(set)
        dataset_sample_counts = defaultdict(int)
        for dataset in schedule:
            used = set()
            batch = []
            for _ in range(self.batch_size):
                stream = self._choose_stream(dataset, sequence_counts, used, rng)
                used.add(stream)
                row = self._sample_row(dataset, stream, rng)
                starts[dataset].add((stream, int(row["start"])))
                dataset_sample_counts[dataset] += 1
                batch.append(row)
            yield batch
        self.last_audit = {
            "epoch": self.epoch,
            "seed": epoch_seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "window_size": self.window_size,
            "steps": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "dataset_weights": self.dataset_weights,
            "by_dataset": {
                dataset: {
                    "batches": dataset_batch_counts.get(dataset, 0),
                    "samples": dataset_sample_counts.get(dataset, 0),
                    "eligible_sequences": len(streams),
                    "sampled_sequences": sum(
                        count > 0 for count in sequence_counts[dataset].values()
                    ),
                    "samples_per_sequence": _distribution(
                        list(sequence_counts[dataset].values())
                    ),
                    "unique_starts": len(starts[dataset]),
                    "possible_starts": sum(
                        sequence["possible_starts"] for sequence in streams.values()
                    ),
                }
                for dataset, streams in sorted(self.sequences.items())
            },
        }

    def audit(self):
        return self.last_audit

    def setup_audit(self):
        by_dataset = {}
        all_datasets = sorted({
            key.split("::", 1)[0] for key in self.reconstruction
        })
        for dataset in all_datasets:
            prefix = dataset + "::"
            reconstructed = [
                stats for key, stats in self.reconstruction.items()
                if key.startswith(prefix)
            ]
            eligible = self.sequences.get(dataset, {})
            by_dataset[dataset] = {
                "source_sequences": len(reconstructed),
                "eligible_sequences": len(eligible),
                "excluded_short_or_conflicting_sequences": (
                    len(reconstructed) - len(eligible)
                ),
                "recoverable_frames": sum(
                    stats["recoverable_frames"] for stats in reconstructed
                ),
                "conflicting_frames": sum(
                    stats["conflicting_frames"] for stats in reconstructed
                ),
                "possible_starts": sum(
                    sequence["possible_starts"] for sequence in eligible.values()
                ),
            }
        return {
            "window_size": self.window_size,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "same_dataset_batches": True,
            "rank": self.rank,
            "world_size": self.world_size,
            "dataset_weights": self.dataset_weights,
            "by_dataset": by_dataset,
        }
