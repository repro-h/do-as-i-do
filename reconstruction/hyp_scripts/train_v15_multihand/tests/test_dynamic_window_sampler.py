import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1] / "dynamic_window_sampler.py"
SPEC = importlib.util.spec_from_file_location("dynamic_window_sampler", MODULE)
dynamic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dynamic)


def row(dataset, stream, start, end):
    frames = list(range(start, end))
    return {
        "dataset": dataset,
        "stream_id": stream,
        "start": start,
        "end": end,
        "frame_indices": frames,
        "image_paths": [f"{stream}/rgb/{frame}.jpg" for frame in frames],
        "label_paths": [f"{stream}/label/{frame}.npz" for frame in frames],
        "tracks_npz": f"{stream}/tracks.npz",
        "visibility_npz": f"{stream}/visibility.npz",
        "intrinsics": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
    }


class DynamicWindowSamplerTests(unittest.TestCase):
    def rows(self):
        rows = []
        for dataset in ("a", "b"):
            for sequence in range(4):
                stream = f"{dataset}_{sequence}"
                rows.extend((row(dataset, stream, 0, 64), row(dataset, stream, 32, 96)))
        return rows

    def test_random_windows_are_contiguous_and_not_limited_to_old_starts(self):
        sampler = dynamic.DynamicSequenceBatchSampler(
            self.rows(), batch_size=2, steps_per_epoch=20,
            window_size=64, seed=7,
        )
        batches = list(sampler)
        sampled = [item for batch in batches for item in batch]
        self.assertTrue(any(item["start"] not in (0, 32) for item in sampled))
        for item in sampled:
            self.assertEqual(len(item["frame_indices"]), 64)
            self.assertEqual(len(item["intrinsics"]), 3)
            self.assertEqual(item["frame_indices"], list(range(item["start"], item["end"])))
            self.assertEqual(len({path.split("/")[0] for path in item["image_paths"]}), 1)

    def test_batches_are_dataset_homogeneous_and_sequences_balanced(self):
        sampler = dynamic.DynamicSequenceBatchSampler(
            self.rows(), batch_size=3, steps_per_epoch=10,
            window_size=64, seed=11,
        )
        batches = list(sampler)
        for batch in batches:
            self.assertEqual(len({item["dataset"] for item in batch}), 1)
            self.assertEqual(len({item["stream_id"] for item in batch}), len(batch))
        audit = sampler.audit()
        self.assertEqual(audit["by_dataset"]["a"]["batches"], 5)
        self.assertEqual(audit["by_dataset"]["b"]["batches"], 5)
        for dataset in ("a", "b"):
            distribution = audit["by_dataset"][dataset]["samples_per_sequence"]
            self.assertLessEqual(distribution["max"] - distribution["min"], 1)

    def test_short_stream_is_excluded(self):
        rows = self.rows() + [row("a", "too_short", 0, 63)]
        sampler = dynamic.DynamicSequenceBatchSampler(
            rows, batch_size=2, steps_per_epoch=2, window_size=64,
        )
        self.assertNotIn("too_short", sampler.sequences["a"])
        self.assertEqual(
            sampler.reconstruction["a::too_short"]["possible_starts"], 0
        )

    def test_ddp_ranks_share_dataset_schedule_but_not_sample_seed(self):
        samplers = [
            dynamic.DynamicSequenceBatchSampler(
                self.rows(), batch_size=1, steps_per_epoch=8,
                window_size=64, seed=5, rank=rank, world_size=2,
            )
            for rank in range(2)
        ]
        batches = [list(sampler) for sampler in samplers]
        schedules = [
            [batch[0]["dataset"] for batch in rank_batches]
            for rank_batches in batches
        ]
        self.assertEqual(schedules[0], schedules[1])
        self.assertNotEqual(samplers[0].audit()["seed"], samplers[1].audit()["seed"])


if __name__ == "__main__":
    unittest.main()
