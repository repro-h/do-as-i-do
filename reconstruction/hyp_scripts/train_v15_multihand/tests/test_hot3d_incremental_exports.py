import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "run_hot3d_incremental_exports.py"
SPEC = importlib.util.spec_from_file_location("hot_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
SETTINGS = {"frame_stride": 3, "window_size": 16, "window_stride": 8, "camera_stream_id": "214-1"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepared(path):
    path.mkdir(parents=True, exist_ok=True)
    rgb = path / "rgb/000000.jpg"
    label = path / "labels/000000.npz"
    rgb.parent.mkdir(exist_ok=True)
    label.parent.mkdir(exist_ok=True)
    rgb.write_bytes(b"rgb fixture")
    label.write_bytes(b"label fixture")
    row = {"image_paths": [str(rgb)], "label_paths": [str(label)], "stream_id": path.name}
    write_json(path / "summary.json", dict(SETTINGS,
        schema_version="hot3d_aria_v15_export_v1", horizontal_mirror=False, windows=1))
    (path / "train_windows.jsonl").write_text(json.dumps(row) + "\n")
    return row


def downloaded(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in pipeline.REQUIRED_FILES:
        (path / name).write_bytes(b"fixture")
    write_json(path / ".download_status.json", dict.fromkeys(pipeline.REQUIRED_GROUPS, True))


class IncrementalTests(unittest.TestCase):
    def test_split_preserves_old_assignments_and_reruns(self):
        old = {"train_sequences": ["P0001_a", "P0002_b"], "val_sequences": ["P0001_c"]}
        new = ["P0001_d", "P0002_e"]
        result = pipeline.make_split(old, new, 2, 42)
        self.assertTrue(set(old["train_sequences"]) <= set(result["train_sequences"]))
        self.assertTrue(set(old["val_sequences"]) <= set(result["val_sequences"]))
        self.assertFalse(set(result["train_sequences"]) & set(result["val_sequences"]))
        self.assertEqual(result, pipeline.make_split(result, new, 2, 42))
        with self.assertRaises(ValueError):
            pipeline.make_split(old, new, 0, 42)

    def test_readiness_requires_status_and_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recording.vrs").write_bytes(b"partial")
            self.assertIsNone(pipeline.ready_signature(root)[0])
            downloaded(root)
            self.assertIsNotNone(pipeline.ready_signature(root)[0])
            write_json(root / ".download_status.json", {"main_vrs": True})
            self.assertIsNone(pipeline.ready_signature(root)[0])
            downloaded(root)
            (root / "mano_hand_pose_trajectory.jsonl").write_bytes(b"")
            self.assertIsNone(pipeline.ready_signature(root)[0])

    def test_cache_missing_files_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared(root)
            self.assertTrue(pipeline.prepared_is_current(root, SETTINGS))
            self.assertFalse(pipeline.prepared_is_current(root, dict(SETTINGS, frame_stride=1)))
            (root / "labels/000000.npz").unlink()
            self.assertFalse(pipeline.prepared_is_current(root, SETTINGS))

    def test_fixed_split_merger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = ["P0001_a", "P0002_b"], ["P0001_c"]
            for sequence in train + val:
                prepared(root / "processed" / sequence)
            (root / "all.txt").write_text("\n".join(train + val))
            write_json(root / "plan.json", {"train_sequences": train, "val_sequences": val})
            command = [sys.executable, "-B", str(SCRIPT.with_name("build_hot3d_sample_split.py")),
                "--processed-root", str(root / "processed"), "--sequence-list", str(root / "all.txt"),
                "--out-dir", str(root / "manifests"), "--fixed-split", str(root / "plan.json"),
                "--val-count", "1", "--overwrite"]
            subprocess.run(command, check=True, capture_output=True)
            result = pipeline.read_json(root / "manifests/split.json")
            self.assertEqual(result["train_sequences"], train)
            self.assertEqual(result["val_sequences"], val)
            write_json(root / "plan.json", {"train_sequences": train, "val_sequences": train})
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def exercise_pipeline(self, fail_visibility=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            process = root / "processed"
            old = {"train_sequences": ["P0001_a", "P0002_b"], "val_sequences": ["P0001_c"]}
            additional = ["P0001_d", "P0002_e"]
            first = None
            for sequence in old["train_sequences"] + old["val_sequences"]:
                row = prepared(process / sequence)
                first = first or row
            write_json(out / "manifests/split.json", old)
            (out / "manifests/train_windows.jsonl").write_text(json.dumps(first) + "\n")
            (root / "aria_train_additional_96.txt").write_text("\n".join(additional))
            for sequence in additional:
                downloaded(root / "data" / sequence)
            sequences = old["train_sequences"] + old["val_sequences"] + additional
            write_json(root / "download_urls/Hot3DAria_download_urls.json", {"sequences": {
                item: {group: {"download_url": "https://example.invalid/fixture"}
                       for group in pipeline.REQUIRED_GROUPS} for item in sequences}})
            for name in ("weights", "MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
                (root / name).touch()
            args = argparse.Namespace(gpus="5,6", hot3d_root=root, out_root=out,
                processed_root=None, hot3d_code_root=root, additional_list=None,
                url_json=None, old_split=None, val_count=2, seed=42,
                poll_seconds=0.001, stable_checks=2, max_wait_hours=0,
                mano_model_folder=root, visibility_python=Path(sys.executable),
                visibility_root=root, visibility_checkpoint=root / "weights",
                hand_uni_root=root, pi3_root=root, pi3x_checkpoint=root / "weights",
                compact_cache_root=root, training_checkpoint=None, dry_run=False)
            calls = []

            def stage(name, command, log_root, env=None):
                calls.append((name, command, env))
                if name.startswith("prepare_"):
                    prepared(Path(command[command.index("--out-dir") + 1]))
                if fail_visibility and name == "visibility_train":
                    raise subprocess.CalledProcessError(1, command)

            with patch.object(pipeline, "parse_args", return_value=args), \
                 patch.object(pipeline, "run_stage", side_effect=stage), \
                 patch.object(pipeline.time, "sleep"), patch.object(pipeline, "log"):
                if fail_visibility:
                    with self.assertRaises(subprocess.CalledProcessError):
                        pipeline.main()
                else:
                    pipeline.main()
            names = [call[0] for call in calls]
            self.assertEqual(names[:2], ["prepare_" + item for item in additional])
            self.assertEqual(names[2:5], ["split", "tracks_train", "tracks_val"])
            if fail_visibility:
                self.assertNotIn("compact_train", names)
                self.assertFalse((out / "logs/incremental_exports/completed.json").exists())
            else:
                self.assertEqual(names[5:], ["visibility_train", "visibility_val", "compact_train", "compact_val"])
                for name, command, env in calls[5:]:
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5,6")
                    self.assertEqual(command[command.index("--num-shards") + 1], 2)
                    self.assertNotIn("--overwrite", command)
                self.assertTrue((out / "logs/incremental_exports/completed.json").is_file())

    def test_pipeline_order_and_incremental_preparation(self):
        self.exercise_pipeline()

    def test_visibility_failure_stops_compact(self):
        self.exercise_pipeline(fail_visibility=True)


if __name__ == "__main__":
    unittest.main()
