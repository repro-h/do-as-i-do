import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "ego_launcher", Path(__file__).resolve().parents[1] / "train_ego_same_batch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EgoSplitTests(unittest.TestCase):
    def test_filter_preserves_rows_and_counts_unique_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            for split in ("train", "val"):
                names = ["dexycb", "hot3d", "oakink2", "taco"]
                if split == "train":
                    names.append("h2o")
                rows = [dict(dataset=n, stream_id=n + split, frame_indices=f,
                             visibility_npz="original/visibility.npz")
                        for n in names for f in ([0, 1], [1, 2])]
                (root / "manifests" / f"{split}_windows.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows))
            selected, report = module.select_and_audit(root)
            self.assertEqual(len(selected["train"]), 8)
            self.assertEqual(report["train"]["h2o"]["unique_frames"], 3)
            self.assertEqual(selected["val"][0]["visibility_npz"], "original/visibility.npz")
            path = root / "manifests/val_windows.jsonl"
            path.write_text(path.read_text().replace("hot3dval", "hot3dtrain"))
            with self.assertRaisesRegex(ValueError, "overlap"):
                module.select_and_audit(root)


if __name__ == "__main__":
    unittest.main()
