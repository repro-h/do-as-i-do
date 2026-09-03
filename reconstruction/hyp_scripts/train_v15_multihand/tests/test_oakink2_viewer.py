import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "visualize_oakink2_v16_mano_object.py"
SPEC = importlib.util.spec_from_file_location("oak_viewer", SCRIPT)
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


class ViewerTests(unittest.TestCase):
    def test_source_frames_are_not_cache_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = []
            for frame, source in ((0, 1), (1, 5)):
                label = Path(tmp) / f"{frame}.npz"
                np.savez(label, source_frame_id=source, joint_3d=np.zeros((2, 21, 3)),
                         hand_sides=["left", "right"], extrinsics=np.eye(4),
                         intrinsics=np.eye(3), image_wh=[848, 480])
                labels.append(str(label))
            row = dict(frame_indices=[0, 1], source_frame_ids=[1, 5], label_paths=labels)
            records = viewer.frame_records([row, row])
            self.assertEqual([records[i]["source"] for i in (0, 1)], [1, 5])
            self.assertEqual(set(records[0]["joints"]), {"left", "right"})
            wrong = dict(row, source_frame_ids=[1, 6])
            with self.assertRaises(ValueError):
                viewer.frame_records([row, wrong])
            with self.assertRaises(ValueError):
                viewer.frame_records([wrong])

    def test_mesh_id_selection_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("O02@0017@00001", "O02@0017@00002"):
                directory = Path(tmp) / name
                directory.mkdir()
                (directory / "model.obj").touch()
            selected = viewer.object_mesh_paths(tmp, ["O02@0017@00001"])
            self.assertEqual(list(selected), ["O02@0017@00001"])
            with self.assertRaises(FileNotFoundError):
                viewer.object_mesh_paths(tmp, ["O02@0017"])
            with self.assertRaises(ValueError):
                viewer.object_mesh_paths(tmp, ["../other"])


if __name__ == "__main__":
    unittest.main()
