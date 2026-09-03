import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "taco_mano_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("taco_diag", SCRIPT)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class DiagnosticsTests(unittest.TestCase):
    def test_rigid_camera(self):
        matrices = np.repeat(np.eye(4)[None], 2, axis=0)
        matrices[1, :2, :2] = [[0, -1], [1, 0]]
        matrices[1, :3, 3] = [1, 2, 3]
        self.assertTrue(diag.camera_rigidity(matrices)["proper_rigid_within_1e_4"])

    def test_nonrigid_camera(self):
        for row, col, value in ((0, 0, 1.2), (0, 1, 0.2), (0, 0, -1), (3, 2, 0.1)):
            with self.subTest(row=row, col=col, value=value):
                matrix = np.eye(4)[None]
                matrix[0, row, col] = value
                self.assertFalse(diag.camera_rigidity(matrix)["proper_rigid_within_1e_4"])
        for matrix in (np.empty((0, 4, 4)), np.eye(3), np.full((1, 4, 4), np.nan)):
            with self.assertRaises(ValueError):
                diag.camera_rigidity(matrix)

    def test_sequential_video_and_release(self):
        for frame_count in (4, 2):
            state = {"read": 0, "released": False, "written": []}

            def read():
                index = state["read"]
                state["read"] += 1
                return index < frame_count, index

            def release():
                state["released"] = True

            def write(path, image):
                state["written"].append(image)
                return True

            cv2 = types.SimpleNamespace(
                VideoCapture=lambda _: types.SimpleNamespace(read=read, release=release),
                imwrite=write,
            )
            with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {"cv2": cv2}):
                video = Path(tmp) / "test.mp4"
                video.touch()
                if frame_count == 4:
                    self.assertEqual(len(diag.export_rgb_frames(video, [1, 3], tmp)), 2)
                    self.assertEqual(state["written"], [1, 3])
                else:
                    with self.assertRaises(RuntimeError):
                        diag.export_rgb_frames(video, [1, 3], tmp)
            self.assertTrue(state["released"])


if __name__ == "__main__":
    unittest.main()
