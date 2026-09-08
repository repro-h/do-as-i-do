import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import DexYCBMultiHandWindowDataset
from compact_model import ray_anchor_relative_uv


class DenseProvider:
    def __init__(self, time):
        self.time = time

    def __call__(self, row):
        return {
            "geometry_patch_features": np.zeros(
                (self.time, 1, 1, 1), dtype=np.float32
            ),
            "geometry_feature_grid_hw": np.asarray([1, 1], dtype=np.int32),
            "metric_window_features": np.zeros((1, 1), dtype=np.float32),
            "intrinsics_resized": np.broadcast_to(
                np.eye(3, dtype=np.float32)[None], (self.time, 3, 3)
            ).copy(),
            "resized_wh": np.asarray([50, 50], dtype=np.int32),
            "horizontal_mirror": np.asarray(False),
        }


class WristAnchorTests(unittest.TestCase):
    def make_dataset(self, root, uv, visibility, max_gap=3):
        root = Path(root)
        labels = []
        for frame, frame_uv in enumerate(uv):
            path = root / f"label_{frame:03d}.npz"
            xyz = np.zeros((1, 21, 3), dtype=np.float32)
            xyz[..., 2] = 1.0
            np.savez(
                path,
                seg=np.zeros((100, 100), dtype=np.uint8),
                joint_2d=frame_uv[None],
                joint_3d=xyz,
            )
            labels.append(str(path))

        visibility_path = root / "visibility.npz"
        np.savez(
            visibility_path,
            frame_indices=np.arange(len(uv), dtype=np.int64),
            joint_visibility=visibility[:, None],
            visibility_valid=np.ones((len(uv), 1), dtype=bool),
        )
        manifest = root / "windows.jsonl"
        row = {
            "dataset": "test",
            "stream_id": "stream",
            "start": 0,
            "end": len(uv),
            "frame_indices": list(range(len(uv))),
            "label_paths": labels,
            "visibility_npz": str(visibility_path),
        }
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return DexYCBMultiHandWindowDataset(
            manifest,
            None,
            max_hands=1,
            visibility_source="detector",
            max_anchor_frames=max_gap,
            near_anchor_frames=1,
            near_missing_weight=0.5,
            far_missing_weight=0.2,
            dense_provider=DenseProvider(len(uv)),
            query_source="gt",
            supervision_source="target",
        )

    @staticmethod
    def base_uv(time):
        return np.full((time, 21, 2), np.nan, dtype=np.float32)

    def test_direct_spatial_and_temporal_sources(self):
        with TemporaryDirectory() as root:
            uv = self.base_uv(5)
            uv[0, 0], uv[0, 1] = (10, 20), (20, 20)
            uv[2, 1] = (40, 20)
            uv[4, 0], uv[4, 1] = (50, 20), (60, 20)
            visibility = np.ones((5, 21), dtype=np.float32)
            dataset = self.make_dataset(root, uv, visibility)
            sample = dataset[0]

            self.assertEqual(sample["ray_anchor_source"][:, 0].tolist(), [1, 3, 2, 3, 1])
            anchor_px = (
                (sample["ray_anchor_uv"][:, 0].numpy() + 1.0)
                * 0.5 * 49.0
            )
            np.testing.assert_allclose(anchor_px[2], [15.0, 10.0], atol=1e-5)
            np.testing.assert_allclose(
                sample["supervision_weight"][:, 0].numpy(),
                [1.0, 0.5, 0.2, 0.5, 1.0],
            )

    def test_low_visibility_joint_falls_back_to_temporal(self):
        with TemporaryDirectory() as root:
            uv = self.base_uv(3)
            uv[0, 0], uv[0, 1] = (10, 20), (20, 20)
            uv[1, 1] = (30, 20)
            uv[2, 0], uv[2, 1] = (30, 20), (40, 20)
            visibility = np.ones((3, 21), dtype=np.float32)
            visibility[1, 1] = 0.49
            dataset = self.make_dataset(root, uv, visibility)
            sample = dataset[0]
            self.assertEqual(int(sample["ray_anchor_source"][1, 0]), 3)

    def test_gap_beyond_max_has_no_anchor_or_supervision(self):
        with TemporaryDirectory() as root:
            uv = self.base_uv(10)
            uv[0, 0], uv[0, 1] = (10, 20), (20, 20)
            uv[9, 1] = (30, 20)
            visibility = np.ones((10, 21), dtype=np.float32)
            dataset = self.make_dataset(root, uv, visibility, max_gap=3)
            sample = dataset[0]
            self.assertEqual(int(sample["ray_anchor_source"][9, 0]), 0)
            self.assertEqual(float(sample["supervision_weight"][9, 0]), 0.0)


class RayAnchorRelativeUvTests(unittest.TestCase):
    @staticmethod
    def inputs(source, wrist, joint, joint_valid=True):
        joint_uv = torch.zeros((1, 1, 1, 21, 2), dtype=torch.float32)
        joint_uv[..., 0, :] = torch.tensor(wrist)
        joint_uv[..., 1, :] = torch.tensor(joint)
        valid = torch.zeros((1, 1, 1, 21), dtype=torch.bool)
        valid[..., 0] = source == 1 and joint_valid
        valid[..., 1] = joint_valid
        anchor = torch.tensor(wrist, dtype=torch.float32).reshape(1, 1, 1, 2)
        anchor_source = torch.tensor([[[source]]], dtype=torch.int64)
        return joint_uv, valid, anchor, anchor_source

    def test_source_1_matches_raw_wrist_reference(self):
        joint_uv, valid, anchor, source = self.inputs(
            1, wrist=(0.2, -0.3), joint=(0.5, 0.1)
        )
        local_uv, relative_valid = ray_anchor_relative_uv(
            joint_uv, anchor, valid, source
        )
        old_local_uv = joint_uv - joint_uv[..., :1, :]
        torch.testing.assert_close(local_uv[..., :2, :], old_local_uv[..., :2, :])
        self.assertTrue(bool(relative_valid[..., 1]))

    def test_source_2_uses_estimated_anchor_not_invalid_wrist(self):
        joint_uv, valid, _, source = self.inputs(
            2, wrist=(-1.0, -1.0), joint=(0.5, 0.6)
        )
        anchor = torch.tensor([0.1, 0.2]).reshape(1, 1, 1, 2)
        local_uv, relative_valid = ray_anchor_relative_uv(
            joint_uv, anchor, valid, source
        )
        torch.testing.assert_close(
            local_uv[..., 1, :], torch.tensor([[[[0.4, 0.4]]]])
        )
        self.assertTrue(bool(relative_valid[..., 1]))

    def test_source_3_invalid_joints_have_zero_relative_metadata(self):
        joint_uv, valid, anchor, source = self.inputs(
            3, wrist=(-1.0, -1.0), joint=(0.4, 0.5), joint_valid=False
        )
        anchor = torch.tensor([0.1, 0.2]).reshape(1, 1, 1, 2)
        local_uv, relative_valid = ray_anchor_relative_uv(
            joint_uv, anchor, valid, source
        )
        self.assertTrue(torch.equal(local_uv, torch.zeros_like(local_uv)))
        self.assertFalse(bool(relative_valid.any()))

    def test_source_0_has_no_relative_metadata(self):
        joint_uv, valid, anchor, source = self.inputs(
            0, wrist=(-1.0, -1.0), joint=(0.4, 0.5)
        )
        local_uv, relative_valid = ray_anchor_relative_uv(
            joint_uv, anchor, valid, source
        )
        self.assertTrue(torch.equal(local_uv, torch.zeros_like(local_uv)))
        self.assertFalse(bool(relative_valid.any()))


if __name__ == "__main__":
    unittest.main()
