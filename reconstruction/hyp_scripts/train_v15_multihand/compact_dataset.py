"""Metadata dataset plus compact Pi3X joint/global candidate tokens."""

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import DexYCBMultiHandWindowDataset


class CompactWindowDataset(Dataset):
    def __init__(self, metadata_dataset, compact_provider):
        self.metadata = metadata_dataset
        self.compact_provider = compact_provider
        self.rows = metadata_dataset.rows

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample = self.metadata[index]
        compact = self.compact_provider(
            self.rows[index], sample["joint_uv"].numpy()
        )
        for key in ("point_features", "grid_uv", "grid_confidence", "grid_valid"):
            sample.pop(key, None)
        for key in (
            "joint_patch_features",
            "joint_patch_uv",
            "joint_patch_confidence",
            "joint_patch_valid",
            "global_features",
            "global_uv",
            "global_confidence",
        ):
            value = np.asarray(compact[key])
            if np.issubdtype(value.dtype, np.floating):
                value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            sample[key] = torch.from_numpy(value.copy())
        metric = np.asarray(compact["metric_window_features"], dtype=np.float32)
        metric = metric.reshape(-1, metric.shape[-1]).mean(axis=0)
        sample["metric_window_features"] = torch.from_numpy(metric)
        time = len(self.rows[index]["frame_indices"])
        intrinsics = np.asarray(compact["intrinsics_resized"], dtype=np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.broadcast_to(intrinsics[None], (time, 3, 3)).copy()
        resized_wh = np.asarray(compact["resized_wh"], dtype=np.float32).reshape(2)
        sample["intrinsics"] = torch.from_numpy(intrinsics.copy())
        sample["image_wh"] = torch.from_numpy(
            np.broadcast_to(resized_wh[None], (time, 2)).copy()
        )
        return sample


def make_metadata_dataset(*args, **kwargs):
    return DexYCBMultiHandWindowDataset(*args, **kwargs)
