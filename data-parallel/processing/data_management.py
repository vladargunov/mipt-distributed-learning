import torch
import numpy as np
from typing import Dict, Optional

from torch.utils.data.distributed import DistributedSampler


class Dataset:
    def __init__(self, data: Dict[str, Optional[np.array]], dtype=torch.float32):
        """
        Default Dataset class that takes in a dict of features and
        targets in the form of numpy arrays

        Example data:
        {
        "features": np.random.rand(12, 2),
        "targets": np.random.randint(2, size=12),
        }
        """

        self._features = torch.from_numpy(data["features"]).to(dtype)
        self._targets = torch.from_numpy(data["targets"]).to(dtype)
        self.device = "cpu"

    def __getitem__(self, index):
        return self._features[index], self._targets[index]

    def __len__(self):
        return self._targets.shape[0]

    def train_dataloader(self, batch_size, distributed_mode=False):
        """
        Returns a train dataloader of features and targets
        """
        sampler = DistributedSampler(self) if distributed_mode else None
        train_dataloader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, collate_fn=self.default_collate_fn,
            sampler=sampler
        )
        return train_dataloader

    def validation_dataloader(self, batch_size, distributed_mode=False):
        """
        Returns a validation dataloader of features and targets
        """
        sampler = DistributedSampler(self) if distributed_mode else None
        val_dataloader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, collate_fn=self.default_collate_fn,
            sampler=sampler
        )
        return val_dataloader

    def default_collate_fn(self, batch):
        """
        Default collate function used for bundling features and targets
        """
        features = []
        targets = []

        for sample in batch:
            feature, target = sample
            features.append(feature)
            targets.append(target)

        return torch.stack(features).to(self.device), torch.stack(targets).reshape(
            -1
        ).to(self.device)

    def _send_to_gpu(self, device_rank=0):
        """Sends data to GPU"""
        self.device = f"cuda:{device_rank}"

    def _send_to_cpu(self):
        """Sends data to CPU"""
        self.device = "cpu"
