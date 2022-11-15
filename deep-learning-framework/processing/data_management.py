import torch
import numpy as np
from typing import Dict, Optional


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

    def __getitem__(self, index):
        return self._features[index], self._targets[index]

    def __len__(self):
        return self._targets.shape[0]

    def train_dataloader(self, batch_size):
        """
        Returns a train dataloader of features and targets
        """
        train_dataloader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, collate_fn=self.default_collate_fn
        )
        return train_dataloader

    def validation_dataloader(self, batch_size):
        """
        Returns a validation dataloader of features and targets
        """
        val_dataloader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, collate_fn=self.default_collate_fn
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

        return torch.stack(features), torch.stack(targets).reshape(-1)
