import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional, Dict
import torch
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from processing.framework import DLFramework

class DistributedDLFramework(DLFramework):
    def __init__(self, dataset,  **kwargs):
        super().__init__(**kwargs)

        self.gpu_id = int(os.environ["LOCAL_RANK"])

        self.prepare_data(dataset)
        self._model = self._model.to(self.gpu_id)
        self._model = DDP(self._model, device_ids=[self.gpu_id])



    def train(self,
                epochs: int = 2,
                batch_size: int = 4,
                validation_frequency: Optional[int] = None):
        """
        Perform train and validation on given data in prepare_data
        """
        print("...Start Training Loop in Distributed Mode...")
        for epoch in range(1, epochs + 1):
            losses_cache = {"train": 0, "validation": 0}
            for features, targets in self._dataset.train_dataloader(batch_size, distributed_mode=True):
                # Send all data to same gpu
                features = features.to(self.gpu_id)
                targets = targets.to(self.gpu_id)

                self._model.zero_grad()
                output = self.forward(features=features)
                loss = self._loss(output, targets)
                losses_cache["train"] += loss
                loss.backward()
                self._optimizer.step()

            if epoch % validation_frequency == 0:
                for features, targets in self._dataset.validation_dataloader(
                    batch_size, distributed_mode=True
                ):
                    with torch.no_grad():
                        output = self.forward(features=features)
                        losses_cache["validation"] += loss

            # Determine print information
            gpu_id = f"GPU : {self.gpu_id}"
            epoch_print = f"Epoch : {epoch}"
            train_print = f"Train Loss : {losses_cache['train']:.2f}"
            val_print = f"Validation Loss : {losses_cache['validation']:.2f}"
            print(gpu_id, epoch_print, train_print, val_print, sep=" | ")


        print("...Training Loop Completed...")
