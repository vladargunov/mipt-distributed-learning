import logging
import torch
import numpy as np

from processing.data_management import Dataset

from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger("__name__")


class DLFramework:
    """
    Basic Deep Learning Framework
    """

    def __init__(
        self, model: Optional[torch.nn.Module], loss: torch.nn.Module, lr: float = 1e-3
    ):

        self._model = model
        self._loss = loss
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)

    def prepare_data(self, dataset: Dataset):
        """
        Include data in the framework
        """
        self._dataset = dataset

    def forward(self, features: np.array):
        """
        Make a forward pass on model
        """
        try:
            output = self._model(features)
        except:
            logger.warning(f"feature: {features}")
            raise

        return output.reshape(-1)

    def _train_single_device(
        self,
        epochs: int = 2,
        batch_size: int = 4,
        validation_frequency: Optional[int] = None,
    ):
        """
        Perform train and validation on given data in prepare_data
        on single CPU/GPU
        """
        print("...Start Training Loop...")
        for epoch in range(1, epochs + 1):
            losses_cache = {"train": 0, "validation": 0}
            for features, targets in self._dataset.train_dataloader(batch_size):
                self._model.zero_grad()
                output = self.forward(features=features)
                loss = self._loss(output, targets)
                losses_cache["train"] += loss
                loss.backward()
                self._optimizer.step()

            if epoch % validation_frequency == 0:
                for features, targets in self._dataset.validation_dataloader(
                    batch_size
                ):
                    with torch.no_grad():
                        output = self.forward(features=features)
                        losses_cache["validation"] += loss

            # Determine print information
            epoch_print = f"Epoch : {epoch}"
            train_print = f"Train Loss : {losses_cache['train']:.2f}"
            val_print = f"Validation Loss : {losses_cache['validation']:.2f}"
            if losses_cache["validation"] != 0:
                print(epoch_print, train_print, val_print, sep=" | ")
            else:
                print(epoch_print, train_print)

        print("...Training Loop Completed...")

    def _train_multiple_devices(
        self,
        epochs: int = 2,
        batch_size: int = 4,
        validation_frequency: Optional[int] = None,
    ):
        """
        Perform train and validation on given data in prepare_data
        on multiple GPUs
        """
        # In distributed setting send model to current gpu
        local_rank = torch.distributed.get_rank()
        device = f"cuda:{local_rank}"
        self._model = torch.nn.parallel.DistributedDataParallel(self._model)

        print("...Start Training Loop...")
        for epoch in range(1, epochs + 1):
            losses_cache = {"train": 0, "validation": 0}
            for features, targets in self._dataset.train_dataloader(batch_size, distributed_mode=True):
                self._model.zero_grad()
                # Send data to current gpu
                features = features.to(device)
                targets = targets.to(device)

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
            epoch_print = f"Epoch : {epoch}"
            train_print = f"Train Loss : {losses_cache['train']:.2f}"
            val_print = f"Validation Loss : {losses_cache['validation']:.2f}"
            if losses_cache["validation"] != 0:
                print(epoch_print, train_print, val_print, sep=" | ")
            else:
                print(epoch_print, train_print)

        print("...Training Loop Completed...")

    def train(self, epochs: int = 2, batch_size: int = 4, validation_frequency: Optional[int] = None):
        """
        Generic train function which determines number of available
        GPU devices and trains data on them accordingly
        """
        number_devices = torch.cuda.device_count()
        params = locals()
        params.pop('self')
        print(f"Number of GPU devices detected: {number_devices}")
        if number_devices == 0:
            print('...Training on CPU...')
            self._train_single_device(**params)
        elif number_devices == 1:
            print('...Training on single GPU...')
            self.send_to_gpu()
            self._train_single_device(**params)
        else:
            print('...Training on multiple GPUs...')
            self._train_multiple_devices(**params)

    def save(self, path: Path):
        """
        Save model
        """
        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        torch.save(state, path)

    def predict(self, dataset: Dataset):
        """
        Predict data given a dataset type
        """
        with torch.no_grad():
            if next(self._model.parameters()).is_cuda:
                return self.forward(dataset._features.to('cuda'))
            else:
                return self.forward(dataset._features)

    def load_model(self, path: Path):
        """
        Load a model into the Framework in the form
        {
        "model": self._model.state_dict(),
        "optimizer": self._optimizer.state_dict(),
        }
        """
        self._model.load(path)

    def send_to_gpu(self):
        """
        Check if gpu is available and then
        send data and model to the gpu
        """
        if torch.cuda.is_available():
            # Send model to gpu
            self._model.to("cuda")

            # Set CUDA flag for data
            if "_dataset" in dir(self):
                self._dataset._send_to_gpu()
            else:
                print(
                    "Dataset was not included into the framework"
                    + "Please call 'prepare_data' method and call this method again."
                )
        else:
            print("CUDA is not supported on this OS.")

    def send_to_cpu(self):
        """
        Send data and model to cpu
        """
        # Send model to cpu
        self._model.to("cpu")

        # Send data to cpu
        if "_dataset" in dir(self):
            self._dataset._send_to_cpu()
        else:
            print(
                "Dataset was not included into the framework"
                + "Please call 'prepare_data' method and call this method again."
            )
