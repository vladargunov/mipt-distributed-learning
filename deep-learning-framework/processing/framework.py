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

    def train(
        self,
        epochs: int = 2,
        batch_size: int = 4,
        validation_frequency: Optional[int] = None,
    ):
        """
        Perform train and validation on given data in prepare_data
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
            train_print = f"Train Loss : {losses_cache['train']}"
            val_print = f"Validation Loss : {losses_cache['validation']}"
            if losses_cache["validation"] != 0:
                print(epoch_print, train_print, val_print, sep=" | ")
            else:
                print(epoch_print, train_print)

        print("...Training Loop Completed...")

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
                    + "Please call 'prepare_data' method and call thsi method again."
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
                + "Please call 'prepare_data' method and call thsi method again."
            )
