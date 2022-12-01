import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np

from processing import framework
from processing.data_management import Dataset


def run_example():
    # Determine the local rank of the model
    local_rank = torch.distributed.get_rank()

    # Sample data
    sample_data = {
            "features": np.random.rand(12, 2),
            "targets": np.random.randint(2, size=12),
        }

    sample_input_data = Dataset(data=sample_data)
    # This code is similar to the 'test_training.py' script

    # Create a model and loss
    test_model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    test_loss = torch.nn.CrossEntropyLoss()

    # Initalise a framework
    test_framework = framework.DLFramework(model=test_model, loss=test_loss, lr=1)

    # Add data to the framework
    test_framework.prepare_data(dataset=sample_input_data)

    # Send model and data to gpu
    test_framework.send_to_gpu(local_rank)

    # Train model
    test_framework.train(epochs=3, batch_size=4, validation_frequency=1)

if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")
    run_example()
