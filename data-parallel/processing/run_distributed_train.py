import argparse
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np

from processing import framework
from processing.data_management import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", default=2)
    return parser.parse_args()


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
    args = parse_args()
    torch.distributed.init_process_group(
       "nccl",
       rank=args.local_rank,
       world_size=args.world_size,
    )
    run_example()
