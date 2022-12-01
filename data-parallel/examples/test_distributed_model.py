from torch.distributed import init_process_group, destroy_process_group
import torch

import numpy as np

from processing import data_parallel_framework as dpf
from processing.data_management import Dataset

def ddp_setup():
    init_process_group(backend="nccl")


def main():
    ddp_setup()

    # Create sample dataset
    # Sample data
    sample_data = {
            "features": np.random.rand(12, 2),
            "targets": np.random.randint(2, size=12),
        }

    sample_input_dataset = Dataset(data=sample_data)

    # Create a model and loss
    test_model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    test_loss = torch.nn.CrossEntropyLoss()

    # Initalise a framework
    test_framework = dpf.DistributedDLFramework(dataset=sample_input_dataset,
                                                model=test_model,
                                                loss=test_loss,
                                                 lr=1
                                                 )

    # Train model
    test_framework.train(epochs=3, batch_size=4, validation_frequency=1)
    destroy_process_group()



if __name__ == "__main__":
    main()
