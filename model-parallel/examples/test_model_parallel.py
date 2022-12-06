import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from processing import framework
from processing.data_management import Dataset
from models.mlp import DistrMLP




def main():
    ddp_setup()

    # Create sample dataset
    # Sample data
    sample_data = {
            "features": np.random.rand(16, 128),
            "targets": np.random.randint(2, size=16),
        }

    sample_input_dataset = Dataset(data=sample_data)

    # Create a model and loss
    num_gpus = 2 # set the number of gpus available
    mlp_distr = DistrMLP(input_shape=(16, 128), output_shape=64, num_gpus=num_gpus)
    test_model = nn.Sequential(mlp_distr, torch.nn.Linear(64 * num_gpus, 1).to(dist.get_rank())) # distributed mlp perceptron
    test_loss = torch.nn.CrossEntropyLoss()

    # Initalise a framework
    test_framework = framework.DLFramework(model=test_model, loss=test_loss, lr=1)

    # Add data to the framework
    test_framework.prepare_data(dataset=sample_input_dataset)

    # Train model
    test_framework.train(epochs=3, batch_size=16, validation_frequency=1)



if __name__ == "__main__":
    main()
