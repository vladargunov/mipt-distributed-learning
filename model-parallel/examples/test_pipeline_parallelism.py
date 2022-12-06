import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.distributed import init_process_group, destroy_process_group
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.distributed.pipeline.sync import Pipe
from processing import framework
from processing.data_management import Dataset
from models.mlp import DistrMLP

def ddp_setup():
    init_process_group(backend="nccl")


def main():
    ddp_setup()

    # Create sample dataset
    # Sample data
    sample_data = {
            "features": np.random.rand(16, 128),
            "targets": np.random.randint(2, size=16),
        }
    if dist.get_rank() == 0:
      torch.distributed.rpc.init_rpc('worker', rank=dist.get_rank(), world_size=1)
      sample_input_dataset = Dataset(data=sample_data)


      # Create a model and loss
      test_model = torch.nn.Sequential(torch.nn.Linear(128, 64).to(0),
                                     torch.nn.Linear(64, 2).to(1),
                                     torch.nn.Sigmoid())

      test_model = Pipe(test_model, chunks=4)

      test_output = test_model(sample_input_dataset._features.to(0))

      print('Output tensor: ', test_output.local_value())

if __name__ == "__main__":
    main()
