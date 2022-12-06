import torch
import torch.nn as nn
import torch.distributed as dist

class DistrMLP(nn.Module):
    def __init__(self, input_shape=128, output_shape=64,
                 activation=nn.ReLU, num_gpus=2):
        super().__init__()
        self.num_gpus = num_gpus


        for idx in range(num_gpus):
            setattr(self, f'fc{idx}', nn.Linear(input_shape // self.num_gpus, output_shape))

        self.relu = nn.ReLU()


    def forward(self, x):
        outs = []
        for idx in range(self.num_gpus):
            x = x.to(idx)
            x = self.relu(getattr(self, f'fc{idx}')(x))
            # Return it to host
            outs.append(x.to(0))
        return torch.cat(outs, -1)
