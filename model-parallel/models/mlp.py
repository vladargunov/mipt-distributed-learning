import torch
import torch.nn as nn
import torch.distributed as dist

class DistrMLP(nn.Module):
    def __init__(self, input_shape=128, output_shape=64,
                 activation=nn.ReLU, num_gpus=2):
        super().__init__()
        self.num_gpus = num_gpus


        for idx in range(num_gpus):
            setattr(self, f'fc{idx}', nn.Linear(input_shape, output_shape // self.num_gpus))

        self.relu = nn.ReLU()


    def forward(self, x):
        outs = []
        for idx in range(self.num_gpus):
            z = x.to(idx)
            layer = getattr(self, f'fc{idx}').to(idx)
            z = self.relu(layer(z))
            # Return it to host
            outs.append(z.to(0))
        return torch.cat(outs, 1)
