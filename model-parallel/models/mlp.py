import torch
import torch.nn as nn
import torch.distributed as dist

class DistrMLP(nn.Module):
    def __init__(self, input_shape=(16, 128), output_shape=64,
                 activation=nn.ReLU, num_gpus=2):
        super().__init__()


        self.current_rank = dist.get_rank()

        self.fc = nn.Linear(input_shape[1], output_shape).to(self.current_rank)
        self.relu = nn.ReLU()

        self.out_tensors = [torch.zeros(input_shape[0], output_shape).to(0) for _ in range(num_gpus)]

    def forward(self, x):
        x = x.to(self.current_rank)
        x = self.relu(self.fc(x))

        if self.current_rank == 0:
            dist.gather(x, gather_list=self.out_tensors)
            return torch.cat(self.out_tensors, -1).to(self.current_rank)
        else:
            dist.gather(x)
            return torch.cat(self.out_tensors, -1).to(self.current_rank)
