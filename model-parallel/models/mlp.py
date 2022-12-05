import torch
import torch.nn as nn
import torch.distributed as dist

class DistrMLP(nn.Module):
    def __init__(self, input_shape=(16, 128), output_shape= 64,
                 activation=nn.ReLU, num_gpus=2):
        super().__init__()


        self.current_rank = dist.get_rank()

        # Linear shapes
        in_shape_linear = input_shape[1] // num_gpus
        out_shape_linear = output_shape // num_gpus

        self.fc = nn.Linear(in_shape_linear, out_shape_linear).to(self.current_rank)
        self.relu = nn.RelU()
        if current_rank == 0:  #when root
            self.out_tensors = [torch.zeros(*input_shape).to(0) for _ in range(num_gpus)]
        else:
            self.out_tensors = None



    def forward(self, x):
        x = x.to(self.current_rank)
        x = self.relu(self.fc(x))

        if self.current_rank == 0:
            dist.gather(x, gather_list=self.out_tensors)
            print(self.out_tensors)
            return torch.cat(self.out_tensors, -1)
        else:
            dist.gather(x)
