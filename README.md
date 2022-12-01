# mipt-distributed-learning
This repo contains homework for distributed learning course from MIPT.

-----
## Homework 1
The framework is available in [deep-learning-framework](deep-learning-framework) folder. Also a usage example can be found in this [notebook](deep-learning-framework/examples/test_cuda_hw1.ipynb).

-----
## Homework 2
The data-parallel code which runs an example on several GPUs is available at [data-parallel](data-parallel) folder. The code seems to work by executing the command `python -m torch.distributed.run run_distributed_train.py` while being in the directory `data-parallel/processing`. However there were connectivity issues while running this code in the server and the code was executing on only one GPU, whichever was specified first. 

The issue reported was as follows (which I was unable to solve):

```
RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
```
