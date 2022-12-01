# mipt-distributed-learning
This repo contains homework for distributed learning course from MIPT.

-----
## Homework 1
The framework is available in [deep-learning-framework](deep-learning-framework) folder. Also a usage example can be found in this [notebook](deep-learning-framework/examples/test_cuda_hw1.ipynb).

-----
## Homework 2

**Original solution (incorrect):**

~~The data-parallel code which runs an example on several GPUs is available at [data-parallel](data-parallel) folder. The code sometimes works with one gpu by executing the command `torchrun run_train.py` while being in the directory `data-parallel`. However there were connectivity issues while running this code in and the following error appears:~~


~~RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).~~

**Update on 01/12/2022 at 20:26 (correct solution)**

I finally worked out the problem (even though missed a deadline a little bit) -- had to rewrite everything again. Now everything works and below is the instruction how to replicate a simplest example of data-parallel training:

1. Log into the server

2. Install miniconda

3. Create conda environment, log into it and run the command `conda install pip`.

4. Clone my repo by command `git clone https://github.com/vladargunov/mipt-distributed-learning.git`

5. Go into the */mipt-distributed-learning/data-parallel/requirements* directory and run command `pip install -r requirements.txt`

6. Then go into the */mipt-distributed-learning/data-parallel/examples* directory and run the command

`CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=gpu test_distributed_model.py`

After that you will start a process of data-parallel training on two gpus, please note that in `CUDA_VISIBLE_DEVICES` you can pass an arbitrary number of gpus that your machine can support.

Thus, if you would like to use several gpus the commands above should be called, and in case of 1 gpu or cpu please refer to the example from previous exercise.
