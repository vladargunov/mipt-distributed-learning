import torch


if __name__ == '__main__':
    world_size = torch.distributed.init_process_group
    torch.distributed.init_process_group(
        "nccl",
        rank=0,
        world_size=world_size
    )
