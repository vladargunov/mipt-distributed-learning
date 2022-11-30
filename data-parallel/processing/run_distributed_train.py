import torch


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.distributed.init_process_group(
        "nccl",
        rank=0,
        world_size=world_size
    )
