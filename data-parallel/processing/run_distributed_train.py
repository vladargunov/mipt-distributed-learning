import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", default=2)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.distributed.init_process_group(
        "nccl",
        rank=args.local_rank,
        world_size=args.world_size,
    )
