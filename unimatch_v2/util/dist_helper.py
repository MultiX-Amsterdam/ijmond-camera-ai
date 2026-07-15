import os
import subprocess

import torch
import torch.distributed as dist


os.environ["USE_LIBUV"] = '0'


def setup_distributed(backend='gloo', port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched by torchrun (works under SLURM or standalone).
        # torchrun sets RANK/LOCAL_RANK/WORLD_SIZE correctly per process,
        # so we must NOT use SLURM_PROCID/SLURM_NTASKS here.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
    elif "SLURM_JOB_ID" in os.environ:
        # Native SLURM launch (srun with --ntasks>1, without torchrun).
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        # Single-GPU / local run without torchrun or SLURM.
        rank = 0
        world_size = 1
        os.environ["MASTER_ADDR"] = "localhost"
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"

    local_rank = int(os.environ.get("LOCAL_RANK", rank % num_gpus))
    torch.cuda.set_device(local_rank if num_gpus > 0 else -1)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank
    )

    return rank, world_size
