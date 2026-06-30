import os
import torch
import torch.distributed as dist
import numpy as np
import random

# ─── Device ───
def _get_device(device_id="cuda:0"):
    if torch.cuda.is_available():
        try:
            d = torch.device(device_id)
            t = torch.zeros(2).to(d)
            t = t + 1
            return d
        except Exception:
            pass
    return torch.device("cpu")


def ddp_setup():
    """Initialize DDP process group. Returns local_rank and world_size."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def init_device():
    """Initialize device after DDP setup (if any)."""
    if dist.is_initialized():
        local_rank = dist.get_rank()
        d = torch.device(f'cuda:{local_rank}')
    else:
        d = _get_device("cuda:0")
    if is_main_process():
        print(f"[Device] Using: {d}")
    return d

device = None
test_device = None

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True



