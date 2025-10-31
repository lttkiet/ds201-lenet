
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DistributedSampler

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
        global_rank = int(os.environ['RANK'])

        if torch.cuda.is_available():
            backend = 'nccl'
            torch.cuda.set_device(local_rank)
        else:
            backend = 'gloo'

        dist.init_process_group(backend=backend)

        return True, local_rank, world_size, global_rank

    return False, 0, 1, 0

def get_device(is_distributed, local_rank):
    if torch.cuda.is_available():
        if is_distributed:
            return torch.device(f'cuda:{local_rank}')
        else:
            return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_distributed_sampler(dataset, is_distributed, shuffle=True):
    if is_distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    return None

def wrap_model_distributed(model, local_rank, is_distributed):
    if is_distributed:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = DP(model)
    return model

def reduce_tensor(tensor, world_size):
    if not dist.is_initialized():
        return tensor.item()

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return (rt / world_size).item()

def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0