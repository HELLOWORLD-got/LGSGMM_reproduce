"""torchrun setup and exact, unpadded evaluation sharding."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def initialize(device):
    size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        if size > 1:
            device = f'cuda:{local_rank}'
        torch.cuda.set_device(torch.device(device))
    if size > 1:
        backend = 'nccl' if device.startswith('cuda') else 'gloo'
        dist.init_process_group(backend, timeout=timedelta(minutes=30))
    return device


def gather(value):
    if world_size() == 1:
        return [value]
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    return values


def reduce_totals(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size() > 1:
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def broadcast_buffers(model):
    if world_size() > 1:
        for buffer in unwrap(model).buffers():
            dist.broadcast(buffer, src=0)


class EvaluationSampler(Sampler):
    """Every example appears on exactly one rank, even for uneven splits."""
    def __init__(self, dataset, replicas=None, process_rank=None):
        self.size = len(dataset)
        self.replicas = world_size() if replicas is None else replicas
        self.process_rank = rank() if process_rank is None else process_rank

    def __iter__(self):
        return iter(range(self.process_rank, self.size, self.replicas))

    def __len__(self):
        return len(range(self.process_rank, self.size, self.replicas))
