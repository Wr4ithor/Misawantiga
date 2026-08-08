from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


class BinaryDataset(IterableDataset):
    def __init__(self, path, block_size, dtype):
        self.path = Path(path)
        self.block_size = block_size
        self.dtype = dtype


    def __iter__(self) -> Iterator:
        tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        # How many complete windows fit
        num_samples = len(tokens) - self.block_size

        # Split across GPUs (DDP Ranks)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Each GPU gets a contiguous chunk of the data
        per_rank = num_samples // world_size
        rank_start = rank * per_rank
        if rank < world_size - 1:
            rank_end = rank_start + per_rank
        else:
            rank_end = num_samples

        # Split across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = rank_start, rank_end
        else:
            per_worker = (rank_end - rank_start) // worker_info.num_workers
            start = rank_start + worker_info.id * per_worker
            if worker_info.id < worker_info.num_workers -1:
                end = start + per_worker
            else:
                end = rank_end

        # Yield the windows
        for i in range(start, end):
            x = tokens[i : i + self.block_size]
            y = tokens[i + 1 : i + self.block_size + 1]

            yield (
                torch.from_numpy(x.copy()).long(),
                torch.from_numpy(y.copy()).long() # If you want int16, do y.astype(np.int16)
            )