import torch
import torch.distributed as dist

class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()

    def forward(self)
