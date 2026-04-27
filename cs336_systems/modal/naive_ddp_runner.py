## Problem naive_ddp 
import json
import os
from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS

def worker(rank, world_size):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from copy import deepcopy

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size) 
    device= f"cuda:{rank}"

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 10, bias=False)
            self.fc2 = nn.Linear(10, 50, bias=False)                                
            self.fc3 = nn.Linear(50, 5, bias=False)
            self.relu = nn.ReLU()                                                   
                    
        def forward(self, x):                                                       
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x) 
    
    model = ToyModel().to(device)
    # broadcast from rank 0 to all other ranks what params we have
    # if i am rank 0 
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)

    if rank == 0:
        baseline = deepcopy(model)
        baseline_optimizer = optim.SGD(baseline.parameters(), lr=0.01)
        
    # should synchronize with torch.cuda.synchronize() when benchmarking
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    n  = 32
    torch.manual_seed(41)
    local_batch_size = n // world_size 
    all_x = torch.randn(n, 10, device=device)
    all_y = torch.randn(n, 5, device=device)

    offset = rank * local_batch_size
    x_local_data = all_x[offset : offset + local_batch_size]
    y_local_data = all_y[offset : offset + local_batch_size]

    for i in range (5):
        optimizer.zero_grad()

        output = model(x_local_data)
        loss = torch.nn.functional.mse_loss(output, y_local_data)
        loss.backward()

        for param in model.parameters():
            dist.all_reduce(param.grad)
            param.grad /= world_size

        optimizer.step()

        if rank == 0:
            baseline_optimizer.zero_grad()
            baseline_output = baseline(all_x)
            baseline_loss = torch.nn.functional.mse_loss(baseline_output, all_y)
            baseline_loss.backward()
            baseline_optimizer.step()

# compare the rank 0 parameters with the baseline parameters
    if rank ==0:
        for param, baseline_param in zip(model.parameters(), baseline.parameters()):
            assert torch.allclose(param, baseline_param)
        print("Model parameters match baseline parameters")

    dist.destroy_process_group()


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200:2", timeout=600)
def run_naive_ddp(world_size):
    import torch.multiprocessing as mp
    print(f"spawning {world_size} workers")
    mp.spawn(fn=worker, args=(world_size,), nprocs=world_size, join=True)
    return "ok"


@app.local_entrypoint()
def main():
    result = run_naive_ddp.remote(world_size=2)
    print(f"got: {result}")