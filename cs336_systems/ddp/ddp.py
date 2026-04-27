import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        # given an instantiated pytorch nn.module construct a DDP container
        super().__init__()
        self.module = module
        # now we want to broadcast the parameters from rank 0 to all other ranks
        with torch.no_grad():
            for param in module.parameters():
                dist.broadcast(param, src=0) # broadcasting

        self.handles = []

        for param in self.module.parameters():
            if param.requires_grad:
                # this runs when we call backward on that specific param
                def hook(p):
                    p.grad /= dist.get_world_size() 
                    handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append(handle) 
                param.register_post_accumulate_grad_hook(hook)


    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        
        self.handles.clear()
