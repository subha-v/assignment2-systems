## section 6 - optimizer state sharding
## PROBLEM optimizer_State_sharding 

import torch
import torch.distributed as dist

class OptimizerSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs): 
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

        self.param_to_rank = []
        self.optimizer = None
        super().__init__(params, defaults=kwargs)


    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)

        with torch.no_grad():
            for param, owner_rank in self.param_to_rank:
                dist.broadcast(param.data, src=owner_rank)
        return loss

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        rank = dist.get_rank() # our rank
        world_size = dist.get_world_size()
        owned_params = []

        next_index = len(self.param_to_rank)
        for param in param_group["params"]: 
            owner_rank = next_index % world_size
            self.param_to_rank.append((param, owner_rank))
            if owner_rank == rank:
                owned_params.append(param)
            next_index += 1 
        
        sharded_group = dict(param_group)
        sharded_group["params"] = owned_params
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([sharded_group], **self.kwargs)
        else:
            self.optimizer.add_param_group(sharded_group)

        