import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding as CS336Embedding
from cs336_basics.model import Linear as CS336Linear
#types we acutally want to shard
SHARDABLE_TYPES = (nn.Linear, nn.Embedding, CS336Linear, CS336Embedding)


class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        # we only want our own parameters from the general module
        self.num_ranks = dist.get_world_size()
        self.rank = dist.get_rank() # my rank!
        self.module = module
        self.fdsp_modules = []
        self.compute_dtype = compute_dtype
        self.gradient_handles = []
    
       # self.my_modules = {} WE DONT NEED THIS WE ARE OVERWRITING MODULE ITSELF
        with torch.no_grad():
            for name, mod in module.named_modules():
                if isinstance(mod, SHARDABLE_TYPES):
                    weight = mod.weight # this is the actual tensor
                    len_dim_0 = weight.shape[0]
                    size_per_rank = len_dim_0 // self.num_ranks 
                    start_idx = self.rank * size_per_rank
                    end_idx = start_idx + size_per_rank
                    # the normal syntax is a view of the original full weight so without hte clone it messes it up
                    my_weight = weight[start_idx:end_idx, :].detach().clone()
                    my_shard = nn.Parameter(my_weight)
                    mod.weight = my_shard # replacing the current module weight with my weight! 
                    mod.local_weight = my_shard
                    self.fdsp_modules.append(mod) # keeping track of the modules in order.
                    self.register_fsdp_hooks(mod) # registering the hooks for the module
                    # We want to shard over dim = 0
                    # Rank 0 will own

    def register_fsdp_hooks(self, mod: nn.Module):
        #triggered before forward
        mod.register_forward_pre_hook(self.forward_pre_hook)
        #triggered after forward
        mod.register_forward_hook(self.forward_hook)


    # all gather step 
    def forward_pre_hook(self, mod, input):
        if hasattr(mod, 'prefetch_handle') and mod.prefetch_handle is not None:
            mod.prefetch_handle.wait() # wait for it to finish loading in
            mod.weight = nn.Parameter(mod.prefetch_tensor)
            mod.prefetch_handle = None
            mod.prefetch_tensor = None
            mod.prefetch_input = None
        else: # Layer 0 or layer 1
            my_shard = mod.local_weight # everyone has a piece of this 
            dtype = self.compute_dtype if self.compute_dtype is not None else my_shard.dtype
            full_shape = (my_shard.shape[0] * self.num_ranks, my_shard.shape[1]) # multiplying it back
            rewritten_tensor = torch.empty(full_shape, device=my_shard.device, dtype=dtype)
            shard_to_gather = my_shard.detach().to(dtype)
            dist.all_gather_into_tensor(rewritten_tensor, shard_to_gather)
            mod.weight = nn.Parameter(rewritten_tensor) # this is the tensor with all the shards so its the full layer
            
        def reduce_scatter_hook(full_grad):
            my_grad_shard = torch.empty_like(mod.local_weight, dtype = full_grad.dtype)
            handle = dist.reduce_scatter_tensor(my_grad_shard, full_grad, async_op=True, op=dist.ReduceOp.AVG)
            self.gradient_handles.append((mod, handle, my_grad_shard))
            return torch.zeros_like(full_grad) 
 
        mod.weight.register_hook(reduce_scatter_hook)

    # After forward
    def forward_hook(self, mod, input, output): # Note that we're passing in 'mod' which is a specifi cmodule we want to rewrite
        with torch.no_grad():
            mod.weight = None # freeing it up
            mod.weight = mod.local_weight 
        
        # load in the weights for the next layer
        current_index = self.fdsp_modules.index(mod) # our current index
        future_index = current_index + 2 
        if future_index < len(self.fdsp_modules):
            # Load it in 
            next_module = self.fdsp_modules[future_index] 
            next_shard = next_module.local_weight # get the weights of the next one
            dtype = self.compute_dtype if self.compute_dtype is not None else next_shard.dtype
            full_shape = (next_shard.shape[0] * self.num_ranks, next_shard.shape[1]) # same logic
            rewritten_tensor = torch.empty(full_shape, device=next_shard.device, dtype=dtype) 
            shard_to_gather = next_shard.detach().to(dtype)
            handle = dist.all_gather_into_tensor(rewritten_tensor, shard_to_gather, async_op=True)
            next_module.prefetch_handle = handle
            next_module.prefetch_tensor = rewritten_tensor
            next_module.prefetch_input = shard_to_gather
       

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
        
    def finish_gradient_synchronization(self):
        for mod, handle, grad_buffer in self.gradient_handles:
            handle.wait()
            # Casting back to FP32
            mod.local_weight.grad = grad_buffer.to(mod.local_weight.dtype)

        # non fsdp params
        fsdp_params = {id(mod.local_weight) for mod in self.fdsp_modules}
        for param in self.module.parameters():
            if id(param) in fsdp_params:
                continue
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        self.gradient_handles.clear()
