import torch
import math

# PROBLEM flash_forward
class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    # produces output O and logsumexp value L
    def forward(ctx, Q, K, V, is_causal):
        # FIRST save for the backward pass
        # ctx standds for context and since it's a static method its first argument
        # is not self, and using ctx instead of self makes that obvious and its how we save thingsd

        # lets set a tile size and then we can torch.split
        b_q = 128
        b_k = 128
        q_tiles = torch.split(Q, b_q, dim=-2)
        k_tiles = torch.split(K, b_k, dim=-2)
        v_tiles = torch.split(V, b_k, dim=-2)
        o_i_list = []
        l_i_list = []

        # loop through tiles of Q
        for q_tile in q_tiles:
            # load q_i from global memory
            # initialize O_i(0) which is B_q x d
            batch_size, b_q, head_dim = q_tile.shape
            o_i = torch.zeros_like(q_tile) 
            l_i = torch.zeros((batch_size, b_q, 1), device = q_tile.device)
            m_i = torch.full((batch_size, b_q, 1), -torch.inf, device = q_tile.device)

            for k_tile, v_tile in zip(k_tiles, v_tiles): #j = k_tile
                s_i = torch.einsum("b q d, b k d -> b q k", q_tile, k_tile)
                s_i = s_i / math.sqrt(head_dim)
                #need to use amax because max you have to do .values
                #keepdim = false means the reduced axis is removed
                #keepdim = true means the reduced axis is kept so the result broadcasts agianst tensors of same shape
                rowmax_si = torch.amax(s_i, dim=-1, keepdim=True) 
                m_prev = m_i
                m_i = torch.maximum(m_prev, rowmax_si)
                p_i = torch.exp(s_i - m_i)
                rowsum_pi = torch.sum(p_i, dim=-1, keepdim=True)
                # Current running sum of the denominator adding the current term
                difference = torch.exp(m_prev - m_i) 
                l_i = difference * l_i + rowsum_pi 
                # need to use @ for matmul
                o_i = difference * o_i + p_i @ v_tile
            
            # divide o_i 
            o_i = o_i / l_i 
            l_i = m_i + torch.log(l_i)
            o_i_list.append(o_i)
            l_i_list.append(l_i.squeeze(-1))
            
        O = torch.cat(o_i_list, dim=-2)
        L = torch.cat(l_i_list, dim=-1)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

