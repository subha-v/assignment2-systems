# PROBLEM flash_forward

import torch
import triton
import triton.language as tl
import math

from cs336_systems.flash_attention.flash_backward import FlashAttentionBackwardPytorch

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False
):
  
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # offset and then adding all the numbers from 0 to Q_TILE_SIZE -1
    absolute_query_index = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
  
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # Start at the top!
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0), # Start at the top!
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    # Now we want to load Q_i from global memory
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    q_i = tl.load(Q_block_ptr)
    m_i = tl.full((Q_TILE_SIZE,1), float("-inf"), dtype=tl.float32)
    l_i = tl.full((Q_TILE_SIZE,1), 0.0, dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(0, T_k):
        if is_causal:
            absolute_key_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) 
            # create a mask that is True for all keys that are before the current query idx
            mask = absolute_query_index[:, None] >= absolute_key_index[None, :]

        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr) 
    
        s_i = tl.dot(q_i, tl.trans(K_j))
        s_i = s_i * scale

        if is_causal:
            s_i = tl.where(mask, s_i, float("-inf"))


        # compute the rowmax of s_i
        rowmax_si = tl.max(s_i, axis=-1, keep_dims=True) 
        m_prev = m_i
        m_i = tl.maximum(m_prev, rowmax_si)
        p_i = tl.exp(s_i - m_i)
        rowsum_pi = tl.sum(p_i, axis=-1, keep_dims=True)
        difference = tl.exp(m_prev - m_i)
        l_i = difference * l_i + rowsum_pi
        p_i = p_i.to(V_block_ptr.type.element_ty)
        o_i = o_i * difference  
        o_i = tl.dot(p_i, V_j, acc = o_i)

        # Move forward the block pointers
        # To advance a block pointer use *_block_ptr = *_block_ptr.advance()
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0)) # moves to the next tile down!
    
    o_i = o_i / l_i
    l_i = m_i + tl.log(l_i)
    # cast o_i to O_block ptr
    o_i = o_i.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, o_i)
    tl.store(L_block_ptr, l_i)


def pick_tile_sizes(seq_len, d, dtype):                              
      if dtype == torch.float32:
          if d >= 128:    
            q_tile = k_tile = 32                         
          elif d >= 64:   
            q_tile = k_tile = 64
          else:           
            q_tile = k_tile = 128                        
      else:  # bf16 / fp16
          if d >= 128:    
            q_tile = k_tile = 64                         
          elif d >= 64:   
            q_tile = k_tile = 128                        
          else:           
            q_tile = k_tile = 128
                                                                       
      if q_tile > seq_len: 
        q_tile = seq_len                            
      if k_tile > seq_len: 
        k_tile = seq_len
      return q_tile, k_tile   

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, N_q, d = Q.shape
        N_k = K.shape[1]

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_q), device=Q.device, dtype=torch.float32)
        Q_TILE_SIZE, K_TILE_SIZE = pick_tile_sizes(N_q, d, Q.dtype)
        scale = 1.0 / math.sqrt(d) 
        grid = (triton.cdiv(N_q, Q_TILE_SIZE), batch_size)
        flash_fwd_kernel[grid](Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
        N_QUERIES=N_q, N_KEYS=N_k, scale=scale, D=d,
        Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
        is_causal=is_causal,
        num_warps=4, num_stages=1,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O


# pytorch backend lol
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = FlashAttentionBackwardPytorch.apply(Q, K, V, O, dO, L)
        return dQ, dK, dV, None