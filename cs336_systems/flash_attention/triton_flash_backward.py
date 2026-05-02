

import torch
import triton
import triton.language as tl
import math

from cs336_systems.flash_attention.triton_flash_attention import pick_tile_sizes


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, dQ_ptr,
    L_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_lb, stride_lq,
    stride_db, stride_dq_in,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
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
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
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

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq_in, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    q_i = tl.load(Q_block_ptr)
    do_i = tl.load(dO_block_ptr)
    l_i = tl.load(L_block_ptr)
    d_i = tl.load(D_block_ptr)
    dq_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(0, T_k):
        if is_causal:
            absolute_key_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = absolute_query_index[:, None] >= absolute_key_index[None, :]

        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        s_ij = tl.dot(q_i, tl.trans(K_j))
        s_ij = s_ij * scale

        if is_causal:
            s_ij = tl.where(mask, s_ij, float("-inf"))

        # p_ij = exp(s - L) since L is the saved logsumexp from forward
        p_ij = tl.exp(s_ij - l_i)

        # dP = dO @ V^T
        dp_ij = tl.dot(do_i, tl.trans(V_j))
        # dS = P * (dP - D_i)
        ds_ij = p_ij * (dp_ij - d_i)
        ds_ij = ds_ij.to(K_j.dtype)
        # accumulate dQ += dS @ K
        dq_i = tl.dot(ds_ij, K_j, acc=dq_i)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # scale only at the end (chain rule from S = QK^T * scale)
    dq_i = dq_i * scale
    dq_i = dq_i.to(dQ_block_ptr.type.element_ty)
    tl.store(dQ_block_ptr, dq_i)


@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    L_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_lb, stride_lq,
    stride_db, stride_dq_in,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False
):

    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    absolute_key_index = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq_in, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    T_q = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)
    dk_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dv_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for i in range(0, T_q):
        if is_causal:
            absolute_query_index = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = absolute_query_index[:, None] >= absolute_key_index[None, :]

        q_i = tl.load(Q_block_ptr)
        do_i = tl.load(dO_block_ptr)
        l_i = tl.load(L_block_ptr)
        d_i = tl.load(D_block_ptr)

        s_ij = tl.dot(q_i, tl.trans(K_j))
        s_ij = s_ij * scale

        if is_causal:
            s_ij = tl.where(mask, s_ij, float("-inf"))

        p_ij = tl.exp(s_ij - l_i)

        # dV += P^T @ dO
        p_ij_typed = p_ij.to(do_i.dtype)
        dv_j = tl.dot(tl.trans(p_ij_typed), do_i, acc=dv_j)

        # dP = dO @ V^T
        dp_ij = tl.dot(do_i, tl.trans(V_j))
        ds_ij = p_ij * (dp_ij - d_i)
        ds_ij = ds_ij.to(q_i.dtype)
        # dK += dS^T @ Q
        dk_j = tl.dot(tl.trans(ds_ij), q_i, acc=dk_j)

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE, 0))

    dk_j = dk_j * scale
    dk_j = dk_j.to(dK_block_ptr.type.element_ty)
    dv_j = dv_j.to(dV_block_ptr.type.element_ty)
    tl.store(dK_block_ptr, dk_j)
    tl.store(dV_block_ptr, dv_j)


class FlashAttentionBackwardTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, O, dO, L, is_causal=False):
        batch_size, N_q, d = Q.shape
        N_k = K.shape[1]

        # D_i = rowsum(O elem dO), in fp32 for stability
        D_vec = (O.float() * dO.float()).sum(dim=-1)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        Q_TILE_SIZE, K_TILE_SIZE = pick_tile_sizes(N_q, d, Q.dtype)
        scale = 1.0 / math.sqrt(d)

        grid_dq = (triton.cdiv(N_q, Q_TILE_SIZE), batch_size)
        flash_bwd_dq_kernel[grid_dq](
            Q, K, V, dO, dQ, L, D_vec,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            N_QUERIES=N_q, N_KEYS=N_k, scale=scale, D=d,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            num_warps=4, num_stages=1,
        )

        grid_dkv = (triton.cdiv(N_k, K_TILE_SIZE), batch_size)
        flash_bwd_dkv_kernel[grid_dkv](
            Q, K, V, dO, dK, dV, L, D_vec,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            N_QUERIES=N_q, N_KEYS=N_k, scale=scale, D=d,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            num_warps=4, num_stages=1,
        )

        return dQ, dK, dV
