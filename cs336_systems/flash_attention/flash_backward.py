import torch
import math

# PROBLEM flash_backward 

class FlashAttentionBackwardPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, O, dO, L): # needs to return dQ, dK, dV
        # first lets calculate D = rowsum(O elem dO)
        #rowsum means we;re getting rid of head dimension

        # do the math in fp32
        orig_dtype = Q.dtype
        Q = Q.float()
        K = K.float()
        V = V.float()
        O = O.float()
        dO = dO.float()
        batch_size, seq_len, head_dim = Q.shape
        D = torch.einsum("b s h , b s h -> b s", O, dO)
        S = torch.einsum("b i h, b j h -> b i j", Q, K)
        S = S / math.sqrt(head_dim)
        #broadcast L t be the same shape as S
        L = L.unsqueeze(-1)
        P = torch.exp(S - L)
        # now must calculate dV 
        dV = torch.einsum("b q k, b q h -> b k h", P, dO)
        #now calculate dP
        dP = torch.einsum("b q h, b k h -> b q k", dO, V)
        # now calculate dS
        # first broadcast D_i to be same as dP
        D_i = D.unsqueeze(-1) 
        subtraction = dP - D_i
        # element wise multiplication
        dS = subtraction * P
        # now calculate dQ
        dQ = torch.einsum("b q k, b k h -> b q h", dS, K)
        dQ = dQ / math.sqrt(head_dim)

        dK = torch.einsum("b q k , b q h -> b k h", dS, Q)
        dK = dK / math.sqrt(head_dim)

        return dQ.to(orig_dtype), dK.to(orig_dtype), dV.to(orig_dtype)
