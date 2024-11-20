
import torch
import triton
import triton.language as tl
import math
import pdb

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Lse, 
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads,
    seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1).to(tl.int64)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[None, :] * stride_kn + offs_d[:, None])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    qk_scale = softmax_scale
    qk_scale *= 1.44269504

    # defined as the original paper
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # pdb.set_trace()
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[None, :] < seqlen_k, other=0.0)
        qk = tl.dot(q, k).to(tl.float32)
    
        qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk = qk * qk_scale + tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
            m_ij = tl.maximum(tl.max(qk, 1), l_i)
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * qk_scale, l_i)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs, mask = offs_n[:, None] < (seqlen_k - start_n))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        m_i = m_ij

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < seqlen_q)


def flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    batch, nheads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = (triton.cdiv(seqlen_q, BLOCK), batch * nheads)
    _fwd_kernel[grid](
        q, k, v, o, lse,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        nheads, seqlen_q, seqlen_k, 
        seqlen_q_rounded, d,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o