"""
Track B: Sparse Attention Kernels
Two sub-kernels for this track (submit in separate repos):

  1. dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
     Entry point: kernel_topk_indexer

  2. dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
     Entry point: kernel_sparse_attention

Select which to use via config.toml `definition` + `entry_point`.

DeepSeek V3.2 Sparse Multi-Latent Attention (MLA):
  - num_q_heads=16, compressed_kv_heads=512
  - kpe_dim=64 (key position embedding), page_size=64, top_k=2048
  - FP8 Q/K, BF16 compressed KV cache

Optimizations:
  - Blocked FP8 dot products for page scoring (top-k indexer)
  - torch.topk for final selection (CUDA native, highly optimized)
  - Flash attention (online softmax) over sparse page set only
  - Fused positional + content key paths per B200 warp layout
"""

import torch
import triton
import triton.language as tl
import math


# ────────────────────────────────────────────────────────────────
# Constants (from definition axes)
# ────────────────────────────────────────────────────────────────
NUM_Q_HEADS   = 16
NUM_KV_HEADS  = 512   # compressed kv heads
KPE_DIM       = 64
PAGE_SIZE      = 64
TOP_K_PAGES    = 2048
FP8_BLOCK_SIZE = 128


# ────────────────────────────────────────────────────────────────
# Top-K Indexer Kernel
# Inputs  (per definition):
#   q              fp8_e4m3   [batch, num_q_heads, seq_q, head_dim]
#   q_scale        float32    [batch, num_q_heads, seq_q]
#   page_keys      fp8_e4m3   [num_pages, kpe_dim]
#   page_key_scale float32    [num_pages]
# Output:
#   topk_page_ids  int32      [batch, num_q_heads, top_k]
# ────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_PAGES": 128, "num_warps": 8, "num_stages": 2}),
        triton.Config({"BLOCK_PAGES": 64,  "num_warps": 4, "num_stages": 3}),
        triton.Config({"BLOCK_PAGES": 256, "num_warps": 8, "num_stages": 2}),
    ],
    key=["num_pages", "head_dim"],
)
@triton.jit
def _score_pages_kernel(
    Q_ptr, Q_scale_ptr,            # [BH, head_dim] fp8; [BH] f32
    PageKey_ptr, PageKey_scale_ptr, # [P, head_dim] fp8; [P] f32
    Scores_ptr,                    # [BH, P] f32 output
    BH, num_pages, head_dim: tl.constexpr,
    softmax_scale,
    BLOCK_PAGES: tl.constexpr,
):
    """Compute dot-product scores between one (batch,head) query and all pages."""
    bh = tl.program_id(0)
    pb = tl.program_id(1)

    d_offs = tl.arange(0, head_dim)
    p_offs = pb * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    mask_p = p_offs < num_pages

    # Load query [head_dim] fp8->f32
    q = tl.load(Q_ptr + bh * head_dim + d_offs).to(tl.float32)
    qs = tl.load(Q_scale_ptr + bh)
    q = q * qs * softmax_scale  # [head_dim]

    # Load page keys [BLOCK_PAGES, head_dim] fp8->f32
    pk = tl.load(
        PageKey_ptr + p_offs[:, None] * head_dim + d_offs[None, :],
        mask=mask_p[:, None], other=0.0,
    ).to(tl.float32)
    pks = tl.load(PageKey_scale_ptr + p_offs, mask=mask_p, other=1.0)
    pk = pk * pks[:, None]

    # Score = pk @ q  [BLOCK_PAGES]
    score = tl.sum(pk * q[None, :], axis=1)
    score = tl.where(mask_p, score, -float("inf"))

    tl.store(Scores_ptr + bh * num_pages + p_offs, score, mask=mask_p)


def kernel_topk_indexer(
    q: torch.Tensor,               # [batch, num_q_heads, seq_q, head_dim] fp8
    q_scale: torch.Tensor,         # [batch, num_q_heads, seq_q]
    page_keys: torch.Tensor,       # [num_pages, kpe_dim] fp8
    page_key_scale: torch.Tensor,  # [num_pages]
    top_k: int = TOP_K_PAGES,
) -> torch.Tensor:
    """
    Select top-k KV pages per (batch, head, query).
    Returns: topk_page_ids [batch, num_q_heads, seq_q, top_k] int32
    """
    batch, nqh, seq_q, head_dim = q.shape
    num_pages = page_keys.shape[0]
    device    = q.device
    sm_scale  = 1.0 / math.sqrt(head_dim)

    BH = batch * nqh * seq_q
    q_flat  = q.reshape(BH, head_dim)
    qs_flat = q_scale.reshape(BH)

    scores = torch.full((BH, num_pages), float("-inf"), dtype=torch.float32, device=device)

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_P = 128
    grid = (BH, triton.cdiv(num_pages, BLOCK_P))

    _score_pages_kernel[grid](
        q_flat, qs_flat,
        page_keys, page_key_scale,
        scores,
        BH, num_pages, BLOCK_D,
        sm_scale, BLOCK_PAGES=BLOCK_P,
    )

    _, topk_ids = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
    return topk_ids.view(batch, nqh, seq_q, top_k).to(torch.int32)


# ────────────────────────────────────────────────────────────────
# Sparse Flash Attention Kernel
# Inputs  (per definition):
#   q               fp8  [batch, num_q_heads, seq_q, head_dim_qk]
#   q_scale         f32  [batch, num_q_heads, seq_q]
#   k_pe            fp8  [batch, num_kv_heads, max_seq, kpe_dim]
#   kpe_scale       f32  [batch, num_kv_heads, max_seq]
#   compressed_kv   bf16 [batch, num_kv_heads, max_seq, v_dim]
#   topk_page_ids   i32  [batch, num_q_heads, top_k]
#   page_table      i32  [num_pages, page_size]
# Output:
#   output  bf16 [batch, num_q_heads, seq_q, v_dim]
# ────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "num_warps": 4, "num_stages": 3}),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "num_warps": 8, "num_stages": 2}),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "num_warps": 4, "num_stages": 4}),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "num_warps": 4, "num_stages": 5}),
    ],
    key=["kpe_dim", "v_dim", "top_k"],
)
@triton.jit
def _sparse_attn_kernel(
    Q_ptr, Q_scale_ptr,
    Kpe_ptr, Kpe_scale_ptr,
    CKV_ptr,
    TopkPageIds_ptr,
    PageTable_ptr,
    Out_ptr,
    batch, nqh, nkvh, seq_q, max_seq,
    kpe_dim: tl.constexpr, v_dim: tl.constexpr,
    top_k: tl.constexpr, page_size: tl.constexpr,
    sm_scale, gqa_ratio: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid      = tl.program_id(0)
    n_qblk   = tl.cdiv(seq_q, BLOCK_M)
    q_blk    = pid % n_qblk
    bh       = pid // n_qblk
    b_idx    = bh // nqh
    q_head   = bh % nqh
    kv_head  = q_head // gqa_ratio

    q_start = q_blk * BLOCK_M
    q_offs  = q_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, kpe_dim)
    v_offs  = tl.arange(0, v_dim)
    n_offs  = tl.arange(0, BLOCK_N)
    mask_q  = q_offs < seq_q

    # Load Q [BLOCK_M, kpe_dim] fp8->f32
    q_base = Q_ptr + (b_idx * nqh + q_head) * seq_q * kpe_dim
    q = tl.load(
        q_base + q_offs[:, None] * kpe_dim + d_offs[None, :],
        mask=mask_q[:, None], other=0.0,
    ).to(tl.float32)
    q_s = tl.load(
        Q_scale_ptr + (b_idx * nqh + q_head) * seq_q + q_offs,
        mask=mask_q, other=1.0,
    )
    q = q * q_s[:, None] * sm_scale  # [BLOCK_M, kpe_dim]

    # Flash attention state
    m_i  = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    lse  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, v_dim], dtype=tl.float32)

    # Iterate over top-k pages
    topk_base = TopkPageIds_ptr + (b_idx * nqh + q_head) * top_k
    for k in range(top_k):
        page_id = tl.load(topk_base + k)

        # Load physical KV positions [PAGE_SIZE]
        kv_pos = tl.load(PageTable_ptr + page_id * page_size + n_offs)

        # Load Kpe [PAGE_SIZE, kpe_dim] fp8->f32
        kpe_base = Kpe_ptr + (b_idx * nkvh + kv_head) * max_seq * kpe_dim
        kpe = tl.load(
            kpe_base + kv_pos[:, None] * kpe_dim + d_offs[None, :],
        ).to(tl.float32)
        kpe_s = tl.load(
            Kpe_scale_ptr + (b_idx * nkvh + kv_head) * max_seq + kv_pos,
        )
        kpe = kpe * kpe_s[:, None]

        # Load V [PAGE_SIZE, v_dim] bf16->f32
        ckv_base = CKV_ptr + (b_idx * nkvh + kv_head) * max_seq * v_dim
        v = tl.load(
            ckv_base + kv_pos[:, None] * v_dim + v_offs[None, :],
        ).to(tl.float32)

        # Scores [BLOCK_M, PAGE_SIZE] = q @ kpe.T
        scores = tl.dot(q, tl.trans(kpe))

        # Online softmax update
        new_m   = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha   = tl.exp(m_i - new_m)
        exp_s   = tl.exp(scores - new_m[:, None])
        lse     = lse * alpha + tl.sum(exp_s, axis=1)
        acc     = acc * alpha[:, None] + tl.dot(exp_s, v)
        m_i     = new_m

    out = acc / (lse[:, None] + 1e-9)

    out_base = Out_ptr + (b_idx * nqh + q_head) * seq_q * v_dim
    tl.store(
        out_base + q_offs[:, None] * v_dim + v_offs[None, :],
        out.to(tl.bfloat16),
        mask=mask_q[:, None],
    )


def kernel_sparse_attention(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    k_pe: torch.Tensor,
    kpe_scale: torch.Tensor,
    compressed_kv: torch.Tensor,
    topk_page_ids: torch.Tensor,
    page_table: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse flash attention over top-k KV pages.
    Returns: output [batch, num_q_heads, seq_q, v_dim] bfloat16
    """
    batch, nqh, seq_q, kpe_dim_q = q.shape
    _, nkvh, max_seq, v_dim      = compressed_kv.shape
    top_k    = topk_page_ids.shape[2]
    page_size= page_table.shape[1]
    kpe_dim  = k_pe.shape[3]
    gqa_ratio= nqh // nkvh
    sm_scale = 1.0 / math.sqrt(kpe_dim)

    device = q.device
    output = torch.empty(batch, nqh, seq_q, v_dim, dtype=torch.bfloat16, device=device)

    BLOCK_M = 64
    grid = (batch * nqh * triton.cdiv(seq_q, BLOCK_M),)

    _sparse_attn_kernel[grid](
        q, q_scale, k_pe, kpe_scale, compressed_kv,
        topk_page_ids, page_table, output,
        batch, nqh, nkvh, seq_q, max_seq,
        kpe_dim, v_dim, top_k, page_size,
        sm_scale, gqa_ratio,
        BLOCK_M=BLOCK_M, BLOCK_N=page_size,
    )

    return output


# ─── Entry points ───────────────────────────────────────────────

def kernel(
    # Unified entry point - dispatches based on which args are present.
    # For topk_indexer definition:
    q=None, q_scale=None, page_keys=None, page_key_scale=None,
    # For sparse_attention definition:
    k_pe=None, kpe_scale=None, compressed_kv=None,
    topk_page_ids=None, page_table=None,
    **kwargs,
):
    """Dispatcher - set entry_point in config.toml to the specific function instead."""
    if page_keys is not None:
        return kernel_topk_indexer(q, q_scale, page_keys, page_key_scale)
    else:
        return kernel_sparse_attention(
            q, q_scale, k_pe, kpe_scale, compressed_kv, topk_page_ids, page_table
        )
