"""
Track A: Fused MoE Kernel
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Signature matches definition exactly (called with keyword args by flashinfer-bench):
  Inputs:
    routing_logits        float32   [seq_len, num_experts]
    routing_bias          bfloat16  [num_experts]
    hidden_states         float8_e4m3fn [seq_len, hidden_size]
    hidden_states_scale   float32   [num_hidden_blocks, seq_len]
    gemm1_weights         float8_e4m3fn [num_local_experts, gemm1_out_size, hidden_size]
    gemm1_weights_scale   float32   [num_local_experts, num_gemm1_out_blocks, num_hidden_blocks]
    gemm2_weights         float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
    gemm2_weights_scale   float32   [num_local_experts, num_hidden_blocks, num_intermediate_blocks]
    local_expert_offset   int32     scalar
    routed_scaling_factor float32   scalar
  Output:
    output  bfloat16  [seq_len, hidden_size]

Optimizations:
  - DeepSeek group-limited top-k routing (ng=8 groups, kg=4 selected, topk=8)
  - FP8 GEMM1 with block-scale dequant + SwiGLU fused in one kernel
  - FP8 GEMM2 with block-scale dequant
  - Token permutation sorted by expert for coalesced memory access
  - Autotuned tile configs targeting B200 (SM100) tensor cores
"""

import torch
import triton
import triton.language as tl
import math


# ─────────────────────────────────────────────────────────────────
# Constants for this workload (from definition axes)
# ─────────────────────────────────────────────────────────────────
NUM_EXPERTS          = 256
NUM_LOCAL_EXPERTS    = 32
HIDDEN_SIZE          = 7168
INTERMEDIATE_SIZE    = 2048
GEMM1_OUT_SIZE       = 4096   # = 2 * INTERMEDIATE_SIZE (SwiGLU gate+up)
NUM_HIDDEN_BLOCKS    = 56     # ceil(HIDDEN_SIZE / 128)
NUM_INTERMEDIATE_BLOCKS = 16  # ceil(INTERMEDIATE_SIZE / 128)
NUM_GEMM1_OUT_BLOCKS = 32     # ceil(GEMM1_OUT_SIZE / 128)
FP8_BLOCK_SIZE       = 128    # block size for scale quantization
TOP_K                = 8
N_GROUP              = 8
TOPK_GROUP           = 4
EXPERTS_PER_GROUP    = NUM_EXPERTS // N_GROUP  # 32


# ─────────────────────────────────────────────────────────────────
# Routing: DeepSeek group-limited top-k
# ─────────────────────────────────────────────────────────────────

def _compute_routing(
    routing_logits: torch.Tensor,   # [seq_len, num_experts] float32
    routing_bias: torch.Tensor,     # [num_experts] bfloat16
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-style group-limited top-k routing.
    Returns topk_ids [seq_len, TOP_K] int32, topk_weights [seq_len, TOP_K] float32.
    """
    seq_len = routing_logits.shape[0]
    bias_f  = routing_bias.float()

    # Biased scores for routing decisions
    scores_biased = routing_logits + bias_f  # [S, E]

    # Group scores: max within each group
    scores_grouped = scores_biased.view(seq_len, N_GROUP, EXPERTS_PER_GROUP)
    group_max, _ = scores_grouped.max(dim=2)           # [S, N_GROUP]
    _, group_topk_ids = group_max.topk(TOPK_GROUP, dim=1)  # [S, TOPK_GROUP]

    # Build mask of selected groups
    group_mask = torch.zeros(seq_len, N_GROUP, dtype=torch.bool, device=routing_logits.device)
    group_mask.scatter_(1, group_topk_ids, True)       # [S, N_GROUP]
    expert_mask = group_mask.unsqueeze(2).expand(-1, -1, EXPERTS_PER_GROUP).reshape(seq_len, NUM_EXPERTS)

    # Mask out non-selected groups
    masked_scores = scores_biased.masked_fill(~expert_mask, float("-inf"))

    # Top-k within selected experts
    topk_weights_raw, topk_ids = masked_scores.topk(TOP_K, dim=1)  # [S, TOP_K]

    # Softmax weights over *unmasked* logits for normalization
    softmax_scores = torch.softmax(routing_logits, dim=1)
    topk_weights = softmax_scores.gather(1, topk_ids)   # [S, TOP_K]
    topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-9)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_ids.to(torch.int32), topk_weights.float()


# ─────────────────────────────────────────────────────────────────
# FP8 GEMM1 + SwiGLU fused kernel
# A: [M, K] fp8_e4m3  x  B: [N, K] fp8_e4m3  ->  C: [M, N//2] bf16
# N = gemm1_out_size (gate+up concatenated)
# C[m, n] = silu(A @ B[:N//2].T)[m,n]  *  (A @ B[N//2:].T)[m,n]
# ─────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128, "num_stages": 4, "num_warps": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "num_stages": 3, "num_warps": 8}),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64,  "num_stages": 4, "num_warps": 4}),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 128, "num_stages": 4, "num_warps": 4}),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 128, "num_stages": 5, "num_warps": 4}),
    ],
    key=["M", "K"],
)
@triton.jit
def _fp8_gemm1_swiglu(
    A_ptr, A_scale_ptr,       # [M, K] fp8; [K_blocks, M] f32
    B_ptr, B_scale_ptr,       # [N, K] fp8; [N_blocks, K_blocks] f32  (N = gemm1_out_size)
    C_ptr,                    # [M, N//2] bf16 output (post SwiGLU)
    M, K,
    stride_am, stride_ak,
    stride_bgate_n, stride_bgate_k,
    stride_cm, stride_cn,
    half_N,                   # = intermediate_size = N // 2
    N_blocks, K_blocks_total,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)
    mask_m  = offs_m < M
    mask_n  = offs_n < half_N

    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_up   = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for kb in range(num_k_blocks):
        k_offs  = kb * BLOCK_K + offs_k
        mask_k  = k_offs < K
        k_blk_i = kb * BLOCK_K // FP8_BLOCK_SIZE

        # Load A tile [BLOCK_M, BLOCK_K] fp8 -> f32 with block scale
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        a_scale = tl.load(A_scale_ptr + k_blk_i * M + offs_m, mask=mask_m, other=1.0)
        a = a * a_scale[:, None]

        n_blk_gate = (pid_n * BLOCK_N) // FP8_BLOCK_SIZE
        n_blk_up   = (pid_n * BLOCK_N + half_N) // FP8_BLOCK_SIZE

        # Gate projection B[offs_n, k_offs]
        b_gate = tl.load(
            B_ptr + offs_n[:, None] * stride_bgate_n + k_offs[None, :] * stride_bgate_k,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        b_gate_s = tl.load(B_scale_ptr + n_blk_gate * K_blocks_total + k_blk_i, other=1.0)
        b_gate = b_gate * b_gate_s

        # Up projection B[offs_n + half_N, k_offs]
        b_up = tl.load(
            B_ptr + (offs_n[:, None] + half_N) * stride_bgate_n + k_offs[None, :] * stride_bgate_k,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        b_up_s = tl.load(B_scale_ptr + n_blk_up * K_blocks_total + k_blk_i, other=1.0)
        b_up = b_up * b_up_s

        acc_gate = tl.dot(a, tl.trans(b_gate), acc=acc_gate)
        acc_up   = tl.dot(a, tl.trans(b_up),   acc=acc_up)

    # SwiGLU: silu(gate) * up
    out = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        out.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ─────────────────────────────────────────────────────────────────
# FP8 GEMM2 kernel
# A: [M, K] bf16  x  B: [N, K] fp8  ->  C: [M, N] bf16
# ─────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64,  "num_stages": 4, "num_warps": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "num_stages": 3, "num_warps": 8}),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64,  "num_stages": 4, "num_warps": 4}),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 256, "BLOCK_K": 64,  "num_stages": 5, "num_warps": 8}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fp8_gemm2(
    A_ptr,                    # [M, K] bf16
    B_ptr, B_scale_ptr,       # [N, K] fp8; [N_blocks, K_blocks] f32
    C_ptr,                    # [M, N] bf16
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    K_blocks_total,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for kb in range(tl.cdiv(K, BLOCK_K)):
        k_offs  = kb * BLOCK_K + offs_k
        mask_k  = k_offs < K
        k_blk_i = kb * BLOCK_K // FP8_BLOCK_SIZE
        n_blk_i = (pid_n * BLOCK_N) // FP8_BLOCK_SIZE

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)

        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + k_offs[None, :] * stride_bk,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        b_s = tl.load(B_scale_ptr + n_blk_i * K_blocks_total + k_blk_i, other=1.0)
        b = b * b_s

        acc = tl.dot(a, tl.trans(b), acc=acc)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ─────────────────────────────────────────────────────────────────
# Main entry point - signature MUST match definition exactly
# ─────────────────────────────────────────────────────────────────

def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """
    Fused MoE forward pass.
    All args are keyword-matched to the definition's input names.
    Returns: output [seq_len, hidden_size] bfloat16
    """
    seq_len = routing_logits.shape[0]
    device  = hidden_states.device

    # ── Routing ──────────────────────────────────────────────────
    topk_ids, topk_weights = _compute_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )
    # topk_ids: [seq_len, TOP_K] int32  (global expert ids)

    # ── Token permutation by local expert ────────────────────────
    local_mask   = (topk_ids >= local_expert_offset) & \
                   (topk_ids < local_expert_offset + NUM_LOCAL_EXPERTS)
    flat_experts  = topk_ids.view(-1)           # [S*TOP_K]
    flat_tokens   = torch.arange(seq_len, device=device) \
                        .unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    flat_weights  = topk_weights.view(-1)
    flat_loc_mask = local_mask.view(-1)

    sel_experts = flat_experts[flat_loc_mask] - local_expert_offset
    sel_tokens  = flat_tokens[flat_loc_mask]
    sel_weights = flat_weights[flat_loc_mask]

    if sel_experts.numel() == 0:
        return torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    sort_idx          = torch.argsort(sel_experts.long(), stable=True)
    sel_experts_s     = sel_experts[sort_idx]
    sel_tokens_s      = sel_tokens[sort_idx]
    sel_weights_s     = sel_weights[sort_idx]
    total_dispatched  = sel_experts_s.shape[0]

    # CSR offsets per expert
    expert_counts  = torch.bincount(sel_experts_s, minlength=NUM_LOCAL_EXPERTS)
    expert_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0).to(torch.int32)

    # Gather hidden states for dispatched tokens
    dispatched_h       = hidden_states[sel_tokens_s]         # [D, hidden_size]  fp8
    dispatched_h_scale = hidden_states_scale[:, sel_tokens_s] # [hidden_blocks, D]

    intermediate = torch.empty(total_dispatched, INTERMEDIATE_SIZE,
                               dtype=torch.bfloat16, device=device)
    expert_out   = torch.zeros(total_dispatched, HIDDEN_SIZE,
                               dtype=torch.bfloat16, device=device)

    # ── Per-expert GEMM1 + SwiGLU + GEMM2 ───────────────────────
    K_blks = math.ceil(HIDDEN_SIZE / FP8_BLOCK_SIZE)
    for e in range(NUM_LOCAL_EXPERTS):
        s = expert_offsets[e].item()
        t = expert_offsets[e + 1].item()
        if s == t:
            continue
        M_e = t - s

        A       = dispatched_h[s:t]                      # [M_e, H] fp8
        A_scale = dispatched_h_scale[:, s:t].contiguous() # [H_blks, M_e]
        B1      = gemm1_weights[e]                        # [G1, H] fp8
        B1s     = gemm1_weights_scale[e]                  # [G1_blks, H_blks]
        C1      = intermediate[s:t]                       # [M_e, I]

        grid1 = (triton.cdiv(M_e, 64), triton.cdiv(INTERMEDIATE_SIZE, 128))
        _fp8_gemm1_swiglu[grid1](
            A, A_scale, B1, B1s, C1,
            M_e, HIDDEN_SIZE,
            A.stride(0), A.stride(1),
            B1.stride(0), B1.stride(1),
            C1.stride(0), C1.stride(1),
            INTERMEDIATE_SIZE,
            math.ceil(GEMM1_OUT_SIZE / FP8_BLOCK_SIZE),
            K_blks,
        )

        B2  = gemm2_weights[e]        # [H, I] fp8
        B2s = gemm2_weights_scale[e]  # [H_blks, I_blks]
        C2  = expert_out[s:t]         # [M_e, H]

        grid2 = (triton.cdiv(M_e, 64), triton.cdiv(HIDDEN_SIZE, 128))
        _fp8_gemm2[grid2](
            C1, B2, B2s, C2,
            M_e, HIDDEN_SIZE, INTERMEDIATE_SIZE,
            C1.stride(0), C1.stride(1),
            B2.stride(0), B2.stride(1),
            C2.stride(0), C2.stride(1),
            math.ceil(INTERMEDIATE_SIZE / FP8_BLOCK_SIZE),
        )

    # ── Scatter-reduce: weighted sum back to output ───────────────
    output = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    for e in range(NUM_LOCAL_EXPERTS):
        s = expert_offsets[e].item()
        t = expert_offsets[e + 1].item()
        if s == t:
            continue
        tokens  = sel_tokens_s[s:t]
        weights = sel_weights_s[s:t].to(torch.bfloat16)
        output.index_add_(
            0, tokens,
            (expert_out[s:t] * weights.unsqueeze(1)),
        )

    return output
