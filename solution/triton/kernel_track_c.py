"""
Track C: Gated Delta Net Kernels (Qwen3-Next)
Two sub-kernels:
  1. gdn_decode_qk16_v32_d128_k_last   -- single token decode
  2. gdn_prefill_qk16_v32_d128_k_last  -- full sequence prefill

Signature matched to definition (called with keyword args):
  Decode inputs:  q, k, v, beta, gate, state
  Decode output:  output, state (updated in-place)

State layout: [batch, heads, v_dim, qk_dim]  ("k_last" = qk innermost)
qk_dim=16, v_dim=32, head_dim=128

Delta rule:
  k_hat  = k / ||k||
  Sk     = S @ k_hat          (state response to key)
  delta  = v - Sk             (residual)
  S_new  = S + delta ⊗ (β * k_hat)^T
  output = (S_new @ q) * gate

Optimizations:
  - State tiles held in registers across the update step
  - SwiGLU-style gate fused into output store
  - k-normalization + beta-scaling fused
  - Chunk-parallel prefill with intra-chunk sequential scan
  - Autotuned BLOCK_V configs for v_dim=32
"""

import torch
import triton
import triton.language as tl
import math


# ─────────────────────────────────────────────────────────────────
# GDN Decode Kernel
# ─────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 32, "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_V": 16, "num_warps": 2, "num_stages": 3}),
    ],
    key=["v_dim", "qk_dim"],
)
@triton.jit
def _gdn_decode_kernel(
    Q_ptr,     # [BH, qk_dim]  bf16
    K_ptr,     # [BH, qk_dim]  bf16
    V_ptr,     # [BH, v_dim]   bf16
    Beta_ptr,  # [BH]          f32
    Gate_ptr,  # [BH, v_dim]   bf16
    State_ptr, # [BH, v_dim, qk_dim]  f32  (in-place update)
    Out_ptr,   # [BH, v_dim]   bf16
    BH,
    qk_dim: tl.constexpr,
    v_dim:  tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    if bh >= BH:
        return

    qk_offs = tl.arange(0, qk_dim)

    # Load k, q, beta
    k    = tl.load(K_ptr    + bh * qk_dim + qk_offs).to(tl.float32)
    q    = tl.load(Q_ptr    + bh * qk_dim + qk_offs).to(tl.float32)
    beta = tl.load(Beta_ptr + bh).to(tl.float32)

    # k_hat = k / ||k||
    k_norm = tl.sqrt(tl.sum(k * k) + 1e-8)
    k_hat  = k / k_norm
    beta_k = beta * k_hat           # [qk_dim]

    # Process in v-tiles
    for v_blk in tl.static_range(tl.cdiv(v_dim, BLOCK_V)):
        v_offs = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = v_offs < v_dim

        # Load state tile [BLOCK_V, qk_dim]
        s = tl.load(
            State_ptr + bh * v_dim * qk_dim
                      + v_offs[:, None] * qk_dim + qk_offs[None, :],
            mask=mask_v[:, None], other=0.0,
        ).to(tl.float32)

        v_tile    = tl.load(V_ptr    + bh * v_dim + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        gate_tile = tl.load(Gate_ptr + bh * v_dim + v_offs, mask=mask_v, other=0.0).to(tl.float32)

        # Delta update
        Sk    = tl.sum(s * k_hat[None, :], axis=1)          # [BLOCK_V]
        delta = v_tile - Sk
        s     = s + delta[:, None] * beta_k[None, :]        # [BLOCK_V, qk_dim]

        # Write updated state
        tl.store(
            State_ptr + bh * v_dim * qk_dim
                      + v_offs[:, None] * qk_dim + qk_offs[None, :],
            s.to(tl.float32),
            mask=mask_v[:, None],
        )

        # Output: (S_new @ q) * gate
        Sq  = tl.sum(s * q[None, :], axis=1)                # [BLOCK_V]
        out = Sq * gate_tile
        tl.store(Out_ptr + bh * v_dim + v_offs, out.to(tl.bfloat16), mask=mask_v)


# ─────────────────────────────────────────────────────────────────
# GDN Prefill Kernel — chunk-sequential within each program
# ─────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"CHUNK_SIZE": 64,  "BLOCK_V": 32, "num_warps": 4, "num_stages": 2}),
        triton.Config({"CHUNK_SIZE": 128, "BLOCK_V": 32, "num_warps": 8, "num_stages": 2}),
        triton.Config({"CHUNK_SIZE": 32,  "BLOCK_V": 32, "num_warps": 4, "num_stages": 3}),
    ],
    key=["seq_len", "v_dim", "qk_dim"],
)
@triton.jit
def _gdn_prefill_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, Beta_ptr, Gate_ptr,   # [BH, seq, dim]
    State_ptr,                                  # [BH, v_dim, qk_dim] f32  (initial state)
    Out_ptr,                                    # [BH, seq, v_dim] bf16
    StateOut_ptr,                               # [BH, v_dim, qk_dim] f32  (chunk final state)
    BH, seq_len, chunk_start,
    qk_dim: tl.constexpr, v_dim: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    if bh >= BH:
        return

    qk_offs = tl.arange(0, qk_dim)

    for v_blk in tl.static_range(tl.cdiv(v_dim, BLOCK_V)):
        v_offs = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = v_offs < v_dim

        # Load initial state tile [BLOCK_V, qk_dim]
        s = tl.load(
            State_ptr + bh * v_dim * qk_dim
                      + v_offs[:, None] * qk_dim + qk_offs[None, :],
            mask=mask_v[:, None], other=0.0,
        ).to(tl.float32)

        # Scan through chunk tokens sequentially
        for t in tl.static_range(CHUNK_SIZE):
            seq_pos = chunk_start + t
            if seq_pos >= seq_len:
                break

            tok = bh * seq_len + seq_pos
            k    = tl.load(K_ptr    + tok * qk_dim + qk_offs).to(tl.float32)
            q    = tl.load(Q_ptr    + tok * qk_dim + qk_offs).to(tl.float32)
            beta = tl.load(Beta_ptr + tok).to(tl.float32)
            vt   = tl.load(V_ptr    + tok * v_dim + v_offs, mask=mask_v, other=0.0).to(tl.float32)
            gt   = tl.load(Gate_ptr + tok * v_dim + v_offs, mask=mask_v, other=0.0).to(tl.float32)

            k_norm = tl.sqrt(tl.sum(k * k) + 1e-8)
            k_hat  = k / k_norm
            beta_k = beta * k_hat

            Sk    = tl.sum(s * k_hat[None, :], axis=1)
            delta = vt - Sk
            s     = s + delta[:, None] * beta_k[None, :]

            Sq  = tl.sum(s * q[None, :], axis=1)
            out = Sq * gt
            tl.store(
                Out_ptr + tok * v_dim + v_offs,
                out.to(tl.bfloat16), mask=mask_v,
            )

        # Write final state for this chunk
        tl.store(
            StateOut_ptr + bh * v_dim * qk_dim
                         + v_offs[:, None] * qk_dim + qk_offs[None, :],
            s.to(tl.float32),
            mask=mask_v[:, None],
        )


# ─────────────────────────────────────────────────────────────────
# Entry points  (names match definition entry_point field)
# ─────────────────────────────────────────────────────────────────

def kernel_decode(
    q: torch.Tensor,      # [batch, heads, qk_dim]
    k: torch.Tensor,
    v: torch.Tensor,      # [batch, heads, v_dim]
    beta: torch.Tensor,   # [batch, heads]
    gate: torch.Tensor,   # [batch, heads, v_dim]
    state: torch.Tensor,  # [batch, heads, v_dim, qk_dim] f32  modified in-place
) -> torch.Tensor:
    """GDN single-step decode. Updates state in-place. Returns output [batch, heads, v_dim]."""
    batch, heads, qk_dim = q.shape
    v_dim  = v.shape[2]
    BH     = batch * heads
    device = q.device

    out = torch.empty(batch, heads, v_dim, dtype=torch.bfloat16, device=device)

    _gdn_decode_kernel[(BH,)](
        q.reshape(BH, qk_dim).contiguous(),
        k.reshape(BH, qk_dim).contiguous(),
        v.reshape(BH, v_dim).contiguous(),
        beta.float().reshape(BH).contiguous(),
        gate.reshape(BH, v_dim).contiguous(),
        state.reshape(BH, v_dim, qk_dim),
        out.reshape(BH, v_dim),
        BH, qk_dim=qk_dim, v_dim=v_dim,
        BLOCK_V=v_dim,
    )
    return out


def kernel_prefill(
    q: torch.Tensor,      # [batch, heads, seq, qk_dim]
    k: torch.Tensor,
    v: torch.Tensor,      # [batch, heads, seq, v_dim]
    beta: torch.Tensor,   # [batch, heads, seq]
    gate: torch.Tensor,   # [batch, heads, seq, v_dim]
    state: torch.Tensor,  # [batch, heads, v_dim, qk_dim] f32  modified in-place
    chunk_size: int = 64,
) -> torch.Tensor:
    """GDN prefill over full sequence. Updates state in-place. Returns output [B,H,S,V]."""
    batch, heads, seq_len, qk_dim = q.shape
    v_dim = v.shape[3]
    BH    = batch * heads

    q_bhs = q.reshape(BH, seq_len, qk_dim).contiguous()
    k_bhs = k.reshape(BH, seq_len, qk_dim).contiguous()
    v_bhs = v.reshape(BH, seq_len, v_dim).contiguous()
    b_bhs = beta.float().reshape(BH, seq_len).contiguous()
    g_bhs = gate.reshape(BH, seq_len, v_dim).contiguous()

    out   = torch.empty(BH, seq_len, v_dim, dtype=torch.bfloat16, device=q.device)
    s_cur = state.reshape(BH, v_dim, qk_dim).contiguous()

    for chunk_start in range(0, seq_len, chunk_size):
        s_next = torch.empty_like(s_cur)
        _gdn_prefill_chunk_kernel[(BH,)](
            q_bhs, k_bhs, v_bhs, b_bhs, g_bhs,
            s_cur, out, s_next,
            BH, seq_len, chunk_start,
            qk_dim=qk_dim, v_dim=v_dim,
            CHUNK_SIZE=chunk_size, BLOCK_V=v_dim,
        )
        s_cur = s_next

    state.copy_(s_cur.view_as(state))
    return out.view(batch, heads, seq_len, v_dim)


# Unified entry point (set specific function in config.toml entry_point for each sub-track)
def kernel(
    q, k, v, beta, gate, state,
    mode: str = "decode",
    **kwargs,
):
    if mode == "decode" or q.dim() == 3:
        return kernel_decode(q, k, v, beta, gate, state)
    else:
        return kernel_prefill(q, k, v, beta, gate, state)
