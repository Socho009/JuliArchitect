# --- PHASE 7 ---
# Multi-Head Latent Attention (MLA)
#
# Inspired by DeepSeek-V2 (Bi et al., 2024), adapted for the FAT architecture.
#
# Standard MHA caches per-head K,V projections:
#     K_h = W_{K,h} x,  V_h = W_{V,h} x   for h = 1,…,n_h
#     KV cache per token: 2 · n_h · d_h = 2 · d_model
#
# MLA compresses through a shared latent bottleneck:
#     c_KV = W^{DKV} x  ∈ ℝ^{d_c},  d_c ≪ d_model
#     K_h = W^{UK}_h · c_KV,  V_h = W^{UV}_h · c_KV
#     KV cache per token: d_c  (e.g., d_model/4)
#
# For FAT, coefficient sequences share spectral structure between adjacent
# positions, making the low-rank compression nearly lossless.
#
# All operations are mutation-free for Zygote AD compatibility.

using LinearAlgebra

# ─── LatentAttention ──────────────────────────────────────────────────────────

"""
    LatentAttention{P, Mask} <: AbstractAttention{P}

Multi-Head Latent Attention (MLA) — a drop-in replacement for `Attention{P}`
that compresses Key-Value projections through a shared low-rank latent space.

## Architecture

Standard path (per token x ∈ ℝ^{d_model}):
    Q = W_Q · x           ∈ ℝ^{d_model}   (standard query projection)
    c_KV = W_DKV · x      ∈ ℝ^{d_c}       (compress to latent)
    K = W_UK · c_KV        ∈ ℝ^{d_model}   (recover keys from latent)
    V = W_UV · c_KV        ∈ ℝ^{d_model}   (recover values from latent)

At inference, only c_KV needs to be cached per position — not K and V separately.

## Memory Savings

For d_model=128, n_head=4, d_c=32:
    Standard MHA KV cache: 2 × 128 = 256 floats/token
    MLA KV cache:          32 floats/token  (8× reduction)

## Type Parameters
- `P`: Floating-point precision
- `Mask`: Boolean matrix type for causal mask

## Fields
- `W_Q`    — Query projection (d_model → d_model), with bias
- `W_DKV`  — KV down-projection (d_model → d_c), no bias
- `W_UK`   — Key up-projection (d_c → d_model), no bias
- `W_UV`   — Value up-projection (d_c → d_model), no bias
- `W_out`  — Output projection (d_model → d_model), with bias
- `mask`   — Causal mask (block_size × block_size)
- `n_head`, `head_dim`, `d_c` — dimensions
- `attn_pdrop`, `resid_pdrop` — dropout rates
"""
struct LatentAttention{P<:AbstractFloat, Mask<:AbstractMatrix{Bool}} <: AbstractAttention{P}
    W_Q::Linear{P, true}             # (d_model → d_model)
    W_DKV::Linear{P, false}          # (d_model → d_c) — shared KV down-projection
    W_UK::Linear{P, false}           # (d_c → d_model) — key up-projection
    W_UV::Linear{P, false}           # (d_c → d_model) — value up-projection
    W_out::Linear{P, true}           # (d_model → d_model)
    mask::Mask
    n_head::Int
    head_dim::Int
    d_c::Int                          # latent compression dimension
    attn_pdrop::P
    resid_pdrop::P
end

# ─── Constructor ──────────────────────────────────────────────────────────────

"""
    LatentAttention{P}(config::AbstractModelConfig{P}; d_c::Int) where P

Construct a LatentAttention layer from a model config.

## Arguments
- `config` — provides n_embd, n_head, block_size, attn_pdrop, resid_pdrop
- `d_c`    — latent compression dimension (default: n_embd ÷ 4)

## Initialization
All projections use small-scale random initialization (0.02σ).
The down-projection W_DKV is initialized at even smaller scale (0.01σ)
to ensure the latent space starts compact.
"""
function LatentAttention{P}(config::AbstractModelConfig{P};
                            d_c::Int = config.n_embd ÷ 4) where {P<:AbstractFloat}
    n_embd = config.n_embd
    n_head = config.n_head
    block_size = config.block_size

    @assert n_embd % n_head == 0 "n_embd ($n_embd) must be divisible by n_head ($n_head)"
    @assert d_c > 0 "d_c must be positive, got $d_c"

    head_dim = n_embd ÷ n_head

    # Query: standard full-rank projection
    W_Q = Linear{P}(n_embd, n_embd; bias=true)

    # KV compression: down-project to shared latent
    W_DKV = Linear{P}(n_embd, d_c; bias=false)

    # KV recovery: up-project from latent to full K,V
    W_UK = Linear{P}(d_c, n_embd; bias=false)
    W_UV = Linear{P}(d_c, n_embd; bias=false)

    # Output projection
    W_out = Linear{P}(n_embd, n_embd; bias=true)

    # Causal mask
    mask = BitMatrix(tril(ones(Bool, block_size, block_size)))

    LatentAttention{P, typeof(mask)}(
        W_Q, W_DKV, W_UK, W_UV, W_out, mask,
        n_head, head_dim, d_c,
        config.attn_pdrop, config.resid_pdrop
    )
end

# ─── Forward Pass ─────────────────────────────────────────────────────────────

"""
    (attn::LatentAttention{P})(x::AbstractArray{P,3}; training::Bool=true) where P

Forward pass of Multi-Head Latent Attention.

Input:  x of shape (n_embd, seq_len, batch)
Output: (n_embd, seq_len, batch)

## Steps
1. Q = W_Q · x                           — standard query projection
2. c_KV = W_DKV · x                      — compress to latent (d_c)
3. K = W_UK · c_KV, V = W_UV · c_KV      — recover K,V from latent
4. Reshape Q,K,V to (head_dim, T, n_head, B)
5. Apply RoPE to Q,K
6. Scaled dot-product attention with causal mask
7. Softmax, dropout, weighted sum of V
8. Recombine heads, output projection, residual dropout
"""
function (attn::LatentAttention{P})(x::AbstractArray{P, 3}; training::Bool=true) where P
    n_embd, T, B = size(x)
    n_head = attn.n_head
    head_dim = attn.head_dim

    # ── Step 1: Query projection (standard) ───────────────────────────────────
    Q = attn.W_Q(x)                                    # (n_embd, T, B)

    # ── Step 2: KV latent compression ─────────────────────────────────────────
    c_KV = attn.W_DKV(x)                               # (d_c, T, B)

    # ── Step 3: Recover K,V from latent ───────────────────────────────────────
    K = attn.W_UK(c_KV)                                 # (n_embd, T, B)
    V = attn.W_UV(c_KV)                                 # (n_embd, T, B)

    # ── Step 4: Reshape for multi-head attention ──────────────────────────────
    Q = reshape(Q, head_dim, T, n_head, B)
    K = reshape(K, head_dim, T, n_head, B)
    V = reshape(V, head_dim, T, n_head, B)

    # ── Step 5: Apply RoPE ────────────────────────────────────────────────────
    if head_dim % 2 == 0
        Q = apply_position_encoding!(Q)
        K = apply_position_encoding!(K)
    end

    # ── Step 6-7: Scaled dot-product attention ────────────────────────────────
    scale = P(1.0 / sqrt(head_dim))
    att = _batched_attention_scores(Q, K, scale)        # (T, T, n_head, B)

    causal_mask = @view attn.mask[1:T, 1:T]
    att = _apply_causal_mask(att, causal_mask, P)

    att = _softmax(att; dims=2)
    att = _dropout(att, attn.attn_pdrop; training=training)

    # ── Step 8: Weighted sum and output projection ────────────────────────────
    out = _batched_attention_output(att, V)              # (head_dim, T, n_head, B)
    out = reshape(out, n_embd, T, B)
    out = attn.W_out(out)
    out = _dropout(out, attn.resid_pdrop; training=training)

    return out
end

# ─── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, attn::LatentAttention{P}) where P
    n_embd = attn.n_head * attn.head_dim
    block_size = size(attn.mask, 1)
    compression = round(100 * (1 - attn.d_c / n_embd); digits=1)
    print(io, "LatentAttention{$P}(d=$(n_embd), d_c=$(attn.d_c) [$(compression)% compression], " *
              "heads=$(attn.n_head), ctx=$(block_size))")
end

# ─── Device Transfer ──────────────────────────────────────────────────────────

function Core.to_device(attn::LatentAttention{P, Mask}, device::Symbol) where {P, Mask}
    LatentAttention{P, typeof(attn.mask)}(
        Core.to_device(attn.W_Q,   device),
        Core.to_device(attn.W_DKV, device),
        Core.to_device(attn.W_UK,  device),
        Core.to_device(attn.W_UV,  device),
        Core.to_device(attn.W_out, device),
        attn.mask,
        attn.n_head, attn.head_dim, attn.d_c,
        attn.attn_pdrop, attn.resid_pdrop
    )
end

# ─── Compatibility helpers ────────────────────────────────────────────────────

# LatentAttention doesn't use W_qkv, but some generic code checks for it
# These accessors allow uniform handling in _collect_fat_params / _fat_extract_grads

"""
    has_latent_kv(attn) → Bool

Return true if the attention layer uses latent KV compression (MLA).
"""
has_latent_kv(::LatentAttention) = true
has_latent_kv(::Any) = false
