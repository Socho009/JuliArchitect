# ─── Transformer Block ─────────────────────────────────────────────────────────

# ─── SiLU Activation ──────────────────────────────────────────────────────────

"""
    silu(x)

Sigmoid Linear Unit activation function (also known as Swish): x * σ(x).
Accepts any Real type (including ForwardDiff.Dual for AD compatibility).
"""
function silu(x::T) where {T<:Real}
    return x / (one(T) + exp(-x))
end

# Broadcast-friendly version for arrays
silu(x::AbstractArray) = silu.(x)

# ─── FeedForward (Feedforward Network) ────────────────────────────────────────────

"""
    FeedForward{P, L1<:Linear{P}, L2<:Linear{P}, L3<:Linear{P}}

FeedForward feedforward network used within a Block.
Architecture: W2(SiLU(W1(x)) ⊙ W3(x))

# Type Parameters
- `P`: Floating-point precision type
- `L1`: Type of the gate projection layer
- `L2`: Type of the down projection layer
- `L3`: Type of the up projection layer

# Fields
- `W1::L1` — Gate projection (n_embd → hidden_dim)
- `W2::L2` — Down projection (hidden_dim → n_embd)
- `W3::L3` — Up projection (n_embd → hidden_dim)
- `resid_pdrop::P` — Dropout probability applied after projection
"""
struct FeedForward{P, L1<:Linear{P}, L2<:Linear{P}, L3<:Linear{P}}
    W1::L1
    W2::L2
    W3::L3
    resid_pdrop::P
end

"""
    FeedForward{P}(config::AbstractModelConfig{P}) where {P<:AbstractFloat}

Construct a FeedForward network from an AbstractModelConfig.
"""
function FeedForward{P}(config::AbstractModelConfig{P}) where {P<:AbstractFloat}
    n_embd = config.n_embd
    hidden_dim = hasproperty(config, :hidden_dim) ? config.hidden_dim : 4 * n_embd
    
    W1 = Linear{P}(n_embd, hidden_dim; bias=false)
    W2 = Linear{P}(hidden_dim, n_embd; bias=false)
    W3 = Linear{P}(n_embd, hidden_dim; bias=false)
    
    FeedForward{P, typeof(W1), typeof(W2), typeof(W3)}(W1, W2, W3, config.resid_pdrop)
end

"""
    (mlp::FeedForward{P})(x::AbstractArray{P, 3}; training::Bool=true) where P

Forward pass of the FeedForward MLP.
Input: (n_embd, seq_len, batch)
Output: (n_embd, seq_len, batch)
"""
function (mlp::FeedForward{P})(x::AbstractArray{P, 3}; training::Bool=true) where P
    gate = mlp.W1(x)          # (hidden_dim, T, B)
    up = mlp.W3(x)            # (hidden_dim, T, B)
    h = silu.(gate) .* up
    out = mlp.W2(h)           # (n_embd, T, B)
    out = _dropout(out, mlp.resid_pdrop; training=training)
    return out
end

# ─── Block ─────────────────────────────────────────────────────────

"""
    Block{P, A<:AbstractAttention{P}, LN, FF} <: AbstractBlock{P}

A single transformer block implementing pre-norm architecture with residual connections.

Architecture (pre-norm):
```
x → Norm → Attention → + → Norm → FeedForward → + → output
|___________________________________|   |__________________|
         residual connection                residual connection
```

# Type Parameters
- `P`: Floating-point precision type
- `A`: Attention layer type
- `LN`: Norm type
- `FF`: Feedforward (FeedForward) type

# Fields
- `ln1::LN` — RMS normalization before attention
- `attention::A` — Causal self-attention layer
- `ln2::LN` — RMS normalization before FeedForward
- `feedforward::FF` — Feedforward FeedForward network
- `resid_pdrop::P` — Dropout probability for residual connections
"""
struct Block{P, A<:AbstractAttention{P}, LN, FF} <: AbstractBlock{P}
    ln1::LN
    attention::A
    ln2::LN
    feedforward::FF
    resid_pdrop::P
end

"""
    Block{P}(config::AbstractModelConfig{P}) where {P<:AbstractFloat}

Construct a Block from a config.
"""
function Block{P}(config::AbstractModelConfig{P}) where {P<:AbstractFloat}
    n_embd = config.n_embd

    ln1 = Norm{P}(n_embd)
    # Use Cayley reparameterization for both output projection and Q/K/V by default
    attention = Attention{P}(config; use_orthogonal_out=true, use_orthogonal_qkv=true)
    ln2 = Norm{P}(n_embd)
    feedforward = FeedForward{P}(config)

    Block{P, typeof(attention), typeof(ln1), typeof(feedforward)}(
        ln1, attention, ln2, feedforward, config.resid_pdrop
    )
end

"""
    Block(config::AbstractModelConfig{P}) where P

Convenience constructor inferring precision from config.
"""
Block(config::AbstractModelConfig{P}) where {P} = Block{P}(config)

"""
    (block::Block{P})(x::AbstractArray{P, 3}; training::Bool=true) where P

Forward pass of the Block using pre-norm architecture.
"""
function (block::Block{P})(x::AbstractArray{P, 3}; training::Bool=true) where P
    # Residual 1: Pre-norm attention
    attn_out = block.attention(block.ln1(x); training=training)
    x = x .+ attn_out

    # Residual 2: Pre-norm FeedForward
    mlp_out = block.feedforward(block.ln2(x); training=training)
    x = x .+ mlp_out

    return x
end

# ─── Display ──────────────────────────────────────────────────────────────────

"""
    Base.show(io::IO, mlp::FeedForward{P}) where P

Pretty-print a FeedForward layer.
"""
function Base.show(io::IO, mlp::FeedForward{P}) where P
    in_dim = size(mlp.W1.weight, 2)
    hidden_dim = size(mlp.W1.weight, 1)
    print(io, "FeedForward{$P}($in_dim → $hidden_dim → $in_dim)")
end

"""
    Base.show(io::IO, block::Block{P}) where P

Pretty-print a Block.
"""
function Base.show(io::IO, block::Block{P}) where P
    print(io, "Block{$P}(")
    print(io, "attention=$(block.attention), ")
    print(io, "mlp=$(block.feedforward))")
end

# Device transfer for FeedForward and Block
import ..Core

function Core.to_device(ff::FeedForward{P, L1, L2, L3}, device::Symbol) where {P, L1<:Linear{P}, L2<:Linear{P}, L3<:Linear{P}}
    W1_new = to_device(ff.W1, device)
    W2_new = to_device(ff.W2, device)
    W3_new = to_device(ff.W3, device)
    return FeedForward{P, typeof(W1_new), typeof(W2_new), typeof(W3_new)}(W1_new, W2_new, W3_new, ff.resid_pdrop)
end

function Core.to_device(block::Block{P, A, LN, FF}, device::Symbol) where {P, A<:AbstractAttention{P}, LN, FF}
    ln1_new = to_device(block.ln1, device)
    attn_new = to_device(block.attention, device)
    ln2_new = to_device(block.ln2, device)
    ff_new = to_device(block.feedforward, device)
    return Block{P, typeof(attn_new), typeof(ln1_new), typeof(ff_new)}(ln1_new, attn_new, ln2_new, ff_new, block.resid_pdrop)
end
