# ─── Causal Self-Attention ─────────────────────────────────────────────────────

"""
    Attention{P, D, Mask} <: AbstractAttention{P}

Multi-head causal self-attention layer with combined Q,K,V projection.
Uses column-major layout (head_dim, seq_len, n_heads, batch) for attention
computation, exploiting Julia's memory layout for cache-friendly access.

# Type Parameters
- `P`: Floating-point precision type (Float16, Float32, Float64)
- `D`: Linear layer type for projections
- `Mask`: Boolean matrix type for the causal mask

# Fields
- `W_qkv::D` — Combined Q,K,V projection (n_embd → 3*n_embd)
- `W_out::D` — Output projection (n_embd → n_embd)
- `mask::Mask` — Lower-triangular causal mask (block_size × block_size)
- `n_head::Int` — Number of attention heads
- `head_dim::Int` — Dimension per head (n_embd ÷ n_head)
- `attn_pdrop::P` — Dropout probability for attention weights
- `resid_pdrop::P` — Dropout probability for output projection
"""
struct Attention{P, Mask<:AbstractMatrix{Bool}} <: AbstractAttention{P}
    # Projections: either `W_qkv` (combined) or `W_q`, `W_k`, `W_v` (separate)
    W_qkv::Any
    W_out::Any
    W_q::Union{Nothing, Any}
    W_k::Union{Nothing, Any}
    W_v::Union{Nothing, Any}
    mask::Mask
    n_head::Int
    head_dim::Int
    attn_pdrop::P
    resid_pdrop::P
    mode::Symbol                    # :softmax or :kernel (linearized kernel/RFF)
    rff_W::Union{Nothing, Array{P,2}}  # RFF projection matrix (m × head_dim) or nothing
    rff_b::Union{Nothing, Array{P,1}}  # RFF bias vector (m) or nothing
    rff_m::Int                      # number of RFF features (0 if unused)
end

"""
    Attention{P}(config::AbstractModelConfig{P}) where {P<:AbstractFloat}

Construct a Attention layer from a ModelConfig.
Creates combined Q,K,V projection, output projection, and causal mask buffer.

# Arguments
- `config` — ModelConfig providing n_embd, n_head, block_size, attn_pdrop, resid_pdrop

# Examples
```julia
config = ModelConfig{Float32}(model_type=:gpt_nano)
attn = Attention{Float32}(config)
```
"""
function Attention{P}(config::AbstractModelConfig{P}, mode::Symbol=:softmax; rff_m::Int=64, use_orthogonal_out::Bool=false, use_orthogonal_qkv::Bool=false) where {P<:AbstractFloat}
    n_embd = config.n_embd
    n_head = config.n_head
    block_size = config.block_size

    @assert n_embd % n_head == 0 "n_embd ($n_embd) must be divisible by n_head ($n_head)"

    head_dim = n_embd ÷ n_head

    # Projections: either combined W_qkv or separate square projections W_q/W_k/W_v
    if use_orthogonal_qkv
        W_qkv = nothing
        W_q = OrthogonalLinear{P}(n_embd; bias=true)
        W_k = OrthogonalLinear{P}(n_embd; bias=true)
        W_v = OrthogonalLinear{P}(n_embd; bias=true)
    else
        W_qkv = Linear{P}(n_embd, 3 * n_embd; bias=true)
        W_q = nothing; W_k = nothing; W_v = nothing
    end

    # Output projection: (n_embd) → (n_embd) (optionally Cayley)
    if use_orthogonal_out
        W_out = OrthogonalLinear{P}(n_embd; bias=true)
    else
        W_out = Linear{P}(n_embd, n_embd; bias=true)
    end

    # Causal mask: lower-triangular boolean matrix
    # mask[i, j] = true means position i can attend to position j (j ≤ i)
    mask = BitMatrix(tril(ones(Bool, block_size, block_size)))

    # RFF params for linearized kernel attention (optional)
    if mode == :kernel
        m = max(4, Int(rff_m))
        # rff_W: (m, head_dim), rff_b: (m)
        rff_W = randn(P, m, head_dim)
        rff_b = (2f0 * pi) .* rand(P, m)
    else
        m = 0
        rff_W = nothing
        rff_b = nothing
    end

    Attention{P, typeof(mask)}(
        W_qkv, W_out, W_q, W_k, W_v, mask, n_head, head_dim,
        config.attn_pdrop, config.resid_pdrop,
        mode, rff_W, rff_b, m
    )
end

"""
    Attention(config::AbstractModelConfig{P}) where P

Convenience constructor inferring precision from config.
"""
Attention(config::AbstractModelConfig{P}) where {P} = Attention{P}(config)

# ─── Dropout utility ──────────────────────────────────────────────────────────

"""
    _dropout(x::AbstractArray{P}, p::P; training::Bool=true) where P

Apply dropout to array `x` with probability `p` during training.
Scales retained values by 1/(1-p) to maintain expected values (inverted dropout).
Returns `x` unchanged when not training or when p ≈ 0.
"""
function _dropout(x::AbstractArray{P}, p::P; training::Bool=true) where {P<:AbstractFloat}
    if !training || p ≤ zero(P)
        return x
    end
    # Inverted dropout: scale by 1/(1-p) so expected value is preserved
    mask = rand(P, size(x)) .> p
    return x .* mask ./ (one(P) - p)
end

# ─── Softmax utility ──────────────────────────────────────────────────────────

"""
    _softmax(x::AbstractArray{P}; dims::Int) where P

Numerically stable softmax along the specified dimension.
Subtracts the maximum value before exponentiation to prevent overflow.
"""
function _softmax(x::AbstractArray{P}; dims::Int) where {P<:AbstractFloat}
    x_max = maximum(x; dims=dims)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x; dims=dims)
end

# ─── Rotary Positional Embedding (RoPE) ───────────────────────────────────────

"""
    apply_position_encoding!(x::AbstractArray{P, 4}) where P

Apply Rotary Positional Embedding (RoPE) to a 4D tensor (head_dim, T, n_heads, B).
Implemented purely mutation-free for Zygote.jl AD compatibility.
"""
function apply_position_encoding!(x::AbstractArray{P, 4}) where {P<:AbstractFloat}
    head_dim, T, n_heads, B = size(x)
    @assert head_dim % 2 == 0 "head_dim must be even for RoPE"
    
    half = head_dim ÷ 2
    inv_freq = P.(1.0 ./ (10000.0 .^ ((0:half-1) .* (2.0 / head_dim))))
    pos = P.(0:T-1)
    angles = inv_freq * pos'  # (half, T)
    
    cosv = reshape(cos.(angles), half, T, 1, 1)
    sinv = reshape(sin.(angles), half, T, 1, 1)
    
    # Separate into even and odd indices
    x1 = x[1:2:end, :, :, :]
    x2 = x[2:2:end, :, :, :]
    
    out1 = x1 .* cosv .- x2 .* sinv
    out2 = x1 .* sinv .+ x2 .* cosv
    
    # Interleave out1 and out2 mutation-free
    out1_r = reshape(out1, 1, half, T, n_heads, B)
    out2_r = reshape(out2, 1, half, T, n_heads, B)
    out = vcat(out1_r, out2_r)  # (2, half, T, n_heads, B)
    
    return reshape(out, head_dim, T, n_heads, B)
end

# ─── Forward pass ─────────────────────────────────────────────────────────────

"""
    (attn::Attention{P})(x::AbstractArray{P, 3}; training::Bool=true) where P

Forward pass of causal self-attention.

Input: x of shape (n_embd, seq_len, batch)
Output: (n_embd, seq_len, batch)

Steps:
1. Project input to Q, K, V via combined linear projection
2. Reshape to (head_dim, seq_len, n_heads, batch)
3. Compute scaled dot-product attention scores
4. Apply causal mask (future positions → -Inf)
5. Apply softmax along key dimension
6. Apply attention dropout
7. Compute weighted sum of values
8. Recombine heads and apply output projection with residual dropout
"""
function (attn::Attention{P})(x::AbstractArray{P, 3}; training::Bool=true) where P
    n_embd, T, B = size(x)
    n_head = attn.n_head
    head_dim = attn.head_dim

    # Step 1: Project input to Q, K, V. Either combined W_qkv or separate W_q/W_k/W_v
    if attn.W_q !== nothing
        Q = attn.W_q(x)
        K = attn.W_k(x)
        V = attn.W_v(x)
    else
        # Combined Q,K,V projection → (3*n_embd, T, B)
        qkv = attn.W_qkv(x)
        # Split into Q, K, V each of shape (n_embd, T, B)
        Q = qkv[1:n_embd, :, :]
        K = qkv[n_embd+1:2*n_embd, :, :]
        V = qkv[2*n_embd+1:3*n_embd, :, :]
    end

    # Step 2: Reshape to (head_dim, T, n_heads, B) for multi-head attention
    Q = reshape(Q, head_dim, T, n_head, B)
    K = reshape(K, head_dim, T, n_head, B)
    V = reshape(V, head_dim, T, n_head, B)

    # Step 3: Apply Rotary Positional Embedding natively to all Q and K
    if head_dim % 2 == 0
        Q = apply_position_encoding!(Q)
        K = apply_position_encoding!(K)
    end

    if attn.mode == :softmax
        # Standard scaled-dot-product path
        scale = P(1.0 / sqrt(head_dim))
        att = _batched_attention_scores(Q, K, scale)
        causal_mask = @view attn.mask[1:T, 1:T]
        att = _apply_causal_mask(att, causal_mask, P)
        att = _softmax(att; dims=2)
        att = _dropout(att, attn.attn_pdrop; training=training)
        out = _batched_attention_output(att, V)
        out = reshape(out, n_embd, T, B)
        out = attn.W_out(out)
        out = _dropout(out, attn.resid_pdrop; training=training)
        return out
    elseif attn.mode == :kernel
        # Linearized (RFF) causal attention prototype (RoPE already applied)
        out = zeros(P, head_dim, T, n_head, B)
        eps = P(1e-6)

        for b in 1:B
            for h in 1:n_head
                q_rope = @view Q[:, :, h, b]   # (head_dim, T)
                k_rope = @view K[:, :, h, b]
                V_slice = @view V[:, :, h, b]

                # Feature map φ via RFF: φ(x) = sqrt(2/m) * cos(W * x + b)
                if attn.rff_W === nothing
                    # fallback: use identity features (no kernelization)
                    phiQ = q_rope
                    phiK = k_rope
                    m_eff = size(phiQ, 1)
                else
                    W_rff = attn.rff_W
                    b_rff = attn.rff_b
                    m_eff = attn.rff_m
                    zQ = W_rff * q_rope .+ b_rff[:, ones(Int, size(q_rope, 2))]
                    zK = W_rff * k_rope .+ b_rff[:, ones(Int, size(k_rope, 2))]
                    s = sqrt(2.0 / float(m_eff))
                    phiQ = s .* cos.(zQ)
                    phiK = s .* cos.(zK)
                end

                # Causal prefix sums: S_k_cum[:, t] = Σ_{j ≤ t} φ(k_j)
                S_k_cum = cumsum(phiK; dims=2)  # (m_eff, T)

                # Build cumulative S_v_cum[:, :, t] = Σ_{j ≤ t} φ(k_j) * v_j^T  (m_eff × head_dim × T)
                H = head_dim
                M3 = Array{P}(undef, m_eff, H, T)
                for j in 1:T
                    # outer product φ(k_j) (m_eff) and v_j (H)
                    M3[:, :, j] = phiK[:, j] * (V_slice[:, j])'
                end
                S_v_cum = cumsum(M3; dims=3)  # (m_eff, H, T)

                # Compute outputs per position t: out[:, t] = (S_v_cum[:,:,t]' * φ(q_t)) / (φ(q_t)' * S_k_cum[:,t])
                for t in 1:T
                    phi_q_t = phiQ[:, t]
                    S_v_t = S_v_cum[:, :, t]   # (m_eff, H)
                    numer = S_v_t' * phi_q_t     # (H,)
                    denom = dot(phi_q_t, S_k_cum[:, t]) + eps
                    out[:, t, h, b] = vec(numer) ./ denom
                end
            end
        end

        # Recombine heads and apply output projection + dropout
        out = reshape(out, n_embd, T, B)
        out = attn.W_out(out)
        out = _dropout(out, attn.resid_pdrop; training=training)
        return out
    else
        error("unknown attention mode: $(attn.mode)")
    end
end

# ─── Batched attention score computation ──────────────────────────────────────

"""
    _batched_attention_scores(Q, K, scale)

Compute scaled attention scores: att[i, j, h, b] = scale * Σ_d Q[d,i,h,b] * K[d,j,h,b]

Input Q, K: (head_dim, T, n_heads, B)
Output: (T, T, n_heads, B)
"""
function _batched_attention_scores(
    Q::AbstractArray{P, 4}, K::AbstractArray{P, 4}, scale::P
) where {P<:AbstractFloat}
    head_dim, T, n_heads, B = size(Q)

    # Use permutedims for pure Q^T K batched matmul
    # Q: (head_dim, T, n_heads, B) → Qt: (T, head_dim, n_heads, B)
    Qt = permutedims(Q, (2, 1, 3, 4))
    # Compute scores: att[i,j,h,b] = sum_d Qt[i,d,h,b] * K[d,j,h,b]
    # Reshape to 3D for matmul: (T, head_dim, HB) × (head_dim, T, HB)
    Qt_3d = reshape(Qt, T, head_dim, n_heads * B)
    K_3d = reshape(K, head_dim, T, n_heads * B)

    # Stack results using comprehension (allocation-based, no mutation)
    att_3d = stack([Qt_3d[:, :, hb] * K_3d[:, :, hb] for hb in 1:(n_heads * B)]; dims=3)
    att = reshape(att_3d, T, T, n_heads, B)

    att = att .* scale
    return att
end

"""
    _apply_causal_mask(att, causal_mask, ::Type{P})

Apply causal mask to attention scores. Positions where mask is false get -Inf.

att: (T, T, n_heads, B)
causal_mask: (T, T) — lower-triangular boolean matrix
"""
function _apply_causal_mask(
    att::AbstractArray{P, 4}, causal_mask::AbstractMatrix{Bool}, ::Type{P}
) where {P<:AbstractFloat}
    # Use large negative instead of -Inf to avoid NaN from 0*Inf in AD
    mask_vals = P.(causal_mask)  # 1.0 for attend, 0.0 for mask
    mask_4d = reshape(mask_vals, size(causal_mask, 1), size(causal_mask, 2), 1, 1)
    large_neg = P(-1e9)
    # Where mask=1: att*1 + 0*large_neg = att
    # Where mask=0: att*0 + 1*large_neg = -1e9  (no NaN because finite*0 = 0)
    return att .* mask_4d .+ (one(P) .- mask_4d) .* large_neg
end

"""
    _batched_attention_output(att, V)

Compute attention-weighted values: out[d, i, h, b] = Σ_j att[i, j, h, b] * V[d, j, h, b]

att: (T, T, n_heads, B)
V: (head_dim, T, n_heads, B)
Output: (head_dim, T, n_heads, B)
"""
function _batched_attention_output(
    att::AbstractArray{P, 4}, V::AbstractArray{P, 4}
) where {P<:AbstractFloat}
    T_q, T_k, n_heads, B = size(att)
    head_dim = size(V, 1)

    V_3d = reshape(V, head_dim, T_k, n_heads * B)
    att_t = permutedims(att, (2, 1, 3, 4))  # (T_k, T_q, n_heads, B)
    att_t_3d = reshape(att_t, T_k, T_q, n_heads * B)

    # out[:,i,hb] = V[:,:,hb] * att_t[:,i,hb]
    out_3d = stack([V_3d[:, :, hb] * att_t_3d[:, :, hb] for hb in 1:(n_heads * B)]; dims=3)
    return reshape(out_3d, head_dim, T_q, n_heads, B)
end

# Device transfer for Attention
import ..Core

function Core.to_device(attn::Attention{P, Mask}, device::Symbol) where {P, Mask<:AbstractMatrix{Bool}}
    W_qkv_new = attn.W_qkv === nothing ? nothing : to_device(attn.W_qkv, device)
    W_out_new = attn.W_out === nothing ? nothing : to_device(attn.W_out, device)
    W_q_new = attn.W_q === nothing ? nothing : to_device(attn.W_q, device)
    W_k_new = attn.W_k === nothing ? nothing : to_device(attn.W_k, device)
    W_v_new = attn.W_v === nothing ? nothing : to_device(attn.W_v, device)
    mask_new = to_device(attn.mask, device)
    rff_W_new = attn.rff_W === nothing ? nothing : to_device(attn.rff_W, device)
    rff_b_new = attn.rff_b === nothing ? nothing : to_device(attn.rff_b, device)

    return Attention{P, typeof(mask_new)}(
        W_qkv_new, W_out_new, W_q_new, W_k_new, W_v_new, mask_new,
        attn.n_head, attn.head_dim, attn.attn_pdrop, attn.resid_pdrop,
        attn.mode, rff_W_new, rff_b_new, attn.rff_m
    )
end

# ─── Display ──────────────────────────────────────────────────────────────────

"""
    Base.show(io::IO, attn::Attention{P}) where P

Pretty-print a Attention layer.
"""
function Base.show(io::IO, attn::Attention{P}) where P
    n_embd = attn.n_head * attn.head_dim
    block_size = size(attn.mask, 1)
    print(io, "Attention{$P}(n_embd=$(n_embd), n_head=$(attn.n_head), head_dim=$(attn.head_dim), block_size=$(block_size))")
end
