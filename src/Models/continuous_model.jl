# ─── Functional Autoregressive Transformer (FAT) ──────────────────────────────
#
# Predict the next FUNCTION, not the next token.
# A strict generalization of autoregressive LMs: when f(x)=constant,
# this reduces to standard next-token prediction.
#
# Each position in the sequence holds a coefficient vector c ∈ ℝ^basis_dim
# representing a function f(x) = Σ cⱼ φⱼ(x) in some basis {φⱼ}.
# The transformer predicts the next coefficient vector.

# ─── Config ───────────────────────────────────────────────────────────────────

"""
    ContinuousConfig{P} <: AbstractModelConfig{P}

Configuration for a Functional Autoregressive Transformer.

Instead of vocab_size (discrete tokens), this has basis_dim (continuous functions).
The transformer blocks are identical to GPT — only the edges change.
"""
struct ContinuousConfig{P<:AbstractFloat} <: AbstractModelConfig{P}
    basis_dim::Int        # dimension of coefficient space (= "vocabulary" of functions)
    basis_type::Symbol    # :monomial, :fourier, :chebyshev, :learned
    n_embd::Int           # transformer embedding dimension
    n_head::Int           # number of attention heads
    n_layer::Int          # number of transformer blocks
    block_size::Int       # max sequence length (max functions in a chain)
    dropout::P            # dropout rate
    # Inherited fields needed by Block
    embd_pdrop::P
    attn_pdrop::P
    resid_pdrop::P
    # For AbstractModelConfig compatibility
    vocab_size::Int       # set to basis_dim (continuous "vocab")
end

function ContinuousConfig{P}(;
    basis_dim::Int = 16,
    basis_type::Symbol = :monomial,
    n_embd::Int = 128,
    n_head::Int = 4,
    n_layer::Int = 4,
    block_size::Int = 32,
    dropout::Real = 0.0,
) where P
    dp = P(dropout)
    ContinuousConfig{P}(basis_dim, basis_type, n_embd, n_head, n_layer,
                         block_size, dp, dp, dp, dp, basis_dim)
end

ContinuousConfig(; kwargs...) = ContinuousConfig{Float32}(; kwargs...)

# ─── Model ────────────────────────────────────────────────────────────────────

"""
    ContinuousModel{P}

Functional Autoregressive Transformer — predicts the next function
in a sequence, represented as coefficient vectors in a chosen basis.

Architecture:
  coefficient_encoder: Linear(basis_dim → n_embd)  ← replaces Embedding lookup
  blocks: Block{P}[]                      ← IDENTICAL to GPT
  ln_f: Norm{P}                                   ← same as GPT
  coefficient_head: Linear(n_embd → basis_dim)       ← replaces softmax head
"""
struct ContinuousModel{P<:AbstractFloat}
    config::ContinuousConfig{P}
    coeff_encoder::Linear{P, true}    # (basis_dim → n_embd) with bias
    blocks::Vector{Block{P}}
    ln_f::Norm{P}
    coeff_head::Linear{P, true}       # (n_embd → basis_dim) regression head
    dropout_rate::P
end

function ContinuousModel(config::ContinuousConfig{P}) where P
    @assert config.n_embd % config.n_head == 0

    coeff_encoder = Linear{P}(config.basis_dim, config.n_embd; bias=true)

    # Build blocks with Linear-only attention (no OrthogonalLinear) for AD compatibility
    n_embd = config.n_embd
    blocks = map(1:config.n_layer) do _
        ln1 = Norm{P}(n_embd)
        attention = Attention{P}(config; use_orthogonal_out=false, use_orthogonal_qkv=false)
        ln2 = Norm{P}(n_embd)
        ff = FeedForward{P}(config)
        Block{P, typeof(attention), typeof(ln1), typeof(ff)}(
            ln1, attention, ln2, ff, config.resid_pdrop
        )
    end

    ln_f = Norm{P}(config.n_embd)
    coeff_head = Linear{P}(config.n_embd, config.basis_dim; bias=true)

    ContinuousModel{P}(config, coeff_encoder, blocks, ln_f,
                        coeff_head, config.dropout)
end

function Base.show(io::IO, m::ContinuousModel{P}) where P
    c = m.config
    print(io, "ContinuousModel{$P}(basis=$(c.basis_type), basis_dim=$(c.basis_dim), n_embd=$(c.n_embd), layers=$(c.n_layer), heads=$(c.n_head))")
end

# ─── Forward Pass ─────────────────────────────────────────────────────────────

"""
    forward(model::ContinuousModel{P}, coeffs; targets=nothing) where P

Forward pass through the Functional Autoregressive Transformer.

# Arguments
- `coeffs`: (basis_dim, T, B) — sequence of T coefficient vectors, batch of B
- `targets`: (basis_dim, T, B) — target coefficient vectors (shifted by 1)

# Returns
- `predicted`: (basis_dim, T, B) — predicted next-function coefficients
- `loss`: MSE loss in coefficient space (or nothing if no targets)
"""
function forward(model::ContinuousModel{P}, coeffs::AbstractArray{P, 3};
                 targets::Union{Nothing, AbstractArray{P, 3}} = nothing,
                 training::Bool = true) where P
    basis_dim, T, B = size(coeffs)

    # 1. Encode coefficients → embeddings (no mutation)
    #    coeff_encoder.weight: (n_embd, basis_dim), coeffs reshaped: (basis_dim, T*B)
    coeffs_2d = reshape(coeffs, basis_dim, T * B)
    x_2d = model.coeff_encoder(coeffs_2d)  # (n_embd, T*B)
    n_embd = size(x_2d, 1)
    x = reshape(x_2d, n_embd, T, B)

    # 3. Transformer blocks
    for block in model.blocks
        x = block(x)
    end

    # 4. Final layer norm
    x = model.ln_f(x)

    # 5. Coefficient head (no mutation)
    x_2d_out = reshape(x, n_embd, T * B)
    pred_2d = model.coeff_head(x_2d_out)  # (basis_dim, T*B)
    predicted = reshape(pred_2d, basis_dim, T, B)

    # 6. Loss: MSE in coefficient space
    loss = nothing
    if targets !== nothing
        diff = predicted .- targets
        loss = sum(abs2, diff) / P(basis_dim * T * B)
    end

    return predicted, loss
end

# ─── Basis Evaluation ─────────────────────────────────────────────────────────

"""
    evaluate_function(coeffs, x; basis=:monomial)

Evaluate f(x) = Σ cⱼ φⱼ(x) for given coefficient vector and basis.
"""
function evaluate_function(coeffs::AbstractVector{P}, x::P;
                           basis::Symbol = :monomial) where P
    if basis == :monomial
        # f(x) = c₁ + c₂x + c₃x² + ... (Horner's method)
        result = zero(P)
        xpow = one(P)
        for c in coeffs
            result += c * xpow
            xpow *= x
        end
        return result
    elseif basis == :fourier
        # f(x) = c₁ + c₂cos(x) + c₃sin(x) + c₄cos(2x) + c₅sin(2x) + ...
        result = coeffs[1]
        for k in 1:((length(coeffs)-1)÷2)
            result += coeffs[2k] * cos(P(k) * x) + coeffs[2k+1] * sin(P(k) * x)
        end
        return result
    elseif basis == :chebyshev
        # Chebyshev recurrence: T₀=1, T₁=x, Tₙ=2xTₙ₋₁-Tₙ₋₂
        d = length(coeffs)
        d == 0 && return zero(P)
        d == 1 && return coeffs[1]
        T_prev = one(P)
        T_curr = x
        result = coeffs[1] * T_prev + coeffs[2] * T_curr
        for j in 3:d
            T_next = P(2) * x * T_curr - T_prev
            result += coeffs[j] * T_next
            T_prev = T_curr
            T_curr = T_next
        end
        return result
    else
        error("Unknown basis: $basis")
    end
end

"""
    evaluate_function(coeffs, xs; basis=:monomial)

Evaluate function at multiple points, returning a vector.
"""
function evaluate_function(coeffs::AbstractVector{P}, xs::AbstractVector{P};
                           basis::Symbol = :monomial) where P
    [evaluate_function(coeffs, x; basis=basis) for x in xs]
end

# ─── Generation ───────────────────────────────────────────────────────────────

"""
    generate(model::ContinuousModel{P}, seed_coeffs; max_new=10) where P

Autoregressively generate the next functions in a sequence.

# Arguments
- `seed_coeffs`: (basis_dim, T_seed) — seed sequence of coefficient vectors
- `max_new`: number of new functions to predict

# Returns
- (basis_dim, T_seed + max_new) — full sequence including predictions
"""
function generate(model::ContinuousModel{P}, seed_coeffs::AbstractMatrix{P};
                  max_new::Int = 10) where P
    basis_dim, T = size(seed_coeffs)
    # Build up the sequence
    sequence = copy(seed_coeffs)  # (basis_dim, T_current)

    for _ in 1:max_new
        T_curr = size(sequence, 2)
        # Use at most block_size positions
        T_use = min(T_curr, model.config.block_size)
        input = reshape(sequence[:, end-T_use+1:end], basis_dim, T_use, 1)  # add batch dim

        predicted, _ = forward(model, input; training=false)

        # Take last position's prediction
        next_coeffs = predicted[:, end, 1]
        sequence = hcat(sequence, next_coeffs)
    end

    return sequence
end

# ─── Device Transfer ───────────────────────────────────────────────────────────

# Device transfer for ContinuousModel
import ..Core

function Core.to_device(model::ContinuousModel{P}, device::Symbol) where P
    coeff_encoder_new = to_device(model.coeff_encoder, device)
    blocks_new = [to_device(b, device) for b in model.blocks]
    ln_f_new = to_device(model.ln_f, device)
    coeff_head_new = to_device(model.coeff_head, device)
    return ContinuousModel{P}(model.config, coeff_encoder_new, blocks_new, ln_f_new, coeff_head_new, model.dropout_rate)
end
