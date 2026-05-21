# --- PHASE 2 ---
# Spectral Head: ℝ^{d_model} → ℝ^{K+1}
#
# Replaces the standard LM head  (Linear(d_model → |V|) → Softmax)
# with a purely linear projection into the Chebyshev coefficient space.
#
# No Softmax, no discrete vocabulary.  The output is the coefficient vector
# Θ_t = (θ₀, θ₁, …, θ_K) ∈ ℝ^{K+1} defining the predicted function
#
#     f_{Θ_t}(x) = Σₙ₌₀^K θₙ Tₙ(x)
#
# Gradients flow from the functional loss (Phase 4) through Θ_t, through
# the weight matrix W and bias b, and back into the transformer backbone
# via standard Zygote autograd — no special treatment needed.
#
# Column-major layout throughout:
#   input   h_t : (d_model,)                 — single hidden state
#   input   H   : (d_model, seq_len, batch)  — batched hidden states
#   output      : (K+1,   seq_len, batch)    — coefficient tensors

using LinearAlgebra

# ─── SpectralHead ─────────────────────────────────────────────────────────────

"""
    SpectralHead{T<:AbstractFloat, K} <: AbstractLayer{T}

Linear spectral projection head implementing

    φ_spec : ℝ^{d_model} → ℝ^{K+1},   h ↦ W h + b

where `W ∈ ℝ^{(K+1)×d_model}` and `b ∈ ℝ^{K+1}`.

This replaces the standard LM head `(d_model → |V|) → Softmax` with a
regression head into the truncated Chebyshev coefficient space V_K.

## Type Parameters
- `T`: floating-point precision (Float32, Float64, …)
- `K`: truncation order; output dimension is `K+1`

## Fields
- `weight :: Matrix{T}` — shape `(K+1, d_model)`; stored row-major in value
  but accessed column-major by Julia's BLAS (no copy needed)
- `bias   :: Vector{T}` — shape `(K+1,)`

## Layout rationale
`weight` is `(K+1) × d_model` so that `weight * h` (matrix-vector product)
produces a `(K+1)` output — consistent with the rest of the Architect Layers
convention where `Linear.weight` is `(out, in)`.
"""
struct SpectralHead{T<:AbstractFloat, K}
    weight::Matrix{T}    # (K+1, d_model)  — out × in, BLAS-friendly
    bias::Vector{T}      # (K+1,)
end

# ─── Constructors ─────────────────────────────────────────────────────────────

"""
    SpectralHead{T,K}(d_model::Int; init_scale::Real=0.02) where {T,K}

Construct a SpectralHead with:
  - `weight` initialized from 𝒩(0, init_scale²)  (same convention as Linear)
  - `bias`   initialized to zero

The small initialization scale (0.02) ensures that early-training outputs
cluster near the origin in coefficient space, keeping the Chebyshev expansion
well-bounded before the first gradient update.
"""
function SpectralHead{T,K}(d_model::Int; init_scale::Real = 0.02) where {T<:AbstractFloat, K}
    W = randn(T, K+1, d_model) .* T(init_scale)
    b = zeros(T, K+1)
    SpectralHead{T,K}(W, b)
end

"""
    SpectralHead{T}(d_model::Int, K::Int; init_scale=0.02)

Convenience constructor when `K` is known at runtime (not type level).
Falls back to the type-parametric form.
"""
SpectralHead{T}(d_model::Int, K::Int; kwargs...) where {T<:AbstractFloat} =
    SpectralHead{T,K}(d_model; kwargs...)

"""
    SpectralHead(d_model::Int, K::Int; kwargs...)

Convenience constructor defaulting to Float32.
"""
SpectralHead(d_model::Int, K::Int; kwargs...) = SpectralHead{Float32,K}(d_model; kwargs...)

# ─── Forward Pass (callable struct) ───────────────────────────────────────────

"""
    (head::SpectralHead{T,K})(h::AbstractVector{T}) -> Vector{T}

Single-vector forward pass.

    Θ = W h + b  ∈ ℝ^{K+1}

Input : `h`  — shape `(d_model,)`
Output: `Θ`  — shape `(K+1,)`
"""
function (head::SpectralHead{T,K})(h::AbstractVector{T}) where {T<:AbstractFloat, K}
    return head.weight * h .+ head.bias
end

"""
    (head::SpectralHead{T,K})(H::AbstractMatrix{T}) -> Matrix{T}

Batched 2-D forward pass.

    Θ = W H + b  ∈ ℝ^{(K+1) × N}

Input : `H`  — shape `(d_model, N)`  (N = seq_len or seq_len×batch)
Output: `Θ`  — shape `(K+1,    N)`

Uses a single BLAS `gemm` call; `b` broadcasts over columns automatically.
"""
function (head::SpectralHead{T,K})(H::AbstractMatrix{T}) where {T<:AbstractFloat, K}
    # head.weight : (K+1, d_model)
    # H           : (d_model, N)
    # result      : (K+1, N)    via BLAS gemm — column-major, cache-optimal
    return head.weight * H .+ head.bias
end

"""
    (head::SpectralHead{T,K})(H::AbstractArray{T,3}) -> Array{T,3}

Batched 3-D forward pass for the standard Architect tensor layout.

Input : `H`  — shape `(d_model, seq_len, batch)`
Output: `Θ`  — shape `(K+1,    seq_len, batch)`

Implementation: reshape to 2-D, apply linear map, reshape back.
The reshape is O(1) (no data copy) for contiguous column-major arrays.
"""
function (head::SpectralHead{T,K})(H::AbstractArray{T,3}) where {T<:AbstractFloat, K}
    d_model, seq_len, batch = size(H)
    # Flatten seq and batch into a single trailing dimension → (d_model, seq_len·batch)
    # This lets us use a single BLAS gemm instead of a loop over batch.
    H_2d  = reshape(H, d_model, seq_len * batch)          # view, O(1)
    Θ_2d  = head.weight * H_2d .+ head.bias               # (K+1, seq_len·batch)
    return reshape(Θ_2d, K+1, seq_len, batch)             # view, O(1)
end

# ─── SpectralState extraction from a tensor ───────────────────────────────────

"""
    extract_state(head::SpectralHead{T,K}, Θ_tensor, t::Int, b::Int) -> SpectralState{T,K}

Extract the `SpectralState` at sequence position `t`, batch index `b`
from a coefficient tensor `Θ_tensor` of shape `(K+1, seq_len, batch)`.

Returns a `SpectralState{T,K}` wrapping a copy of the relevant coefficient
slice.  Primarily useful for autoregressive generation and diagnostics.
"""
function extract_state(::SpectralHead{T,K},
                       Θ_tensor::AbstractArray{T,3},
                       t::Int, b::Int) where {T<:AbstractFloat, K}
    SpectralState{T,K}(Vector{T}(Θ_tensor[:, t, b]))
end

# ─── Parameter helpers (Zygote-compatible) ────────────────────────────────────

"""
    params(head::SpectralHead{T,K}) -> NamedTuple

Return the learnable parameters as a NamedTuple for use with
`Zygote.withgradient`.  Mirrors the convention used by `Linear` in layers.jl.
"""
params(head::SpectralHead{T,K}) where {T,K} = (weight = head.weight, bias = head.bias)

"""
    n_params(head::SpectralHead{T,K}) -> Int

Count the total number of scalar parameters.
"""
n_params(head::SpectralHead{T,K}) where {T,K} = length(head.weight) + length(head.bias)

# ─── Device transfer ──────────────────────────────────────────────────────────

"""
    Core.to_device(head::SpectralHead{T,K}, device::Symbol) -> SpectralHead{T,K}

Transfer a SpectralHead to `:cpu` or `:cuda`.
Uses the same `to_device` dispatch machinery as the rest of Architect.
"""
function Core.to_device(head::SpectralHead{T,K}, device::Symbol) where {T<:AbstractFloat, K}
    W_new = Core.to_device(head.weight, device)
    b_new = Core.to_device(head.bias,   device)
    SpectralHead{T,K}(W_new, b_new)
end

# ─── Display ─────────────────────────────────────────────────────────────────

function Base.show(io::IO, head::SpectralHead{T,K}) where {T,K}
    d_model = size(head.weight, 2)
    print(io, "SpectralHead{$T,K=$K}(d_model=$d_model → $(K+1) Chebyshev coeffs)")
end
