# ─── Core Layers ───

using LinearAlgebra

# ─── Embedding Layer ─────────────────────────────────────────────────────────

"""
    Embedding{P, M<:AbstractMatrix{P}} <: AbstractLayer{P}

Embedding lookup table storing dense vectors in column-major layout.
Weight matrix has shape (n_embd, vocab_size) so that each token's embedding
is a contiguous column — exploiting Julia's column-major memory layout for
cache-friendly access during lookup.

# Type Parameters
- `P`: Floating-point precision type (Float16, Float32, Float64)
- `M`: Matrix type (Array{P,2}, CuArray{P,2}, etc.) for device abstraction

# Fields
- `weight::M` — Embedding weight matrix of shape (n_embd, vocab_size)
"""
struct Embedding{P, M<:AbstractMatrix{P}} <: AbstractLayer{P}
    weight::M  # (n_embd, vocab_size) — column-major for fast lookup
end

"""
    Embedding{P}(n_embd::Int, vocab_size::Int) where {P<:AbstractFloat}

Construct an Embedding layer with random Normal(0, 0.02) initialization.
Weight matrix is stored as (n_embd, vocab_size) in column-major layout.

# Arguments
- `n_embd::Int` — Embedding dimension (number of rows)
- `vocab_size::Int` — Vocabulary size (number of columns)

# Examples
```julia
emb = Embedding{Float32}(768, 50257)
size(emb.weight)  # (768, 50257)
```
"""
function Embedding{P}(n_embd::Int, vocab_size::Int) where {P<:AbstractFloat}
    weight = randn(P, n_embd, vocab_size) .* P(0.02)
    Embedding{P, typeof(weight)}(weight)
end

"""
    Embedding(n_embd::Int, vocab_size::Int)

Convenience constructor defaulting to Float32 precision.
"""
Embedding(n_embd::Int, vocab_size::Int) = Embedding{Float32}(n_embd, vocab_size)

"""
    (e::Embedding)(idx::AbstractVector{<:Integer})

Perform embedding lookup for a vector of token indices.
Returns matrix of shape (n_embd, length(idx)).

Each index selects a column from the weight matrix, leveraging
column-major layout for contiguous memory access.
Uses Julia's 1-based indexing convention.
"""
function (e::Embedding)(idx::AbstractVector{<:Integer})
    # Vector input: select columns directly → (n_embd, T)
    return e.weight[:, idx]
end

"""
    (e::Embedding)(idx::AbstractMatrix{<:Integer})

Perform embedding lookup for a batch of token indices.
Input: idx of shape (seq_len, batch)
Output: (n_embd, seq_len, batch)

Each index selects a column from the weight matrix, leveraging
column-major layout for contiguous memory access.
Uses Julia's 1-based indexing convention.
"""
function (e::Embedding)(idx::AbstractMatrix{<:Integer})
    # Matrix input of shape (seq_len, batch) → (n_embd, seq_len, batch)
    n_embd = size(e.weight, 1)
    seq_len, batch_size = size(idx)

    # Allocate output tensor
    output = similar(e.weight, n_embd, seq_len, batch_size)
    for b in 1:batch_size
        for t in 1:seq_len
            @views output[:, t, b] .= e.weight[:, idx[t, b]]
        end
    end
    return output
end

"""
    Base.show(io::IO, e::Embedding{P}) where P

Pretty-print an Embedding layer showing its precision and dimensions.
"""
function Base.show(io::IO, e::Embedding{P}) where P
    n_embd, vocab_size = size(e.weight)
    print(io, "Embedding{$P}(n_embd=$n_embd, vocab_size=$vocab_size)")
end

# ─── Norm Layer ─────────────────────────────────────────────────────────────

"""
    Norm{P} <: AbstractLayer{P}

Root Mean Square Layer Normalization.
Normalizes input by its RMS value, then applies learnable scale (γ). No mean-centering or bias.

# Type Parameters
- `P`: Floating-point precision type

# Fields
- `weight::Vector{P}` — Learnable scale parameter γ, shape (n_embd,)
- `eps::P` — Small constant for numerical stability
"""
struct Norm{P} <: AbstractLayer{P}
    weight::Vector{P}  # γ (scale), shape (n_embd,)
    eps::P
end

"""
    Norm{P}(n_embd::Int; eps::Real=1e-5) where P

Construct an Norm layer with weight initialized to ones.

# Arguments
- `n_embd::Int` — Dimension to normalize over (embedding dimension)
- `eps::Real` — Small constant for numerical stability (default: 1e-5)

# Examples
```julia
ln = Norm{Float32}(768)
ln.weight  # ones(Float32, 768)
```
"""
function Norm{P}(n_embd::Int; eps::Real=1e-5) where {P<:AbstractFloat}
    weight = ones(P, n_embd)
    Norm{P}(weight, P(eps))
end

"""
    Norm(n_embd::Int; eps::Real=1e-5)

Convenience constructor defaulting to Float32 precision.
"""
Norm(n_embd::Int; kwargs...) = Norm{Float32}(n_embd; kwargs...)

"""
    (ln::Norm{P})(x::AbstractArray{P, 3}) where P

Apply RMS normalization over the first dimension (embedding dimension)
for 3D input of shape (n_embd, seq_len, batch).
"""
function (ln::Norm{P})(x::AbstractArray{P, 3}) where P
    # Compute RMS over embedding dimension (dim 1)
    rms = sqrt.(mean(x .^ 2; dims=1) .+ ln.eps)
    return ln.weight .* (x ./ rms)
end

"""
    (ln::Norm{P})(x::AbstractArray{P, 2}) where P

Apply RMS normalization over the first dimension (embedding dimension)
for 2D input of shape (n_embd, seq_len).
"""
function (ln::Norm{P})(x::AbstractArray{P, 2}) where P
    rms = sqrt.(mean(x .^ 2; dims=1) .+ ln.eps)
    return ln.weight .* (x ./ rms)
end

"""
    Base.show(io::IO, ln::Norm{P}) where P

Pretty-print an Norm layer showing its precision and dimension.
"""
function Base.show(io::IO, ln::Norm{P}) where P
    n_embd = length(ln.weight)
    print(io, "Norm{$P}(n_embd=$n_embd, eps=$(ln.eps))")
end

# ─── Linear (Dense) Layer ──────────────────────────────────────────────────────

"""
    Linear{P, HasBias} <: AbstractLayer{P}

Linear (dense/fully-connected) layer implementing y = W*x + b.
Weight matrix stored in column-major layout (out_features, in_features).
Parameterized on precision type `P` and whether bias is present (`HasBias`).

The `HasBias` type parameter (Bool) controls dispatch: when `true`, the forward
pass computes y = W*x + b; when `false`, it computes y = W*x.

# Fields
- `weight::Matrix{P}` — Weight matrix of shape (out_features, in_features)
- `bias::Vector{P}` — Bias vector of shape (out_features,); unused when HasBias=false
"""
struct Linear{P, HasBias} <: AbstractLayer{P}
    weight::Matrix{P}   # (out_features, in_features)
    bias::Vector{P}     # (out_features,)
end

"""
    Linear{P}(in_features::Int, out_features::Int; bias::Bool=true) where P

Construct a Linear layer with random weight initialization and optional bias.
Weights initialized from Normal(0, 0.02), bias initialized to zeros.

# Arguments
- `in_features::Int` — Input dimension (number of columns in weight matrix)
- `out_features::Int` — Output dimension (number of rows in weight matrix)
- `bias::Bool` — Whether to include a bias term (default: true)

# Examples
```julia
l = Linear{Float32}(768, 3072)          # with bias
l_nb = Linear{Float32}(768, 3072; bias=false)  # without bias
```
"""
function Linear{P}(in_features::Int, out_features::Int; bias::Bool=true) where {P<:AbstractFloat}
    weight = randn(P, out_features, in_features) .* P(0.02)
    b = zeros(P, out_features)
    Linear{P, bias}(weight, b)
end

"""
    Linear(in_features::Int, out_features::Int; bias::Bool=true)

Convenience constructor defaulting to Float32 precision.
"""
Linear(in_features::Int, out_features::Int; kwargs...) = Linear{Float32}(in_features, out_features; kwargs...)

"""
    has_bias(::Linear{P, HB}) where {P, HB}

Check whether a Linear layer has bias enabled.
"""
has_bias(::Linear{P, HB}) where {P, HB} = HB

# ─── Linear forward pass: with bias ───

"""
    (l::Linear{P, true})(x::AbstractVector{P}) where P

Forward pass for 1D input with bias: y = W*x + b.
Input: (in_features,) → Output: (out_features,)
"""
function (l::Linear{P, true})(x::AbstractVector{P}) where P
    return l.weight * x .+ l.bias
end

"""
    (l::Linear{P, true})(x::AbstractMatrix{P}) where P

Forward pass for 2D input with bias: y = W*x + b.
Input: (in_features, N) → Output: (out_features, N)
"""
function (l::Linear{P, true})(x::AbstractMatrix{P}) where P
    return l.weight * x .+ l.bias
end

"""
    (l::Linear{P, true})(x::AbstractArray{P, 3}) where P

Forward pass for 3D tensor input with bias.
Reshapes (in_features, seq_len, batch) to 2D, applies W*x + b, reshapes back.
Output: (out_features, seq_len, batch)
"""
function (l::Linear{P, true})(x::AbstractArray{P, 3}) where P
    in_dim, T, B = size(x)
    out_dim = size(l.weight, 1)
    x_2d = reshape(x, in_dim, T * B)
    y_2d = l.weight * x_2d .+ l.bias
    return reshape(y_2d, out_dim, T, B)
end

# ─── Linear forward pass: without bias ───

"""
    (l::Linear{P, false})(x::AbstractVector{P}) where P

Forward pass for 1D input without bias: y = W*x.
Input: (in_features,) → Output: (out_features,)
"""
function (l::Linear{P, false})(x::AbstractVector{P}) where P
    return l.weight * x
end

"""
    (l::Linear{P, false})(x::AbstractMatrix{P}) where P

Forward pass for 2D input without bias: y = W*x.
Input: (in_features, N) → Output: (out_features, N)
"""
function (l::Linear{P, false})(x::AbstractMatrix{P}) where P
    return l.weight * x
end

"""
    (l::Linear{P, false})(x::AbstractArray{P, 3}) where P

Forward pass for 3D tensor input without bias.
Reshapes (in_features, seq_len, batch) to 2D, applies W*x, reshapes back.
Output: (out_features, seq_len, batch)
"""
function (l::Linear{P, false})(x::AbstractArray{P, 3}) where P
    in_dim, T, B = size(x)
    out_dim = size(l.weight, 1)
    x_2d = reshape(x, in_dim, T * B)
    y_2d = l.weight * x_2d
    return reshape(y_2d, out_dim, T, B)
end

"""
    Base.show(io::IO, l::Linear{P, HB}) where {P, HB}

Pretty-print a Linear layer showing its precision, dimensions, and bias status.
"""
function Base.show(io::IO, l::Linear{P, HB}) where {P, HB}
    out_features, in_features = size(l.weight)
    print(io, "Linear{$P}($in_features → $out_features, bias=$HB)")
end


# ─── Cayley (Orthogonal) Linear Layer ───────────────────────────────────────

"""
    OrthogonalLinear{P, HB} <: AbstractLayer{P}

Orthogonal linear layer parameterized via the Cayley transform.
Only supports square mappings (`in_features == out_features`). The layer
stores a skew-symmetric parameter `A` and computes the forward product
efficiently using linear solves without forming the dense orthogonal
matrix explicitly: y = (I - A) * ((I + A) \\ x).

Fields:
- `A::Matrix{P}` — skew-symmetric parameter (n, n)
- `bias::Vector{P}` — bias (n,)
"""
struct OrthogonalLinear{P, HB} <: AbstractLayer{P}
    A::Matrix{P}
    bias::Vector{P}
end

"""
    OrthogonalLinear{P}(dim::Int; bias::Bool=true, init_scale::Real=1e-3)

Construct a square OrthogonalLinear of size `dim` with small random
skew-symmetric initialization. Defaults to Float32 precision.
"""
function OrthogonalLinear{P}(dim::Int; bias::Bool=true, init_scale::Real=1e-3) where {P<:AbstractFloat}
    R = randn(P, dim, dim) .* P(init_scale)
    S = 0.5 .* (R .- R')                # make skew-symmetric
    b = zeros(P, dim)
    return OrthogonalLinear{P, bias}(S, b)
end

"""Convenience constructor defaulting to Float32."""
OrthogonalLinear(dim::Int; kwargs...) = OrthogonalLinear{Float32}(dim; kwargs...)

has_bias(::OrthogonalLinear{P, HB}) where {P, HB} = HB

function _cayley_solve_A_plus_I!(A::AbstractMatrix{P}) where P
    # Return M = I + A (no mutation — use Matrix(I) addition)
    n = size(A, 1)
    return A + Matrix{P}(LinearAlgebra.I, n, n)
end

function _apply_cayley_mul(A::AbstractMatrix{P}, x::AbstractMatrix{P}) where P
    # Compute (I - A) * ((I + A) \ x) without forming (I ± A) explicitly
    n = size(A, 1)
    @assert size(x, 1) == n "OrthogonalLinear input dimension mismatch"
    M = _cayley_solve_A_plus_I!(A)
    y = M \ x
    return y .- A * y
end

function _apply_cayley_mul(A::AbstractMatrix{P}, x::AbstractVector{P}) where P
    n = size(A, 1)
    @assert length(x) == n "OrthogonalLinear input dimension mismatch"
    M = _cayley_solve_A_plus_I!(A)
    y = M \ x
    return y .- A * y
end

function (c::OrthogonalLinear{P, true})(x::AbstractVector{P}) where P
    out = _apply_cayley_mul(c.A, x)
    return out .+ c.bias
end

function (c::OrthogonalLinear{P, true})(x::AbstractMatrix{P}) where P
    out = _apply_cayley_mul(c.A, x)
    return out .+ c.bias
end

function (c::OrthogonalLinear{P, true})(x::AbstractArray{P, 3}) where P
    in_dim, T, B = size(x)
    n = size(c.A, 1)
    @assert in_dim == n "OrthogonalLinear expects square input dimension"
    x_2d = reshape(x, in_dim, T * B)
    y_2d = _apply_cayley_mul(c.A, x_2d) .+ c.bias
    return reshape(y_2d, n, T, B)
end

function (c::OrthogonalLinear{P, false})(x::AbstractVector{P}) where P
    return _apply_cayley_mul(c.A, x)
end

function (c::OrthogonalLinear{P, false})(x::AbstractMatrix{P}) where P
    return _apply_cayley_mul(c.A, x)
end

function (c::OrthogonalLinear{P, false})(x::AbstractArray{P, 3}) where P
    in_dim, T, B = size(x)
    n = size(c.A, 1)
    @assert in_dim == n "OrthogonalLinear expects square input dimension"
    x_2d = reshape(x, in_dim, T * B)
    y_2d = _apply_cayley_mul(c.A, x_2d)
    return reshape(y_2d, n, T, B)
end

function Base.show(io::IO, c::OrthogonalLinear{P, HB}) where {P, HB}
    n = size(c.A, 1)
    print(io, "OrthogonalLinear{$P}($n → $n, bias=$HB)")
end

"""
    orthogonal_weight(c::OrthogonalLinear)

Return the dense orthogonal matrix represented by the Cayley parameterization.
This constructs the full matrix (expensive), useful for diagnostics or exports.
"""
function orthogonal_weight(c::OrthogonalLinear{P, HB}) where {P, HB}
    A = c.A
    n = size(A, 1)
    M = _cayley_solve_A_plus_I!(A)
    # Compute each column by solving (I + A) y = e_j then applying (I - A)*y
    W = similar(A)
    for j in 1:n
        e = zeros(P, n); e[j] = one(P)
        y = M \ e
        W[:, j] = y .- A * y
    end
    return W
end

# ─── Device transfer (`Core.to_device`) implementations for Layers ─────────
# These add methods to the `Core.to_device` generic defined in Core/device.jl.
import ..Core

function Core.to_device(e::Embedding{P, M}, device::Symbol) where {P, M}
    wnew = to_device(e.weight, device)
    return Embedding{P, typeof(wnew)}(wnew)
end

function Core.to_device(ln::Norm{P}, device::Symbol) where {P}
    wnew = to_device(ln.weight, device)
    return Norm{P}(wnew, ln.eps)
end

function Core.to_device(l::Linear{P, HB}, device::Symbol) where {P, HB}
    wnew = to_device(l.weight, device)
    bnew = to_device(l.bias, device)
    return Linear{P, HB}(wnew, bnew)
end

function Core.to_device(c::OrthogonalLinear{P, HB}, device::Symbol) where {P, HB}
    Anew = to_device(c.A, device)
    bnew = to_device(c.bias, device)
    return OrthogonalLinear{P, HB}(Anew, bnew)
end
