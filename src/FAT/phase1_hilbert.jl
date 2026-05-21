# --- PHASE 1 ---
# Core Hilbert Space Abstractions for FAT
#
# Implements the mathematical foundation of L²([-1,1], w) where
# w(x) = 1/(π√(1-x²)) is the Chebyshev probability measure.
#
# Key types:
#   AbstractFunctionalBasis         — dispatch root for all basis families
#   ChebyshevBasis{K}               — Chebyshev polynomials of the first kind, degree ≤ K
#   SpectralState{T,K}              — an element of V_K = span{T₀,...,T_K} ⊂ H_w
#
# All arrays are column-major (features × sequence × batch) consistent with
# the rest of the Architect codebase.

using LinearAlgebra

# ─── Abstract Basis Root ──────────────────────────────────────────────────────

"""
    AbstractFunctionalBasis

Abstract supertype for all spectral bases used by the FAT framework.
Subtypes must provide:
  - `basis_dim(::B)::Int`         — number of basis functions
  - `evaluate(::B, coeffs, x)`   — evaluate Σ cₙ φₙ(x) at scalar x
  - `inner_product_matrix(::B)`  — Gram matrix G where Gᵢⱼ = ⟨φᵢ, φⱼ⟩_μ

Multiple dispatch on basis type (not a runtime symbol) lets the compiler
specialize and inline all arithmetic at compile time.
"""
abstract type AbstractFunctionalBasis end

# ─── Chebyshev Basis ──────────────────────────────────────────────────────────

"""
    ChebyshevBasis{K}

Chebyshev polynomials of the first kind {T₀, T₁, …, T_K} on [-1,1],
orthonormal with respect to the Chebyshev probability measure

    w(x) = 1 / (π √(1 - x²)),     x ∈ (-1, 1).

The orthonormality relation is:
    ⟨Tₙ, Tₘ⟩_w = δₙₘ     (with the convention that T₀ is already normalized).

Type parameter `K` (::Int, >= 0) encodes the truncation order at the type
level, enabling compile-time size assertions and stack-allocated coefficient
vectors.  The coefficient space has dimension `K+1`.
"""
struct ChebyshevBasis{K} <: AbstractFunctionalBasis end

"""
    basis_dim(::ChebyshevBasis{K}) -> Int

Return the number of basis functions (= truncation order + 1).
"""
@inline basis_dim(::ChebyshevBasis{K}) where {K} = K + 1

# ─── SpectralState ────────────────────────────────────────────────────────────

"""
    SpectralState{T<:AbstractFloat, K}

An element of V_K = span{T₀, T₁, …, T_K} ⊂ H = L²([-1,1], w).

Stores the coefficient vector Θ = (θ₀, θ₁, …, θ_K) ∈ ℝ^{K+1} such that

    f_Θ(x) = Σₙ₌₀^K θₙ Tₙ(x).

# Layout
`coeffs` is a `Vector{T}` of length `K+1`.  This is the **column** of the
(K+1, seq_len, batch) tensor that the transformer head produces — each
column is one SpectralState.

# Type Parameters
- `T`: floating-point precision (Float32, Float64, …)
- `K`: truncation order (Int, encoded at the type level)
"""
struct SpectralState{T<:AbstractFloat, K}
    coeffs::Vector{T}   # length K+1; column-major convention: coefficients are contiguous

    function SpectralState{T, K}(coeffs::Vector{T}) where {T<:AbstractFloat, K}
        @assert length(coeffs) == K + 1 "Expected $(K+1) coefficients, got $(length(coeffs))"
        new{T, K}(coeffs)
    end
end

"""
    SpectralState{T,K}(coeffs::AbstractVector)

Outer constructor that converts any AbstractVector to the required Vector{T}.
"""
SpectralState{T,K}(v::AbstractVector) where {T,K} = SpectralState{T,K}(Vector{T}(v))

"""
    SpectralState{T,K}()

Zero state: f_Θ ≡ 0 (all coefficients are zero).
"""
SpectralState{T,K}() where {T,K} = SpectralState{T,K}(zeros(T, K+1))

Base.length(s::SpectralState{T,K}) where {T,K} = K + 1
Base.eltype(::SpectralState{T,K}) where {T,K}  = T

function Base.show(io::IO, s::SpectralState{T,K}) where {T,K}
    print(io, "SpectralState{$T,$K}(θ = $(round.(s.coeffs; digits=4)))")
end

# ─── Clenshaw's Algorithm for Chebyshev Evaluation ────────────────────────────

"""
    evaluate(::ChebyshevBasis{K}, coeffs::AbstractVector{T}, x::T) -> T

Evaluate the truncated Chebyshev expansion

    f(x) = Σₙ₌₀^K θₙ Tₙ(x)

at a scalar point `x ∈ [-1, 1]` using **Clenshaw's recurrence**, which is
numerically superior to explicit polynomial evaluation.

## Clenshaw recurrence (backward)

Given θ₀, …, θ_K and x, define:
    b_{K+1} = 0
    b_K     = 0
    bₙ      = θₙ + 2x bₙ₊₁ - bₙ₊₂,  for n = K-1, …, 1
    f(x)    = θ₀ + x b₁ - b₂

This achieves O(K) multiplications with no explicit basis evaluation,
and is Zygote-differentiable since it uses only +, *, -.

## Arguments
- `coeffs`: length-`(K+1)` vector [θ₀, θ₁, …, θ_K]
- `x`: evaluation point in [-1, 1]
"""
function evaluate(::ChebyshevBasis{K}, coeffs::AbstractVector{T}, x::T) where {T<:AbstractFloat, K}
    @assert length(coeffs) == K + 1 "coeffs length $(length(coeffs)) ≠ K+1 = $(K+1)"

    # Edge cases
    K == 0 && return coeffs[1]          # f(x) = θ₀ T₀(x) = θ₀

    # Clenshaw backward recurrence
    # Index convention: coeffs[n+1] = θₙ (1-based Julia indexing)
    b_next = zero(T)    # b_{K+1}
    b_curr = zero(T)    # b_K (will be set to θ_K below)

    # Start: n = K → b_K = θ_K + 2x·0 - 0 = θ_K
    b_curr = coeffs[K+1]

    # Recur from n = K-1 down to n = 1
    for n in K-1:-1:1
        b_prev = coeffs[n+1] + T(2) * x * b_curr - b_next
        b_next = b_curr
        b_curr = b_prev
    end

    # Final value: f(x) = θ₀ + x·b₁ - b₂
    return coeffs[1] + x * b_curr - b_next
end

"""
    evaluate(::ChebyshevBasis{K}, state::SpectralState{T,K}, x::T) -> T

Dispatch overload accepting a `SpectralState` directly.
"""
evaluate(basis::ChebyshevBasis{K}, state::SpectralState{T,K}, x::T) where {T,K} =
    evaluate(basis, state.coeffs, x)

"""
    evaluate(::ChebyshevBasis{K}, coeffs, xs::AbstractVector{T}) -> Vector{T}

Evaluate the expansion at multiple points xs ∈ [-1,1].
Returns a vector of length `length(xs)`.
"""
function evaluate(basis::ChebyshevBasis{K}, coeffs::AbstractVector{T},
                  xs::AbstractVector{T}) where {T<:AbstractFloat, K}
    [evaluate(basis, coeffs, x) for x in xs]
end

# ─── Gauss-Chebyshev Quadrature ───────────────────────────────────────────────

"""
    gauss_chebyshev_nodes_weights(N::Int, ::Type{T}) -> (nodes::Vector{T}, weights::Vector{T})

Return the `N`-point Gauss-Chebyshev quadrature nodes and weights for the
Chebyshev probability measure w(x) = 1/(π√(1-x²)) on (-1,1).

The `N`-point rule integrates polynomials of degree ≤ 2N-1 exactly.

## Nodes and weights
    xₖ = cos((2k-1)π / (2N)),    k = 1, …, N
    wₖ = 1/N                     (equal weights under the probability measure)

This is the Type-I (midpoint) Gauss-Chebyshev rule.  All weights are equal
under the Chebyshev probability measure, making quadrature trivially O(N).
"""
function gauss_chebyshev_nodes_weights(N::Int, ::Type{T}) where {T<:AbstractFloat}
    nodes   = T[cos(T(π) * T(2k - 1) / T(2N)) for k in 1:N]
    weights = fill(one(T) / T(N), N)   # uniform under the probability measure
    return nodes, weights
end

# ─── Inner Product ⟨f, g⟩_w ──────────────────────────────────────────────────

"""
    inner_product(::ChebyshevBasis{K}, f_coeffs, g_coeffs, N::Int) -> T

Compute the Hilbert space inner product

    ⟨f, g⟩_w = ∫₋₁¹ f(x) g(x) w(x) dx

via `N`-point Gauss-Chebyshev quadrature.

Since the Chebyshev polynomials are orthonormal under w, this equals
    ⟨f, g⟩_w = Σₙ₌₀^K f̂ₙ ĝₙ
when both expansions are truncated to the same degree.  The quadrature
form is provided for cross-verification and for cases where one argument
is given as a callable rather than a coefficient vector.

## Arguments
- `f_coeffs`, `g_coeffs`: coefficient vectors of length K+1
- `N`: quadrature order; default is `4*(K+1)` for exact integration
"""
function inner_product(basis::ChebyshevBasis{K},
                       f_coeffs::AbstractVector{T},
                       g_coeffs::AbstractVector{T},
                       N::Int = 4*(K+1)) where {T<:AbstractFloat, K}
    nodes, weights = gauss_chebyshev_nodes_weights(N, T)
    acc = zero(T)
    for (xₖ, wₖ) in zip(nodes, weights)
        fval = evaluate(basis, f_coeffs, xₖ)
        gval = evaluate(basis, g_coeffs, xₖ)
        acc += wₖ * fval * gval
    end
    return acc
end

"""
    inner_product_exact(f_coeffs, g_coeffs) -> T

Closed-form inner product exploiting orthonormality of {Tₙ} under w:

    ⟨f, g⟩_w = Σₙ₌₀^K θₙ^f θₙ^g = dot(f_coeffs, g_coeffs)

This is O(K) and exact (no quadrature error).  Used internally where
both arguments are already in the Chebyshev basis.
"""
inner_product_exact(f_coeffs::AbstractVector{T}, g_coeffs::AbstractVector{T}) where T =
    dot(f_coeffs, g_coeffs)

# ─── Chebyshev Projection ⟨f, Tₙ⟩_w ─────────────────────────────────────────

"""
    project_coefficient(::ChebyshevBasis{K}, f_fn, n::Int, N::Int) -> T

Compute the n-th Chebyshev coefficient of a function `f_fn : T -> T`:

    θₙ = ⟨f, Tₙ⟩_w = ∫₋₁¹ f(x) Tₙ(x) w(x) dx

via N-point Gauss-Chebyshev quadrature.  Recovers the exact DCT-I
formula when N = K+1.

## Arguments
- `f_fn`: callable (takes one Float argument, returns Float)
- `n`: basis index (0-based)
- `N`: quadrature order (default: `4*(K+1)`)
"""
function project_coefficient(::ChebyshevBasis{K}, f_fn, n::Int, ::Type{T},
                              N::Int = 4*(K+1)) where {T<:AbstractFloat, K}
    @assert 0 <= n <= K "Basis index n=$n out of range [0, $K]"
    nodes, weights = gauss_chebyshev_nodes_weights(N, T)
    acc = zero(T)
    for (xₖ, wₖ) in zip(nodes, weights)
        Tnk = cos(T(n) * acos(clamp(xₖ, -one(T), one(T))))   # Tₙ(xₖ) = cos(n·arccos(xₖ))
        acc += wₖ * f_fn(xₖ) * Tnk
    end
    return acc
end

"""
    project(::ChebyshevBasis{K}, f_fn, ::Type{T}; N=4*(K+1)) -> SpectralState{T,K}

Project a continuous function `f_fn` onto V_K by computing all K+1 coefficients:

    θₙ = ⟨f, Tₙ⟩_w,   n = 0, 1, …, K

Returns a `SpectralState{T,K}` whose `.coeffs` vector is the Chebyshev expansion.
"""
function project(basis::ChebyshevBasis{K}, f_fn, ::Type{T};
                 N::Int = 4*(K+1)) where {T<:AbstractFloat, K}
    coeffs = T[project_coefficient(basis, f_fn, n, T, N) for n in 0:K]
    return SpectralState{T,K}(coeffs)
end

# ─── Chebyshev Derivative Matrix ─────────────────────────────────────────────

"""
    chebyshev_derivative_matrix(::ChebyshevBasis{K}, ::Type{T}) -> Matrix{T}

Return the (K × (K+1)) matrix D such that, for a coefficient vector Θ ∈ ℝ^{K+1},

    D Θ = Θ'  ∈ ℝ^K

gives the Chebyshev coefficients of f'(x) = d/dx f_Θ(x).

The recurrence relation for Chebyshev derivatives is:
    θ'_{K-1} = 2K θ_K
    θ'_n     = 2(n+1) θ_{n+1} + θ'_{n+2},    n = K-2, K-3, …, 0
    θ'_0    /= 2

This gives a linear map on coefficients, enabling exact symbolic
differentiation with zero numerical error and O(K) cost.

The returned matrix is dense but K × (K+1) — small for typical K ≤ 32.
"""
function chebyshev_derivative_matrix(::ChebyshevBasis{K}, ::Type{T}) where {T<:AbstractFloat, K}
    D = zeros(T, K, K+1)   # output has K coefficients (degree drops by 1)
    # Recurrence: for a coefficient vector θ of length K+1,
    # the derivative coefficients are computed by the "Chebyshev differentiation matrix"
    # using the standard result: θ'_n = 2 Σ_{j>n, j+n odd} j·θ_j
    for n in 0:K-1          # output index (0-based)
        for j in n+1:K      # source index (0-based)
            if (j + n) % 2 == 1   # odd parity condition
                D[n+1, j+1] = T(2) * T(j)
            end
        end
    end
    # Halve the n=0 row (normalization convention for T₀ in the probability measure)
    D[1, :] ./= T(2)
    return D
end

# ─── H_w Norm of a SpectralState ─────────────────────────────────────────────

"""
    hilbert_norm(state::SpectralState{T,K}) -> T

Compute the Hilbert space norm ‖f_Θ‖_{H_w} = √⟨f_Θ, f_Θ⟩_w.

By orthonormality of {Tₙ} this reduces to the Euclidean norm of the
coefficient vector:
    ‖f_Θ‖_{H_w} = ‖Θ‖_2 = √(Σₙ θₙ²)
"""
hilbert_norm(state::SpectralState{T,K}) where {T,K} = norm(state.coeffs)

"""
    hilbert_dist(s1::SpectralState{T,K}, s2::SpectralState{T,K}) -> T

Hilbert-space distance ‖f_Θ₁ - f_Θ₂‖_{H_w} = ‖Θ₁ - Θ₂‖_2.
"""
hilbert_dist(s1::SpectralState{T,K}, s2::SpectralState{T,K}) where {T,K} =
    norm(s1.coeffs .- s2.coeffs)
