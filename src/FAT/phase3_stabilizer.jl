# --- PHASE 3 ---
# Stabilization Pipeline: Π = P_p ∘ P_k
#
# Resolution of the norm-equivalence flaw identified in the audit:
#
#   The original paper proposed Π = P_F ∘ P_p ∘ P_k, composing a Fourier
#   projection P_F (orthogonal in L²([0,L])) with Chebyshev projections
#   (orthogonal in L²([-1,1], w)).  These live in *different* Hilbert spaces
#   with different inner products and measures.  The composition is therefore
#   *not* covered by Lemma 4.2 (which applies within a single Hilbert space),
#   and the claimed L_Π ≤ 1 bound is broken by the norm-equivalence constant
#   between the two spaces.
#
#   Fix: drop P_F entirely.  Work exclusively in H_w = L²([-1,1], w) with
#   the Chebyshev inner product.  The corrected pipeline is:
#
#       Π = P_p ∘ P_k,   0 ≤ p ≤ k ≤ K
#
#   where both projections are orthogonal in H_w.  The proof of L_Π ≤ 1
#   now follows immediately, with no norm-equivalence gap.
#
# All operations reduce to simple coefficient truncation because the
# Chebyshev polynomials are *already* orthonormal — projection onto V_n
# is just zeroing coefficients n+1 through K.  This is O(1) in memory
# (no data copy for contiguous slices) and O(K) in arithmetic.

using LinearAlgebra

# ─── ChebyshevEconomizer ─────────────────────────────────────────────────────

"""
    ChebyshevEconomizer{P_order} <: AbstractLayer{Float32}

Orthogonal truncation operator

    P_p : V_K → V_p,   f ↦ Σₙ₌₀^p θₙ Tₙ

in the Hilbert space H_w = L²([-1,1], w).

## Non-expansivity proof (inline justification)

**Claim:** ‖P_p f - P_p g‖_{H_w} ≤ ‖f - g‖_{H_w} for all f, g ∈ H_w.

**Proof:**

(1) V_p = span{T₀, …, T_p} is a closed (finite-dimensional) linear subspace
    of H_w, because it is the span of a finite set of continuous functions.

(2) Because {Tₙ} is an orthonormal basis of H_w, the formula
        P_p[f] = Σₙ₌₀^p ⟨f, Tₙ⟩_w Tₙ
    satisfies:
      - Idempotency: P_p(P_p f) = P_p f   (truncating twice changes nothing)
      - Self-adjointness: ⟨P_p f, g⟩_w = ⟨f, P_p g⟩_w
        (because ⟨Σ θₙ^f Tₙ, g⟩_w = Σ θₙ^f ⟨Tₙ,g⟩_w = ⟨f, Σ θₙ^g Tₙ⟩_w)
    These two properties characterise the orthogonal projection onto V_p.

(3) For any h ∈ H_w, the Pythagorean theorem in H_w gives:
        ‖h‖_{H_w}² = ‖P_p h‖_{H_w}² + ‖(I - P_p) h‖_{H_w}²
                   ≥ ‖P_p h‖_{H_w}²
    Substituting h = f - g:
        ‖P_p(f - g)‖_{H_w} ≤ ‖f - g‖_{H_w}   □

(4) In the coefficient representation: ‖P_p Θ‖_2² = Σₙ₌₀^p θₙ² ≤ Σₙ₌₀^K θₙ² = ‖Θ‖_2²
    confirming (3) algebraically.

## Implementation
In the Chebyshev basis, P_p is just zeroing θ_{p+1}, …, θ_K — a coefficient
slice.  No matrix multiply, no data copy (for contiguous arrays).

## Type Parameter
- `P_order`: output degree (Int, encoded at type level for compile-time checks)
"""
struct ChebyshevEconomizer{P_order} end

"""
    ChebyshevEconomizer(p::Int) -> ChebyshevEconomizer{p}

Construct an economizer that projects onto V_p (polynomials of degree ≤ p).
"""
ChebyshevEconomizer(p::Int) = ChebyshevEconomizer{p}()

"""
    (proj::ChebyshevEconomizer{P_order})(coeffs::AbstractVector{T}) -> Vector{T}

Apply P_p to a coefficient vector Θ ∈ ℝ^{K+1}, returning Θ' ∈ ℝ^{P_order+1}.

Input : `coeffs` — length K+1, K ≥ P_order
Output: length P_order+1 (truncated, zeroing θ_{P_order+1}…θ_K)

The return type is a new Vector rather than a view so that downstream
AD (Zygote) can differentiate through the selection without mutation.
"""
function (proj::ChebyshevEconomizer{P_order})(
        coeffs::AbstractVector{T}) where {T<:AbstractFloat, P_order}
    K_in = length(coeffs) - 1
    @assert K_in >= P_order "Input degree $K_in < target degree $P_order"
    # Slice the first P_order+1 entries — Zygote differentiates array indexing
    return coeffs[1:P_order+1]
end

"""
    (proj::ChebyshevEconomizer{P_order})(state::SpectralState{T,K}) -> SpectralState{T,P_order}

Apply P_p to a SpectralState, returning a lower-degree SpectralState.
"""
function (proj::ChebyshevEconomizer{P_order})(
        state::SpectralState{T,K}) where {T<:AbstractFloat, P_order, K}
    @assert K >= P_order "SpectralState degree $K < target degree $P_order"
    return SpectralState{T, P_order}(state.coeffs[1:P_order+1])
end

"""
    (proj::ChebyshevEconomizer{P_order})(Θ::AbstractArray{T,3}) -> Array{T,3}

Apply P_p to a coefficient tensor of shape (K+1, seq_len, batch).
Output shape: (P_order+1, seq_len, batch).

The leading-dimension slice is O(1) for column-major arrays.
"""
function (proj::ChebyshevEconomizer{P_order})(
        Θ::AbstractArray{T,3}) where {T<:AbstractFloat, P_order}
    K_in = size(Θ, 1) - 1
    @assert K_in >= P_order "Tensor K_in=$K_in < target P_order=$P_order"
    return Θ[1:P_order+1, :, :]
end

# ─── Composite Spectral Projection Operator Π ─────────────────────────────────

"""
    SpectralStabilizer{K_anchor, P_econ} <: AbstractLayer{Float32}

The corrected composite projection operator

    Π = P_p ∘ P_k : H_w → V_p ⊂ V_k ⊂ H_w

consisting of:
  1. **Local Taylor Anchor** P_k : truncate to degree k  (damp high-frequency noise)
  2. **Global Chebyshev Economization** P_p : truncate further to degree p < k
     (guard against Runge-like boundary growth)

Both projections are orthogonal in H_w = L²([-1,1], w); their composition
is therefore also non-expansive in H_w (see detailed proof below).

## Non-expansivity of Π (main theorem)

**Theorem:** L_Π ≤ 1.

**Proof:**

Let f, g ∈ H_w.  By Proposition for P_k (ChebyshevEconomizer{k}):
    ‖P_k f - P_k g‖_{H_w} ≤ ‖f - g‖_{H_w}           … (i)

By Proposition for P_p (ChebyshevEconomizer{p}), applied to P_k f and P_k g,
which live in H_w (same space — no norm-equivalence issue):
    ‖P_p(P_k f) - P_p(P_k g)‖_{H_w} ≤ ‖P_k f - P_k g‖_{H_w}  … (ii)

Note: Because p ≤ k, P_p ∘ P_k = P_p (the smaller projection dominates).
Chaining (i) and (ii):
    ‖Π f - Π g‖_{H_w} = ‖P_p f - P_p g‖_{H_w} ≤ ‖f - g‖_{H_w}    □

The key point is that both P_k and P_p are orthogonal projections **in the
same Hilbert space H_w**.  This is what was absent in the original paper's
proof, which mixed H_w (Chebyshev) with L²([0,L]) (Fourier).

## Type Parameters
- `K_anchor`: degree of the local anchor projection P_k
- `P_econ`  : degree of the economization projection P_p  (P_econ ≤ K_anchor)
"""
struct SpectralStabilizer{K_anchor, P_econ}
    Pk::ChebyshevEconomizer{K_anchor}   # local Taylor anchor
    Pp::ChebyshevEconomizer{P_econ}     # global Chebyshev economization

    function SpectralStabilizer{K_anchor, P_econ}() where {K_anchor, P_econ}
        @assert P_econ <= K_anchor "P_econ ($P_econ) must be ≤ K_anchor ($K_anchor)"
        new{K_anchor, P_econ}(
            ChebyshevEconomizer{K_anchor}(),
            ChebyshevEconomizer{P_econ}()
        )
    end
end

"""
    SpectralStabilizer(k::Int, p::Int)

Construct Π = P_p ∘ P_k with runtime-specified degrees.
"""
SpectralStabilizer(k::Int, p::Int) = SpectralStabilizer{k, p}()

# ─── Forward calls ────────────────────────────────────────────────────────────

"""
    (Π::SpectralStabilizer{K_anchor, P_econ})(coeffs::AbstractVector) -> Vector

Apply Π to a coefficient vector.
Because P_econ ≤ K_anchor, P_p ∘ P_k = P_p, so we apply P_p directly.

Input : length K+1 (K ≥ K_anchor)
Output: length P_econ+1
"""
function (Π::SpectralStabilizer{K_anchor, P_econ})(
        coeffs::AbstractVector{T}) where {T<:AbstractFloat, K_anchor, P_econ}
    # Step 1: anchor to V_{K_anchor}
    anchored = Π.Pk(coeffs)     # length K_anchor+1
    # Step 2: economize to V_{P_econ}
    return Π.Pp(anchored)       # length P_econ+1
end

"""
    (Π::SpectralStabilizer{K_anchor, P_econ})(state::SpectralState) -> SpectralState

Apply Π to a SpectralState.
"""
function (Π::SpectralStabilizer{K_anchor, P_econ})(
        state::SpectralState{T,K}) where {T<:AbstractFloat, K_anchor, P_econ, K}
    anchored = Π.Pk(state)
    return Π.Pp(anchored)
end

"""
    (Π::SpectralStabilizer{K_anchor, P_econ})(Θ::AbstractArray{T,3}) -> Array{T,3}

Apply Π to a full coefficient tensor (K+1, seq_len, batch).
Output shape: (P_econ+1, seq_len, batch).

This is the hot path called once per autoregressive step.  Implementation
is a single leading-dimension slice — O(1) for column-major contiguous arrays.
"""
function (Π::SpectralStabilizer{K_anchor, P_econ})(
        Θ::AbstractArray{T,3}) where {T<:AbstractFloat, K_anchor, P_econ}
    anchored = Π.Pk(Θ)    # (K_anchor+1, seq_len, batch)
    return Π.Pp(anchored) # (P_econ+1,   seq_len, batch)
end

# ─── Display ─────────────────────────────────────────────────────────────────

function Base.show(io::IO, ::ChebyshevEconomizer{P}) where P
    print(io, "ChebyshevEconomizer{P=$P}(V_K → V_$P, L_op ≤ 1)")
end

function Base.show(io::IO, Π::SpectralStabilizer{K_anchor, P_econ}) where {K_anchor, P_econ}
    print(io, "SpectralStabilizer{k=$K_anchor, p=$P_econ}(Π = P_$P_econ ∘ P_$K_anchor, L_Π ≤ 1)")
end

# ─── Lipschitz constant verification (diagnostic) ─────────────────────────────

"""
    verify_nonexpansive(Π::SpectralStabilizer; n_trials=1000, K_test=nothing, T=Float64) -> Bool

Monte Carlo verification that Π is non-expansive.
Samples random pairs (f, g) ∈ H_w and checks ‖Πf - Πg‖ ≤ ‖f - g‖ + ε.

Returns `true` if all trials pass (Lipschitz constant ≤ 1 + ε).
Intended for unit testing, not production use.
"""
function verify_nonexpansive(Π::SpectralStabilizer{K_anchor, P_econ};
                             n_trials::Int = 1000,
                             K_test::Union{Nothing,Int} = nothing,
                             T::Type = Float64,
                             ε::Float64 = 1e-10) where {K_anchor, P_econ}
    K = K_test === nothing ? K_anchor : K_test
    @assert K >= K_anchor "K_test must be ≥ K_anchor=$K_anchor"
    violations = 0
    for _ in 1:n_trials
        f = randn(T, K+1)
        g = randn(T, K+1)
        lhs = norm(Π(f) .- Π(g))
        rhs = norm(f .- g)
        if lhs > rhs + ε
            violations += 1
        end
    end
    return violations == 0
end
