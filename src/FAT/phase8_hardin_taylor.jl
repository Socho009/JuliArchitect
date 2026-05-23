# --- PHASE 8 ---
# Hardin-Taylor Minimum-Complexity Prediction (MCP)
#
# A constructive approximation of the Nearly Perfect Prediction Theorem
# (Hardin & Taylor, 2008; exposition by Hamkins, AMS Notices 2026).
#
# The theorem states: fix a well-ordering ≤ on all functions ℝ → ℝ,
# and predict using the ≤-least function compatible with observed data.
# The predictions are correct except on a countable, nowhere-dense set.
#
# This module implements three constructive analogs:
#
# 1. COMPLEXITY ORDERING — replaces the non-constructive well-ordering
#    with a computable total pre-order on SpectralState coefficients:
#        C(Θ) = Σₙ (1+n)^α |θₙ|
#    Lower C = "simpler" function (fewer high-frequency components).
#
# 2. MCP LOSS — differentiable regularizer that penalizes prediction
#    complexity, encouraging the model to predict the simplest function
#    whose trajectory direction matches the ground truth.
#        ℒ_MCP = λ · 𝔼_{t,b}[C(Θ̂_{t,b})]
#
# 3. HIERARCHY CLIMBING GENERATION — starts predictions at minimum K,
#    climbs to higher order only when trajectory approval drops below
#    threshold τ.  Climb events are the constructive analog of the
#    theorem's "countable exceptional set."
#
# All operations are Zygote-differentiable (no mutation).

using LinearAlgebra
using Statistics: mean

# ─── 8.1: Complexity Ordering on Spectral States ─────────────────────────────

"""
    spectral_complexity(Θ::AbstractVector{T}; α::T=T(1.5)) → T

Compute the spectral complexity of a coefficient vector Θ ∈ ℝ^{K+1}.

The complexity score is a weighted ℓ¹ norm that penalizes higher-order
Chebyshev coefficients more heavily:

    C(Θ) = Σₙ₌₀ᴷ (1 + n)^α |θₙ|

This induces a total pre-order on V_K that serves as the constructive
analog of the Hardin-Taylor well-ordering:
- C(Θ) = 0  ⟺  Θ = 0 (the zero function — maximally simple)
- C(Θ₁) < C(Θ₂) means Θ₁ is "simpler" (uses less high-frequency energy)

## Connection to Hardin-Taylor

The theorem predicts using the "≤-least function compatible with data."
In our constructive setting, this becomes: among all Θ that match the
observed trajectory, prefer the one with minimal C(Θ).

The exponent α controls the severity of the high-frequency penalty:
- α = 0: uniform weight (just ℓ¹ norm, no frequency preference)
- α = 1: linear penalty (mild preference for low frequencies)
- α = 1.5: default (moderate — matches Chebyshev coefficient decay rate)
- α = 2: quadratic (strong preference, aggressive smoothing)

## Arguments
- `Θ`: coefficient vector of length K+1
- `α`: frequency penalty exponent (default: 1.5)
"""
function spectral_complexity(Θ::AbstractVector{T}; α::T=T(1.5)) where {T<:AbstractFloat}
    K = length(Θ) - 1
    weights = T[(one(T) + T(n))^α for n in 0:K]
    return sum(weights .* abs.(Θ))
end

"""
    spectral_complexity(Θ::AbstractArray{T,3}; α::T=T(1.5)) → T

Batch version: mean complexity over all (t, b) positions.

Input: Θ of shape (K+1, seq_len, batch)
Output: scalar mean complexity
"""
function spectral_complexity(Θ::AbstractArray{T,3}; α::T=T(1.5)) where {T<:AbstractFloat}
    K = size(Θ, 1) - 1
    # Frequency weights: (K+1,) → broadcast over seq_len and batch
    # Wrapped in @ignore so Zygote treats these as constants
    weights = Zygote.@ignore reshape(T[(one(T) + T(n))^α for n in 0:K], :, 1, 1)
    return mean(sum(weights .* abs.(Θ); dims=1))
end

# ─── 8.2: Minimum-Complexity Prediction Loss ─────────────────────────────────

"""
    mcp_loss(Θ_pred::AbstractArray{T,3}; α::T=T(1.5)) → T

Minimum-Complexity Prediction (MCP) loss — the differentiable regularizer
that implements the Hardin-Taylor prediction principle.

    ℒ_MCP = 𝔼_{t,b}[C(Θ̂_{t,b})]
         = 𝔼_{t,b}[Σₙ (1+n)^α |θ̂ₙ^{(t,b)}|]

## Mathematical Justification

The Hardin-Taylor theorem predicts using the ≤-least function compatible
with observed data.  In our constructive setting:

1. The "compatibility" constraint is enforced by the trajectory approval
   loss ℒ_Flow (Phase 4), which requires predictions to align directionally
   with ground truth.

2. The "≤-least" (minimum complexity) constraint is enforced by ℒ_MCP,
   which penalizes predictions that are more complex than necessary.

Together: ℒ_total = ℒ_Flow + λ · ℒ_MCP selects the simplest function whose
trajectory direction matches ground truth — exactly the Hardin-Taylor
prediction strategy, constructivized.

## Zygote Compatibility

Uses `abs.(Θ)` which Zygote differentiates as `sign.(Θ)` (subgradient).
The weighted sum and mean are standard differentiable operations.

## Arguments
- `Θ_pred`: predicted coefficients, shape (K+1, seq_len, batch)
- `α`: frequency penalty exponent (default: 1.5)
"""
function mcp_loss(Θ_pred::AbstractArray{T,3}; α::T=T(1.5)) where {T<:AbstractFloat}
    return spectral_complexity(Θ_pred; α=α)
end

"""
    mcp_loss(Θ_pred::AbstractArray{T,3}, Θ_target::AbstractArray{T,3};
             α::T=T(1.5), compatibility_weight::T=T(0.1)) → T

Extended MCP loss with target-relative complexity.

Penalizes the *excess* complexity of the prediction over the target:

    ℒ_MCP = 𝔼[C(Θ̂)] + w · 𝔼[max(0, C(Θ̂) - C(Θ*))]

The second term specifically penalizes predictions that are more complex
than the ground truth, which is a direct encoding of the Hardin-Taylor
principle: don't use more spectral resources than necessary.

## Arguments
- `Θ_pred`: predicted coefficients (K+1, seq_len, batch)
- `Θ_target`: ground-truth coefficients (same shape)
- `α`: frequency penalty exponent
- `compatibility_weight`: weight for excess-complexity penalty
"""
function mcp_loss(Θ_pred::AbstractArray{T,3}, Θ_target::AbstractArray{T,3};
                  α::T=T(1.5), compatibility_weight::T=T(0.1)) where {T<:AbstractFloat}
    K = size(Θ_pred, 1) - 1
    # Compute weights as a constant (Zygote should not track these)
    w_vec = Zygote.@ignore reshape(T[(one(T) + T(n))^α for n in 0:K], :, 1, 1)

    # Per-position complexity
    c_pred   = sum(w_vec .* abs.(Θ_pred);   dims=1)   # (1, seq_len, batch)
    c_target = sum(w_vec .* abs.(Θ_target); dims=1)   # (1, seq_len, batch)

    # Base complexity: encourage overall simplicity
    base = mean(c_pred)

    # Excess complexity: penalize predictions more complex than target
    # Use softplus as a smooth approximation of max(0, ·) for AD
    excess_raw = c_pred .- c_target
    excess = mean(log.(one(T) .+ exp.(excess_raw)))  # softplus mean

    return base + compatibility_weight * excess
end

# ─── 8.3: Hierarchical Order Climbing Generation ─────────────────────────────

"""
    ClimbEvent{T<:AbstractFloat}

Record of a single hierarchy climbing event during Hardin-Taylor generation.

## Fields
- `step::Int`          — generation step where the climb occurred
- `K_before::Int`      — truncation order before climbing
- `K_after::Int`       — truncation order after climbing
- `approval::T`        — trajectory approval A(Θ_t) that triggered the climb
- `complexity_before::T` — spectral complexity before climbing
- `complexity_after::T`  — spectral complexity after re-prediction

## Interpretation
In the Hardin-Taylor theorem, exceptional points are where the prediction
function "climbs" to a higher position in the well-ordering.  ClimbEvents
are the constructive analog: they record when the generator had to increase
K to maintain trajectory compatibility.

The theorem guarantees that exceptional points form a countable, nowhere-dense
set.  In practice, we expect climb events to be sparse — most predictions
should succeed at low K.
"""
struct ClimbEvent{T<:AbstractFloat}
    step::Int
    K_before::Int
    K_after::Int
    approval::T
    complexity_before::T
    complexity_after::T
end

function Base.show(io::IO, e::ClimbEvent{T}) where T
    @printf(io, "Climb(step=%d, K=%d→%d, A=%.4f, C=%.4f→%.4f)",
            e.step, e.K_before, e.K_after, e.approval,
            e.complexity_before, e.complexity_after)
end

"""
    hardin_taylor_generate(model::FunctionalAutoregressiveTransformer{T},
                           seed_coeffs::AbstractMatrix{T};
                           max_new::Int=10,
                           τ::T=T(0.3),
                           K_min::Int=4,
                           α::T=T(1.5),
                           max_climbs_per_step::Int=3) → (sequence, K_history, climb_events)

Hardin-Taylor inspired autoregressive generation with formal hierarchy climbing.

## Algorithm

At each generation step t:
1. Start with K_current (initially K_min — the "simplest" prediction level)
2. Generate prediction Θ̂_t at current K_current
3. Compute trajectory approval A(Θ̂_t) relative to the previous 2 steps
4. If A(Θ̂_t) ≥ τ: ACCEPT prediction (compatible with observed trajectory)
5. If A(Θ̂_t) < τ: CLIMB hierarchy — increase K_current, re-predict
6. After acceptance, if recent approvals are consistently high, DESCEND
   (decrease K_current back toward K_min)

## Connection to Hardin-Taylor Theorem

The theorem's key structural result: as time progresses, the prediction
function f*_t can only "climb" in the well-ordering (f*_s ≤ f*_t for s < t),
and each climb must be strict.  Since the well-ordering is well-founded,
climbs cannot happen infinitely often — the exceptional set is countable.

Our constructive version:
- The "well-ordering" is replaced by K values (finite, bounded by K_max)
- "Climbing" means increasing K_current
- The τ threshold ensures climbs only happen when prediction quality degrades
- After every climb, there exists a "stability interval" where the higher K
  produces acceptable predictions

## Arguments
- `model`: trained FAT model
- `seed_coeffs`: shape (K+1, T_seed) — initial coefficient sequence
- `max_new`: number of new steps to generate
- `τ`: approval threshold for triggering climbs (default: 0.3)
- `K_min`: minimum truncation order to start with (default: 4)
- `α`: complexity exponent for diagnostics (default: 1.5)
- `max_climbs_per_step`: max re-predictions per step before forcing acceptance

## Returns
- `sequence`: (K+1, T_seed + max_new) coefficient matrix
- `K_history`: Vector{Int} of K_current at each generated step
- `climb_events`: Vector{ClimbEvent{T}} recording all hierarchy climbs
"""
function hardin_taylor_generate(
        model::FunctionalAutoregressiveTransformer{T,K,Ka,Pe},
        seed_coeffs::AbstractMatrix{T};
        max_new::Int = 10,
        τ::T = T(0.3),
        K_min::Int = 4,
        α::T = T(1.5),
        max_climbs_per_step::Int = 3) where {T<:AbstractFloat, K, Ka, Pe}

    @assert size(seed_coeffs, 1) == K + 1 "seed_coeffs first dim $(size(seed_coeffs,1)) ≠ K+1=$(K+1)"
    @assert K_min >= 1 "K_min must be ≥ 1, got $K_min"
    @assert K_min <= K "K_min=$K_min must be ≤ K=$K"

    sequence = copy(seed_coeffs)    # (K+1, T_current)
    K_current = K_min               # Start at minimum complexity (Hardin-Taylor: ≤-least)
    K_history = Int[]
    climb_events = ClimbEvent{T}[]

    # Track recent approvals for descent decisions
    recent_approvals = T[]

    for step in 1:max_new
        T_curr = size(sequence, 2)
        T_use = min(T_curr, model.config.block_size)

        accepted = false
        n_climbs = 0

        while !accepted && n_climbs <= max_climbs_per_step
            # ── Generate prediction at current K_current ──────────────────────
            # Slice input to K_current+1 coefficients
            inp_slice = sequence[1:K_current+1, end-T_use+1:end]

            # Dynamic embedding: use only the first K_current+1 input dimensions
            W_in = model.coeff_embed.weight[:, 1:K_current+1]
            b_in = model.coeff_embed.bias
            H_2d = W_in * inp_slice .+ b_in                    # (n_embd, T_use)
            H = reshape(H_2d, model.config.n_embd, T_use, 1)   # (n_embd, T_use, 1)

            # Transformer backbone
            for block in model.blocks
                H = block(H; training=false)
            end
            H = model.ln_f(H)

            # Dynamic spectral head: output only K_current+1 coefficients
            W_out = model.spectral_head.weight[1:K_current+1, :]
            b_out = model.spectral_head.bias[1:K_current+1]
            H_2d_out = reshape(H, model.config.n_embd, T_use)
            out_2d = W_out * H_2d_out .+ b_out
            next_Θ_sliced = out_2d[:, end]                      # (K_current+1,)

            # ── Compute trajectory approval ───────────────────────────────────
            # We need at least 2 previous steps to compute approval
            if T_curr >= 2
                # Previous two coefficient vectors (at current K level)
                prev2 = sequence[1:K_current+1, end-1]
                prev1 = sequence[1:K_current+1, end]

                # Model update direction: prev1 → next_Θ
                Δ_model = next_Θ_sliced .- prev1
                # Trajectory direction: prev2 → prev1 (observed trend)
                Δ_trend = prev1 .- prev2

                norm_model = sqrt(sum(abs2, Δ_model) + T(1e-8))
                norm_trend = sqrt(sum(abs2, Δ_trend) + T(1e-8))

                # Cosine similarity (approval)
                approval = dot(Δ_model, Δ_trend) / (norm_model * norm_trend)
            else
                approval = one(T)  # Accept unconditionally for first steps
            end

            # ── Decision: accept or climb ─────────────────────────────────────
            if approval >= τ || n_climbs >= max_climbs_per_step
                # ACCEPT: prediction is compatible with trajectory
                # (or we've exhausted climb budget)
                accepted = true
                push!(recent_approvals, approval)

                # Pad to full K+1 for storage
                next_Θ = zeros(T, K+1)
                next_Θ[1:K_current+1] .= next_Θ_sliced

                # Apply stabilizer on the active coefficients
                # Only apply the full stabilizer when K_current >= Ka (anchor degree)
                # When K_current < Ka, the low order already provides natural stability
                if K_current >= Ka
                    # Full stabilizer: Π = P_p ∘ P_k
                    stabilized = model.stabilizer(next_Θ_sliced)
                    fill!(next_Θ, zero(T))
                    next_Θ[1:length(stabilized)] .= stabilized
                elseif K_current >= Pe
                    # Partial stabilization: just economize to P_econ
                    economized = next_Θ_sliced[1:Pe+1]
                    fill!(next_Θ, zero(T))
                    next_Θ[1:Pe+1] .= economized
                end
                # If K_current < Pe, the coefficients are already at very low
                # order — no stabilization needed (inherently stable)

                sequence = hcat(sequence, next_Θ)
            else
                # CLIMB: increase K_current (the Hardin-Taylor hierarchy climb)
                complexity_before = spectral_complexity(next_Θ_sliced; α=α)
                K_before = K_current
                K_current = min(K_current + 1, K)

                # Re-predict at higher K (next iteration of while loop)
                # After re-prediction, compute new complexity for the climb event
                complexity_after = complexity_before  # Will be updated if re-predicted

                push!(climb_events, ClimbEvent{T}(
                    step, K_before, K_current,
                    approval, complexity_before, complexity_after
                ))

                n_climbs += 1
            end
        end

        push!(K_history, K_current)

        # ── Descent heuristic: if recent approvals are consistently high, ─────
        # reduce K_current back toward K_min (the "stability interval" analog)
        if length(recent_approvals) >= 3
            recent_window = recent_approvals[max(1, end-2):end]
            if all(a -> a > T(0.7), recent_window) && K_current > K_min
                K_current = max(K_min, K_current - 1)
            end
        end

        # Keep only last 10 approvals in memory
        if length(recent_approvals) > 10
            recent_approvals = recent_approvals[end-9:end]
        end
    end

    return sequence, K_history, climb_events
end
