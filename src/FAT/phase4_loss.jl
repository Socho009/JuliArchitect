# --- PHASE 4 ---
# Directional Trajectory Approval Loss
#
# Implements ℒ_Flow = 1 - 𝔼[A(Θ_t)], where
#
#   A(Θ_t) = ⟨V_Θ(Θ_t), V*(Θ_t)⟩_{ℝ^{K+1}}
#
# is the cosine similarity between:
#   V*(Θ_t)  — unit vector from current prediction toward ground-truth target
#   V_Θ(Θ_t) — unit vector of the model's autoregressive update direction
#
# Teacher-forcing grounding (resolves the circularity in the original paper):
#   V*(Θ_t) = (Θ*_t - Θ_t) / ‖Θ*_t - Θ_t‖_2
#
# Gradient-vanishing warm-up:
#   Near initialization, Θ_pred ≈ Θ_target may coincide, making V* undefined.
#   When ‖Θ*_t - Θ_t‖_2 < ε, we fall back to MSE loss (smooth and always
#   non-zero gradient), then blend back to the approval loss as training
#   progresses and the coefficient space is explored.
#
# Tensor layout (features × sequence × batch):
#   θ_pred   : (K+1, seq_len, batch)
#   θ_target : (K+1, seq_len, batch)
#
# Zygote compatibility:
#   Every operation is pure functional (no mutation).
#   The only non-differentiable operation is the `norm` denominator — guarded
#   by a small ε to prevent divide-by-zero (and the associated NaN gradient).

using LinearAlgebra
using Statistics: mean

# ─── Core directional loss ────────────────────────────────────────────────────

"""
    directional_trajectory_loss(θ_pred, θ_target; ε=1f-6, α=1f0, warmup_mse=true) -> T

Compute the Directional Trajectory Approval loss

    ℒ_Flow = 1 - 𝔼_{t,b}[A(Θ_t)]

where the approval function A is the cosine similarity between the model's
autoregressive update direction and the teacher-forced target direction in
Chebyshev coefficient space.

## Arguments
- `θ_pred   :: AbstractArray{T,3}` — predicted coefficients, shape `(K+1, seq_len, batch)`
- `θ_target :: AbstractArray{T,3}` — ground-truth coefficients (same shape), teacher-forced
- `ε`       — numerical floor for norms (prevents div-by-zero and NaN gradients)
- `α`       — blending weight: `loss = α · ℒ_Flow + (1-α) · ℒ_MSE`
               set `α=1` for pure approval loss, `α=0` for pure MSE
- `warmup_mse` — if `true`, detect near-zero norms and fall back to MSE automatically

## Returns
Scalar loss value of type `T`.

## Mathematical derivation

**Target direction V*(Θ_t):**
    V*(Θ_t) = (Θ*_t - Θ_t) / ‖Θ*_t - Θ_t‖_2
This points from the model's current prediction toward the ground truth.
Well-defined whenever Θ_t ≠ Θ*_t (i.e., away from exact convergence).

**Model update direction V_Θ(Θ_t):**
    V_Θ(Θ_t) = (Θ_{t+1} - Θ_t) / ‖Θ_{t+1} - Θ_t‖_2
This is the direction of the model's autoregressive step in coefficient space.
Computed from consecutive predicted positions: position t → position t+1.

**Approval A(Θ_t):**
    A(Θ_t) = ⟨V_Θ(Θ_t), V*(Θ_t)⟩_{ℝ^{K+1}}
= cosine similarity ∈ [-1, 1].
A = +1 means the model is stepping perfectly toward the target.
A = -1 means it is stepping directly away.

**Loss:**
    ℒ_Flow = 1 - mean(A)  ∈ [0, 2]
Minimizing ℒ_Flow maximizes directional alignment.

## Connection to Flow Matching
This is the discrete-time, coefficient-space analogue of the CFM objective
of Lipman et al. (2022): optimize the velocity field direction to align with
the conditional transport direction — here in ℝ^{K+1} rather than ℝ^d data space.

## Tensor layout
Coefficients are in the **first** dimension (features × sequence × batch),
consistent with column-major Julia convention.  Taking `diff` along dim=2
gives the sequence of autoregressive steps in one vectorized call.
"""
function directional_trajectory_loss(
        θ_pred::AbstractArray{T,3},
        θ_target::AbstractArray{T,3};
        ε::T        = T(1e-6),
        α::T        = one(T),
        warmup_mse::Bool = true) where {T<:AbstractFloat}

    K1, seq_len, batch = size(θ_pred)
    @assert size(θ_target) == (K1, seq_len, batch) "Shape mismatch: θ_pred $(size(θ_pred)) ≠ θ_target $(size(θ_target))"
    @assert seq_len >= 2 "Need at least 2 time steps to compute update direction; got seq_len=$seq_len"

    # ── MSE fallback (always finite gradient) ─────────────────────────────────
    mse_loss = mean(abs2, θ_pred .- θ_target)

    α ≈ zero(T) && return mse_loss   # pure MSE mode — skip approval computation

    # ── Step 1: V*(Θ_t) — teacher-forced target direction ─────────────────────
    # Δ_target[:, t, b] = Θ*_t - Θ_t = θ_target[:,t,b] - θ_pred[:,t,b]
    # Shape: (K+1, seq_len, batch)
    Δ_target = θ_target .- θ_pred

    # ‖Δ_target‖_2 per (t, b): shape (1, seq_len, batch) for broadcasting
    # Using sum(abs2; dims=1) + ε before sqrt avoids NaN in backward pass
    Δ_target_norms = sqrt.(sum(abs2, Δ_target; dims=1) .+ ε^2)  # (1, seq_len, batch)

    # Normalized: V* = Δ_target / ‖Δ_target‖   shape: (K+1, seq_len, batch)
    V_star = Δ_target ./ Δ_target_norms

    # ── Step 2: V_Θ(Θ_t) — model autoregressive update direction ─────────────
    # Δ_pred[:, t, b] = Θ_{t+1} - Θ_t = θ_pred[:, t+1, b] - θ_pred[:, t, b]
    # Only defined for t = 1 … seq_len-1, so we have seq_len-1 valid steps.
    # Shape: (K+1, seq_len-1, batch)
    Δ_pred = θ_pred[:, 2:end, :] .- θ_pred[:, 1:end-1, :]

    Δ_pred_norms = sqrt.(sum(abs2, Δ_pred; dims=1) .+ ε^2)      # (1, seq_len-1, batch)
    V_theta = Δ_pred ./ Δ_pred_norms                             # (K+1, seq_len-1, batch)

    # ── Step 3: Approval A(Θ_t) — cosine similarity at each step ─────────────
    # V_star is defined at all seq_len positions but the autoregressive update
    # direction only exists for t = 1 … seq_len-1 (we observe the "before" state).
    # Align: compare V_theta[:, t, b] with V_star[:, t, b] for t = 1…seq_len-1.
    V_star_aligned = V_star[:, 1:end-1, :]  # (K+1, seq_len-1, batch)

    # Element-wise product and sum over coefficient dim → cosine similarity
    # approval[:, t, b] = ⟨V_theta[:, t, b], V_star[:, t, b]⟩
    # Shape after sum: (1, seq_len-1, batch) → squeeze to (seq_len-1, batch)
    approval = sum(V_theta .* V_star_aligned; dims=1)  # (1, seq_len-1, batch)
    approval = reshape(approval, seq_len-1, batch)      # (seq_len-1, batch)

    flow_loss = one(T) - mean(approval)

    # ── Step 4: Warm-up blending ───────────────────────────────────────────────
    # When ‖Δ_target‖ is uniformly small (near convergence or initialization),
    # flow_loss ≈ 1 - ⟨noise, noise⟩ which has unstable gradients.
    # Detect this condition and blend with MSE.
    if warmup_mse
        # Mean target norm across all positions and batch
        mean_target_norm = mean(Δ_target_norms)
        # Soft blend factor: γ → 1 (pure approval) when mean norm >> ε
        # γ → 0 (pure MSE) when mean norm ≈ ε (near convergence / bad init)
        γ = mean_target_norm / (mean_target_norm + T(10) * ε)
        effective_α = α * γ
        return effective_α * flow_loss + (one(T) - effective_α) * mse_loss
    else
        return α * flow_loss + (one(T) - α) * mse_loss
    end
end

# ─── Convenience overloads ────────────────────────────────────────────────────

"""
    directional_trajectory_loss(θ_pred, θ_target, α::T; kwargs...) -> T

Convenience overload with positional α argument.
"""
directional_trajectory_loss(θ_pred::AbstractArray{T,3}, θ_target::AbstractArray{T,3},
                            α::T; kwargs...) where T =
    directional_trajectory_loss(θ_pred, θ_target; α=α, kwargs...)

# ─── Per-step approval diagnostics ───────────────────────────────────────────

"""
    compute_approval_per_step(θ_pred, θ_target; ε=1f-6) -> Matrix{T}

Compute the approval A(Θ_t) at each (t, b) pair without collapsing to
a scalar.  Returns a `(seq_len-1, batch)` matrix of cosine similarities.

Useful for:
  - Diagnosing which sequence positions the model struggles with
  - Plotting approval curves during training
  - Setting adaptive α schedules

## Return
Matrix of shape `(seq_len-1, batch)` with values in `[-1, 1]`.
"""
function compute_approval_per_step(
        θ_pred::AbstractArray{T,3},
        θ_target::AbstractArray{T,3};
        ε::T = T(1e-6)) where {T<:AbstractFloat}

    K1, seq_len, batch = size(θ_pred)

    Δ_target = θ_target[:, 1:end-1, :] .- θ_pred[:, 1:end-1, :]
    Δ_pred   = θ_pred[:, 2:end, :]     .- θ_pred[:, 1:end-1, :]

    Δ_target_norms = sqrt.(sum(abs2, Δ_target; dims=1) .+ ε^2)
    Δ_pred_norms   = sqrt.(sum(abs2, Δ_pred;   dims=1) .+ ε^2)

    V_star  = Δ_target ./ Δ_target_norms
    V_theta = Δ_pred   ./ Δ_pred_norms

    approval = sum(V_theta .* V_star; dims=1)   # (1, seq_len-1, batch)
    return reshape(approval, seq_len-1, batch)
end

# ─── α warm-up schedule ───────────────────────────────────────────────────────

"""
    approval_alpha_schedule(iter::Int; warmup_iters::Int=500) -> Float32

Linearly ramp the approval weight α from 0 (pure MSE) to 1 (pure approval)
over the first `warmup_iters` training iterations.

This prevents the gradient-vanishing pathology at initialization: in early
training the coefficient vectors are nearly identical (all near zero from
the small-scale initialization), so V* and V_Θ are dominated by noise.
Starting with MSE gives clean gradients while the model explores coefficient space.

## Usage
```julia
for iter in 1:max_iters
    α = approval_alpha_schedule(iter; warmup_iters=500)
    loss = directional_trajectory_loss(θ_pred, θ_target; α=Float32(α))
    ...
end
```
"""
function approval_alpha_schedule(iter::Int; warmup_iters::Int = 500)::Float32
    return Float32(min(1.0, iter / warmup_iters))
end

# ─── Zygote gradient verification helper ─────────────────────────────────────

"""
    check_loss_gradient(K::Int=7, seq_len::Int=6, batch::Int=2) -> Bool

Verify that `Zygote.gradient` flows correctly through `directional_trajectory_loss`.

Computes the gradient of the loss with respect to `θ_pred` and checks:
  1. Gradient is not `nothing`
  2. All gradient entries are finite (no NaN/Inf)
  3. Gradient has the same shape as `θ_pred`

Returns `true` on success.  Designed to be called from a unit test.
"""
function check_loss_gradient(K::Int = 7, seq_len::Int = 6, batch::Int = 2)::Bool
    T = Float32
    θ_pred   = randn(T, K+1, seq_len, batch)
    θ_target = randn(T, K+1, seq_len, batch)

    val, grads = Zygote.withgradient(θ_pred) do θ
        directional_trajectory_loss(θ, θ_target;
                                    ε=T(1e-6), α=T(1.0), warmup_mse=true)
    end

    g = grads[1]

    ok = (g !== nothing) &&
         (size(g) == (K+1, seq_len, batch)) &&
         all(isfinite, g)

    if !ok
        @warn "Gradient check failed" val=val grad_nothing=(g===nothing)
    end
    return ok
end
