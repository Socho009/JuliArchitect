# Canonical rollout evaluation utilities.
#
# FAT's scientific value is mostly in long-horizon behavior, so these helpers
# make rollout evaluation a first-class API rather than script-local code.

using Statistics: mean

"""
    RolloutMetrics{T}

Metrics for comparing a generated coefficient rollout with a ground-truth
coefficient sequence.

Fields:
- `one_step_mse`: MSE at the first generated step.
- `rollout_mse`: MSE across all generated future steps.
- `final_error`: coefficient-space norm at the last generated step.
- `final_hilbert_error`: Hilbert/coefficient norm at the last generated step.
- `step_errors`: coefficient-space norm per generated step.
- `hilbert_errors`: Hilbert/coefficient norm per generated step.
- `spectral_energy`: mean squared coefficient energy per generated step.
- `spectral_energy_drift`: predicted minus truth spectral energy per generated step.
- `stabilizer_contraction_ratios`: `norm(Pi(x)) / norm(x)` per generated step, if a stabilizer is provided.
- `complexity`: weighted l1 spectral complexity per generated step.
- `error_growth_ratios`: `step_errors[t+1] / step_errors[t]` with epsilon floor.
"""
struct RolloutMetrics{T<:AbstractFloat}
    one_step_mse::T
    rollout_mse::T
    final_error::T
    final_hilbert_error::T
    step_errors::Vector{T}
    hilbert_errors::Vector{T}
    spectral_energy::Vector{T}
    spectral_energy_drift::Vector{T}
    stabilizer_contraction_ratios::Vector{T}
    complexity::Vector{T}
    error_growth_ratios::Vector{T}
end

"""
    rollout(model, seed_coeffs; max_new, apply_stabilizer=true)

Generate an autoregressive coefficient rollout from a seed matrix.
For FAT models, `apply_stabilizer` controls use of `SpectralStabilizer`.
For `ContinuousModel`, the keyword is accepted for API symmetry and ignored.
"""
function rollout(model::FunctionalAutoregressiveTransformer{T},
                 seed_coeffs::AbstractMatrix{T};
                 max_new::Int,
                 apply_stabilizer::Bool=true) where {T<:AbstractFloat}
    return fat_generate(model, seed_coeffs; max_new=max_new, apply_stabilizer=apply_stabilizer)
end

function rollout(model::ContinuousModel{T},
                 seed_coeffs::AbstractMatrix{T};
                 max_new::Int,
                 apply_stabilizer::Bool=true) where {T<:AbstractFloat}
    return generate(model, seed_coeffs; max_new=max_new)
end

_rollout_stabilizer(model) = nothing
_rollout_stabilizer(model::FunctionalAutoregressiveTransformer) = model.stabilizer

"""
    rollout_metrics(predicted, truth; seed_len, eps=1e-8, complexity_alpha=1.5, stabilizer=nothing)

Compute coefficient-space rollout metrics. `predicted` and `truth` must both
have shape `(coefficients, total_steps)`, where the first `seed_len` columns are
the seed and the remaining columns are generated/evaluated steps.
"""
function rollout_metrics(predicted::AbstractMatrix{T},
                         truth::AbstractMatrix{T};
                         seed_len::Int,
                         eps::T=T(1e-8),
                         complexity_alpha::T=T(1.5),
                         stabilizer=nothing) where {T<:AbstractFloat}
    @assert size(predicted, 1) == size(truth, 1) "coefficient dimensions differ"
    @assert size(predicted, 2) <= size(truth, 2) "truth must cover predicted horizon"
    @assert 1 <= seed_len < size(predicted, 2) "seed_len must leave at least one future step"

    total_steps = size(predicted, 2)
    future = seed_len+1:total_steps
    pred_future = predicted[:, future]
    truth_future = truth[:, future]
    diff = pred_future .- truth_future

    step_errors = T[norm(diff[:, i]) for i in axes(diff, 2)]
    hilbert_errors = copy(step_errors)
    one_step_mse = mean(abs2, diff[:, 1])
    rollout_mse = mean(abs2, diff)
    final_error = step_errors[end]
    final_hilbert_error = hilbert_errors[end]

    spectral_energy = T[mean(abs2, pred_future[:, i]) for i in axes(pred_future, 2)]
    truth_energy = T[mean(abs2, truth_future[:, i]) for i in axes(truth_future, 2)]
    spectral_energy_drift = spectral_energy .- truth_energy
    weights = T[(one(T) + T(n))^complexity_alpha for n in 0:(size(predicted, 1)-1)]
    complexity = T[sum(weights .* abs.(pred_future[:, i])) for i in axes(pred_future, 2)]

    contraction = T[]
    if stabilizer !== nothing
        for i in axes(pred_future, 2)
            v = pred_future[:, i]
            push!(contraction, T(norm(stabilizer(v)) / (norm(v) + eps)))
        end
    end

    ratios = T[]
    for i in 1:(length(step_errors)-1)
        push!(ratios, step_errors[i+1] / (step_errors[i] + eps))
    end

    return RolloutMetrics{T}(
        one_step_mse,
        rollout_mse,
        final_error,
        final_hilbert_error,
        step_errors,
        hilbert_errors,
        spectral_energy,
        spectral_energy_drift,
        contraction,
        complexity,
        ratios,
    )
end

"""
    evaluate_rollout(model, truth_sequence; seed_len, max_new, apply_stabilizer=true)

Use the first `seed_len` columns of `truth_sequence` as seed, generate
`max_new` steps, and return `(prediction, metrics)`.
"""
function evaluate_rollout(model,
                          truth_sequence::AbstractMatrix{T};
                          seed_len::Int,
                          max_new::Int=size(truth_sequence, 2)-seed_len,
                          apply_stabilizer::Bool=true,
                          complexity_alpha::T=T(1.5)) where {T<:AbstractFloat}
    @assert seed_len + max_new <= size(truth_sequence, 2)
    seed = truth_sequence[:, 1:seed_len]
    predicted = rollout(model, seed; max_new=max_new, apply_stabilizer=apply_stabilizer)
    truth = truth_sequence[:, 1:(seed_len + max_new)]
    metrics = rollout_metrics(predicted, truth;
                              seed_len=seed_len,
                              complexity_alpha=complexity_alpha,
                              stabilizer=_rollout_stabilizer(model))
    return predicted, metrics
end

"""
    evaluate_rollout(model, seed, truth_sequence; apply_stabilizer=true)

Evaluate a model from an explicit seed against a full truth sequence that
contains the seed followed by the desired future trajectory.
"""
function evaluate_rollout(model,
                          seed::AbstractMatrix{T},
                          truth_sequence::AbstractMatrix{T};
                          apply_stabilizer::Bool=true,
                          complexity_alpha::T=T(1.5)) where {T<:AbstractFloat}
    seed_len = size(seed, 2)
    max_new = size(truth_sequence, 2) - seed_len
    predicted = rollout(model, seed; max_new=max_new, apply_stabilizer=apply_stabilizer)
    metrics = rollout_metrics(predicted, truth_sequence;
                              seed_len=seed_len,
                              complexity_alpha=complexity_alpha,
                              stabilizer=_rollout_stabilizer(model))
    return predicted, metrics
end
