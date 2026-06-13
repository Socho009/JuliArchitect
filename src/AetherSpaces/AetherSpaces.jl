module AetherSpaces

using ..Spaces

export AbstractMathSpace, SpaceState
export MetricSpace, MeasureSpace, TopologicalChartSpace, FractionalOperatorSpace
export TropicalSpace, UltrametricSpace, SheafConsistencySpace, CategoricalBridgeSpace
export to_space, from_space, space_distance, space_project, space_regularizer, space_metadata
export apply_space_stack, space_stack_regularizer, space_stack_metrics
export chart_residual, fractional_operator_weights, gluing_residual, commutator_residual
export stable_aether_spaces, diagnostic_aether_spaces, default_aether_spaces, aether_space_metrics

const AbstractMathSpace = Spaces.AbstractMathSpace
const SpaceState = Spaces.SpaceState
const MetricSpace = Spaces.MetricSpace
const MeasureSpace = Spaces.MeasureSpace
const TopologicalChartSpace = Spaces.TopologicalChartSpace
const FractionalOperatorSpace = Spaces.FractionalOperatorSpace
const TropicalSpace = Spaces.TropicalSpace
const UltrametricSpace = Spaces.UltrametricSpace
const SheafConsistencySpace = Spaces.SheafConsistencySpace
const CategoricalBridgeSpace = Spaces.CategoricalBridgeSpace

const to_space = Spaces.to_space
const from_space = Spaces.from_space
const space_distance = Spaces.space_distance
const space_project = Spaces.space_project
const space_regularizer = Spaces.space_regularizer
const space_metadata = Spaces.space_metadata
const apply_space_stack = Spaces.apply_space_stack
const space_stack_regularizer = Spaces.space_stack_regularizer
const space_stack_metrics = Spaces.space_stack_metrics
const chart_residual = Spaces.chart_residual
const fractional_operator_weights = Spaces.fractional_operator_weights
const gluing_residual = Spaces.gluing_residual
const commutator_residual = Spaces.commutator_residual

function _coefficient_weights(::Type{T}, K::Int) where {T<:AbstractFloat}
    return T[one(T) / T(1 + n) for n in 0:K]
end

"""
    stable_aether_spaces(T=Float32; K, fractional_alpha=1.25)

Return the v1 Aether spaces intended for opt-in training regularization.
These are the metric, measure, and fractional adapters.
"""
function stable_aether_spaces(::Type{T}=Float32; K::Int, fractional_alpha::Real=1.25) where {T<:AbstractFloat}
    weights = _coefficient_weights(T, K)
    return (
        MetricSpace{T}(weights=weights, metric=:weighted),
        MeasureSpace{T}(weights),
        FractionalOperatorSpace{T}(alpha=T(fractional_alpha)),
    )
end

"""
    diagnostic_aether_spaces(T=Float32; K)

Return the full experimental eight-space Aether stack. The last five spaces are
diagnostic-first until benchmark evidence justifies using them as losses.
"""
function diagnostic_aether_spaces(::Type{T}=Float32; K::Int) where {T<:AbstractFloat}
    weights = _coefficient_weights(T, K)
    return (
        MetricSpace{T}(weights=weights, metric=:weighted),
        MeasureSpace{T}(weights),
        TopologicalChartSpace{T}(radius=T(2)),
        FractionalOperatorSpace{T}(alpha=T(1.25)),
        TropicalSpace{T}(),
        UltrametricSpace{T}(p=2),
        SheafConsistencySpace{T}(patch_size=2),
        CategoricalBridgeSpace{T}(),
    )
end

default_aether_spaces(::Type{T}=Float32; K::Int) where {T<:AbstractFloat} =
    stable_aether_spaces(T; K=K)

aether_space_metrics(coeffs, target, spaces) =
    space_stack_metrics(coeffs, target, spaces)

end # module AetherSpaces
