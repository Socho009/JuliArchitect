module Spaces

using LinearAlgebra
using Statistics: mean
using Zygote

export AbstractMathSpace, SpaceState
export MetricSpace, MeasureSpace, TopologicalChartSpace, FractionalOperatorSpace
export TropicalSpace, UltrametricSpace, SheafConsistencySpace, CategoricalBridgeSpace
export to_space, from_space, space_distance, space_project, space_regularizer, space_metadata
export apply_space_stack, space_stack_regularizer, space_stack_metrics
export chart_residual, fractional_operator_weights, gluing_residual, commutator_residual

abstract type AbstractMathSpace end

struct SpaceState{T,S<:AbstractMathSpace}
    coords::AbstractArray{T}
    space::S
end

_coords(x::AbstractArray) = x
_coords(s::SpaceState) = s.coords
_coords(x) = hasproperty(x, :coeffs) ? getproperty(x, :coeffs) : x

_same_array(x) = copy(_coords(x))

function _weights_array(weights::AbstractVector{T}, x::AbstractArray{T}) where {T}
    @assert length(weights) == size(x, 1) "weights length $(length(weights)) must match first dimension $(size(x, 1))"
    return reshape(weights, length(weights), ntuple(_ -> 1, ndims(x) - 1)...)
end

Zygote.@nograd _weights_array

_finite_project(x::AbstractArray{T}) where {T<:AbstractFloat} =
    ifelse.(isfinite.(x), x, zero(T))

to_space(space::AbstractMathSpace, x) = SpaceState(space_project(space, _to_space_coords(space, _coords(x))), space)
from_space(state::SpaceState) = _from_space_coords(state.space, state.coords)

_to_space_coords(::AbstractMathSpace, x::AbstractArray) = copy(x)
_from_space_coords(::AbstractMathSpace, x::AbstractArray) = copy(x)
space_project(::AbstractMathSpace, x::AbstractArray) = _finite_project(x)
space_regularizer(space::AbstractMathSpace, x) = mean(abs2, _coords(to_space(space, x)))
space_metadata(space::AbstractMathSpace) = Dict{String,Any}("name" => string(nameof(typeof(space))))

# 1. Metric Space
struct MetricSpace{T<:AbstractFloat} <: AbstractMathSpace
    weights::Union{Nothing,Vector{T}}
    metric::Symbol
end

MetricSpace{T}(; weights=nothing, metric::Symbol=:euclidean) where {T<:AbstractFloat} =
    MetricSpace{T}(weights === nothing ? nothing : Vector{T}(weights), metric)
MetricSpace(; weights=nothing, metric::Symbol=:euclidean) =
    MetricSpace{Float32}(; weights=weights, metric=metric)

function space_distance(space::MetricSpace{T}, a, b) where {T}
    diff = T.(_coords(a) .- _coords(b))
    if space.weights === nothing || space.metric == :euclidean
        return sqrt(sum(abs2, diff))
    end
    w = _weights_array(space.weights, diff)
    return sqrt(sum(w .* abs2.(diff)))
end

space_regularizer(space::MetricSpace, x) = space_distance(space, x, zero.(_coords(x)))
space_metadata(space::MetricSpace) = Dict{String,Any}("name" => "MetricSpace", "metric" => string(space.metric), "weighted" => space.weights !== nothing)

# 2. Measure Space
struct MeasureSpace{T<:AbstractFloat} <: AbstractMathSpace
    weights::Vector{T}
    normalize::Bool
end

function MeasureSpace{T}(weights::AbstractVector; normalize::Bool=true) where {T<:AbstractFloat}
    w = Vector{T}(weights)
    w = normalize ? w ./ max(sum(w), eps(T)) : w
    return MeasureSpace{T}(w, normalize)
end
MeasureSpace(weights::AbstractVector; normalize::Bool=true) =
    MeasureSpace{Float32}(weights; normalize=normalize)

function space_distance(space::MeasureSpace{T}, a, b) where {T}
    diff = T.(_coords(a) .- _coords(b))
    w = _weights_array(space.weights, diff)
    return sqrt(sum(w .* abs2.(diff)))
end

space_regularizer(space::MeasureSpace, x) = space_distance(space, x, zero.(_coords(x)))
space_metadata(space::MeasureSpace) = Dict{String,Any}("name" => "MeasureSpace", "mass" => sum(space.weights), "normalize" => space.normalize)

# 3. Topological Chart Space
struct TopologicalChartSpace{T<:AbstractFloat} <: AbstractMathSpace
    radius::T
    overlap_weight::T
end

TopologicalChartSpace{T}(; radius::Real=1, overlap_weight::Real=1) where {T<:AbstractFloat} =
    TopologicalChartSpace{T}(T(radius), T(overlap_weight))
TopologicalChartSpace(; radius::Real=1, overlap_weight::Real=1) =
    TopologicalChartSpace{Float32}(; radius=radius, overlap_weight=overlap_weight)

function space_project(space::TopologicalChartSpace{T}, x::AbstractArray{T}) where {T}
    y = _finite_project(x)
    n = norm(y)
    scale = max(one(T), n / max(space.radius, eps(T)))
    return y ./ scale
end

function chart_residual(space::TopologicalChartSpace{T}, x) where {T}
    c = _coords(x)
    size(c, 1) <= 1 && return zero(T)
    mat = reshape(c, size(c, 1), :)
    return space.overlap_weight * mean(abs2, mat[2:end, :] .- mat[1:end-1, :])
end

space_distance(::TopologicalChartSpace{T}, a, b) where {T} = T(norm(_coords(a) .- _coords(b)))
space_regularizer(space::TopologicalChartSpace, x) = chart_residual(space, x)
space_metadata(space::TopologicalChartSpace) = Dict{String,Any}("name" => "TopologicalChartSpace", "radius" => space.radius)

# 4. Fractional Operator Space
struct FractionalOperatorSpace{T<:AbstractFloat} <: AbstractMathSpace
    alpha::T
    eps::T
end

FractionalOperatorSpace{T}(; alpha::Real=1.5, eps::Real=1e-6) where {T<:AbstractFloat} =
    FractionalOperatorSpace{T}(T(alpha), T(eps))
FractionalOperatorSpace(; alpha::Real=1.5, eps::Real=1e-6) =
    FractionalOperatorSpace{Float32}(; alpha=alpha, eps=eps)

fractional_operator_weights(space::FractionalOperatorSpace{T}, n::Int) where {T} =
    T[(one(T) + T(k))^space.alpha for k in 0:n-1]

Zygote.@nograd fractional_operator_weights

function _to_space_coords(space::FractionalOperatorSpace{T}, x::AbstractArray{T}) where {T}
    w = _weights_array(fractional_operator_weights(space, size(x, 1)), x)
    return x .* w
end

function _from_space_coords(space::FractionalOperatorSpace{T}, x::AbstractArray{T}) where {T}
    w = _weights_array(fractional_operator_weights(space, size(x, 1)), x)
    return x ./ (w .+ space.eps)
end

space_distance(space::FractionalOperatorSpace, a, b) =
    norm(_coords(to_space(space, a)) .- _coords(to_space(space, b)))
space_regularizer(space::FractionalOperatorSpace, x) = mean(abs2, _coords(to_space(space, x)))
space_metadata(space::FractionalOperatorSpace) = Dict{String,Any}("name" => "FractionalOperatorSpace", "alpha" => space.alpha)

# 5. Tropical Space
struct TropicalSpace{T<:AbstractFloat} <: AbstractMathSpace
    mode::Symbol
    eps::T
    clamp::T
end

TropicalSpace{T}(; mode::Symbol=:maxplus, eps::Real=1e-6, clamp::Real=32) where {T<:AbstractFloat} =
    TropicalSpace{T}(mode, T(eps), T(clamp))
TropicalSpace(; mode::Symbol=:maxplus, eps::Real=1e-6, clamp::Real=32) =
    TropicalSpace{Float32}(; mode=mode, eps=eps, clamp=clamp)

_to_space_coords(space::TropicalSpace{T}, x::AbstractArray{T}) where {T} =
    sign.(x) .* log1p.(abs.(x) ./ space.eps)

_from_space_coords(space::TropicalSpace{T}, x::AbstractArray{T}) where {T} =
    sign.(x) .* space.eps .* expm1.(abs.(x))

space_project(space::TropicalSpace{T}, x::AbstractArray{T}) where {T} =
    clamp.(_finite_project(x), -space.clamp, space.clamp)
space_distance(::TropicalSpace{T}, a, b) where {T} = maximum(abs.(_coords(a) .- _coords(b)))
space_regularizer(space::TropicalSpace, x) = mean(abs.(_coords(to_space(space, x))))
space_metadata(space::TropicalSpace) = Dict{String,Any}("name" => "TropicalSpace", "mode" => string(space.mode))

# 6. Ultrametric Space
struct UltrametricSpace{T<:AbstractFloat} <: AbstractMathSpace
    p::Int
    eps::T
    clamp::T
end

UltrametricSpace{T}(; p::Int=2, eps::Real=1e-6, clamp::Real=64) where {T<:AbstractFloat} =
    UltrametricSpace{T}(p, T(eps), T(clamp))
UltrametricSpace(; p::Int=2, eps::Real=1e-6, clamp::Real=64) =
    UltrametricSpace{Float32}(; p=p, eps=eps, clamp=clamp)

function _valuation(space::UltrametricSpace{T}, x::AbstractArray{T}) where {T}
    base = log(T(space.p))
    return -log.(abs.(x) .+ space.eps) ./ base
end

_to_space_coords(space::UltrametricSpace{T}, x::AbstractArray{T}) where {T} =
    sign.(x) .* _valuation(space, x)

_from_space_coords(space::UltrametricSpace{T}, x::AbstractArray{T}) where {T} =
    sign.(x) .* T(space.p) .^ (-abs.(x))

space_project(space::UltrametricSpace{T}, x::AbstractArray{T}) where {T} =
    clamp.(_finite_project(x), -space.clamp, space.clamp)

function space_distance(space::UltrametricSpace{T}, a, b) where {T}
    diff = abs.(_coords(a) .- _coords(b))
    isempty(diff) && return zero(T)
    maximum(diff) <= space.eps && return zero(T)
    return maximum(T(space.p) .^ (-_valuation(space, diff)))
end

space_regularizer(space::UltrametricSpace, x) = mean(abs.(_coords(to_space(space, x))))
space_metadata(space::UltrametricSpace) = Dict{String,Any}("name" => "UltrametricSpace", "p" => space.p)

# 7. Sheaf Consistency Space
struct SheafConsistencySpace{T<:AbstractFloat} <: AbstractMathSpace
    patch_size::Int
    gluing_weight::T
end

SheafConsistencySpace{T}(; patch_size::Int=2, gluing_weight::Real=1) where {T<:AbstractFloat} =
    SheafConsistencySpace{T}(max(1, patch_size), T(gluing_weight))
SheafConsistencySpace(; patch_size::Int=2, gluing_weight::Real=1) =
    SheafConsistencySpace{Float32}(; patch_size=patch_size, gluing_weight=gluing_weight)

function gluing_residual(space::SheafConsistencySpace{T}, x) where {T}
    c = _coords(x)
    n = size(c, 1)
    n <= space.patch_size && return zero(T)
    mat = reshape(c, n, :)
    left = mat[1:n-space.patch_size, :]
    right = mat[space.patch_size+1:n, :]
    return space.gluing_weight * mean(abs2, right .- left)
end

space_distance(::SheafConsistencySpace{T}, a, b) where {T} = T(norm(_coords(a) .- _coords(b)))
space_regularizer(space::SheafConsistencySpace, x) = gluing_residual(space, x)
space_metadata(space::SheafConsistencySpace) = Dict{String,Any}("name" => "SheafConsistencySpace", "patch_size" => space.patch_size)

# 8. Categorical Bridge Space
struct CategoricalBridgeSpace{T<:AbstractFloat} <: AbstractMathSpace
    source_scale::T
    target_scale::T
end

CategoricalBridgeSpace{T}(; source_scale::Real=1, target_scale::Real=1) where {T<:AbstractFloat} =
    CategoricalBridgeSpace{T}(T(source_scale), T(target_scale))
CategoricalBridgeSpace(; source_scale::Real=1, target_scale::Real=1) =
    CategoricalBridgeSpace{Float32}(; source_scale=source_scale, target_scale=target_scale)

function commutator_residual(f::Function, g::Function, path_a::Function, path_b::Function, x)
    return norm(path_b(f(x)) .- g(path_a(x)))
end

function commutator_residual(space::CategoricalBridgeSpace{T}, x) where {T}
    f = y -> space.source_scale .* y
    g = y -> space.target_scale .* y
    path_a = y -> y
    path_b = y -> y
    return T(commutator_residual(f, g, path_a, path_b, _coords(x)))
end

space_distance(::CategoricalBridgeSpace{T}, a, b) where {T} = T(norm(_coords(a) .- _coords(b)))
space_regularizer(space::CategoricalBridgeSpace, x) = commutator_residual(space, x)
space_metadata(space::CategoricalBridgeSpace) = Dict{String,Any}("name" => "CategoricalBridgeSpace", "source_scale" => space.source_scale, "target_scale" => space.target_scale)

# Stack utilities
function apply_space_stack(coeffs, spaces)
    current = _same_array(coeffs)
    for space in spaces
        current = from_space(to_space(space, current))
    end
    return current
end

function space_stack_regularizer(coeffs, spaces)
    isempty(spaces) && return zero(eltype(_coords(coeffs)))
    regs = [space_regularizer(space, coeffs) for space in spaces]
    return sum(regs)
end

function space_stack_metrics(coeffs, target, spaces)
    metrics = Dict{String,Any}()
    for (i, space) in enumerate(spaces)
        key = "$(i)_$(nameof(typeof(space)))"
        metrics[key * "_distance"] = space_distance(space, coeffs, target)
        metrics[key * "_regularizer"] = space_regularizer(space, coeffs)
        metrics[key * "_metadata"] = space_metadata(space)
    end
    return metrics
end

end # module Spaces
