#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Architect
using JSON
using LinearAlgebra
using Printf
using Random
using Statistics

function parse_args(args)
    opts = Dict{String,String}()
    for arg in args
        occursin("=", arg) || continue
        k, v = split(arg, "=", limit=2)
        opts[k] = v
    end
    return opts
end

function synthetic_coeffs(K::Int, steps::Int, ::Type{T}) where {T<:AbstractFloat}
    coeffs = zeros(T, K + 1, steps)
    for t in 1:steps
        phase = T(0.2) * T(t - 1)
        for n in 0:K
            coeffs[n+1, t] = sin(T(n + 1) * phase) / T(n + 1)
        end
    end
    return coeffs
end

function default_spaces(K::Int)
    weights = Float32[1 / (1 + n) for n in 0:K]
    return AbstractMathSpace[
        MetricSpace{Float32}(weights=weights, metric=:weighted),
        MeasureSpace{Float32}(weights),
        TopologicalChartSpace{Float32}(radius=2.0f0, overlap_weight=0.5f0),
        FractionalOperatorSpace{Float32}(alpha=1.5f0),
        TropicalSpace{Float32}(mode=:maxplus),
        UltrametricSpace{Float32}(p=2),
        SheafConsistencySpace{Float32}(patch_size=2, gluing_weight=0.75f0),
        CategoricalBridgeSpace{Float32}(source_scale=1.1f0, target_scale=0.9f0),
    ]
end

function main(args=ARGS)
    opts = parse_args(args)
    K = parse(Int, get(opts, "K", "7"))
    steps = parse(Int, get(opts, "steps", "12"))
    seed = parse(Int, get(opts, "seed", "42"))
    outdir = get(opts, "outdir", joinpath(@__DIR__, "..", "results", "unified_spaces"))
    mkpath(outdir)

    Random.seed!(seed)
    coeffs = synthetic_coeffs(K, steps, Float32)
    spaces = default_spaces(K)
    csv_path = joinpath(outdir, "space_probe.csv")

    open(csv_path, "w") do io
        println(io, "space,self_distance,regularizer,roundtrip_error,metadata")
        for space in spaces
            state = to_space(space, coeffs)
            back = from_space(state)
            self_dist = space_distance(space, state, state)
            reg = space_regularizer(space, coeffs)
            roundtrip = norm(back .- coeffs)
            metadata = JSON.json(space_metadata(space))
            @printf(io, "%s,%.8g,%.8g,%.8g,%s\n",
                    nameof(typeof(space)), self_dist, reg, roundtrip, metadata)
        end
    end

    stack_out = apply_space_stack(coeffs, spaces)
    @assert size(stack_out) == size(coeffs)

    println("Wrote unified-space probe to $csv_path")
end

main()
