using Test

include(joinpath(@__DIR__, "..", "src", "Architect.jl"))
using .Architect
using LinearAlgebra

@testset "Unified mathematical spaces" begin
    K = 5
    coeffs = reshape(Float32[sin(i / 3) for i in 1:((K+1)*4)], K+1, 4)
    target = coeffs .* 0.9f0
    weights = Float32[1 / (1 + n) for n in 0:K]

    spaces = AbstractMathSpace[
        MetricSpace{Float32}(weights=weights, metric=:weighted),
        MeasureSpace{Float32}(weights),
        TopologicalChartSpace{Float32}(radius=2.0f0),
        FractionalOperatorSpace{Float32}(alpha=1.25f0),
        TropicalSpace{Float32}(),
        UltrametricSpace{Float32}(p=2),
        SheafConsistencySpace{Float32}(patch_size=2),
        CategoricalBridgeSpace{Float32}(source_scale=1.1f0, target_scale=0.9f0),
    ]

    for space in spaces
        state = to_space(space, coeffs)
        back = from_space(state)
        @test size(state.coords) == size(coeffs)
        @test size(back) == size(coeffs)
        @test isapprox(space_distance(space, state, state), 0f0; atol=1f-5)
        projected = space_project(space, state.coords)
        @test size(projected) == size(coeffs)
        @test isapprox(space_project(space, projected), projected; atol=1f-5)
        @test isfinite(space_regularizer(space, coeffs))
        @test haskey(space_metadata(space), "name")
    end

    @test size(apply_space_stack(coeffs, spaces)) == size(coeffs)
    @test isfinite(space_stack_regularizer(coeffs, spaces))
    metrics = space_stack_metrics(coeffs, target, spaces)
    @test !isempty(metrics)

    @test isfinite(chart_residual(TopologicalChartSpace{Float32}(), coeffs))
    @test length(fractional_operator_weights(FractionalOperatorSpace{Float32}(), K+1)) == K+1
    @test isfinite(gluing_residual(SheafConsistencySpace{Float32}(), coeffs))
    @test isfinite(commutator_residual(CategoricalBridgeSpace{Float32}(), coeffs))

    f = x -> 2f0 .* x
    g = x -> 3f0 .* x
    path_a = x -> x .+ 1f0
    path_b = x -> x .- 1f0
    @test isfinite(commutator_residual(f, g, path_a, path_b, coeffs))
end
