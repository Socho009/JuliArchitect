using Test
using Statistics
using LinearAlgebra
using Random

# Include source files directly for testing
include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "config.jl"))
include(joinpath(@__DIR__, "..", "src", "layers.jl"))

@testset "LayerNorm" begin

    @testset "Construction and type parameterization" begin
        ln32 = LayerNorm{Float32}(64)
        @test ln32 isa LayerNorm{Float32}
        @test ln32 isa AbstractLayer{Float32}
        @test length(ln32.weight) == 64
        @test length(ln32.bias) == 64
        @test all(ln32.weight .== 1.0f0)
        @test all(ln32.bias .== 0.0f0)
        @test ln32.eps == Float32(1e-5)

        ln64 = LayerNorm{Float64}(128)
        @test ln64 isa LayerNorm{Float64}
        @test ln64 isa AbstractLayer{Float64}
        @test ln64.weight isa Vector{Float64}
        @test ln64.bias isa Vector{Float64}
    end

    @testset "Custom epsilon" begin
        ln = LayerNorm{Float32}(32; eps=1e-6)
        @test ln.eps == Float32(1e-6)
    end

    @testset "Convenience constructor defaults to Float32" begin
        ln = LayerNorm(48)
        @test ln isa LayerNorm{Float32}
    end

    @testset "Forward pass 3D - normalization correctness" begin
        n_embd, T, B = 48, 10, 2
        ln = LayerNorm{Float32}(n_embd)
        Random.seed!(42)
        x = randn(Float32, n_embd, T, B)

        y = ln(x)
        @test size(y) == (n_embd, T, B)
        @test eltype(y) == Float32

        # With default weight=1, bias=0, output should be approximately
        # zero-mean and unit-variance along embedding dimension
        for b in 1:B
            for t in 1:T
                col = y[:, t, b]
                @test abs(mean(col)) < 1e-5
                @test abs(std(col; corrected=false) - 1.0) < 1e-4
            end
        end
    end

    @testset "Forward pass 2D" begin
        n_embd, T = 32, 5
        ln = LayerNorm{Float32}(n_embd)
        Random.seed!(123)
        x = randn(Float32, n_embd, T)

        y = ln(x)
        @test size(y) == (n_embd, T)
        @test eltype(y) == Float32

        for t in 1:T
            col = y[:, t]
            @test abs(mean(col)) < 1e-5
            @test abs(std(col; corrected=false) - 1.0) < 1e-4
        end
    end

    @testset "Learnable parameters affect output" begin
        n_embd = 16
        ln = LayerNorm{Float32}(n_embd)
        Random.seed!(7)
        x = randn(Float32, n_embd, 4, 2)

        y_default = ln(x)

        # Modify scale
        ln.weight .= 2.0f0
        y_scaled = ln(x)
        @test !isapprox(y_default, y_scaled)

        # Modify bias
        ln.weight .= 1.0f0
        ln.bias .= 0.5f0
        y_biased = ln(x)
        @test isapprox(y_biased, y_default .+ 0.5f0; atol=1e-6)
    end

    @testset "Does not mutate input" begin
        n_embd = 32
        ln = LayerNorm{Float32}(n_embd)
        x = randn(Float32, n_embd, 5, 2)
        x_copy = copy(x)
        _ = ln(x)
        @test x == x_copy
    end

end

@testset "Linear" begin

    @testset "Construction with bias" begin
        l = Linear{Float32}(64, 128; bias=true)
        @test l isa Linear{Float32, true}
        @test l isa AbstractLayer{Float32}
        @test size(l.weight) == (128, 64)
        @test length(l.bias) == 128
        @test all(l.bias .== 0.0f0)
        @test has_bias(l) == true
    end

    @testset "Construction without bias" begin
        l = Linear{Float32}(64, 128; bias=false)
        @test l isa Linear{Float32, false}
        @test l isa AbstractLayer{Float32}
        @test size(l.weight) == (128, 64)
        @test has_bias(l) == false
    end

    @testset "Type parameterization" begin
        l64 = Linear{Float64}(32, 16)
        @test l64 isa Linear{Float64, true}
        @test l64.weight isa Matrix{Float64}
        @test l64.bias isa Vector{Float64}
    end

    @testset "Convenience constructor defaults to Float32" begin
        l = Linear(32, 16)
        @test l isa Linear{Float32, true}
    end

    @testset "Forward pass 1D with bias" begin
        l = Linear{Float32}(4, 3; bias=true)
        l.weight .= Float32[1 0 0 0; 0 1 0 0; 0 0 1 0]
        l.bias .= Float32[0.1, 0.2, 0.3]

        x = Float32[1.0, 2.0, 3.0, 4.0]
        y = l(x)
        @test size(y) == (3,)
        @test isapprox(y, Float32[1.1, 2.2, 3.3]; atol=1e-6)
    end

    @testset "Forward pass 1D without bias" begin
        l = Linear{Float32}(4, 3; bias=false)
        l.weight .= Float32[1 0 0 0; 0 1 0 0; 0 0 1 0]

        x = Float32[1.0, 2.0, 3.0, 4.0]
        y = l(x)
        @test size(y) == (3,)
        @test isapprox(y, Float32[1.0, 2.0, 3.0]; atol=1e-6)
    end

    @testset "Forward pass 2D with bias" begin
        l = Linear{Float32}(4, 3; bias=true)
        l.weight .= Float32[1 0 0 0; 0 1 0 0; 0 0 1 0]
        l.bias .= Float32[0.5, 0.5, 0.5]

        x = Float32[1 5; 2 6; 3 7; 4 8]  # (4, 2)
        y = l(x)
        @test size(y) == (3, 2)
        @test isapprox(y[:, 1], Float32[1.5, 2.5, 3.5]; atol=1e-6)
        @test isapprox(y[:, 2], Float32[5.5, 6.5, 7.5]; atol=1e-6)
    end

    @testset "Forward pass 3D with bias" begin
        in_dim, out_dim = 8, 16
        T, B = 5, 3
        l = Linear{Float32}(in_dim, out_dim; bias=true)
        Random.seed!(99)
        x = randn(Float32, in_dim, T, B)

        y = l(x)
        @test size(y) == (out_dim, T, B)
        @test eltype(y) == Float32

        # Verify equivalence with manual 2D computation
        x_2d = reshape(x, in_dim, T * B)
        y_expected = l.weight * x_2d .+ l.bias
        y_expected_3d = reshape(y_expected, out_dim, T, B)
        @test isapprox(y, y_expected_3d; atol=1e-5)
    end

    @testset "Forward pass 3D without bias" begin
        in_dim, out_dim = 8, 16
        T, B = 5, 3
        l = Linear{Float32}(in_dim, out_dim; bias=false)
        Random.seed!(99)
        x = randn(Float32, in_dim, T, B)

        y = l(x)
        @test size(y) == (out_dim, T, B)

        # Verify equivalence with manual 2D computation
        x_2d = reshape(x, in_dim, T * B)
        y_expected = l.weight * x_2d
        y_expected_3d = reshape(y_expected, out_dim, T, B)
        @test isapprox(y, y_expected_3d; atol=1e-5)
    end

    @testset "Does not mutate input" begin
        l = Linear{Float32}(8, 16)
        x = randn(Float32, 8, 5, 2)
        x_copy = copy(x)
        _ = l(x)
        @test x == x_copy
    end

end
