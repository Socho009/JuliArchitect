using Test
using Statistics
using LinearAlgebra
using Random

# Include source files directly for testing
include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "config.jl"))
include(joinpath(@__DIR__, "..", "src", "layers.jl"))
include(joinpath(@__DIR__, "..", "src", "attention.jl"))

@testset "CausalSelfAttention" begin

    @testset "Construction and type hierarchy" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        attn = CausalSelfAttention{Float32}(config)

        @test attn isa CausalSelfAttention{Float32}
        @test attn isa AbstractAttention{Float32}
        @test attn isa AbstractLayer{Float32}
        @test attn.n_head == 3
        @test attn.head_dim == 16  # 48 ÷ 3
        @test attn.attn_pdrop == Float32(0.1)
        @test attn.resid_pdrop == Float32(0.1)
    end

    @testset "Convenience constructor" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        attn = CausalSelfAttention(config)
        @test attn isa CausalSelfAttention{Float32}
    end

    @testset "Causal mask is lower-triangular" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        attn = CausalSelfAttention{Float32}(config)

        mask = attn.mask
        block_size = config.block_size
        @test size(mask) == (block_size, block_size)

        # Lower triangular: mask[i,j] = true iff j ≤ i
        for i in 1:min(10, block_size)
            for j in 1:min(10, block_size)
                if j <= i
                    @test mask[i, j] == true
                else
                    @test mask[i, j] == false
                end
            end
        end
    end

    @testset "W_qkv and W_out dimensions" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        attn = CausalSelfAttention{Float32}(config)

        # W_qkv: (n_embd) → (3*n_embd) means weight is (3*n_embd, n_embd)
        @test size(attn.W_qkv.weight) == (3 * 48, 48)
        @test length(attn.W_qkv.bias) == 3 * 48

        # W_out: (n_embd) → (n_embd) means weight is (n_embd, n_embd)
        @test size(attn.W_out.weight) == (48, 48)
        @test length(attn.W_out.bias) == 48
    end

    @testset "Forward pass output shape" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        attn = CausalSelfAttention{Float32}(config)

        n_embd = config.n_embd  # 48
        T, B = 8, 2
        Random.seed!(42)
        x = randn(Float32, n_embd, T, B)

        y = attn(x; training=false)
        @test size(y) == (n_embd, T, B)
        @test eltype(y) == Float32
    end

    @testset "Forward pass preserves shape for various seq lengths" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        attn = CausalSelfAttention{Float32}(config)

        n_embd = config.n_embd
        for T in [1, 4, 16, 32]
            for B in [1, 3]
                x = randn(Float32, n_embd, T, B)
                y = attn(x; training=false)
                @test size(y) == (n_embd, T, B)
            end
        end
    end

    @testset "Does not mutate input" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        attn = CausalSelfAttention{Float32}(config)

        Random.seed!(123)
        x = randn(Float32, 48, 8, 2)
        x_copy = copy(x)

        _ = attn(x; training=false)
        @test x == x_copy
    end

    @testset "Attention weights are valid probability distributions" begin
        # Test the internal attention computation produces valid distributions
        config = GPTConfig{Float32}(model_type=:gpt_nano, attn_pdrop=0.0, resid_pdrop=0.0)
        attn = CausalSelfAttention{Float32}(config)

        n_embd = config.n_embd
        T, B = 6, 2
        Random.seed!(42)
        x = randn(Float32, n_embd, T, B)

        # Manually compute attention weights to verify properties
        n_head = attn.n_head
        head_dim = attn.head_dim

        qkv = attn.W_qkv(x)
        Q = reshape(qkv[1:n_embd, :, :], head_dim, T, n_head, B)
        K = reshape(qkv[n_embd+1:2*n_embd, :, :], head_dim, T, n_head, B)

        scale = Float32(1.0 / sqrt(head_dim))

        # Compute attention scores manually for one head/batch
        for b in 1:B
            for h in 1:n_head
                Q_slice = Q[:, :, h, b]  # (head_dim, T)
                K_slice = K[:, :, h, b]  # (head_dim, T)
                scores = Q_slice' * K_slice .* scale  # (T, T)

                # Apply causal mask
                causal = attn.mask[1:T, 1:T]
                for j in 1:T
                    for i in 1:T
                        if !causal[i, j]
                            scores[i, j] = -Inf32
                        end
                    end
                end

                # Softmax along dim 2 (key dimension)
                for i in 1:T
                    row = scores[i, :]
                    max_val = maximum(row)
                    exp_row = exp.(row .- max_val)
                    scores[i, :] = exp_row ./ sum(exp_row)
                end

                # Check: non-negative
                @test all(scores .>= 0.0f0)

                # Check: rows sum to 1
                for i in 1:T
                    @test isapprox(sum(scores[i, :]), 1.0f0; atol=1e-5)
                end

                # Check: future positions are zero (causal)
                for i in 1:T
                    for j in (i+1):T
                        @test scores[i, j] ≈ 0.0f0 atol=1e-7
                    end
                end
            end
        end
    end

    @testset "Float64 precision" begin
        config = GPTConfig{Float64}(model_type=:gpt_nano)
        attn = CausalSelfAttention{Float64}(config)

        x = randn(Float64, 48, 8, 2)
        y = attn(x; training=false)
        @test size(y) == (48, 8, 2)
        @test eltype(y) == Float64
    end

    @testset "Dropout disabled in eval mode" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano, attn_pdrop=0.5, resid_pdrop=0.5)
        attn = CausalSelfAttention{Float32}(config)

        Random.seed!(42)
        x = randn(Float32, 48, 8, 2)

        # With training=false, dropout should be disabled → deterministic output
        y1 = attn(x; training=false)
        y2 = attn(x; training=false)
        @test y1 ≈ y2
    end

    @testset "Dropout active in training mode produces different outputs" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano, attn_pdrop=0.5, resid_pdrop=0.5)
        attn = CausalSelfAttention{Float32}(config)

        x = randn(Float32, 48, 8, 2)

        # With training=true and high dropout, outputs should differ
        Random.seed!(1)
        y1 = attn(x; training=true)
        Random.seed!(2)
        y2 = attn(x; training=true)
        @test !isapprox(y1, y2; atol=1e-6)
    end

    @testset "Single token sequence" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano, attn_pdrop=0.0, resid_pdrop=0.0)
        attn = CausalSelfAttention{Float32}(config)

        # T=1: single token should attend only to itself
        x = randn(Float32, 48, 1, 1)
        y = attn(x; training=false)
        @test size(y) == (48, 1, 1)
        @test all(isfinite.(y))
    end

    @testset "n_embd not divisible by n_head raises error" begin
        # Create a config where n_embd % n_head != 0
        # We need to bypass the config constructor, so test the assertion directly
        @test_throws AssertionError CausalSelfAttention{Float32}(
            GPTConfig{Float32}(n_layer=3, n_head=5, n_embd=48)  # 48 % 5 != 0
        )
    end

end
