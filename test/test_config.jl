using Test

using Architect

@testset "GPTConfig" begin

    @testset "Requirement 1.1: Precision type propagation" begin
        # Float32
        config32 = GPTConfig{Float32}(model_type=:gpt_nano)
        @test config32.embd_pdrop isa Float32
        @test config32.resid_pdrop isa Float32
        @test config32.attn_pdrop isa Float32

        # Float64
        config64 = GPTConfig{Float64}(model_type=:gpt_nano)
        @test config64.embd_pdrop isa Float64
        @test config64.resid_pdrop isa Float64
        @test config64.attn_pdrop isa Float64

        # Float16
        config16 = GPTConfig{Float16}(model_type=:gpt_nano)
        @test config16.embd_pdrop isa Float16
        @test config16.resid_pdrop isa Float16
        @test config16.attn_pdrop isa Float16
    end

    @testset "Requirement 1.2: Model presets" begin
        # :gpt_nano
        c = GPTConfig{Float32}(model_type=:gpt_nano)
        @test c.n_layer == 3
        @test c.n_head == 3
        @test c.n_embd == 48

        # :gpt_micro
        c = GPTConfig{Float32}(model_type=:gpt_micro)
        @test c.n_layer == 4
        @test c.n_head == 4
        @test c.n_embd == 128

        # :gpt_mini
        c = GPTConfig{Float32}(model_type=:gpt_mini)
        @test c.n_layer == 6
        @test c.n_head == 6
        @test c.n_embd == 192

        # :gpt2
        c = GPTConfig{Float32}(model_type=:gpt2)
        @test c.n_layer == 12
        @test c.n_head == 12
        @test c.n_embd == 768

        # :gpt2_medium
        c = GPTConfig{Float32}(model_type=:gpt2_medium)
        @test c.n_layer == 24
        @test c.n_head == 16
        @test c.n_embd == 1024

        # :gpt2_large
        c = GPTConfig{Float32}(model_type=:gpt2_large)
        @test c.n_layer == 36
        @test c.n_head == 20
        @test c.n_embd == 1280

        # :gpt2_xl
        c = GPTConfig{Float32}(model_type=:gpt2_xl)
        @test c.n_layer == 48
        @test c.n_head == 25
        @test c.n_embd == 1600
    end

    @testset "Requirement 1.3: Keyword override merging" begin
        # Override n_embd from preset
        c = GPTConfig{Float32}(model_type=:gpt2, n_embd=512)
        @test c.n_layer == 12   # from preset
        @test c.n_head == 12    # from preset
        @test c.n_embd == 512   # overridden

        # Override vocab_size
        c = GPTConfig{Float32}(model_type=:gpt_nano, vocab_size=32000)
        @test c.vocab_size == 32000
        @test c.n_layer == 3    # from preset

        # Override dropout rates
        c = GPTConfig{Float32}(model_type=:gpt_nano, embd_pdrop=0.2, attn_pdrop=0.3)
        @test c.embd_pdrop == Float32(0.2)
        @test c.attn_pdrop == Float32(0.3)
        @test c.resid_pdrop == Float32(0.1)  # default retained
    end

    @testset "Requirement 1.4: Immutability" begin
        c = GPTConfig{Float32}(model_type=:gpt_nano)
        # GPTConfig is a struct (not mutable struct), so setproperty! should error
        @test !ismutable(c)
    end

    @testset "Requirement 1.5: Default to Float32" begin
        c = GPTConfig(model_type=:gpt_nano)
        @test c isa GPTConfig{Float32}
        @test c.embd_pdrop isa Float32
    end

    @testset "Requirement 1.6: Default values" begin
        c = GPTConfig{Float32}(n_layer=6, n_head=6, n_embd=192)
        @test c.vocab_size == 50257
        @test c.block_size == 1024
        @test c.embd_pdrop == Float32(0.1)
        @test c.resid_pdrop == Float32(0.1)
        @test c.attn_pdrop == Float32(0.1)
    end

    @testset "Requirement 1.7: Error for unrecognized preset" begin
        @test_throws ErrorException GPTConfig{Float32}(model_type=:gpt3)
        @test_throws ErrorException GPTConfig{Float32}(model_type=:unknown)
    end

    @testset "Abstract type hierarchy" begin
        c = GPTConfig{Float32}(model_type=:gpt_nano)
        @test c isa AbstractGPTConfig{Float32}
        @test c isa AbstractGPTConfig
    end

    @testset "Custom config without preset" begin
        c = GPTConfig{Float32}(n_layer=8, n_head=8, n_embd=256, vocab_size=10000, block_size=512)
        @test c.n_layer == 8
        @test c.n_head == 8
        @test c.n_embd == 256
        @test c.vocab_size == 10000
        @test c.block_size == 512
    end

    @testset "Error when architecture params missing" begin
        # No preset and no explicit n_layer/n_head/n_embd
        @test_throws ErrorException GPTConfig{Float32}(vocab_size=1000)
    end

end
