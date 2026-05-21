using Test
using JuliaNativeGPT

@testset "GPTModel" begin
    @testset "Construction with gpt_nano" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        model = GPTModel(config)

        # Verify struct fields exist and have correct shapes
        @test size(model.tok_emb.weight) == (48, 50257)
        @test size(model.pos_emb.weight) == (48, 1024)
        @test length(model.blocks) == 3
        @test length(model.ln_f.weight) == 48
        @test length(model.ln_f.bias) == 48
        @test size(model.lm_head.weight) == (50257, 48)
        @test has_bias(model.lm_head) == false
        @test model.embd_pdrop == Float32(0.1)
        @test model.config === config
    end

    @testset "Construction with gpt_micro" begin
        config = GPTConfig{Float32}(model_type=:gpt_micro)
        model = GPTModel(config)

        @test size(model.tok_emb.weight) == (128, 50257)
        @test size(model.pos_emb.weight) == (128, 1024)
        @test length(model.blocks) == 4
        @test size(model.lm_head.weight) == (50257, 128)
    end

    @testset "Float64 precision" begin
        config = GPTConfig{Float64}(model_type=:gpt_nano)
        model = GPTModel(config)

        @test typeof(model.embd_pdrop) == Float64
        @test eltype(model.tok_emb.weight) == Float64
        @test eltype(model.pos_emb.weight) == Float64
        @test eltype(model.ln_f.weight) == Float64
        @test eltype(model.lm_head.weight) == Float64
    end

    @testset "Validation: n_embd % n_head != 0" begin
        bad_config = GPTConfig{Float32}(n_layer=3, n_head=5, n_embd=48)
        @test_throws AssertionError GPTModel(bad_config)

        bad_config2 = GPTConfig{Float32}(n_layer=2, n_head=7, n_embd=64)
        @test_throws AssertionError GPTModel(bad_config2)
    end

    @testset "Convenience constructor" begin
        config = GPTConfig(model_type=:gpt_nano)
        model = GPTModel(config)
        @test typeof(model.embd_pdrop) == Float32
    end

    @testset "Custom config" begin
        config = GPTConfig{Float32}(
            n_layer=2, n_head=4, n_embd=64,
            vocab_size=1000, block_size=128,
            embd_pdrop=0.2
        )
        model = GPTModel(config)

        @test size(model.tok_emb.weight) == (64, 1000)
        @test size(model.pos_emb.weight) == (64, 128)
        @test length(model.blocks) == 2
        @test size(model.lm_head.weight) == (1000, 64)
        @test model.embd_pdrop == Float32(0.2)
    end

    @testset "Blocks are TransformerBlocks" begin
        config = GPTConfig{Float32}(model_type=:gpt_nano)
        model = GPTModel(config)

        for block in model.blocks
            @test block isa TransformerBlock{Float32}
        end
    end
end
