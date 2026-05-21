using Test
using JuliaNativeGPT
using Random

@testset "Training" begin
    @testset "TrainConfig construction" begin
        # Default Float32
        config = TrainConfig{Float32}()
        @test config.learning_rate == Float32(3e-4)
        @test config.weight_decay == Float32(0.1)
        @test config.max_iters == 1000
        @test config.grad_clip == Float32(1.0)
        @test config.batch_size == 64
        @test config.betas == (Float32(0.9), Float32(0.95))
        @test config.eps == Float32(1e-8)

        # Custom values
        config2 = TrainConfig{Float64}(
            learning_rate = 1e-3,
            weight_decay = 0.01,
            max_iters = 500,
            grad_clip = 0.5,
            batch_size = 32,
            betas = (0.9, 0.999),
            eps = 1e-7,
        )
        @test config2.learning_rate == 1e-3
        @test config2.weight_decay == 0.01
        @test config2.max_iters == 500
        @test config2.grad_clip == 0.5
        @test config2.batch_size == 32
        @test config2.betas == (0.9, 0.999)
        @test config2.eps == 1e-7

        # Convenience constructor defaults to Float32
        config3 = TrainConfig()
        @test typeof(config3.learning_rate) == Float32
    end

    @testset "ParamGroup dispatch" begin
        # Embedding → NoDecayGroup
        emb = Embedding{Float32}(16, 10)
        @test param_group(emb) isa NoDecayGroup

        # LayerNorm → NoDecayGroup
        ln = LayerNorm{Float32}(16)
        @test param_group(ln) isa NoDecayGroup

        # Linear → DecayGroup
        lin = Linear{Float32}(16, 32)
        @test param_group(lin) isa DecayGroup
    end

    @testset "Weight decay separation completeness" begin
        config = GPTConfig{Float32}(
            n_layer = 1, n_head = 2, n_embd = 4,
            vocab_size = 10, block_size = 8,
        )
        model = GPTModel(config)
        params = collect_params(model)

        # All decay params should be 2D (weight matrices)
        decay_params = filter(p -> p.decay, params)
        @test all(p -> ndims(p.param) == 2, decay_params)

        # No-decay group should include 1D params (biases, LN weights)
        no_decay_params = filter(p -> !p.decay, params)
        @test any(p -> ndims(p.param) == 1, no_decay_params)

        # Embeddings should be in no-decay group (they are 2D but no decay)
        @test any(p -> !p.decay && ndims(p.param) == 2, no_decay_params)

        # Total params should cover all model parameters
        @test length(params) > 0
        @test length(decay_params) > 0
        @test length(no_decay_params) > 0
    end

    @testset "Gradient norm clipping" begin
        # Create gradients with known norm
        grads = Any[ones(Float32, 3, 3) .* 2.0f0, ones(Float32, 4) .* 3.0f0]
        # norm = sqrt(9*4 + 4*9) = sqrt(72) ≈ 8.485
        expected_norm = sqrt(Float32(72))

        max_norm = 1.0f0
        original_norm = JuliaNativeGPT.clip_grad_norm!(grads, max_norm)

        @test original_norm ≈ expected_norm atol=1e-4
        # After clipping, norm should be ≤ max_norm
        clipped_norm = sqrt(sum(sum(abs2, g) for g in grads))
        @test clipped_norm ≤ max_norm + 1e-5

        # When norm is already below max, no clipping should occur
        grads2 = Any[ones(Float32, 2, 2) .* 0.1f0]
        original_norm2 = JuliaNativeGPT.clip_grad_norm!(grads2, 10.0f0)
        @test original_norm2 < 10.0f0
        # Gradients should be unchanged
        @test all(grads2[1] .≈ 0.1f0)
    end

    @testset "NaN/Inf loss halting" begin
        Random.seed!(123)
        config = GPTConfig{Float32}(
            n_layer = 1, n_head = 2, n_embd = 4,
            vocab_size = 10, block_size = 8,
            embd_pdrop = 0.0, resid_pdrop = 0.0, attn_pdrop = 0.0,
        )
        model = GPTModel(config)
        initialize!(model)

        # Corrupt weights to produce NaN loss
        model.tok_emb.weight .= NaN32

        function nan_batch()
            input = reshape(collect(1:8), 8, 1)
            target = reshape(collect(2:9), 8, 1)
            return input, target
        end

        train_config = TrainConfig{Float32}(
            learning_rate = 1f-3,
            max_iters = 10,
            grad_clip = 1.0f0,
            batch_size = 1,
        )

        err = nothing
        try
            train!(model, nan_batch, train_config)
        catch e
            err = e
        end

        @test err !== nothing
        @test err isa ErrorException
        @test occursin("iteration", err.msg)
        @test occursin("NaN", err.msg) || occursin("Inf", err.msg)
    end

    @testset "Training convergence on toy data" begin
        Random.seed!(42)
        # Use a very small model for fast testing
        config = GPTConfig{Float32}(
            n_layer = 1, n_head = 2, n_embd = 4,
            vocab_size = 10, block_size = 8,
            embd_pdrop = 0.0, resid_pdrop = 0.0, attn_pdrop = 0.0,
        )
        model = GPTModel(config)
        initialize!(model)

        # Simple memorizable pattern
        function toy_batch()
            input = reshape(collect(1:8), 8, 1)
            target = reshape(collect(2:9), 8, 1)
            return input, target
        end

        train_config = TrainConfig{Float32}(
            learning_rate = 1f-3,
            weight_decay = 0.0f0,
            max_iters = 30,
            grad_clip = 1.0f0,
            batch_size = 1,
            betas = (0.9f0, 0.99f0),
        )

        state = train!(model, toy_batch, train_config)

        # Loss should decrease
        @test state.losses[end] < state.losses[1]
        @test state.iter == 30
        @test length(state.losses) == 30
    end

    @testset "In-place parameter update (mutating convention)" begin
        Random.seed!(42)
        config = GPTConfig{Float32}(
            n_layer = 1, n_head = 2, n_embd = 4,
            vocab_size = 10, block_size = 8,
            embd_pdrop = 0.0, resid_pdrop = 0.0, attn_pdrop = 0.0,
        )
        model = GPTModel(config)
        initialize!(model)

        # Save initial weights
        initial_weight = copy(model.blocks[1].attention.W_qkv.weight)

        function simple_batch()
            input = reshape(collect(1:8), 8, 1)
            target = reshape(collect(2:9), 8, 1)
            return input, target
        end

        train_config = TrainConfig{Float32}(
            learning_rate = 1f-3,
            max_iters = 3,
            grad_clip = 1.0f0,
            batch_size = 1,
        )

        train!(model, simple_batch, train_config)

        # Weights should have changed (in-place update)
        @test model.blocks[1].attention.W_qkv.weight != initial_weight
    end

    @testset "AdamW weight decay separation" begin
        # Verify that weight decay is applied to decay group but not no-decay group
        # by directly testing the adamw_step! function
        Random.seed!(42)

        config = GPTConfig{Float32}(
            n_layer = 1, n_head = 2, n_embd = 4,
            vocab_size = 10, block_size = 8,
            embd_pdrop = 0.0, resid_pdrop = 0.0, attn_pdrop = 0.0,
        )
        model = GPTModel(config)
        initialize!(model)

        params = collect_params(model)

        # Find a decay param and a no-decay param
        decay_idx = findfirst(p -> p.decay, params)
        no_decay_idx = findfirst(p -> !p.decay && ndims(p.param) == 1, params)

        # Save initial values
        initial_decay = copy(params[decay_idx].param)
        initial_no_decay = copy(params[no_decay_idx].param)

        # Create zero gradients (so only weight decay affects the update)
        grads = [zeros(Float32, size(p.param)) for p in params]

        # Create optimizer state and step
        opt_state = JuliaNativeGPT.AdamWState(params)
        train_config = TrainConfig{Float32}(
            learning_rate = 0.1f0,  # Large LR to make effect visible
            weight_decay = 0.5f0,   # Large weight decay
            max_iters = 1,
            grad_clip = 1.0f0,
            batch_size = 1,
            betas = (0.9f0, 0.99f0),
        )

        JuliaNativeGPT.adamw_step!(params, grads, opt_state, train_config)

        # With zero gradients, only weight decay should affect decay params
        # Decay param: θ -= lr * (0 + wd * θ) = θ - lr*wd*θ = θ*(1 - lr*wd)
        expected_decay = initial_decay .* (1.0f0 - 0.1f0 * 0.5f0)
        @test params[decay_idx].param ≈ expected_decay atol=1e-6

        # No-decay param: θ -= lr * 0 = θ (unchanged)
        @test params[no_decay_idx].param ≈ initial_no_decay atol=1e-6
    end
end
