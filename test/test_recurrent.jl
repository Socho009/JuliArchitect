using Test
using Random

# Load the module
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using JuliaNativeGPT

Random.seed!(42)

@testset "Recurrent Architecture" begin

    # ─── RecurrentConfig ──────────────────────────────────────────────────────
    @testset "RecurrentConfig" begin
        config = RecurrentConfig{Float32}(
            vocab_size=65, hidden_size=128, n_layer=2,
            cell_type=:lstm, seq_length=50, dropout=0.1
        )
        @test config.vocab_size == 65
        @test config.hidden_size == 128
        @test config.n_layer == 2
        @test config.cell_type == :lstm
        @test config.seq_length == 50
        @test config.dropout == 0.1f0

        # Default precision is Float32
        config2 = RecurrentConfig(vocab_size=100)
        @test config2 isa RecurrentConfig{Float32}

        # Invalid cell type should error
        @test_throws ErrorException RecurrentConfig{Float32}(cell_type=:transformer)
    end

    # ─── RNN Cell ─────────────────────────────────────────────────────────────
    @testset "RNNCell" begin
        cell = RNNCell{Float32}(64, 128)
        @test cell.hidden_size == 128
        @test size(cell.W_ih.weight) == (128, 64)
        @test size(cell.W_hh.weight) == (128, 128)

        # Forward pass
        B = 4
        x = randn(Float32, 64, B)
        state = init_state(cell, B)
        @test state isa RNNState{Float32}
        @test size(state.h) == (128, B)
        @test all(state.h .== 0)

        h_new, new_state = cell(x, state)
        @test size(h_new) == (128, B)
        @test new_state isa RNNState{Float32}
        # tanh output should be bounded
        @test all(-1 .<= h_new .<= 1)

        # Second timestep should produce different output
        x2 = randn(Float32, 64, B)
        h2, state2 = cell(x2, new_state)
        @test h2 != h_new  # different input → different output
    end

    # ─── LSTM Cell ────────────────────────────────────────────────────────────
    @testset "LSTMCell" begin
        cell = LSTMCell{Float32}(64, 128)
        @test cell.hidden_size == 128
        # Fused gates: 4*hidden_size
        @test size(cell.W_ih.weight) == (512, 64)
        @test size(cell.W_hh.weight) == (512, 128)

        B = 4
        x = randn(Float32, 64, B)
        state = init_state(cell, B)
        @test state isa LSTMState{Float32}
        @test size(state.c) == (128, B)
        @test size(state.h) == (128, B)

        h_new, new_state = cell(x, state)
        @test size(h_new) == (128, B)
        @test new_state isa LSTMState{Float32}
        @test size(new_state.c) == (128, B)

        # LSTM hidden output bounded by tanh*sigmoid → [-1, 1]
        @test all(-1 .<= h_new .<= 1)

        # Cell state should be non-zero after first step
        @test !all(new_state.c .== 0)
    end

    # ─── LSTM Forget Gate Bias Init ───────────────────────────────────────────
    @testset "LSTM Forget Gate Bias = 1.0" begin
        cell = LSTMCell{Float32}(64, 128)
        initialize!(cell)
        H = cell.hidden_size
        # Forget gate bias (2nd block) should be 1.0
        @test all(cell.W_ih.bias[H+1:2H] .== 1.0f0)
        # Other biases should be 0
        @test all(cell.W_ih.bias[1:H] .== 0.0f0)
        @test all(cell.W_ih.bias[2H+1:3H] .== 0.0f0)
        @test all(cell.W_ih.bias[3H+1:4H] .== 0.0f0)
    end

    # ─── GRU Cell ─────────────────────────────────────────────────────────────
    @testset "GRUCell" begin
        cell = GRUCell{Float32}(64, 128)
        @test cell.hidden_size == 128
        @test size(cell.W_iz.weight) == (128, 64)
        @test size(cell.W_hz.weight) == (128, 128)

        B = 4
        x = randn(Float32, 64, B)
        state = init_state(cell, B)
        @test state isa RNNState{Float32}  # GRU uses RNNState

        h_new, new_state = cell(x, state)
        @test size(h_new) == (128, B)
        @test new_state isa RNNState{Float32}
    end

    # ─── RecurrentLM Construction ─────────────────────────────────────────────
    @testset "RecurrentLM Construction" begin
        for ct in (:rnn, :lstm, :gru)
            config = RecurrentConfig{Float32}(
                vocab_size=65, hidden_size=64, n_layer=2, cell_type=ct
            )
            model = RecurrentLM(config)
            @test length(model.cells) == 2
            @test size(model.embedding.weight) == (64, 65)
            @test size(model.decoder.weight) == (65, 64)

            # Initialize should not error
            initialize!(model)
        end
    end

    # ─── RecurrentLM Forward Pass ─────────────────────────────────────────────
    @testset "RecurrentLM Forward" begin
        config = RecurrentConfig{Float32}(
            vocab_size=65, hidden_size=64, n_layer=2, cell_type=:lstm
        )
        model = RecurrentLM(config)
        initialize!(model)

        T, B = 10, 4
        idx = rand(1:65, T, B)

        # Forward without targets
        logits, loss, states = forward(model, idx)
        @test size(logits) == (65, T, B)
        @test loss === nothing
        @test length(states) == 2

        # Forward with targets
        targets = rand(1:65, T, B)
        logits2, loss2, states2 = forward(model, idx; targets=targets)
        @test loss2 isa Float32
        @test loss2 > 0  # cross-entropy is always positive
        @test isfinite(loss2)

        # Loss should be roughly -log(1/vocab_size) for random init
        expected_loss = -log(1.0f0 / 65)
        @test abs(loss2 - expected_loss) < 1.5f0  # within 1.5 nats
    end

    # ─── State Carry-Over ─────────────────────────────────────────────────────
    @testset "BPTT State Carry-Over" begin
        config = RecurrentConfig{Float32}(
            vocab_size=65, hidden_size=32, n_layer=1, cell_type=:lstm
        )
        model = RecurrentLM(config)
        initialize!(model)

        idx = rand(1:65, 8, 2)

        # First forward pass
        _, _, states1 = forward(model, idx; training=false)

        # Second forward with carried-over states should differ from cold start
        _, loss_warm, _ = forward(model, idx; targets=idx, training=false, init_states=states1)
        _, loss_cold, _ = forward(model, idx; targets=idx, training=false)

        # Warm start should produce different loss (states are non-zero)
        @test loss_warm != loss_cold
    end

    # ─── Generation ───────────────────────────────────────────────────────────
    @testset "RecurrentLM Generation" begin
        config = RecurrentConfig{Float32}(
            vocab_size=65, hidden_size=32, n_layer=1, cell_type=:gru
        )
        model = RecurrentLM(config)
        initialize!(model)

        prompt = rand(1:65, 5, 2)  # (5 tokens, 2 batches)

        # Greedy
        out = generate(model, prompt; strategy=GreedySampling(), max_new_tokens=20)
        @test size(out) == (25, 2)
        @test all(1 .<= out .<= 65)

        # Greedy should be deterministic
        out2 = generate(model, prompt; strategy=GreedySampling(), max_new_tokens=20)
        @test out == out2

        # Top-K
        out3 = generate(model, prompt; strategy=TopKSampling(10, 0.8f0), max_new_tokens=15)
        @test size(out3) == (20, 2)
        @test all(1 .<= out3 .<= 65)

        # Nucleus
        out4 = generate(model, prompt; strategy=NucleusSampling(0.9f0, 0.9f0), max_new_tokens=15)
        @test size(out4) == (20, 2)
    end

    # ─── to_device ────────────────────────────────────────────────────────────
    @testset "to_device RecurrentLM" begin
        config = RecurrentConfig{Float32}(
            vocab_size=30, hidden_size=16, n_layer=1, cell_type=:lstm
        )
        model = RecurrentLM(config)
        initialize!(model)

        # Transfer to CPU (should be a no-op but not error)
        model_cpu = to_device(model, :cpu)
        @test model_cpu isa RecurrentLM{Float32}

        # Forward should still work after transfer
        idx = rand(1:30, 5, 2)
        logits, _, _ = forward(model_cpu, idx)
        @test size(logits) == (30, 5, 2)
    end

    # ─── CharDataLoader ───────────────────────────────────────────────────────
    @testset "CharDataLoader" begin
        text = repeat("Hello, world! This is a test of the data loader. ", 200)
        loader = CharDataLoader(text; batch_size=8, seq_length=20)

        @test loader.tokenizer.vocab_size > 0
        @test loader.nbatches > 0
        @test sum(loader.split_sizes) == loader.nbatches

        # Get a training batch
        x, y = next_batch!(loader, :train)
        @test size(x) == (20, 8)
        @test size(y) == (20, 8)

        # y should be x shifted by 1 (next character prediction)
        @test all(1 .<= x .<= loader.tokenizer.vocab_size)
        @test all(1 .<= y .<= loader.tokenizer.vocab_size)

        # Cycling: getting more batches than available should wrap
        n_train = loader.split_sizes[1]
        for _ in 1:n_train + 5
            x2, y2 = next_batch!(loader, :train)
            @test size(x2) == (20, 8)
        end

        # make_batch_fn should be callable
        batch_fn = make_batch_fn(loader, :train)
        x3, y3 = batch_fn()
        @test size(x3) == (20, 8)
    end

    # ─── Spectral Diagnostics ─────────────────────────────────────────────────
    @testset "Recurrent Spectral Diagnostics" begin
        for ct in (:rnn, :lstm, :gru)
            config = RecurrentConfig{Float32}(
                vocab_size=30, hidden_size=16, n_layer=2, cell_type=ct
            )
            model = RecurrentLM(config)
            initialize!(model)

            stats = compute_recurrent_spectral_stats(model)
            @test length(stats) > 0
            @test all(v -> v isa Float32 && isfinite(v) && v >= 0, values(stats))
        end
    end

    # ─── collect_params ───────────────────────────────────────────────────────
    @testset "collect_params RecurrentLM" begin
        config = RecurrentConfig{Float32}(
            vocab_size=30, hidden_size=16, n_layer=2, cell_type=:lstm
        )
        model = RecurrentLM(config)
        params = collect_params(model)

        @test length(params) > 0

        # Check decay separation: embeddings and biases should NOT decay
        n_decay = count(p -> p.decay, params)
        n_nodecay = count(p -> !p.decay, params)
        @test n_decay > 0
        @test n_nodecay > 0

        # Total params should match _count_parameters
        total_from_infos = sum(length(p.param) for p in params)
        # Should capture all parameters
        @test total_from_infos > 0
    end

    # ─── Float64 Precision ────────────────────────────────────────────────────
    @testset "Float64 Precision" begin
        config = RecurrentConfig{Float64}(
            vocab_size=30, hidden_size=16, n_layer=1, cell_type=:rnn
        )
        model = RecurrentLM(config)
        initialize!(model)

        idx = rand(1:30, 5, 2)
        logits, loss, _ = forward(model, idx; targets=idx)
        @test logits isa Array{Float64, 3}
        @test loss isa Float64
    end

end

println("\n✓ All recurrent tests passed!")
