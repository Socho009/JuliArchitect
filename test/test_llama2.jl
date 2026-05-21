using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using JuliaNativeGPT

Random.seed!(42)

# Helper: build a small Llama2 transformer with random weights for testing
function make_test_transformer(; P=Float32, dim=64, hidden_dim=128, n_layers=2,
                                 n_heads=4, n_kv_heads=2, vocab_size=32, seq_len=64)
    config = Llama2Config(Int32(dim), Int32(hidden_dim), Int32(n_layers),
                          Int32(n_heads), Int32(n_kv_heads), Int32(vocab_size), Int32(seq_len))
    hs = dim ÷ n_heads
    kv_dim = (dim * n_kv_heads) ÷ n_heads
    w = Llama2Weights{P}(
        randn(P, dim, vocab_size), ones(P, dim, n_layers),
        randn(P, dim, n_heads*hs, n_layers)*P(0.02),
        randn(P, dim, kv_dim, n_layers)*P(0.02),
        randn(P, dim, kv_dim, n_layers)*P(0.02),
        randn(P, n_heads*hs, dim, n_layers)*P(0.02),
        ones(P, dim, n_layers),
        randn(P, dim, hidden_dim, n_layers)*P(0.02),
        randn(P, hidden_dim, dim, n_layers)*P(0.02),
        randn(P, dim, hidden_dim, n_layers)*P(0.02),
        ones(P, dim), randn(P, dim, vocab_size)*P(0.02),
    )
    state = Llama2State{P}(config)
    Llama2Transformer{P}(config, w, state)
end

@testset "Llama 2 Inference" begin

    @testset "Llama2Config" begin
        c = Llama2Config(Int32(256), Int32(512), Int32(4), Int32(8), Int32(4), Int32(32000), Int32(512))
        @test c.dim == 256
        @test c.n_kv_heads == 4
        @test c.vocab_size == 32000
    end

    @testset "Forward Pass" begin
        t = make_test_transformer()
        logits = copy(JuliaNativeGPT._llama_forward!(t, 1, 0))
        @test length(logits) == 32
        @test all(isfinite, logits)

        # Second position on same transformer
        logits2 = JuliaNativeGPT._llama_forward!(t, 5, 1)
        @test length(logits2) == 32
        @test logits2 != logits  # different token+position → different output
    end

    @testset "GQA (n_kv_heads < n_heads)" begin
        # 8 query heads, 2 KV heads → kv_mul = 4
        t = make_test_transformer(n_heads=8, n_kv_heads=2)
        logits = JuliaNativeGPT._llama_forward!(t, 1, 0)
        @test all(isfinite, logits)
    end

    @testset "Full MHA (n_kv_heads == n_heads)" begin
        t = make_test_transformer(n_heads=4, n_kv_heads=4)
        logits = JuliaNativeGPT._llama_forward!(t, 1, 0)
        @test all(isfinite, logits)
    end

    @testset "Multi-step Causal" begin
        t = make_test_transformer()
        logits_seq = []
        for pos in 0:9
            l = JuliaNativeGPT._llama_forward!(t, rand(1:32), pos)
            push!(logits_seq, copy(l))
        end
        @test length(logits_seq) == 10
        @test all(l -> all(isfinite, l), logits_seq)
    end

    @testset "Generation — Greedy" begin
        t = make_test_transformer()
        out = generate(t, [1, 5, 10]; strategy=GreedySampling(), max_tokens=20)
        @test length(out) >= 3
        @test all(1 .<= out .<= 32)

        # Greedy is deterministic — same weights, reset state
        t.state.key_cache .= 0
        t.state.value_cache .= 0
        out2 = generate(t, [1, 5, 10]; strategy=GreedySampling(), max_tokens=20)
        @test out == out2
    end

    @testset "Generation — TopK" begin
        t = make_test_transformer()
        out = generate(t, [1]; strategy=TopKSampling(5, 0.8f0), max_tokens=30)
        @test length(out) >= 1
        @test all(1 .<= out .<= 32)
    end

    @testset "Generation — Nucleus" begin
        t = make_test_transformer()
        out = generate(t, [1]; strategy=NucleusSampling(0.9f0, 0.9f0), max_tokens=30)
        @test length(out) >= 1
        @test all(1 .<= out .<= 32)
    end

    @testset "Tokenizer" begin
        tok_path = joinpath(@__DIR__, "..", "llama2.c", "tokenizer.bin")
        if isfile(tok_path)
            tok = load_llama2_tokenizer(tok_path, 32000)
            @test tok.vocab_size == 32000
            @test tok.max_token_length > 0

            # Encode/decode round-trip
            tokens = encode(tok, "Hello")
            @test tokens[1] == 1  # BOS
            @test length(tokens) >= 2

            text = decode(tok, tokens)
            @test occursin("Hello", text)

            # Without BOS
            tokens_no_bos = encode(tok, "Hello"; bos=false)
            @test tokens_no_bos[1] != 1
        else
            @info "tokenizer.bin not found, skipping tokenizer tests"
        end
    end

    @testset "Float64 Precision" begin
        t = make_test_transformer(P=Float64)
        logits = JuliaNativeGPT._llama_forward!(t, 1, 0)
        @test eltype(logits) == Float64
        @test all(isfinite, logits)
    end
end

println("\n✓ All Llama 2 tests passed!")
