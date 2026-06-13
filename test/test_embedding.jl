using Test
using Random

using Architect

@testset "Embedding Layer" begin

    @testset "Construction and shape" begin
        # Float32 default
        emb = Embedding{Float32}(48, 100)
        @test size(emb.weight) == (48, 100)
        @test eltype(emb.weight) == Float32
        @test emb isa AbstractLayer{Float32}

        # Float64
        emb64 = Embedding{Float64}(128, 5000)
        @test size(emb64.weight) == (128, 5000)
        @test eltype(emb64.weight) == Float64
        @test emb64 isa AbstractLayer{Float64}

        # Float16
        emb16 = Embedding{Float16}(64, 256)
        @test size(emb16.weight) == (64, 256)
        @test eltype(emb16.weight) == Float16
        @test emb16 isa AbstractLayer{Float16}

        # Convenience constructor defaults to Float32
        emb_default = Embedding(48, 100)
        @test emb_default isa Embedding{Float32}
    end

    @testset "Column-major layout: weight is (n_embd, vocab_size)" begin
        emb = Embedding{Float32}(48, 100)
        # n_embd is first axis (rows) — each column is a token embedding
        @test size(emb.weight, 1) == 48   # n_embd
        @test size(emb.weight, 2) == 100  # vocab_size

        # Verify column access gives a contiguous vector
        col = emb.weight[:, 1]
        @test length(col) == 48
    end

    @testset "Vector lookup (1D indices)" begin
        Random.seed!(42)
        n_embd, vocab_size = 48, 100
        emb = Embedding{Float32}(n_embd, vocab_size)

        # Single index in a vector
        idx = [5]
        result = emb(idx)
        @test size(result) == (n_embd, 1)
        @test result[:, 1] == emb.weight[:, 5]

        # Multiple indices
        idx = [1, 10, 50, 100]
        result = emb(idx)
        @test size(result) == (n_embd, 4)
        for (i, token_id) in enumerate(idx)
            @test result[:, i] == emb.weight[:, token_id]
        end
    end

    @testset "Matrix lookup (2D batch)" begin
        Random.seed!(42)
        n_embd, vocab_size = 48, 100
        emb = Embedding{Float32}(n_embd, vocab_size)

        # (seq_len=3, batch=2)
        idx = [1 4; 2 5; 3 6]
        result = emb(idx)
        @test size(result) == (n_embd, 3, 2)

        # Verify each position matches the correct column
        @test result[:, 1, 1] == emb.weight[:, 1]
        @test result[:, 2, 1] == emb.weight[:, 2]
        @test result[:, 3, 1] == emb.weight[:, 3]
        @test result[:, 1, 2] == emb.weight[:, 4]
        @test result[:, 2, 2] == emb.weight[:, 5]
        @test result[:, 3, 2] == emb.weight[:, 6]
    end

    @testset "1-based indexing (Julia convention)" begin
        n_embd, vocab_size = 16, 10
        emb = Embedding{Float32}(n_embd, vocab_size)

        # First valid index is 1
        result = emb([1])
        @test result[:, 1] == emb.weight[:, 1]

        # Last valid index is vocab_size
        result = emb([vocab_size])
        @test result[:, 1] == emb.weight[:, vocab_size]

        # Index 0 should throw BoundsError
        @test_throws BoundsError emb([0])

        # Index > vocab_size should throw BoundsError
        @test_throws BoundsError emb([vocab_size + 1])
    end

    @testset "Embedding lookup consistency (Property 6)" begin
        # For any embedding and batch of valid token indices,
        # the result at position (t, b) equals the column indexed by idx[t, b]
        Random.seed!(123)
        n_embd, vocab_size = 64, 200
        emb = Embedding{Float32}(n_embd, vocab_size)

        seq_len, batch_size = 8, 4
        idx = rand(1:vocab_size, seq_len, batch_size)
        result = emb(idx)

        @test size(result) == (n_embd, seq_len, batch_size)
        for b in 1:batch_size
            for t in 1:seq_len
                @test result[:, t, b] == emb.weight[:, idx[t, b]]
            end
        end
    end

    @testset "Type hierarchy" begin
        emb = Embedding{Float32}(48, 100)
        @test emb isa Embedding{Float32}
        @test emb isa AbstractLayer{Float32}
        @test emb isa AbstractLayer
    end

    @testset "show method" begin
        emb = Embedding{Float32}(768, 50257)
        str = sprint(show, emb)
        @test contains(str, "Embedding")
        @test contains(str, "Float32")
        @test contains(str, "768")
        @test contains(str, "50257")
    end

end
