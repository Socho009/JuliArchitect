using Test
using JSON

# Include source files directly for testing
include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "tokenizer.jl"))

@testset "CharTokenizer" begin

    @testset "Requirement 8.1: Sorted unique character vocabulary with 1-based indexing" begin
        tok = CharTokenizer("hello")
        # Unique chars: 'e', 'h', 'l', 'o' → sorted: ['e', 'h', 'l', 'o']
        @test tok.vocab_size == 4
        @test tok.idx_to_char == ['e', 'h', 'l', 'o']
        @test tok.char_to_idx == Dict('e' => 1, 'h' => 2, 'l' => 3, 'o' => 4)

        # Encoding uses 1-based indices
        ids = encode(tok, "hello")
        @test ids == [2, 1, 3, 3, 4]
    end

    @testset "Requirement 8.2: Round-trip correctness" begin
        text = "the quick brown fox jumps over the lazy dog"
        tok = CharTokenizer(text)
        @test decode(tok, encode(tok, text)) == text

        # Also test with substrings
        @test decode(tok, encode(tok, "fox")) == "fox"
        @test decode(tok, encode(tok, "the")) == "the"
    end

    @testset "Requirement 8.6: Empty string handling" begin
        tok = CharTokenizer("abc")

        # Encode empty string → empty vector
        @test encode(tok, "") == Int[]

        # Decode empty vector → empty string
        @test decode(tok, Int[]) == ""
    end

    @testset "Requirement 8.7: KeyError for unknown characters" begin
        tok = CharTokenizer("abc")
        @test_throws KeyError encode(tok, "xyz")
        @test_throws KeyError encode(tok, "abcd")
    end

    @testset "Requirement 8.8: BoundsError for invalid indices" begin
        tok = CharTokenizer("abc")
        @test tok.vocab_size == 3

        # Index 0 is invalid (1-based)
        @test_throws BoundsError decode(tok, [0])

        # Index > vocab_size is invalid
        @test_throws BoundsError decode(tok, [4])

        # Negative index
        @test_throws BoundsError decode(tok, [-1])
    end

    @testset "Vocabulary is sorted" begin
        tok = CharTokenizer("zyxwvu")
        @test tok.idx_to_char == ['u', 'v', 'w', 'x', 'y', 'z']
    end

    @testset "Unicode characters" begin
        text = "héllo wörld"
        tok = CharTokenizer(text)
        @test decode(tok, encode(tok, text)) == text
    end

    @testset "Single character text" begin
        tok = CharTokenizer("a")
        @test tok.vocab_size == 1
        @test encode(tok, "a") == [1]
        @test decode(tok, [1]) == "a"
    end

end


# ─── BPETokenizer Tests ───────────────────────────────────────────────────────

# Helper to get vocab file paths
const ENCODER_PATH = expanduser("~/.cache/mingpt/encoder.json")
const VOCAB_BPE_PATH = expanduser("~/.cache/mingpt/vocab.bpe")

# Only run BPE tests if vocab files are available
if isfile(ENCODER_PATH) && isfile(VOCAB_BPE_PATH)

@testset "BPETokenizer" begin

    @testset "Construction and vocabulary loading" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)
        @test length(tok.encoder) == 50257
        @test length(tok.decoder) == 50257
        @test length(tok.bpe_ranks) == 50000
        @test length(tok.byte_encoder) == 256
        @test length(tok.byte_decoder) == 256
    end

    @testset "Requirement 8.3: BPE encoding with pre-tokenization and merges" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)

        # Known GPT-2 encodings from the Python reference
        ids = encode(tok, "Hello")
        @test ids == [15496]

        ids = encode(tok, "!!")
        @test ids == [3228]

        ids = encode(tok, " I")
        @test ids == [314]

        ids = encode(tok, "'m")
        @test ids == [1101]
    end

    @testset "Requirement 8.4: BPE round-trip correctness" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)

        texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D",
            "   spaces   and\ttabs\n",
            "Numbers: 12345 and 3.14159",
            "Special chars: @#\$%^&*()",
        ]

        for text in texts
            ids = encode(tok, text)
            decoded = decode(tok, ids)
            @test decoded == text
        end
    end

    @testset "Requirement 8.5: BPE merge caching" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)

        # Encode a word to populate cache
        encode(tok, "hello")
        cache_size_after_first = length(tok.cache)
        @test cache_size_after_first > 0

        # Encode same word again — cache should not grow
        encode(tok, "hello")
        @test length(tok.cache) == cache_size_after_first
    end

    @testset "Requirement 8.6: Empty string handling" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)

        @test encode(tok, "") == Int[]
        @test decode(tok, Int[]) == ""
    end

    @testset "Requirement 8.8: BoundsError for invalid token indices" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)

        @test_throws BoundsError decode(tok, [999999])
        @test_throws BoundsError decode(tok, [-1])
    end

    @testset "Unicode text round-trip" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)

        # Emoji and multi-byte characters
        text = "Hello 🤗 world"
        ids = encode(tok, text)
        @test decode(tok, ids) == text
    end

    @testset "Pre-tokenization regex splitting" begin
        tok = BPETokenizer(ENCODER_PATH, VOCAB_BPE_PATH)

        # Contractions should be split properly
        text = "I'm don't we're they've"
        ids = encode(tok, text)
        decoded = decode(tok, ids)
        @test decoded == text
    end

end

else
    @warn "Skipping BPETokenizer tests: vocab files not found at $ENCODER_PATH and $VOCAB_BPE_PATH"
end
