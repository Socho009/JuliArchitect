# ─── Character-Level Tokenizer ────────────────────────────────────────────────
# Maps individual characters to 1-based integer indices using sorted unique characters
# from a training corpus. Provides round-trip correctness: decode(encode(text)) == text.

"""
    CharTokenizer

A character-level tokenizer that maps individual characters to unique 1-based integer indices.
The vocabulary is built from sorted unique characters of the training text.

# Fields
- `char_to_idx::Dict{Char, Int}` — character → 1-based index mapping
- `idx_to_char::Vector{Char}` — 1-based index → character mapping
- `vocab_size::Int` — total number of unique characters in vocabulary
"""
struct CharTokenizer
    char_to_idx::Dict{Char, Int}
    idx_to_char::Vector{Char}
    vocab_size::Int
end

"""
    CharTokenizer(text::String)

Construct a CharTokenizer from a training text. The vocabulary is built by
extracting all unique characters, sorting them, and assigning 1-based indices.

# Examples
```julia
tok = CharTokenizer("hello world")
encode(tok, "hello") # → [3, 2, 4, 4, 5]
```
"""
function CharTokenizer(text::String)
    chars = sort(collect(Set(text)))
    char_to_idx = Dict{Char, Int}(c => i for (i, c) in enumerate(chars))
    idx_to_char = chars
    vocab_size = length(chars)
    CharTokenizer(char_to_idx, idx_to_char, vocab_size)
end

"""
    encode(tokenizer::CharTokenizer, text::String) -> Vector{Int}

Encode a string into a vector of 1-based token IDs.

Returns an empty vector for empty input. Raises `KeyError` if any character
in `text` is not present in the tokenizer's vocabulary.
"""
function encode(tokenizer::CharTokenizer, text::String)::Vector{Int}
    isempty(text) && return Int[]
    return [tokenizer.char_to_idx[c] for c in text]
end

"""
    decode(tokenizer::CharTokenizer, ids::Vector{Int}) -> String

Decode a vector of 1-based token IDs back into a string.

Returns an empty string for empty input. Raises `BoundsError` if any index
is outside the range 1 to vocab_size.
"""
function decode(tokenizer::CharTokenizer, ids::Vector{Int})::String
    isempty(ids) && return ""
    return String([tokenizer.idx_to_char[i] for i in ids])
end

# ─── BPE Tokenizer ────────────────────────────────────────────────────────────
# Byte-Pair Encoding tokenizer compatible with GPT-2's vocabulary.
# Converts text → bytes → unicode representation → BPE merges → token IDs.
# Provides round-trip correctness: decode(encode(text)) == text.

"""
    bytes_to_unicode()

Build the byte-to-unicode mapping used by GPT-2. Every possible byte (0..255)
gets mapped to a unicode character for visual representation. Bytes that already
render nicely (printable ASCII, Latin-1 supplement) keep their character; others
get shifted to avoid control characters.

Returns a Dict{UInt8, Char}.
"""
function bytes_to_unicode()::Dict{UInt8, Char}
    # The 188 bytes that render fine in their original form
    bs = UInt8[]
    append!(bs, UInt8(0x21):UInt8(0x7E))   # '!' to '~'
    append!(bs, UInt8(0xA1):UInt8(0xAC))   # '¡' to '¬'
    append!(bs, UInt8(0xAE):UInt8(0xFF))   # '®' to 'ÿ'

    cs = Int[Int(b) for b in bs]

    # Map the remaining 68 bytes to characters starting at codepoint 256
    n = 0
    for b in 0x00:0xFF
        if !(b in bs)
            push!(bs, b)
            push!(cs, 256 + n)
            n += 1
        end
    end

    return Dict{UInt8, Char}(bs[i] => Char(cs[i]) for i in eachindex(bs))
end

"""
    get_pairs(word::Vector{String})

Return all consecutive bigram pairs from a word (vector of subword strings).
Returns a Vector of (String, String) tuples.
"""
function get_pairs(word::Vector{String})::Vector{Tuple{String, String}}
    pairs = Tuple{String, String}[]
    for i in 1:(length(word) - 1)
        push!(pairs, (word[i], word[i+1]))
    end
    return pairs
end

"""
    BPETokenizer

A Byte-Pair Encoding tokenizer compatible with GPT-2's vocabulary files.
Converts arbitrary UTF-8 text to integer token sequences and back.

# Fields
- `encoder::Dict{String, Int}` — token string → token ID mapping
- `decoder::Dict{Int, String}` — token ID → token string mapping
- `bpe_ranks::Dict{Tuple{String, String}, Int}` — merge pair → priority rank
- `byte_encoder::Dict{UInt8, Char}` — byte value → unicode character
- `byte_decoder::Dict{Char, UInt8}` — unicode character → byte value
- `pattern::Regex` — pre-tokenization regex (GPT-2 pattern)
- `cache::Dict{String, String}` — memoized BPE merge results
"""
mutable struct BPETokenizer
    encoder::Dict{String, Int}
    decoder::Dict{Int, String}
    bpe_ranks::Dict{Tuple{String, String}, Int}
    byte_encoder::Dict{UInt8, Char}
    byte_decoder::Dict{Char, UInt8}
    pattern::Regex
    cache::Dict{String, String}
end

"""
    BPETokenizer(encoder_path::String, vocab_bpe_path::String)

Construct a BPETokenizer from GPT-2 vocabulary files.

# Arguments
- `encoder_path`: path to `encoder.json` (token string → ID mapping)
- `vocab_bpe_path`: path to `vocab.bpe` (BPE merge rules)

# Examples
```julia
tok = BPETokenizer("encoder.json", "vocab.bpe")
ids = encode(tok, "Hello world")
text = decode(tok, ids)
```
"""
function BPETokenizer(encoder_path::String, vocab_bpe_path::String)
    # Load encoder.json
    encoder_data = open(encoder_path, "r") do f
        JSON.parse(f)
    end
    encoder = Dict{String, Int}(String(k) => Int(v) for (k, v) in encoder_data)

    # Build decoder (inverse mapping)
    decoder = Dict{Int, String}(v => k for (k, v) in encoder)

    # Load vocab.bpe and parse merge rules
    bpe_data = read(vocab_bpe_path, String)
    lines = split(bpe_data, '\n')
    # Skip the first line (version header) and last empty line
    merge_lines = filter(!isempty, lines[2:end])
    bpe_merges = [Tuple(split(line)) for line in merge_lines]
    bpe_ranks = Dict{Tuple{String, String}, Int}(
        (String(m[1]), String(m[2])) => i - 1 for (i, m) in enumerate(bpe_merges)
    )

    # Build byte encoder/decoder
    byte_enc = bytes_to_unicode()
    byte_dec = Dict{Char, UInt8}(v => k for (k, v) in byte_enc)

    # GPT-2 pre-tokenization regex pattern
    # Matches contractions, letters, numbers, non-letter/number/space, and whitespace
    pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    cache = Dict{String, String}()

    return BPETokenizer(encoder, decoder, bpe_ranks, byte_enc, byte_dec, pattern, cache)
end

"""
    bpe_merge(tokenizer::BPETokenizer, token::String) -> String

Apply BPE merges to a token string iteratively, merging the lowest-rank pair
at each step until no more merges are possible. Results are cached for efficiency.

Returns a space-separated string of BPE subword pieces.
"""
function bpe_merge(tokenizer::BPETokenizer, token::String)::String
    # Check cache first
    haskey(tokenizer.cache, token) && return tokenizer.cache[token]

    # Split token into individual characters as strings
    word = [string(c) for c in token]

    pairs = get_pairs(word)
    if isempty(pairs)
        tokenizer.cache[token] = token
        return token
    end

    while true
        # Find the lowest-rank bigram
        best_pair = nothing
        best_rank = typemax(Int)
        for pair in pairs
            rank = get(tokenizer.bpe_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_pair = pair
            end
        end

        # If no pair has a valid rank, we're done
        if best_pair === nothing || !haskey(tokenizer.bpe_ranks, best_pair)
            break
        end

        first, second = best_pair

        # Merge all occurrences of (first, second) in word
        new_word = String[]
        i = 1
        while i <= length(word)
            # Find next occurrence of `first`
            j = nothing
            for k in i:length(word)
                if word[k] == first
                    j = k
                    break
                end
            end

            if j === nothing
                # No more occurrences of `first`; append rest
                append!(new_word, word[i:end])
                break
            end

            # Append everything before the found position
            append!(new_word, word[i:j-1])
            i = j

            # Check if this occurrence is followed by `second`
            if i <= length(word) && word[i] == first && i < length(word) && word[i+1] == second
                push!(new_word, first * second)
                i += 2
            else
                push!(new_word, word[i])
                i += 1
            end
        end

        word = new_word
        if length(word) == 1
            break
        else
            pairs = get_pairs(word)
        end
    end

    result = join(word, ' ')
    tokenizer.cache[token] = result
    return result
end

"""
    encode(tokenizer::BPETokenizer, text::String) -> Vector{Int}

Encode a string into a vector of BPE token IDs.

The encoding process:
1. Pre-tokenize text using the GPT-2 regex pattern
2. Convert each pre-token to bytes, then map bytes to unicode characters
3. Apply BPE merges iteratively (lowest rank pair first)
4. Look up final tokens in the encoder dictionary to get IDs

Returns an empty vector for empty input.
"""
function encode(tokenizer::BPETokenizer, text::String)::Vector{Int}
    isempty(text) && return Int[]

    bpe_idx = Int[]

    # Pre-tokenize using the regex pattern
    tokens = [m.match for m in eachmatch(tokenizer.pattern, text)]

    for token in tokens
        # Encode token to bytes, then map through byte_encoder
        token_bytes = Vector{UInt8}(token)
        token_translated = String([tokenizer.byte_encoder[b] for b in token_bytes])

        # Apply BPE merges
        merged = bpe_merge(tokenizer, token_translated)

        # Split on spaces and look up each piece in the encoder
        for piece in split(merged, ' ')
            push!(bpe_idx, tokenizer.encoder[piece])
        end
    end

    return bpe_idx
end

"""
    decode(tokenizer::BPETokenizer, ids::Vector{Int}) -> String

Decode a vector of BPE token IDs back into a string.

Returns an empty string for empty input. Raises `BoundsError` if any token ID
is not present in the decoder vocabulary.
"""
function decode(tokenizer::BPETokenizer, ids::Vector{Int})::String
    isempty(ids) && return ""

    # Look up each ID in the decoder
    tokens_merged = String[]
    for id in ids
        if !haskey(tokenizer.decoder, id)
            throw(BoundsError(tokenizer.decoder, id))
        end
        push!(tokens_merged, tokenizer.decoder[id])
    end

    # Concatenate all token strings
    tokens_flat = join(tokens_merged)

    # Map each character back through byte_decoder to get bytes
    token_bytes = UInt8[]
    for c in tokens_flat
        if !haskey(tokenizer.byte_decoder, c)
            throw(BoundsError(tokenizer.byte_decoder, c))
        end
        push!(token_bytes, tokenizer.byte_decoder[c])
    end

    # Decode bytes as UTF-8
    return String(copy(token_bytes))
end
