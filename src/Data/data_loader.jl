# ─── Character-Level Data Loader ──────────────────────────────────────────────
#
# Architect-native reimagining of char-rnn's CharSplitLMMinibatchLoader.
# Loads text, splits into train/val/test, serves sequential (x, y) batches
# where y is x shifted by one character (next-character prediction).
#
# Reuses the existing CharTokenizer for vocabulary management.

"""
    CharDataLoader

Batched character-level language model data loader with train/val/test splits.

Faithful to char-rnn's design:
1. Read text → encode via CharTokenizer
2. Create targets shifted by 1 (y[i] = x[i+1])
3. Reshape into (total_per_batch, batch_size) — column-major for cache efficiency
4. Chunk into seq_length windows
5. Split chunks into train/val/test by fraction

# Fields
- `x_batches::Vector{Matrix{Int}}` — Input batches, each (seq_length, batch_size)
- `y_batches::Vector{Matrix{Int}}` — Target batches (shifted by 1)
- `tokenizer::CharTokenizer` — Character vocabulary
- `split_sizes::Tuple{Int,Int,Int}` — (n_train, n_val, n_test) batch counts
- `batch_ix::Vector{Int}` — Mutable cursor per split [train, val, test]
- `nbatches::Int` — Total number of batches
"""
mutable struct CharDataLoader
    x_batches::Vector{Matrix{Int}}
    y_batches::Vector{Matrix{Int}}
    tokenizer::CharTokenizer
    split_sizes::Tuple{Int, Int, Int}
    batch_ix::Vector{Int}
    nbatches::Int
end

"""
    CharDataLoader(text::String; batch_size=50, seq_length=50,
                   split_fractions=(0.95, 0.05, 0.0))

Construct a CharDataLoader from raw text.

# Arguments
- `text` — Raw input text
- `batch_size` — Number of parallel sequences per batch
- `seq_length` — Length of each sequence chunk (BPTT window)
- `split_fractions` — (train, val, test) fractions, must sum to ≤ 1.0

# Examples
```julia
text = read("shakespeare.txt", String)
loader = CharDataLoader(text; batch_size=32, seq_length=64)
x, y = next_batch!(loader, :train)
# x, y: (64, 32) matrices of character indices
```
"""
function CharDataLoader(
    text::String;
    batch_size::Int = 50,
    seq_length::Int = 50,
    split_fractions::Tuple{Float64, Float64, Float64} = (0.95, 0.05, 0.0)
)
    @assert sum(split_fractions) ≤ 1.0 + 1e-10 "split_fractions must sum to ≤ 1.0"
    @assert all(f -> f >= 0, split_fractions) "split_fractions must be non-negative"

    # Build tokenizer and encode
    tokenizer = CharTokenizer(text)
    data = encode(tokenizer, text)
    total_len = length(data)

    # Truncate to fit evenly into batch_size × seq_length chunks
    usable = batch_size * seq_length * div(total_len, batch_size * seq_length)
    if usable == 0
        error("Text too short ($total_len chars) for batch_size=$batch_size × seq_length=$seq_length = $(batch_size * seq_length)")
    end
    data = data[1:usable]

    # Create targets: y[i] = data[i+1], y[end] = data[1] (circular shift)
    ydata = similar(data)
    ydata[1:end-1] = data[2:end]
    ydata[end] = data[1]

    # Reshape: (total_per_batch, batch_size) in column-major layout
    # Column j contains the j-th contiguous text segment
    total_per_batch = usable ÷ batch_size
    x_mat = reshape(data, total_per_batch, batch_size)
    y_mat = reshape(ydata, total_per_batch, batch_size)

    # Chunk into seq_length windows along the time axis (dim 1)
    n_chunks = total_per_batch ÷ seq_length
    x_batches = Vector{Matrix{Int}}(undef, n_chunks)
    y_batches = Vector{Matrix{Int}}(undef, n_chunks)
    for i in 1:n_chunks
        t_start = (i - 1) * seq_length + 1
        t_end = i * seq_length
        x_batches[i] = x_mat[t_start:t_end, :]  # (seq_length, batch_size)
        y_batches[i] = y_mat[t_start:t_end, :]
    end

    # Split into train/val/test by batch count
    n_train = floor(Int, n_chunks * split_fractions[1])
    n_val = floor(Int, n_chunks * split_fractions[2])
    n_test = n_chunks - n_train - n_val

    if n_chunks < 50
        @warn "Only $n_chunks batches total — consider smaller batch_size or seq_length"
    end

    batch_ix = [0, 0, 0]

    CharDataLoader(x_batches, y_batches, tokenizer, (n_train, n_val, n_test), batch_ix, n_chunks)
end

"""
    CharDataLoader(filepath::String, args...; kwargs...)

Convenience constructor that reads from a file path.
"""
function CharDataLoader(::Val{:file}, filepath::String; kwargs...)
    text = read(filepath, String)
    CharDataLoader(text; kwargs...)
end

# ─── Batch Access ─────────────────────────────────────────────────────────────

"""
    next_batch!(loader::CharDataLoader, split::Symbol)

Get the next (x, y) batch for the specified split.
Cycles through batches sequentially, wrapping around at the end.

# Arguments
- `split` — `:train`, `:val`, or `:test`

# Returns
- `(x, y)` — Tuple of (seq_length, batch_size) integer matrices
"""
function next_batch!(loader::CharDataLoader, split::Symbol)
    split_idx = _split_index(split)
    n = loader.split_sizes[split_idx]
    n == 0 && error("No data in $split split")

    # Advance cursor (cycle)
    loader.batch_ix[split_idx] = mod1(loader.batch_ix[split_idx] + 1, n)

    # Compute absolute batch index with offset
    ix = loader.batch_ix[split_idx]
    if split_idx == 2
        ix += loader.split_sizes[1]
    elseif split_idx == 3
        ix += loader.split_sizes[1] + loader.split_sizes[2]
    end

    return loader.x_batches[ix], loader.y_batches[ix]
end

"""
    reset!(loader::CharDataLoader, split::Symbol)

Reset the batch cursor for the specified split to the beginning.
"""
function reset!(loader::CharDataLoader, split::Symbol)
    loader.batch_ix[_split_index(split)] = 0
end

"""
    make_batch_fn(loader::CharDataLoader, split::Symbol=:train)

Create a zero-argument callable that returns (x, y) batches on each call.
Compatible with the existing `train!` interface which expects `dataset()`.
"""
function make_batch_fn(loader::CharDataLoader, split::Symbol=:train)
    return () -> next_batch!(loader, split)
end

# ─── Utilities ────────────────────────────────────────────────────────────────

function _split_index(split::Symbol)
    split == :train && return 1
    split == :val && return 2
    split == :test && return 3
    error("Unknown split: $split. Must be :train, :val, or :test")
end

function Base.show(io::IO, loader::CharDataLoader)
    n_train, n_val, n_test = loader.split_sizes
    vocab = loader.tokenizer.vocab_size
    seq_len = size(loader.x_batches[1], 1)
    batch_sz = size(loader.x_batches[1], 2)
    print(io, "CharDataLoader(vocab=$vocab, seq=$seq_len, batch=$batch_sz, ")
    print(io, "splits=(train=$n_train, val=$n_val, test=$n_test))")
end
