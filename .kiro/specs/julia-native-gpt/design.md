# Design Document: Julia-Native GPT (Iteration 1)

## Overview

A from-scratch GPT implementation leveraging Julia's unique strengths: multiple dispatch, parametric types, metaprogramming, and high-performance array operations. Rather than porting Python OOP patterns, this design uses Julia's type system for zero-cost abstractions, multiple dispatch for extensible behavior, and generated functions for compile-time optimizations. Iteration 1 establishes the foundational architecture that will scale toward GPT-5.5 performance across subsequent iterations.

## Main Algorithm/Workflow

```mermaid
sequenceDiagram
    participant User as User Code
    participant Config as GPTConfig{P}
    participant Model as GPTModel{P}
    participant Attn as attention(...)
    participant Train as train!(...)
    participant Gen as generate(...)

    User->>Config: GPTConfig{Float32}(model_type=:gpt2)
    User->>Model: GPTModel(config)
    User->>Train: train!(model, dataset, optimizer_config)
    Train->>Model: forward(model, tokens)
    Model->>Attn: attention(Q, K, V, mask)
    Attn-->>Model: context vectors
    Model-->>Train: logits, loss
    Train-->>User: trained model
    User->>Gen: generate(model, prompt; strategy=TopKSampling(k=40))
    Gen->>Model: forward(model, tokens)
    Model-->>Gen: logits
    Gen-->>User: generated token sequence
```


## Core Interfaces/Types

```julia
# ─── Parametric Precision: All types parameterized on floating-point type ───

"""
    AbstractGPTConfig{P<:AbstractFloat}

Abstract supertype for all GPT configurations, enabling dispatch on config variants.
"""
abstract type AbstractGPTConfig{P<:AbstractFloat} end

"""
    GPTConfig{P} <: AbstractGPTConfig{P}

Immutable configuration with compile-time type information for precision.
Uses Julia's parametric types for zero-cost precision abstraction.
"""
struct GPTConfig{P<:AbstractFloat} <: AbstractGPTConfig{P}
    vocab_size::Int
    block_size::Int
    n_layer::Int
    n_head::Int
    n_embd::Int
    embd_pdrop::P
    resid_pdrop::P
    attn_pdrop::P
end

# Named model presets via multiple dispatch
function GPTConfig{P}(; model_type::Symbol=:custom, kwargs...) where P
    presets = Dict(
        :gpt_nano  => (n_layer=3,  n_head=3,  n_embd=48),
        :gpt_micro => (n_layer=4,  n_head=4,  n_embd=128),
        :gpt_mini  => (n_layer=6,  n_head=6,  n_embd=192),
        :gpt2      => (n_layer=12, n_head=12, n_embd=768),
        :gpt2_medium => (n_layer=24, n_head=16, n_embd=1024),
        :gpt2_large  => (n_layer=36, n_head=20, n_embd=1280),
        :gpt2_xl     => (n_layer=48, n_head=25, n_embd=1600),
    )
    base = model_type == :custom ? NamedTuple() : presets[model_type]
    merged = merge(base, kwargs)
    GPTConfig{P}(
        get(merged, :vocab_size, 50257),
        get(merged, :block_size, 1024),
        merged.n_layer, merged.n_head, merged.n_embd,
        P(get(merged, :embd_pdrop, 0.1)),
        P(get(merged, :resid_pdrop, 0.1)),
        P(get(merged, :attn_pdrop, 0.1)),
    )
end

# Convenience: default to Float32
GPTConfig(; kwargs...) = GPTConfig{Float32}(; kwargs...)

# ─── Abstract Layer Hierarchy ───

"""
Abstract type hierarchy enabling multiple dispatch on layer types.
This replaces Python's isinstance() checks with Julia's type dispatch.
"""
abstract type AbstractLayer{P} end
abstract type AbstractAttention{P} <: AbstractLayer{P} end
abstract type AbstractBlock{P} <: AbstractLayer{P} end

# ─── Embedding with proper Julia array semantics ───

struct Embedding{P, M<:AbstractMatrix{P}} <: AbstractLayer{P}
    weight::M  # (n_embd, vocab_size) — column-major for fast lookup
end

# ─── Multi-Head Causal Self-Attention ───

struct CausalSelfAttention{P, D, Mask<:AbstractMatrix{Bool}} <: AbstractAttention{P}
    W_qkv::D       # Combined Q,K,V projection (n_embd → 3*n_embd)
    W_out::D       # Output projection (n_embd → n_embd)
    mask::Mask     # Causal mask (block_size × block_size)
    n_head::Int
    head_dim::Int
    attn_pdrop::P
    resid_pdrop::P
end

# ─── Transformer Block ───

struct TransformerBlock{P, A<:AbstractAttention{P}, LN, FF} <: AbstractBlock{P}
    ln1::LN
    attention::A
    ln2::LN
    feedforward::FF
    resid_pdrop::P
end

# ─── Full GPT Model ───

struct GPTModel{P, C<:AbstractGPTConfig{P}, E, PE, B, LN, LH}
    config::C
    tok_emb::E       # Token embedding
    pos_emb::PE      # Position embedding
    blocks::B        # Vector of TransformerBlocks
    ln_f::LN        # Final layer norm
    lm_head::LH     # Language model head
    embd_pdrop::P
end
```


## Key Functions with Formal Specifications

### Function 1: `attention(Q, K, V, mask; dropout_p, training)`

```julia
"""
    attention(Q::A, K::A, V::A, mask::AbstractMatrix{Bool};
              dropout_p::P=0.0, training::Bool=true) where {P, A<:AbstractArray{P,4}}

Scaled dot-product attention with causal masking.
Operates on 4D tensors: (head_dim, seq_len, n_heads, batch).

Julia-native: uses multiple dispatch for GPU/CPU specialization,
column-major layout for cache-friendly access patterns.
"""
function attention(
    Q::A, K::A, V::A, mask::AbstractMatrix{Bool};
    dropout_p::P = zero(P), training::Bool = true
) where {P<:AbstractFloat, A<:AbstractArray{P,4}}
    # Implementation dispatches on array type (CuArray vs Array)
end
```

**Preconditions:**
- `size(Q) == size(K) == size(V)` — all (head_dim, seq_len, n_heads, batch)
- `size(Q, 1) == head_dim > 0`
- `size(mask, 1) >= size(Q, 2)` and `size(mask, 2) >= size(Q, 2)`
- `0 ≤ dropout_p ≤ 1`
- `n_embd % n_head == 0`

**Postconditions:**
- Returns array of same shape as V: (head_dim, seq_len, n_heads, batch)
- Attention weights sum to 1.0 along key dimension (within numerical tolerance)
- Future positions are masked (causal): `att[i, j, :, :] ≈ 0` for `j > i`
- No mutation of input arrays

**Loop Invariants:** N/A (vectorized operations)

---

### Function 2: `forward(model::GPTModel, idx; targets=nothing)`

```julia
"""
    forward(model::GPTModel{P}, idx::AbstractMatrix{<:Integer};
            targets::Union{Nothing, AbstractMatrix{<:Integer}}=nothing) where P

Forward pass through the full GPT model.
Returns (logits, loss) where loss is nothing if targets not provided.

Multiple dispatch enables specialization:
- forward(model, idx) → inference mode (no loss computation)
- forward(model, idx; targets=y) → training mode (computes cross-entropy)
"""
function forward(
    model::GPTModel{P}, idx::AbstractMatrix{<:Integer};
    targets::Union{Nothing, AbstractMatrix{<:Integer}} = nothing
) where P
end
```

**Preconditions:**
- `size(idx, 2) ≤ model.config.block_size` (sequence length within bounds)
- `all(1 .≤ idx .≤ model.config.vocab_size)` (valid token indices)
- If `targets !== nothing`: `size(targets) == size(idx)`

**Postconditions:**
- `logits` has shape `(vocab_size, seq_len, batch_size)` — column-major for Julia
- If `targets !== nothing`: `loss` is a scalar `P` value, `loss ≥ 0`
- If `targets === nothing`: `loss === nothing`
- Model parameters are not mutated

**Loop Invariants:**
- After processing block `i`: tensor shape remains `(n_embd, seq_len, batch)`
- Residual connections preserve gradient flow: `x_out = x_in + f(x_in)`

---

### Function 3: `generate(model, idx; strategy, max_new_tokens)`

```julia
"""
    abstract type SamplingStrategy end
    struct GreedySampling <: SamplingStrategy end
    struct TopKSampling <: SamplingStrategy
        k::Int
        temperature::Float32
    end
    struct NucleusSampling <: SamplingStrategy
        p::Float32
        temperature::Float32
    end

    generate(model::GPTModel, idx::AbstractMatrix{<:Integer};
             strategy::SamplingStrategy=GreedySampling(),
             max_new_tokens::Int=100) -> AbstractMatrix{<:Integer}

Autoregressive generation using multiple dispatch on SamplingStrategy.
This is the Julia-native pattern: behavior varies by type, not by if/else flags.
"""
function generate(
    model::GPTModel{P}, idx::AbstractMatrix{<:Integer};
    strategy::SamplingStrategy = GreedySampling(),
    max_new_tokens::Int = 100
) where P
end

# Dispatch on strategy type for sampling logic
function sample_next_token(logits::AbstractVector{P}, ::GreedySampling) where P
    argmax(logits)
end

function sample_next_token(logits::AbstractVector{P}, s::TopKSampling) where P
    # Top-k filtering then categorical sampling
end

function sample_next_token(logits::AbstractVector{P}, s::NucleusSampling) where P
    # Nucleus (top-p) filtering then categorical sampling
end
```

**Preconditions:**
- `size(idx, 2) ≥ 1` (at least one conditioning token)
- `max_new_tokens ≥ 1`
- For TopKSampling: `1 ≤ k ≤ vocab_size`
- For NucleusSampling: `0 < p ≤ 1`
- `temperature > 0` (for non-greedy strategies)

**Postconditions:**
- Returns matrix of shape `(batch, original_seq_len + max_new_tokens)`
- All generated tokens are in range `1:vocab_size`
- Original conditioning tokens are preserved unchanged in output
- For GreedySampling: output is deterministic given same input

**Loop Invariants:**
- At iteration `t`: `size(result, 2) == original_len + t`
- Conditioning window never exceeds `block_size`

---

### Function 4: `train!(model, dataset, opt_config; callbacks)`

```julia
"""
    TrainConfig{P<:AbstractFloat}

Training configuration using Julia's parametric types.
"""
struct TrainConfig{P<:AbstractFloat}
    learning_rate::P
    weight_decay::P
    betas::Tuple{P, P}
    grad_norm_clip::P
    batch_size::Int
    max_iters::Int
    device::Symbol  # :cpu, :cuda, :metal
end

"""
    abstract type TrainCallback end

Callback system using multiple dispatch instead of string-keyed dictionaries.
"""
abstract type TrainCallback end
struct OnBatchEnd <: TrainCallback end
struct OnEpochEnd <: TrainCallback end

"""
    train!(model::GPTModel, dataset, config::TrainConfig;
           callbacks=TrainCallback[]) -> TrainState

In-place training loop. Uses Julia's convention of ! for mutating functions.
Separates weight decay groups via multiple dispatch on parameter metadata.
"""
function train!(
    model::GPTModel{P}, dataset, config::TrainConfig{P};
    callbacks::Vector{<:TrainCallback} = TrainCallback[]
) where P
end
```

**Preconditions:**
- `config.learning_rate > 0`
- `config.batch_size ≥ 1`
- `config.max_iters ≥ 1`
- `0 ≤ config.weight_decay`
- `all(0 .< config.betas .< 1)`
- `dataset` provides iteration interface yielding `(x, y)` batches

**Postconditions:**
- Model parameters are updated in-place
- Returns `TrainState` with loss history, iteration count
- Gradient norms are clipped to `config.grad_norm_clip`
- Weight decay applied only to weight matrices (not biases, not layer norms)

**Loop Invariants:**
- At each iteration: gradient norm ≤ `grad_norm_clip` after clipping
- Adam moment estimates maintain running averages
- Loss is finite (no NaN/Inf propagation)


## Algorithmic Pseudocode

### Attention Algorithm (Column-Major Julia Layout)

```julia
function attention(Q::Array{P,4}, K::Array{P,4}, V::Array{P,4},
                   mask::Matrix{Bool}; dropout_p::P=zero(P), training::Bool=true) where P
    head_dim, T, n_heads, B = size(Q)
    scale = P(1.0 / sqrt(head_dim))

    # Scaled dot-product: (T, T, n_heads, B)
    # K transposed on first two dims: (T, head_dim, n_heads, B)
    # att[i,j,h,b] = sum_d Q[d,i,h,b] * K[d,j,h,b] * scale
    att = batched_matmul_transA(K, Q) .* scale  # (T, T, n_heads, B)

    # Apply causal mask: future positions → -Inf
    causal = mask[1:T, 1:T]  # (T, T)
    att .= ifelse.(reshape(causal, T, T, 1, 1), att, P(-Inf))

    # Softmax along key dimension (dim=2 in column-major)
    att .= softmax(att; dims=2)

    # Dropout during training
    if training && dropout_p > zero(P)
        att .= dropout(att, dropout_p)
    end

    # Weighted sum of values: (head_dim, T, n_heads, B)
    output = batched_matmul(V, att)  # V * att^T effectively

    return output
end
```

### Forward Pass Algorithm

```julia
function forward(model::GPTModel{P}, idx::AbstractMatrix{<:Integer};
                 targets=nothing) where P
    B, T = size(idx)
    @assert T ≤ model.config.block_size

    # Token + positional embeddings (Julia 1-indexed)
    tok_emb = embed_lookup(model.tok_emb, idx)      # (n_embd, T, B)
    pos_emb = embed_lookup(model.pos_emb, 1:T)      # (n_embd, T, 1) — broadcasts
    x = dropout(tok_emb .+ pos_emb, model.embd_pdrop)

    # Transformer blocks — sequential with residual connections
    for block in model.blocks
        x = transformer_block(block, x)
    end

    # Final layer norm + projection to vocabulary
    x = layer_norm(model.ln_f, x)                   # (n_embd, T, B)
    logits = model.lm_head * reshape(x, :, T*B)     # (vocab_size, T*B)
    logits = reshape(logits, :, T, B)               # (vocab_size, T, B)

    # Loss computation (only if targets provided)
    loss = if targets !== nothing
        cross_entropy_loss(logits, targets)
    else
        nothing
    end

    return logits, loss
end

function transformer_block(block::TransformerBlock{P}, x::Array{P,3}) where P
    # Pre-norm architecture (GPT-2 style)
    # Residual 1: attention
    x = x .+ attention_forward(block.attention, layer_norm(block.ln1, x))

    # Residual 2: feedforward (MLP)
    h = layer_norm(block.ln2, x)
    h = gelu.(block.feedforward.W_fc * reshape(h, :, size(h,2)*size(h,3)))
    h = block.feedforward.W_proj * h
    h = dropout(reshape(h, :, size(x,2), size(x,3)), block.resid_pdrop)
    x = x .+ h

    return x
end
```

### Weight Initialization Algorithm (Julia Multiple Dispatch)

```julia
"""
Weight initialization via multiple dispatch — each layer type gets its own method.
This replaces Python's isinstance() pattern with Julia's type system.
"""
function initialize!(layer::Embedding{P}) where P
    randn!(layer.weight)
    layer.weight .*= P(0.02)
end

function initialize!(layer::CausalSelfAttention{P}) where P
    # Standard init for Q,K,V projection
    randn!(layer.W_qkv.weight)
    layer.W_qkv.weight .*= P(0.02)
    layer.W_qkv.bias .= zero(P)

    # Scaled init for output projection (per GPT-2 paper)
    randn!(layer.W_out.weight)
    layer.W_out.weight .*= P(0.02 / sqrt(2 * n_layers))
    layer.W_out.bias .= zero(P)
end

function initialize!(ln::LayerNorm{P}) where P
    ln.weight .= one(P)
    ln.bias .= zero(P)
end

# Recursive initialization using Julia's functor pattern
function initialize!(model::GPTModel)
    for layer in trainable_layers(model)
        initialize!(layer)
    end
end
```

### Optimizer Configuration (Weight Decay Separation)

```julia
"""
Separate parameters into decay/no-decay groups using Julia's type dispatch.
Replaces Python's string-matching on parameter names.
"""
abstract type ParamGroup end
struct DecayGroup <: ParamGroup end    # Weight matrices
struct NoDecayGroup <: ParamGroup end  # Biases, LayerNorm, Embeddings

# Dispatch on layer type to determine decay behavior
param_group(::Embedding) = NoDecayGroup()
param_group(::LayerNorm) = NoDecayGroup()

function configure_optimizer(model::GPTModel{P}, config::TrainConfig{P}) where P
    decay_params = P[]
    no_decay_params = P[]

    for (layer, params) in parameter_pairs(model)
        group = param_group(layer)
        if group isa DecayGroup
            append!(decay_params, params.weight)
        else
            append!(no_decay_params, params.weight)
            if hasproperty(params, :bias)
                append!(no_decay_params, params.bias)
            end
        end
    end

    # AdamW with separate weight decay groups
    optimizer = AdamW([
        (params=decay_params, weight_decay=config.weight_decay),
        (params=no_decay_params, weight_decay=zero(P)),
    ]; lr=config.learning_rate, betas=config.betas)

    return optimizer
end
```

### BPE Tokenizer Algorithm

```julia
"""
    BPETokenizer{V}

Byte-Pair Encoding tokenizer using Julia's type system.
V is the vocabulary size (known at construction time).
"""
struct BPETokenizer
    encoder::Dict{String, Int}       # token string → index
    decoder::Vector{String}          # index → token string
    bpe_ranks::Dict{Tuple{String,String}, Int}  # merge priority
    byte_encoder::Dict{UInt8, Char}  # byte → unicode visual repr
    byte_decoder::Dict{Char, UInt8}  # unicode → byte
    pattern::Regex                   # pre-tokenization regex
    cache::Dict{String, String}      # memoized BPE results
end

function encode(tokenizer::BPETokenizer, text::AbstractString)::Vector{Int}
    tokens = collect(eachmatch(tokenizer.pattern, text))
    bpe_idx = Int[]

    for token in tokens
        # Encode to bytes, map through byte_encoder
        token_bytes = Vector{UInt8}(token.match)
        token_translated = String([tokenizer.byte_encoder[b] for b in token_bytes])

        # Apply BPE merges
        merged = bpe_merge(tokenizer, token_translated)

        # Map to integer indices
        for piece in split(merged, ' ')
            push!(bpe_idx, tokenizer.encoder[piece])
        end
    end

    return bpe_idx
end

function bpe_merge(tokenizer::BPETokenizer, token::String)::String
    # Check cache first
    haskey(tokenizer.cache, token) && return tokenizer.cache[token]

    word = collect(string.(collect(token)))  # individual characters

    while length(word) > 1
        # Find lowest-rank bigram
        pairs = [(word[i], word[i+1]) for i in 1:length(word)-1]
        best = argmin(p -> get(tokenizer.bpe_ranks, p, typemax(Int)), pairs)

        if !haskey(tokenizer.bpe_ranks, pairs[best])
            break  # No more merges possible
        end

        # Merge all occurrences of the best pair
        word = merge_pair(word, pairs[best]...)
    end

    result = join(word, ' ')
    tokenizer.cache[token] = result
    return result
end
```


## Example Usage

```julia
using JuliaNativeGPT

# ─── Configuration via parametric types ───
config = GPTConfig{Float32}(model_type=:gpt2, vocab_size=50257, block_size=1024)

# ─── Model construction ───
model = GPTModel(config)
println("Parameters: $(count_parameters(model) / 1e6)M")

# ─── Training with Julia-native patterns ───
train_config = TrainConfig{Float32}(
    learning_rate = 3f-4,
    weight_decay = 0.1f0,
    betas = (0.9f0, 0.95f0),
    grad_norm_clip = 1.0f0,
    batch_size = 64,
    max_iters = 5000,
    device = :cuda,
)

# Dataset using Julia's iterator protocol
dataset = CharDataset("input.txt", config.block_size)

# Training with callback dispatch
state = train!(model, dataset, train_config;
    callbacks = [
        LogCallback(every=100),       # Dispatches on_batch_end
        CheckpointCallback(every=1000),
    ]
)

# ─── Generation with strategy dispatch ───
tokenizer = BPETokenizer()
prompt = encode(tokenizer, "The meaning of life is")
prompt_matrix = reshape(prompt, 1, :)  # (1, seq_len)

# Greedy
greedy_output = generate(model, prompt_matrix;
    strategy = GreedySampling(),
    max_new_tokens = 100
)

# Top-k sampling
sampled_output = generate(model, prompt_matrix;
    strategy = TopKSampling(k=40, temperature=0.8f0),
    max_new_tokens = 100
)

# Nucleus sampling
nucleus_output = generate(model, prompt_matrix;
    strategy = NucleusSampling(p=0.95f0, temperature=0.9f0),
    max_new_tokens = 100
)

println(decode(tokenizer, vec(sampled_output)))

# ─── Loading pretrained weights ───
pretrained = from_pretrained(GPTModel{Float32}, :gpt2)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Causal Masking

*For any* valid GPT model configuration, batch of token indices, and sequence length T ≤ block_size, the logits at position t in the full sequence SHALL be identical (within numerical tolerance) to the logits at position t when only tokens 1:t are provided as input — proving that future tokens cannot influence past predictions.

**Validates: Requirements 3.3, 5.1**

### Property 2: Attention Weights Form Valid Causal Probability Distributions

*For any* random Q, K, V tensors of shape (head_dim, T, n_heads, batch) and causal mask, the attention weights SHALL be non-negative, sum to 1.0 along the key dimension (within tolerance), and be exactly zero for all future positions (j > i).

**Validates: Requirements 3.3, 3.4**

### Property 3: Generation Preserves Prefix and Produces Valid Tokens

*For any* model, conditioning prefix of valid tokens, and generation length, the output matrix SHALL contain the original prefix unchanged in its first columns, have exactly (prefix_len + max_new_tokens) columns, and contain only token values in the range 1 to vocab_size.

**Validates: Requirements 6.4, 6.6**

### Property 4: Greedy Generation is Deterministic

*For any* model and input token sequence, calling generate with GreedySampling twice with identical inputs SHALL produce identical output sequences.

**Validates: Requirements 6.1, 6.7**

### Property 5: Training Convergence on Memorizable Data

*For any* small model configuration and memorizable toy dataset, training for sufficient iterations SHALL result in a final loss strictly less than the initial loss.

**Validates: Requirements 7.5**

### Property 6: Embedding Lookup Consistency with 1-Indexing

*For any* embedding of size (n_embd, vocab_size) and batch of valid token indices, the embedding lookup result at position (b, t) SHALL exactly equal the column of the weight matrix indexed by idx[b, t].

**Validates: Requirements 2.2, 10.1**

### Property 7: Weight Decay Separation Completeness and Correctness

*For any* GPT model, separating parameters into decay and no-decay groups SHALL place only 2D+ weight matrices of linear layers in the decay group, place all biases, layer norm parameters, and embedding weights in the no-decay group, and ensure no parameter appears in both groups.

**Validates: Requirements 7.1, 7.2**

### Property 8: CharTokenizer Round-Trip

*For any* text string used to build a CharTokenizer, encoding then decoding that text SHALL produce the original text exactly.

**Validates: Requirements 8.1, 8.2**

### Property 9: BPETokenizer Round-Trip

*For any* valid UTF-8 text string, encoding with the BPETokenizer then decoding the resulting token indices SHALL reconstruct the original text exactly.

**Validates: Requirements 8.3, 8.4**

### Property 10: Forward Pass Output Shape

*For any* valid GPT model, batch size B, and sequence length T ≤ block_size, the forward pass SHALL return logits of shape (vocab_size, T, B), and when targets are provided the loss SHALL be a non-negative scalar, and when targets are not provided the loss SHALL be nothing.

**Validates: Requirements 5.2, 5.3, 5.6**

### Property 11: Config Precision Type Propagation

*For any* AbstractFloat type P and valid configuration parameters, constructing GPTConfig{P} SHALL result in all floating-point fields having type P, and providing keyword overrides alongside a preset SHALL produce a config with the overridden values taking precedence.

**Validates: Requirements 1.1, 1.3**

### Property 12: Transformer Block Shape Preservation

*For any* valid TransformerBlock and input tensor of shape (n_embd, T, B), the output tensor SHALL have the identical shape (n_embd, T, B).

**Validates: Requirements 4.5**

### Property 13: Attention Input Immutability

*For any* input array passed to the CausalSelfAttention layer, the input array SHALL remain unchanged after the attention computation completes.

**Validates: Requirements 3.6**
