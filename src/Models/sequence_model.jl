# ─── Recurrent Language Model ──────────────────────────────────────────────────
#
# Multi-layer recurrent language model paralleling SequenceModel in structure:
# Embedding → Stacked Cells (with dropout) → LayerNorm → Decoder
#
# Supports RNN, LSTM, and GRU cells via dispatch on cell_type config.
# Uses the same forward/generate interface as SequenceModel where possible.

"""
    SequenceModel{P, C, E, LN, LH}

Multi-layer recurrent language model with token embedding and linear decoder.

Architecture:
    token → Embedding → [Cell₁ → Dropout → Cell₂ → ... → Cellₙ] → Dropout → LayerNorm → Linear → logits

# Type Parameters
- `P`: Floating-point precision type
- `C`: Config type
- `E`: Embedding type
- `LN`: Final LayerNorm type
- `LH`: Decoder (Linear) type

# Fields
- `config::C` — SequenceConfig
- `embedding::E` — Token embedding (hidden_size, vocab_size)
- `cells::Vector` — Stacked recurrent cells (n_layer)
- `ln_f::LN` — Final layer normalization
- `decoder::LH` — Hidden → vocab projection
- `dropout_rate::P`
"""

# ─── SequenceConfig ──────────────────────────────────────────────────────────

"""
    SequenceConfig{P<:AbstractFloat}

Configuration for recurrent language models, parameterized on precision type `P`.

# Fields
- `vocab_size::Int` — Size of the token vocabulary
- `seq_length::Int` — BPTT unroll length (number of timesteps per training chunk)
- `n_layer::Int` — Number of stacked recurrent layers
- `hidden_size::Int` — Hidden state dimension (rnn_size in char-rnn)
- `cell_type::Symbol` — `:rnn`, `:lstm`, or `:gru`
- `dropout::P` — Dropout probability between layers
"""
struct SequenceConfig{P<:AbstractFloat}
    vocab_size::Int
    seq_length::Int
    n_layer::Int
    hidden_size::Int
    cell_type::Symbol
    dropout::P
end

"""
    SequenceConfig{P}(; kwargs...) where P

Construct a SequenceConfig with keyword arguments and sensible defaults.
"""
function SequenceConfig{P}(;
    vocab_size::Int = 65,
    seq_length::Int = 50,
    n_layer::Int = 2,
    hidden_size::Int = 128,
    cell_type::Symbol = :lstm,
    dropout::Real = 0.0,
) where {P<:AbstractFloat}
    cell_type in (:rnn, :lstm, :gru) || error("Unknown cell_type: $cell_type. Must be :rnn, :lstm, or :gru")
    SequenceConfig{P}(vocab_size, seq_length, n_layer, hidden_size, cell_type, P(dropout))
end

SequenceConfig(; kwargs...) = SequenceConfig{Float32}(; kwargs...)

function Base.show(io::IO, c::SequenceConfig{P}) where P
    print(io, "SequenceConfig{$P}(cell=$(c.cell_type), hidden=$(c.hidden_size), layers=$(c.n_layer), vocab=$(c.vocab_size), seq=$(c.seq_length))")
end


struct SequenceModel{P, C, E, LN, LH}
    config::C
    embedding::E
    cells::Vector
    ln_f::LN
    decoder::LH
    dropout_rate::P
end

"""
    SequenceModel{P}(config::SequenceConfig{P}) where P

Construct a SequenceModel from a SequenceConfig.
Dispatches on `config.cell_type` to create the appropriate cell type.
"""
function SequenceModel{P}(config::SequenceConfig{P}) where {P<:AbstractFloat}
    H = config.hidden_size

    # Token embedding: maps vocab indices → hidden_size vectors
    embedding = Embedding{P}(H, config.vocab_size)

    # Stacked recurrent cells — all layers use hidden_size for both input and output
    cells = if config.cell_type == :rnn
        [RNNCell{P}(H, H) for _ in 1:config.n_layer]
    elseif config.cell_type == :lstm
        [LSTMCell{P}(H, H) for _ in 1:config.n_layer]
    elseif config.cell_type == :gru
        [GRUCell{P}(H, H) for _ in 1:config.n_layer]
    else
        error("Unknown cell_type: $(config.cell_type)")
    end

    # Final layer norm (stabilizes recurrent output)
    ln_f = Norm{P}(H)

    # Decoder: hidden → vocab logits (no bias, weight-tied with embedding possible)
    decoder = Linear{P}(H, config.vocab_size; bias=true)

    SequenceModel{P, typeof(config), typeof(embedding), typeof(ln_f), typeof(decoder)}(
        config, embedding, cells, ln_f, decoder, config.dropout
    )
end

SequenceModel(config::SequenceConfig{P}) where {P} = SequenceModel{P}(config)

# ─── Single-Step Forward ──────────────────────────────────────────────────────

"""
    forward_step(model::SequenceModel{P}, x_t::AbstractVector{<:Integer},
                 states::Vector; training::Bool=true) where P

Process a single timestep through the recurrent language model.

# Arguments
- `x_t` — Token indices for this timestep, shape (batch,)
- `states` — Vector of per-layer recurrent states
- `training` — Whether to apply dropout

# Returns
- `logits` — Output logits of shape (vocab_size, batch)
- `new_states` — Updated per-layer recurrent states
"""
function forward_step(
    model::SequenceModel{P}, x_t::AbstractVector{<:Integer},
    states::Vector; training::Bool=true
) where P
    # Embed tokens: (hidden_size, batch)
    h = model.embedding(x_t)

    # Process through stacked cells with inter-layer dropout
    new_states = Any[]
    for (i, cell) in enumerate(model.cells)
        h, s = cell(h, states[i])
        push!(new_states, s)
        # Apply dropout between layers (not after last)
        if training && i < length(model.cells)
            h = _dropout(h, model.dropout_rate; training=training)
        end
    end

    # Final dropout + layer norm + decode
    h = _dropout(h, model.dropout_rate; training=training)
    h = model.ln_f(h)
    logits = model.decoder(h)  # (vocab_size, batch)

    return logits, new_states
end

# ─── Full Sequence Forward (for training with BPTT) ───────────────────────────

"""
    forward(model::SequenceModel{P}, idx::AbstractMatrix{<:Integer};
            targets=nothing, training::Bool=true,
            init_states::Union{Nothing, Vector}=nothing) where P

Full sequence forward pass through the recurrent language model.
Unrolls through all timesteps, optionally computing cross-entropy loss.

# Arguments
- `idx` — Token indices of shape (seq_len, batch)
- `targets` — Optional target indices of shape (seq_len, batch)
- `training` — Whether to apply dropout
- `init_states` — Optional initial states (for state carry-over between batches)

# Returns
- `logits` — Output logits of shape (vocab_size, seq_len, batch)
- `loss` — Cross-entropy loss scalar if targets provided, else nothing
- `final_states` — Final per-layer states (for carry-over to next batch)
"""
function forward(
    model::SequenceModel{P}, idx::AbstractMatrix{<:Integer};
    targets::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
    training::Bool = true,
    init_states::Union{Nothing, Vector} = nothing
) where P
    T, B = size(idx)  # (seq_len, batch)
    V = model.config.vocab_size

    # Initialize or reuse states
    states = if init_states !== nothing
        init_states
    else
        [init_state(cell, B) for cell in model.cells]
    end

    # Unroll through time using Any[] for Zygote compatibility (avoids array mutation)
    all_logits_list = Any[]
    for t in 1:T
        x_t = @view idx[t, :]  # (batch,)
        logits_t, states = forward_step(model, x_t, states; training=training)
        push!(all_logits_list, logits_t)
    end
    
    # all_logits_list is a list of T matrices of shape (V, B).
    # We want to combine them into (V, T, B).
    # cat with dims=3 gives (V, B, T), then we permute.
    logits_3d = cat(all_logits_list...; dims=3)
    all_logits = permutedims(logits_3d, (1, 3, 2))

    # Compute loss if targets provided
    loss = if targets !== nothing
        _cross_entropy_loss(all_logits, targets, P)
    else
        nothing
    end

    return all_logits, loss, states
end

# ─── Generation ───────────────────────────────────────────────────────────────

"""
    generate(model::SequenceModel{P}, idx::AbstractMatrix{<:Integer};
             strategy::SamplingStrategy=GreedySampling(),
             max_new_tokens::Int=100) where P

Autoregressive text generation from a recurrent language model.

1. Processes the prompt tokens to warm up hidden states
2. Generates `max_new_tokens` by sampling from the model's output distribution
3. Uses the same SamplingStrategy dispatch as SequenceModel

# Arguments
- `model` — SequenceModel to generate from
- `idx` — Input token indices of shape (seq_len, batch)
- `strategy` — SamplingStrategy for token selection
- `max_new_tokens` — Number of new tokens to generate

# Returns
- Matrix of shape (original_seq_len + max_new_tokens, batch)
"""
function generate(
    model::SequenceModel{P}, idx::AbstractMatrix{<:Integer};
    strategy::SamplingStrategy = GreedySampling(),
    max_new_tokens::Int = 100
) where P
    T, B = size(idx)

    # Initialize states
    states = [init_state(cell, B) for cell in model.cells]

    # Pre-allocate result
    result = Matrix{Int}(undef, T + max_new_tokens, B)
    result[1:T, :] = idx

    # Warm up: process prompt tokens to build up hidden state
    for t in 1:T
        x_t = @view idx[t, :]
        _, states = forward_step(model, x_t, states; training=false)
    end

    # Generate new tokens autoregressively
    for i in 1:max_new_tokens
        # Use the last generated token
        last_token = @view result[T + i - 1, :]
        logits, states = forward_step(model, last_token, states; training=false)

        # Sample next token for each batch element
        for b in 1:B
            result[T + i, b] = sample_next_token(logits[:, b], strategy)
        end
    end

    return result
end

# ─── Parameter Counting ───────────────────────────────────────────────────────

function _count_parameters(model::SequenceModel)
    count = 0

    # Embedding
    count += length(model.embedding.weight)

    # Cells
    for cell in model.cells
        count += _count_cell_params(cell)
    end

    # Final layer norm
    count += length(model.ln_f.weight) + length(model.ln_f.bias)

    # Decoder
    count += length(model.decoder.weight)
    if has_bias(model.decoder)
        count += length(model.decoder.bias)
    end

    return count
end

function _count_cell_params(cell::RNNCell)
    return (length(cell.W_ih.weight) + length(cell.W_ih.bias)
          + length(cell.W_hh.weight) + length(cell.W_hh.bias))
end

function _count_cell_params(cell::LSTMCell)
    return (length(cell.W_ih.weight) + length(cell.W_ih.bias)
          + length(cell.W_hh.weight) + length(cell.W_hh.bias))
end

function _count_cell_params(cell::GRUCell)
    return (length(cell.W_iz.weight) + length(cell.W_iz.bias)
          + length(cell.W_hz.weight) + length(cell.W_hz.bias)
          + length(cell.W_ir.weight) + length(cell.W_ir.bias)
          + length(cell.W_hr.weight) + length(cell.W_hr.bias)
          + length(cell.W_in.weight) + length(cell.W_in.bias)
          + length(cell.W_hn.weight) + length(cell.W_hn.bias))
end

# ─── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, model::SequenceModel{P}) where P
    c = model.config
    n_params = _count_parameters(model)
    print(io, "SequenceModel{$P}(")
    print(io, "cell=$(c.cell_type), ")
    print(io, "hidden=$(c.hidden_size), ")
    print(io, "layers=$(c.n_layer), ")
    print(io, "vocab=$(c.vocab_size), ")
    print(io, "params≈$(round(n_params / 1e6; digits=2))M)")
end

# Device transfer for SequenceModel
import ..Core

function Core.to_device(model::SequenceModel{P, C, E, LN, LH}, device::Symbol) where {P, C, E, LN, LH}
    embedding_new = to_device(model.embedding, device)
    cells_new = [to_device(c, device) for c in model.cells]
    ln_f_new = to_device(model.ln_f, device)
    decoder_new = to_device(model.decoder, device)
    return SequenceModel{P, typeof(model.config), typeof(embedding_new), typeof(ln_f_new), typeof(decoder_new)}(
        model.config, embedding_new, cells_new, ln_f_new, decoder_new, model.dropout_rate
    )
end
