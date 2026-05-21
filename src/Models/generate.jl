# ─── Text Generation with Strategy Dispatch ───────────────────────────────────

# ─── Sampling Strategy Type Hierarchy ─────────────────────────────────────────

"""
    SamplingStrategy

Abstract type for text generation sampling strategies.
Enables multiple dispatch on strategy type — the Architect-native pattern
for behavior specialization without conditional branching.
"""
abstract type SamplingStrategy end

"""
    GreedySampling <: SamplingStrategy

Deterministic sampling strategy that always selects the token with the
highest logit (argmax). Produces identical outputs for identical inputs.
"""
struct GreedySampling <: SamplingStrategy end

"""
    TopKSampling{P<:AbstractFloat} <: SamplingStrategy

Top-k sampling strategy: filters logits to only the top-k values,
applies temperature scaling, then samples from the resulting distribution.

# Fields
- `k::Int` — Number of top logits to keep (must be ≥ 1)
- `temperature::P` — Temperature for scaling logits (must be > 0)
"""
struct TopKSampling{P<:AbstractFloat} <: SamplingStrategy
    k::Int
    temperature::P

    function TopKSampling(k::Int, temperature::P) where {P<:AbstractFloat}
        k < 1 && error("TopKSampling: k must be ≥ 1, got k=$k")
        temperature ≤ zero(P) && error("TopKSampling: temperature must be > 0, got temperature=$temperature")
        new{P}(k, temperature)
    end
end

"""
    NucleusSampling{P<:AbstractFloat} <: SamplingStrategy

Nucleus (top-p) sampling strategy: sorts logits by descending probability,
accumulates probabilities until the cumulative sum reaches p, then samples
from that filtered subset with temperature scaling.

# Fields
- `p::P` — Cumulative probability threshold (must be in (0, 1])
- `temperature::P` — Temperature for scaling logits (must be > 0)
"""
struct NucleusSampling{P<:AbstractFloat} <: SamplingStrategy
    p::P
    temperature::P

    function NucleusSampling(p::P, temperature::P) where {P<:AbstractFloat}
        (p ≤ zero(P) || p > one(P)) && error("NucleusSampling: p must be in (0, 1], got p=$p")
        temperature ≤ zero(P) && error("NucleusSampling: temperature must be > 0, got temperature=$temperature")
        new{P}(p, temperature)
    end
end

# ─── sample_next_token: Multiple Dispatch on Strategy ─────────────────────────

"""
    sample_next_token(logits::AbstractVector{P}, ::GreedySampling) where P

Greedy sampling: return the index of the maximum logit (deterministic).
"""
function sample_next_token(logits::AbstractVector{P}, ::GreedySampling) where {P<:AbstractFloat}
    return argmax(logits)
end

"""
    sample_next_token(logits::AbstractVector{P}, s::TopKSampling{P}) where P

Top-k sampling: keep only the top-k logits (set rest to -Inf),
apply temperature scaling, compute softmax, and sample from the distribution.
"""
function sample_next_token(logits::AbstractVector{P}, s::TopKSampling{P}) where {P<:AbstractFloat}
    V = length(logits)
    k = min(s.k, V)

    # Find the top-k logit values
    sorted_logits = sort(logits; rev=true)
    threshold = sorted_logits[k]

    # Filter: keep only logits ≥ threshold, set rest to -Inf
    filtered = copy(logits)
    for i in eachindex(filtered)
        if filtered[i] < threshold
            filtered[i] = P(-Inf)
        end
    end

    # Apply temperature
    filtered ./= s.temperature

    # Softmax
    probs = _softmax(filtered)

    # Sample from categorical distribution
    return _categorical_sample(probs)
end

"""
    sample_next_token(logits::AbstractVector{P}, s::NucleusSampling{P}) where P

Nucleus (top-p) sampling: sort logits descending, compute cumulative softmax
probabilities, keep tokens until cumulative prob ≥ p, then sample from
the filtered distribution with temperature scaling.
"""
function sample_next_token(logits::AbstractVector{P}, s::NucleusSampling{P}) where {P<:AbstractFloat}
    # Sort logits descending and track original indices
    sorted_indices = sortperm(logits; rev=true)
    sorted_logits = logits[sorted_indices]

    # Compute softmax probabilities of sorted logits (before temperature)
    sorted_probs = _softmax(sorted_logits)

    # Compute cumulative probabilities
    cum_probs = cumsum(sorted_probs)

    # Find cutoff: keep tokens until cumulative prob ≥ p
    # Always keep at least one token
    cutoff_idx = 1
    for i in eachindex(cum_probs)
        cutoff_idx = i
        if cum_probs[i] ≥ s.p
            break
        end
    end

    # Create filtered logits: keep only the nucleus tokens, set rest to -Inf
    filtered = fill(P(-Inf), length(logits))
    for i in 1:cutoff_idx
        orig_idx = sorted_indices[i]
        filtered[orig_idx] = logits[orig_idx]
    end

    # Apply temperature
    filtered[filtered .!= P(-Inf)] ./= s.temperature

    # Softmax over filtered logits
    probs = _softmax(filtered)

    # Sample from categorical distribution
    return _categorical_sample(probs)
end

# ─── Generate Function ────────────────────────────────────────────────────────

"""
    generate(model, idx::AbstractMatrix{<:Integer};
             strategy::SamplingStrategy=GreedySampling(),
             max_new_tokens::Int=100) where P

Autoregressive text generation using multiple dispatch on SamplingStrategy.

Generates `max_new_tokens` new tokens by repeatedly:
1. Truncating the conditioning window to at most block_size tokens
2. Running a forward pass to get logits for the last position
3. Sampling the next token using the specified strategy
4. Appending the new token to the sequence

# Arguments
- `model` — SequenceModel to generate from
- `idx` — Input token indices of shape (seq_len, batch) with 1-based indices
- `strategy` — SamplingStrategy determining how to select next tokens
- `max_new_tokens` — Number of new tokens to generate

# Returns
- Matrix of shape (original_seq_len + max_new_tokens, batch) containing
  the original prefix followed by generated tokens

# Examples
```julia
config = ModelConfig{Float32}(model_type=:gpt_nano, vocab_size=100, block_size=32)
model = SequenceModel(config)
idx = rand(1:100, 5, 1)  # (seq_len=5, batch=1)
output = generate(model, idx; strategy=GreedySampling(), max_new_tokens=10)
# output shape: (15, 1)
```
"""
function generate(
    model, idx::AbstractMatrix{<:Integer};
    strategy::SamplingStrategy = GreedySampling(),
    max_new_tokens::Int = 100
) where P
    block_size = model.config.block_size
    B = size(idx, 2)  # batch size

    # Work with a mutable copy to accumulate generated tokens
    result = copy(idx)

    for _ in 1:max_new_tokens
        # Truncate conditioning window to block_size if needed
        T_current = size(result, 1)
        if T_current > block_size
            idx_cond = result[end-block_size+1:end, :]
        else
            idx_cond = result
        end

        # Forward pass (inference mode, no dropout)
        logits, _ = forward(model, idx_cond; training=false)

        # Get logits for the last position: (vocab_size, batch)
        # logits shape is (vocab_size, T_cond, B)
        last_logits = logits[:, end, :]  # (vocab_size, B)

        # Sample next token for each batch element
        next_tokens = Matrix{Int}(undef, 1, B)
        for b in 1:B
            next_tokens[1, b] = sample_next_token(last_logits[:, b], strategy)
        end

        # Append to sequence
        result = vcat(result, next_tokens)
    end

    return result
end

# ─── Helper Functions ─────────────────────────────────────────────────────────

"""
    _softmax(x::AbstractVector{P}) where P

Numerically stable softmax for a 1D vector.
"""
function _softmax(x::AbstractVector{P}) where {P<:AbstractFloat}
    x_max = maximum(x)
    exps = exp.(x .- x_max)
    return exps ./ sum(exps)
end

"""
    _categorical_sample(probs::AbstractVector{P}) where P

Sample an index from a categorical distribution defined by `probs`.
Uses inverse CDF sampling.
"""
function _categorical_sample(probs::AbstractVector{P}) where {P<:AbstractFloat}
    u = rand(P)
    cumulative = zero(P)
    for i in eachindex(probs)
        cumulative += probs[i]
        if u ≤ cumulative
            return i
        end
    end
    # Fallback for numerical edge cases (rounding)
    return length(probs)
end
