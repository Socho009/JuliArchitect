# ─── Model Configuration with Parametric Types and Presets ───

"""
    MODEL_PRESETS

Dictionary mapping preset symbols to their (n_layer, n_head, n_embd) configurations.
"""
const MODEL_PRESETS = Dict{Symbol, NamedTuple{(:n_layer, :n_head, :n_embd), Tuple{Int, Int, Int}}}(
    :nano        => (n_layer=3,  n_head=3,  n_embd=48),
    :micro       => (n_layer=4,  n_head=4,  n_embd=128),
    :mini        => (n_layer=6,  n_head=6,  n_embd=192),
    :base        => (n_layer=12, n_head=12, n_embd=768),
    :medium      => (n_layer=24, n_head=16, n_embd=1024),
    :large       => (n_layer=36, n_head=20, n_embd=1280),
    :xl          => (n_layer=48, n_head=25, n_embd=1600),
    # Legacy aliases for test compatibility
    :gpt_nano    => (n_layer=3,  n_head=3,  n_embd=48),
    :gpt2        => (n_layer=12, n_head=12, n_embd=768),
    :gpt2_medium => (n_layer=24, n_head=16, n_embd=1024),
    :gpt2_large  => (n_layer=36, n_head=20, n_embd=1280),
    :gpt2_xl     => (n_layer=48, n_head=25, n_embd=1600),
)

"""
    ModelConfig{P<:AbstractFloat} <: AbstractModelConfig{P}

Immutable configuration struct for models, parameterized on floating-point
precision type `P`.

# Fields
- `vocab_size::Int` — Size of the token vocabulary
- `block_size::Int` — Maximum sequence length (context window)
- `n_layer::Int` — Number of transformer blocks
- `n_head::Int` — Number of attention heads
- `n_embd::Int` — Embedding dimension
- `embd_pdrop::P` — Dropout probability for embeddings
- `resid_pdrop::P` — Dropout probability for residual connections
- `attn_pdrop::P` — Dropout probability for attention weights
"""
struct ModelConfig{P<:AbstractFloat} <: AbstractModelConfig{P}
    vocab_size::Int
    block_size::Int
    n_layer::Int
    n_head::Int
    n_embd::Int
    embd_pdrop::P
    resid_pdrop::P
    attn_pdrop::P
end

"""
    ModelConfig{P}(; model_type::Symbol=:custom, kwargs...) where P

Construct a ModelConfig with precision type `P`. Optionally specify a `model_type`
preset symbol to populate `n_layer`, `n_head`, and `n_embd` from predefined values.
"""
function ModelConfig{P}(;
    model_type::Symbol = :custom,
    vocab_size::Int = 50257,
    block_size::Int = 1024,
    n_layer::Union{Int, Nothing} = nothing,
    n_head::Union{Int, Nothing} = nothing,
    n_embd::Union{Int, Nothing} = nothing,
    embd_pdrop::Real = 0.1,
    resid_pdrop::Real = 0.1,
    attn_pdrop::Real = 0.1,
) where {P<:AbstractFloat}

    if model_type != :custom
        if !haskey(MODEL_PRESETS, model_type)
            supported = join(sort(collect(keys(MODEL_PRESETS))), ", ", " and ")
            error("Unrecognized model_type preset: :$model_type. Supported presets are: $supported")
        end
        preset = MODEL_PRESETS[model_type]
        n_layer = n_layer === nothing ? preset.n_layer : n_layer
        n_head = n_head === nothing ? preset.n_head : n_head
        n_embd = n_embd === nothing ? preset.n_embd : n_embd
    end

    if n_layer === nothing || n_head === nothing || n_embd === nothing
        error("Must specify model_type preset or provide n_layer, n_head, and n_embd explicitly")
    end

    ModelConfig{P}(vocab_size, block_size, n_layer, n_head, n_embd, P(embd_pdrop), P(resid_pdrop), P(attn_pdrop))
end

"""
    ModelConfig(; kwargs...)

Convenience constructor defaulting to Float32 precision.
"""
ModelConfig(; kwargs...) = ModelConfig{Float32}(; kwargs...)

# ─── Utility functions ───

function Base.show(io::IO, config::ModelConfig{P}) where P
    print(io, "ModelConfig{$P}(")
    print(io, "vocab_size=$(config.vocab_size), ")
    print(io, "block_size=$(config.block_size), ")
    print(io, "n_layer=$(config.n_layer), ")
    print(io, "n_head=$(config.n_head), ")
    print(io, "n_embd=$(config.n_embd), ")
    print(io, "embd_pdrop=$(config.embd_pdrop), ")
    print(io, "resid_pdrop=$(config.resid_pdrop), ")
    print(io, "attn_pdrop=$(config.attn_pdrop))")
end
