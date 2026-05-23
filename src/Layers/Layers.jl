module Layers

using ..Core
using LinearAlgebra
using Random
using Statistics: mean

# ─── Exports ────────────────────────────────────────────────────────────────
export Embedding, Norm, Linear, OrthogonalLinear, has_bias, orthogonal_weight
export AbstractAttention, Attention
export _batched_attention_scores, _apply_causal_mask, _batched_attention_output
export _softmax, _dropout, apply_position_encoding!
export FeedForward, Block
export RNNCell, LSTMCell, GRUCell

# ─── Includes ───────────────────────────────────────────────────────────────
include("layers.jl")
include("attention.jl")
include("block.jl")
include("recurrent.jl")

end # module Layers
