# ─── Abstract Type Hierarchy ───
# Defines the core abstract types for the Architect-native GPT implementation.
# These enable multiple dispatch on layer types and precision-parameterized configs.

"""
    AbstractModelConfig{P<:AbstractFloat}

Abstract supertype for all Model configurations, enabling dispatch on config variants.
Parameterized on floating-point precision type `P`.
"""
abstract type AbstractModelConfig{P<:AbstractFloat} end

"""
    AbstractLayer{P<:AbstractFloat}

Abstract supertype for all neural network layers, parameterized on precision type `P`.
Enables multiple dispatch on layer types for initialization, parameter traversal,
and device transfer — replacing Python's isinstance() checks with Julia's type system.
"""
abstract type AbstractLayer{P<:AbstractFloat} end

"""
    AbstractAttention{P<:AbstractFloat} <: AbstractLayer{P}

Abstract type for attention mechanisms, enabling dispatch-based specialization.
Subtypes implement causal masking, multi-head splitting, and scaled dot-product attention.
"""
abstract type AbstractAttention{P<:AbstractFloat} <: AbstractLayer{P} end

"""
    AbstractBlock{P<:AbstractFloat} <: AbstractLayer{P}

Abstract type for transformer blocks, enabling dispatch-based specialization.
Subtypes implement pre-norm architecture with residual connections.
"""
abstract type AbstractBlock{P<:AbstractFloat} <: AbstractLayer{P} end

# ─── Recurrent Architecture Types ─────────────────────────────────────────────

"""
    AbstractRecurrentCell{P<:AbstractFloat} <: AbstractLayer{P}

Abstract type for recurrent cells (RNN, LSTM, GRU), enabling dispatch-based
specialization. Subtypes implement single-timestep forward passes that take
(input, state) → (output, new_state).
"""
abstract type AbstractRecurrentCell{P<:AbstractFloat} <: AbstractLayer{P} end

"""
    AbstractRecurrentState{P<:AbstractFloat}

Abstract type for typed recurrent state containers. Each cell type carries
its own state structure — RNN/GRU have a single hidden vector, LSTM has
both cell state and hidden state.
"""
abstract type AbstractRecurrentState{P<:AbstractFloat} end
