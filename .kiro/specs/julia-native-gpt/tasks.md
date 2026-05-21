# Implementation Plan: Julia-Native GPT

## Overview

Rebuild the existing Julia GPT implementation from scratch using Julia-native paradigms: parametric types for precision abstraction, abstract type hierarchies for extensibility, multiple dispatch for behavior specialization, and column-major array layouts for performance. The implementation replaces the current Flux-dependent OOP-style code with idiomatic Julia patterns that form the foundation for scaling toward GPT-5.5.

## Tasks

- [x] 1. Set up project structure, abstract types, and configuration
  - [x] 1.1 Define the module structure and abstract type hierarchy
    - Create `src/JuliaNativeGPT.jl` module file with exports
    - Define `AbstractLayer{P}`, `AbstractAttention{P} <: AbstractLayer{P}`, `AbstractBlock{P} <: AbstractLayer{P}` abstract types
    - Define `AbstractGPTConfig{P<:AbstractFloat}` abstract type
    - Set up `Project.toml` with dependencies (Flux, CUDA, Random, LinearAlgebra)
    - _Requirements: 2.1_

  - [x] 1.2 Implement GPTConfig with parametric types and presets
    - Create `src/config.jl` with `GPTConfig{P} <: AbstractGPTConfig{P}` struct
    - Implement all 7 model presets (:gpt_nano, :gpt_micro, :gpt_mini, :gpt2, :gpt2_medium, :gpt2_large, :gpt2_xl)
    - Implement keyword override merging with presets
    - Default to Float32 when no type parameter specified
    - Raise error for unrecognized preset symbols
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 1.3 Write property test for Config Precision Type Propagation
    - **Property 11: Config Precision Type Propagation**
    - **Validates: Requirements 1.1, 1.3**

- [x] 2. Implement core layers (Embedding, LayerNorm, Linear)
  - [x] 2.1 Implement Embedding layer with column-major layout
    - Create `src/layers.jl` with `Embedding{P, M<:AbstractMatrix{P}} <: AbstractLayer{P}`
    - Store weights as (n_embd, vocab_size) for cache-friendly column lookups
    - Implement `(e::Embedding)(idx)` functor for batch embedding lookup
    - Support 1-based indexing for Julia convention
    - _Requirements: 2.2, 2.3_

  - [x] 2.2 Write property test for Embedding Lookup Consistency
    - **Property 6: Embedding Lookup Consistency with 1-Indexing**
    - **Validates: Requirements 2.2, 10.1**

  - [x] 2.3 Implement LayerNorm and Linear (Dense) layers
    - Implement `LayerNorm{P}` with weight and bias parameters
    - Implement `Linear{P}` (Dense) layer with optional bias
    - Support 3D tensor operations (n_embd, seq_len, batch) via reshape
    - _Requirements: 4.1, 4.3_

- [x] 3. Implement Causal Self-Attention
  - [x] 3.1 Implement CausalSelfAttention struct and forward pass
    - Create `src/attention.jl` with `CausalSelfAttention{P, D, Mask} <: AbstractAttention{P}`
    - Implement combined Q,K,V projection and head splitting
    - Implement scaled dot-product attention with column-major layout (head_dim, seq_len, n_heads, batch)
    - Apply causal mask (set future positions to -Inf before softmax)
    - Apply attention dropout and residual dropout
    - Recombine heads and apply output projection
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 3.2 Write property test for Attention Weights Distribution
    - **Property 2: Attention Weights Form Valid Causal Probability Distributions**
    - **Validates: Requirements 3.3, 3.4**

  - [x] 3.3 Write property test for Attention Input Immutability
    - **Property 13: Attention Input Immutability**
    - **Validates: Requirements 3.6**

- [x] 4. Implement Transformer Block
  - [x] 4.1 Implement TransformerBlock with pre-norm and residual connections
    - Create `src/block.jl` with `TransformerBlock{P, A, LN, FF} <: AbstractBlock{P}`
    - Implement pre-norm architecture: LN → Attention → Residual → LN → MLP → Residual
    - Implement feedforward MLP: Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd)
    - Apply residual dropout before each residual addition
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 4.2 Write property test for Transformer Block Shape Preservation
    - **Property 12: Transformer Block Shape Preservation**
    - **Validates: Requirements 4.5**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement GPTModel and Forward Pass
  - [x] 6.1 Implement GPTModel struct and construction
    - Create `src/model.jl` with `GPTModel{P, C, E, PE, B, LN, LH}` struct
    - Construct token embeddings (n_embd, vocab_size), position embeddings (n_embd, block_size)
    - Create n_layer TransformerBlocks, final LayerNorm, and language model head
    - Validate n_embd % n_head == 0 at construction time
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

  - [x] 6.2 Implement forward pass
    - Implement `forward(model::GPTModel{P}, idx; targets=nothing)`
    - Combine token + positional embeddings with 1-based positions
    - Process through all transformer blocks sequentially
    - Apply final layer norm and project to vocab logits (vocab_size, T, B)
    - Compute cross-entropy loss when targets provided
    - Assert sequence length ≤ block_size
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 6.3 Write property test for Forward Pass Output Shape
    - **Property 10: Forward Pass Output Shape**
    - **Validates: Requirements 5.2, 5.3, 5.6**

  - [x] 6.4 Write property test for Causal Masking
    - **Property 1: Causal Masking**
    - **Validates: Requirements 3.3, 5.1**

- [x] 7. Implement Weight Initialization via Multiple Dispatch
  - [x] 7.1 Implement dispatch-based weight initialization
    - Create `src/init.jl` with `initialize!` methods dispatching on layer type
    - Embedding: Normal(0, 0.02)
    - CausalSelfAttention Q/K/V: Normal(0, 0.02), biases = 0
    - CausalSelfAttention output projection: Normal(0, 0.02/sqrt(2*n_layers))
    - LayerNorm: weights = 1.0, biases = 0.0
    - MLP layers: Normal(0, 0.02), biases = 0
    - Language model head: Normal(0, 0.02)
    - Recursive initialization visiting all layers exactly once
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [x] 7.2 Write property test for Weight Decay Separation
    - **Property 7: Weight Decay Separation Completeness and Correctness**
    - **Validates: Requirements 7.1, 7.2**

- [x] 8. Implement Text Generation with Strategy Dispatch
  - [x] 8.1 Implement SamplingStrategy type hierarchy and generate function
    - Create `src/generate.jl` with `SamplingStrategy` abstract type
    - Implement `GreedySampling`, `TopKSampling(k, temperature)`, `NucleusSampling(p, temperature)`
    - Implement `generate(model, idx; strategy, max_new_tokens)` with autoregressive loop
    - Implement `sample_next_token` dispatching on strategy type
    - Handle block_size truncation of conditioning window
    - Validate sampling parameters (k range, p range, temperature > 0)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 8.2 Write property test for Generation Preserves Prefix
    - **Property 3: Generation Preserves Prefix and Produces Valid Tokens**
    - **Validates: Requirements 6.4, 6.6**

  - [x] 8.3 Write property test for Greedy Generation Determinism
    - **Property 4: Greedy Generation is Deterministic**
    - **Validates: Requirements 6.1, 6.7**

- [x] 9. Implement Training Loop
  - [x] 9.1 Implement TrainConfig and train! function
    - Create `src/train.jl` with `TrainConfig{P}` struct
    - Implement weight decay separation via `param_group` dispatch (DecayGroup/NoDecayGroup)
    - Implement AdamW optimizer with separate decay groups
    - Implement gradient norm clipping
    - Implement `train!(model, dataset, config)` with mutating convention
    - Halt on NaN/Inf loss with descriptive error
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 9.2 Write property test for Training Convergence
    - **Property 5: Training Convergence on Memorizable Data**
    - **Validates: Requirements 7.5**

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement Tokenizers
  - [x] 11.1 Implement CharTokenizer with round-trip correctness
    - Create `src/tokenizer.jl` with `CharTokenizer` struct
    - Build vocabulary from sorted unique characters (1-based indexing)
    - Implement `encode(tokenizer, text)` and `decode(tokenizer, ids)`
    - Handle empty string encoding/decoding
    - Raise KeyError for unknown characters, BoundsError for invalid indices
    - _Requirements: 8.1, 8.2, 8.6, 8.7, 8.8_

  - [x] 11.2 Write property test for CharTokenizer Round-Trip
    - **Property 8: CharTokenizer Round-Trip**
    - **Validates: Requirements 8.1, 8.2**

  - [x] 11.3 Implement BPETokenizer with merge caching
    - Implement `BPETokenizer` struct with encoder, decoder, bpe_ranks, byte_encoder/decoder, pattern, cache
    - Implement pre-tokenization regex splitting
    - Implement `bpe_merge` with rank-ordered merges and caching
    - Implement `encode(tokenizer, text)` and `decode(tokenizer, ids)`
    - Handle empty string encoding/decoding
    - Raise BoundsError for invalid token indices
    - _Requirements: 8.3, 8.4, 8.5, 8.6, 8.8_

  - [x] 11.4 Write property test for BPETokenizer Round-Trip
    - **Property 9: BPETokenizer Round-Trip**
    - **Validates: Requirements 8.3, 8.4**

- [x] 12. Implement GPU/Device Abstraction
  - [x] 12.1 Implement device transfer and dispatch
    - Create `src/device.jl` with `to_device(model, device::Symbol)` function
    - Support :cpu (Array), :cuda (CuArray), :metal (future)
    - Move all parameters, buffers, and masks to target device
    - Ensure forward pass returns arrays on same device as model
    - Graceful fallback to CPU when CUDA.jl unavailable
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 13. Wire module together and create examples
  - [x] 13.1 Wire all components into the module and create training example
    - Update `src/JuliaNativeGPT.jl` to include all source files and export public API
    - Create `examples/train_char_native.jl` demonstrating character-level training
    - Create `examples/generate_text.jl` demonstrating generation with all strategies
    - Ensure all components integrate correctly end-to-end
    - _Requirements: 1.1, 2.1, 5.1, 6.1, 7.4, 8.1_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation uses Julia-native patterns throughout: parametric types, multiple dispatch, abstract type hierarchies, column-major arrays, and mutating conventions (!)
- This is Iteration 1 — the architecture is designed to scale toward GPT-5.5 in subsequent iterations

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1", "2.3"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "4.1"] },
    { "id": 4, "tasks": ["4.2", "6.1"] },
    { "id": 5, "tasks": ["6.2", "7.1"] },
    { "id": 6, "tasks": ["6.3", "6.4", "7.2", "8.1"] },
    { "id": 7, "tasks": ["8.2", "8.3", "9.1"] },
    { "id": 8, "tasks": ["9.2", "11.1"] },
    { "id": 9, "tasks": ["11.2", "11.3", "12.1"] },
    { "id": 10, "tasks": ["11.4", "13.1"] }
  ]
}
```
