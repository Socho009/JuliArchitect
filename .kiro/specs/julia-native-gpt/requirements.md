# Requirements Document

## Introduction

This document specifies the requirements for the Julia-native GPT implementation, now transitioning into **Iteration 2**. The core focus of Iteration 2 is transitioning from a standard GPT-2 textual modeling baseline towards experimental continuous autoregressive modeling using **Functional Autoregressive Transformers (FAT)** and **Recurrent Language Models (RecurrentLM)**. 

The architecture aggressively abandons legacy standards (e.g. standard LayerNorm, GELU, absolute positional embeddings, and pure OOP representations) in favor of Llama-2-inspired primitives (RMSNorm, SwiGLU, RoPE), orthogonal parameterization via Cayley layers, and mutation-free forward passes for seamless O(1) backpropagation compatibility with `Zygote.jl`.

## Glossary

- **FunctionalModel**: The primary experimental model designed to learn operations over continuous functional coefficients instead of discrete text tokens.
- **RecurrentLM**: A BPTT-trained multi-layer recurrent model (RNN, GRU, LSTM) acting as a foundational baseline for sequence modeling.
- **ModelConfig**: An abstract configuration struct holding model hyperparameters parameterized by precision type `P`.
- **CayleyLinear**: A linear layer constrained to orthogonal weight matrices using the Cayley transform, mitigating vanishing gradients.
- **RMSNorm**: Root Mean Square Normalization replacing legacy LayerNorm for improved performance and stability.
- **SwiGLU**: A gated feed-forward activation function replacing legacy MLP + GELU.
- **RoPE (Rotary Positional Embedding)**: Mutation-free rotational sequence encodings replacing absolute learned embeddings.
- **Mutation-Free Zygote Compatibility**: The strict requirement that no array allocations are mutated in-place during the forward pass, ensuring O(1) reverse-mode automatic differentiation.

## Requirements

### Requirement 1: Iteration 2 Experimental Primitives

**User Story:** As an AI researcher, I want the core architectural primitives to reflect modern state-of-the-art designs (Llama-2 stack) rather than legacy GPT-2 baselines, so that models are more performant and parameter-efficient.

#### Acceptance Criteria
1. The codebase SHALL replace standard `LayerNorm` implementations entirely with `RMSNorm`, eliminating mean-centering operations.
2. The codebase SHALL replace standard `MLP` + `gelu` feedforward networks with `SwiGLU` (SiLU + Gated Linear Units), utilizing a 3-matrix transformation instead of 2.
3. The `CausalSelfAttention` module SHALL integrate `RoPE` (Rotary Positional Embeddings) by default, completely deprecating statically learned absolute positional embeddings in the core Transformer stack.

### Requirement 2: Mutation-Free Forward Passes for AD

**User Story:** As a differential computing researcher, I want all models to be perfectly compatible with `Zygote.jl` via strictly mutation-free array operations, enabling highly efficient automatic differentiation.

#### Acceptance Criteria
1. The `CausalSelfAttention` forward pass SHALL execute without utilizing `A .= ...` or indexed assignments for masking or positional scaling.
2. `RoPE` application SHALL utilize non-mutating complex interleaving or batched matrix transformations.
3. The `RecurrentLM` forward pass SHALL accumulate hidden states and outputs using Zygote-friendly comprehensions or allocations (e.g. `Any[]` arrays or tuples) rather than pre-allocating and assigning via `all_logits[:, t, :] = ...`.
4. `CayleyLinear` matrices SHALL compute the orthogonal approximation `(I - A) * inv(I + A)` without in-place mutation.

### Requirement 3: Continuous Functional Modeling (FAT)

**User Story:** As an experimental modeler, I want to use `FunctionalModel` to predict sequences of continuous coefficients representing symbolic basis functions instead of discrete vocabulary logits.

#### Acceptance Criteria
1. The `FunctionalModel` SHALL accept a sequence of dense vectors (coefficients) directly instead of utilizing a discrete `Embedding` layer.
2. The `FunctionalModel` SHALL output dense target coefficients via a linear `coeff_head` projection rather than standard vocabulary logits.
3. The underlying blocks of `FunctionalModel` SHALL utilize the modernized Transformer stack (`RMSNorm`, `SwiGLU`, `RoPE`).
4. Training routines SHALL optimize the `FunctionalModel` against continuous targets (e.g. Mean Squared Error or spectral divergence).

### Requirement 4: Orthogonal Parameterization (Cayley Layers)

**User Story:** As a researcher combating vanishing gradients in deep recurrent or functional networks, I want to use orthogonal weight parameterization seamlessly across the architecture.

#### Acceptance Criteria
1. Attention modules (`CausalSelfAttention`) SHALL support `use_cayley_out` and `use_cayley_qkv` boolean flags to instantiate `CayleyLinear` projections.
2. `CayleyLinear` layer parameterization SHALL cleanly split into an antisymmetric matrix `A` serving as the learnable parameters in the weight decay group.
3. Evaluation functions (like `compute_spectral_stats`) SHALL correctly identify and measure the orthogonality error `norm(W' * W - I)` for all Cayley layers during the training loop.

### Requirement 5: Weight Decay and Optimization Grouping

**User Story:** As an optimizer developer, I want all models to systematically group weights using Julia's multiple dispatch, applying decay only to specific matrices.

#### Acceptance Criteria
1. `param_group` dispatch SHALL return `DecayGroup` for `Linear` weights and the learnable matrix `A` inside `CayleyLinear`.
2. `param_group` dispatch SHALL return `NoDecayGroup` for `RMSNorm`, embedding matrices, and all layer biases.
3. The `collect_params` interface SHALL correctly extract variables for both `FunctionalModel` and `RecurrentLM` recursively and transparently.

### Requirement 6: Legacy Code Cleanup

**User Story:** As a maintainer, I want to ensure that all GPT-2 legacy code is completely stripped out of the main logic path so that the project focuses entirely on the new architectural goals.

#### Acceptance Criteria
1. The core `JuliaNativeGPT.jl` module SHALL NOT export `GPTModel`, `GPTConfig`, `LayerNorm`, `MLP`, or `gelu`.
2. All generic functions (e.g. `train!`, `_compute_loss_and_grads`) SHALL use generic parameter types or duck-typing instead of being strictly constrained to `::GPTModel{P}`.
3. Legacy documentation, presets, and tokenization logic heavily dependent on `GPT-2` should be isolated, deprecated, or abstracted to fit the experimental configurations (`AbstractModelConfig`, `ModelConfig`).
