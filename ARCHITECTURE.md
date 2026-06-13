# AetherSpaces / Architect Architecture

This repository is currently loaded as `Architect.jl`, but the project identity
and research direction is **AetherSpaces.jl**: stable functional
autoregression plus opt-in mathematical space diagnostics and regularization.

## System Shape

```text
Architect.jl
  Core/             abstract types, model config, device movement
  Layers/           Linear, Embedding, Norm, Attention, Block, RNN/LSTM/GRU
  Models/           ContinuousModel, SequenceModel, Llama2 checkpoint reader
  Data/             tokenizers, text loaders, functional coefficient loaders
  Training/         initialization, AdamW-style loops, functional training
  Spaces/           typed mathematical spaces and stack utilities
  AetherSpaces/     public facade over Spaces with stable/diagnostic presets
  FAT/              Chebyshev FAT model, stabilizer, rollout metrics, folding
```

The dependency order in `src/Architect.jl` is deliberate:

```text
Core -> Layers -> Models/Data -> Training -> Spaces -> AetherSpaces -> FAT
```

`Spaces` is loaded before `FAT` so FAT training can use
`space_stack_regularizer` without creating a circular dependency.

## Package Identity

- `using Architect` is the current package entry point.
- `Architect.AetherSpaces` is the public subsystem for the Aether direction.
- `FAT` is the stable spectral autoregressive core.
- `Spaces` remains the compatibility namespace for mathematical space adapters.

The repository has not been mechanically renamed. Documentation should call the
research system AetherSpaces and the implementation package Architect.

## Core Modules

### Core

`src/Core/` defines the common type language:

- `AbstractModelConfig`
- `AbstractLayer`
- `AbstractAttention`
- `AbstractBlock`
- recurrent state/cell abstract types
- `ModelConfig`
- `MODEL_PRESETS`
- `to_device`

`ModelConfig` is for discrete sequence models. Functional coefficient models
use `ContinuousConfig` or `FATConfig`.

### Layers

`src/Layers/` contains reusable neural building blocks:

- `Embedding`
- `Norm`
- `Linear`
- `OrthogonalLinear`
- `Attention`
- `FeedForward`
- `Block`
- `RNNCell`, `LSTMCell`, `GRUCell`

The main tensor convention for transformer-like code is:

```text
features x sequence x batch
```

### Models

`src/Models/` contains non-FAT model families:

- `ContinuousModel`: transformer over coefficient sequences using
  `ContinuousConfig`.
- `SequenceModel`: recurrent language model using `SequenceConfig`.
- `Llama2Transformer`: pure Julia reader/runtime for llama2.c-style
  checkpoints.

`ContinuousModel` now uses canonical field vocabulary:

```julia
coeff_embed
spectral_head
```

Compatibility aliases for older `coeff_encoder` / `coeff_head` access remain,
but new code should use shared accessors:

```julia
coefficient_encoder(model)
coefficient_head(model)
coefficient_dimension(model)
context_length(model)
```

### Training

`src/Training/` contains:

- `TrainConfig` for classical sequence-model training.
- `ContinuousTrainConfig` for coefficient models.
- `train!` dispatch entry points.
- `initialize!`, parameter collection, spectral stats, and related utilities.

Training state is represented by `TrainState` or `FATTrainState`, both of which
store loss history.

### FAT

`src/FAT/` is the main research implementation:

- `phase1_hilbert.jl`: Chebyshev basis, projection, evaluation, Hilbert metrics.
- `phase2_spectral_head.jl`: coefficient output head.
- `phase3_stabilizer.jl`: corrected non-expansive Chebyshev stabilizer.
- `phase4_loss.jl`: directional trajectory approval loss.
- `phase5_fat_model.jl`: FAT config, model, generation, training.
- `phase6_data.jl`: continuous-time loaders.
- `phase7_mla.jl`: latent-attention variant.
- `phase8_hardin_taylor.jl`: minimum-complexity prediction utilities.
- `rollout.jl`: canonical rollout evaluator and metrics.
- `folding.jl`: attention probing and soft head folding.
- `fractional.jl`: fractional collocation utilities.

FAT configuration uses Chebyshev order, not `basis_dim`:

```julia
cfg = FATConfig{Float32}(
    K = 9,
    K_anchor = 9,
    P_econ = 4,
    n_embd = 48,
    n_head = 4,
    n_layer = 2,
    block_size = 16,
)
```

### Spaces and AetherSpaces

`src/Spaces/Spaces.jl` defines the experimental space interface:

```julia
to_space(space, coeffs)
from_space(state)
space_distance(space, a, b)
space_project(space, x)
space_regularizer(space, x)
space_metadata(space)
apply_space_stack(coeffs, spaces)
space_stack_regularizer(coeffs, spaces)
space_stack_metrics(coeffs, target, spaces)
```

`src/AetherSpaces/AetherSpaces.jl` provides public presets:

```julia
stable_aether_spaces(Float32; K=9)
diagnostic_aether_spaces(Float32; K=9)
default_aether_spaces(Float32; K=9)
aether_space_metrics(coeffs, target, spaces)
```

Stable training spaces in v1:

- `MetricSpace`
- `MeasureSpace`
- `FractionalOperatorSpace`

Diagnostic-first spaces:

- `TopologicalChartSpace`
- `TropicalSpace`
- `UltrametricSpace`
- `SheafConsistencySpace`
- `CategoricalBridgeSpace`

## Training and Evaluation Flow

```text
coefficient trajectory
  -> FAT or ContinuousModel
  -> one-step prediction loss
  -> optional approval loss
  -> optional MCP regularizer
  -> optional Aether space regularizer
  -> rollout generation
  -> rollout metrics and Aether metrics
```

Default FAT behavior is unchanged unless:

```julia
FATTrainConfig(;
    aether_lambda = 1f-3,
    aether_spaces = stable_aether_spaces(Float32; K=K),
)
```

## Proof Boundary

The proof-backed stability claim is narrow:

- The corrected Chebyshev stabilizer is non-expansive.
- Rollout proofs reason about bounded perturbation growth with that stabilizer.
- AetherSpaces regularizers are experimental diagnostics/losses.
- Invariant preservation and gauge covariance are research goals, not current
  global guarantees.

## Main Scripts

- `scripts/benchmark_function_rollouts.jl`: derivative, oscillator, heat suite.
- `scripts/benchmark_aether_physics.jl`: harmonic, damped, heat, wave,
  fractional physics suite with Aether metrics.
- `scripts/probe_unified_spaces.jl`: deterministic space diagnostics.
- `scripts/ablation_stabilizer_approval.jl`: rollout ablation wrapper.
- `scripts/fold_probe_train_and_fold.jl`: train/probe/fold workflow.
- `scripts/fractional_collocation.jl`: fractional solver probe.

## Extension Rules

- Prefer new dispatch methods over central conditionals.
- Keep coefficient tensors in `features x sequence x batch` order.
- Keep `Spaces` deterministic and typed; make training use opt-in.
- Do not weaken the default Chebyshev FAT path when adding Aether experiments.
- Add tests for shape preservation, finite losses/metrics, and default behavior.
