# AetherSpaces.jl Documentation

AetherSpaces.jl is the research direction of this repository: continuous
functional AI built from a stable spectral autoregressive core plus
mathematical space diagnostics and regularizers.

The current Julia package is still loaded as:

```julia
using Architect
```

## Project Map

- `Architect.jl`: current package implementation.
- `FAT`: Functional Autoregressive Transformer over Chebyshev coefficients.
- `AetherSpaces`: public facade for mathematical spaces.
- `Spaces`: compatibility namespace and concrete space implementations.
- `ContinuousModel`: coefficient-transformer baseline.
- `SequenceModel`: recurrent language-model family.
- `Llama2Transformer`: llama2.c checkpoint reader/runtime.

## Installation

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Architect; println("ok")'
```

## Minimal FAT Example

```julia
using Architect

cfg = FATConfig{Float32}(
    K = 5,
    K_anchor = 5,
    P_econ = 3,
    n_embd = 16,
    n_head = 4,
    n_layer = 1,
    block_size = 8,
)

model = FunctionalAutoregressiveTransformer(cfg)
seed = randn(Float32, cfg.K + 1, 4)
generated = fat_generate(model, seed; max_new=2, apply_stabilizer=true)
```

Tensor shape is:

```text
coefficient_dimension x sequence_length x batch
```

For matrix rollouts without a batch axis:

```text
coefficient_dimension x sequence_length
```

## Minimal AetherSpaces Example

```julia
using Architect

K = 5
coeffs = randn(Float32, K + 1, 8)
target = 0.95f0 .* coeffs

spaces = stable_aether_spaces(Float32; K=K)
reg = space_stack_regularizer(coeffs, spaces)
metrics = aether_space_metrics(coeffs, target, spaces)
```

Use `diagnostic_aether_spaces(Float32; K=K)` for all eight spaces.

## Optional Aether-Regularized FAT Training

```julia
using Architect

K = 5
model = FunctionalAutoregressiveTransformer(FATConfig{Float32}(
    K = K,
    K_anchor = K,
    P_econ = 3,
    n_embd = 16,
    n_head = 4,
    n_layer = 1,
    block_size = 8,
))

spaces = stable_aether_spaces(Float32; K=K)

train_cfg = FATTrainConfig{Float32}(
    learning_rate = 3f-4,
    max_iters = 10,
    aether_lambda = 1f-3,
    aether_spaces = spaces,
)

batch_fn = function ()
    x = randn(Float32, K + 1, 8, 2)
    y = 0.98f0 .* x
    return x, y
end

state = train!(model, batch_fn, train_cfg)
```

Defaults keep Aether regularization off:

```julia
FATTrainConfig{Float32}().aether_lambda == 0f0
```

## Benchmarks

Rollout suite:

```bash
julia --project=. scripts/benchmark_function_rollouts.jl
```

Aether physics suite:

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```

Small Aether smoke:

```bash
julia --project=. scripts/benchmark_aether_physics.jl \
  tasks=heat iters=1 K=1 seq_len=2 seed_len=1 horizon=2 \
  batch_size=1 n_embd=4 n_head=1 n_layer=0 outdir=/tmp/aether_smoke
```

Outputs include config, git metadata, losses, rollout metrics, Aether metrics,
samples, CSV curves, and SVG plots.

## Main APIs

### FAT

```julia
FATConfig
FunctionalAutoregressiveTransformer
FATTrainConfig
FATTrainState
fat_forward
fat_train!
fat_generate
rollout
evaluate_rollout
rollout_metrics
```

### AetherSpaces

```julia
AbstractMathSpace
SpaceState
MetricSpace
MeasureSpace
TopologicalChartSpace
FractionalOperatorSpace
TropicalSpace
UltrametricSpace
SheafConsistencySpace
CategoricalBridgeSpace

to_space
from_space
space_distance
space_project
space_regularizer
space_metadata
apply_space_stack
space_stack_regularizer
space_stack_metrics

stable_aether_spaces
diagnostic_aether_spaces
default_aether_spaces
aether_space_metrics
```

### Baselines

```julia
ContinuousConfig
ContinuousModel
ContinuousTrainConfig
SequenceConfig
SequenceModel
Llama2Config
Llama2Transformer
```

## Theory Boundary

The proof-backed stability statement is:

- the corrected Chebyshev stabilizer is non-expansive;
- rollout error has bounded growth under bounded perturbations.

AetherSpaces currently provides experimental diagnostics and opt-in
regularization. It does not yet prove global invariant preservation or gauge
covariance.

## Related Docs

- [README](../README.md)
- [Architecture](../ARCHITECTURE.md)
- [Design Notes](../DESIGN.md)
- [Aether mathematical frame](aether_math.md)
- [Theory](theory.md)
- [AetherSpaces subsystem](unified_spaces.md)
- [Guides](guides.md)
- [FAQ](faq.md)
