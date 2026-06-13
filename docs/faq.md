# FAQ

## What is this project called?

The research direction is **AetherSpaces.jl**.

The current Julia package is still loaded as:

```julia
using Architect
```

This keeps the codebase stable while the project identity moves toward
AetherSpaces.

## What is AetherSpaces?

AetherSpaces is a physics-grounded, math-aware research system for continuous
functional trajectories. The current implementation combines:

- FAT: a Chebyshev spectral autoregressive transformer;
- a proof-backed non-expansive Chebyshev stabilizer;
- rollout metrics for long-horizon behavior;
- AetherSpaces: opt-in mathematical spaces for diagnostics and regularization.

## What is FAT?

FAT means Functional Autoregressive Transformer. Instead of predicting the next
token class, it predicts the next coefficient vector of a function:

```math
f_{\Theta_t}(x) = \sum_{n=0}^{K} \theta_n^{(t)} T_n(x)
```

The model autoregresses over coefficient trajectories.

## Is AetherSpaces an LLM?

No. It is not a token LLM. It is a continuous trajectory modeling system.

The relationship to LLMs is conceptual: FAT generalizes autoregression from
discrete token IDs to continuous function descriptors.

## Is the project production-ready?

No. Treat it as a research prototype.

The code has tests and reproducible benchmarks, but the main claims are still
being validated. Use it for research, experiments, and development, not
production-critical deployment.

## What is actually proven?

The proof-backed claim is narrow:

- the corrected Chebyshev stabilizer is non-expansive;
- rollout error has bounded growth under bounded perturbations.

AetherSpaces does not yet prove global invariant preservation, gauge
covariance, or impossibility of rollout drift.

## What are the eight Aether spaces?

- `MetricSpace`
- `MeasureSpace`
- `TopologicalChartSpace`
- `FractionalOperatorSpace`
- `TropicalSpace`
- `UltrametricSpace`
- `SheafConsistencySpace`
- `CategoricalBridgeSpace`

The first three used for regularization in v1 are:

- `MetricSpace`
- `MeasureSpace`
- `FractionalOperatorSpace`

The others are diagnostic-first.

## How do I enable Aether regularization?

```julia
using Architect

K = 9
spaces = stable_aether_spaces(Float32; K=K)

cfg = FATTrainConfig{Float32}(
    aether_lambda = 1f-3,
    aether_spaces = spaces,
)
```

Default FAT training is unchanged when `aether_lambda = 0` or `aether_spaces`
is empty.

## How do I run the main Aether benchmark?

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```

Small smoke test:

```bash
julia --project=. scripts/benchmark_aether_physics.jl \
  tasks=heat iters=1 K=1 seq_len=2 seed_len=1 horizon=2 \
  batch_size=1 n_embd=4 n_head=1 n_layer=0 outdir=/tmp/aether_smoke
```

## What does the benchmark save?

The Aether physics benchmark writes:

- `config.json`
- `git_commit.txt`
- `summary.csv`
- training losses
- rollout metrics
- Aether per-space metrics
- rollout samples
- rollout error CSVs
- SVG plots

Default output directory:

```text
results/aether_physics/
```

## How do I construct a FAT model?

```julia
using Architect

cfg = FATConfig{Float32}(
    K = 9,
    K_anchor = 9,
    P_econ = 4,
    n_embd = 48,
    n_head = 4,
    n_layer = 2,
    block_size = 16,
)

model = FunctionalAutoregressiveTransformer(cfg)
```

Use `K`, not `basis_dim`, for FAT. `basis_dim` belongs to
`ContinuousConfig` and `FunctionSequenceLoader`.

## What shape should coefficient data have?

For training:

```text
coefficient_dimension x sequence_length x batch
```

For rollout:

```text
coefficient_dimension x sequence_length
```

For FAT, `coefficient_dimension = K + 1`.

## What is the difference between ContinuousModel and FAT?

`ContinuousModel` is a generic coefficient transformer baseline:

- uses `ContinuousConfig`;
- trains with MSE;
- no FAT stabilizer;
- no directional approval loss.

FAT:

- uses `FATConfig`;
- predicts Chebyshev coefficient trajectories;
- supports directional approval loss;
- supports the Chebyshev stabilizer during generation;
- supports optional Aether regularization.

## Why does Julia/Zygote take a long time on first run?

The first run compiles model, AD, and benchmark paths. A small Aether training
smoke can still take a while the first time. Later runs are usually faster.

## Why do I see doc replacement warnings for `Linear`?

The current code has duplicate docstrings for several `Linear` constructors.
They are warnings, not failures. They are unrelated to AetherSpaces behavior.

## How do I run focused tests?

```bash
julia --project=. test/test_spaces.jl
julia --project=. test/test_aether_training.jl
```

Full suite:

```bash
julia --project=. test/runtests.jl
```

## Which docs should I read first?

Recommended order:

1. `README.md`
2. `ARCHITECTURE.md`
3. `DESIGN.md`
4. `docs/theory.md`
5. `docs/unified_spaces.md`
6. `docs/guides.md`
