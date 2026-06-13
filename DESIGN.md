# Design Notes

This document explains the implementation contracts that matter when changing
the codebase. It reflects the current AetherSpaces direction while keeping the
package reality clear: code is loaded through `Architect.jl`.

## 1. Multiple Dispatch as the API Boundary

The package uses Julia dispatch instead of large runtime switches.

Examples:

```julia
train!(model, data_fn, config)
rollout(model, seed; max_new, apply_stabilizer=true)
space_regularizer(space, coeffs)
space_distance(space, a, b)
```

New model families, spaces, or data loaders should add methods rather than
editing central branches unless a shared contract must change.

## 2. Tensor Layout

The dominant model layout is:

```text
features x sequence x batch
```

For FAT:

```text
(K + 1) x sequence_length x batch
```

For `ContinuousModel`:

```text
basis_dim x sequence_length x batch
```

This layout matches existing layer code and Julia column-major memory behavior.

## 3. Model Configuration Split

Use the right config for the job:

- `ModelConfig`: discrete sequence model presets.
- `ContinuousConfig`: generic coefficient transformer baseline.
- `FATConfig`: Chebyshev FAT model with stabilizer parameters.
- `TrainConfig`: classical/discrete training.
- `ContinuousTrainConfig`: `ContinuousModel` training.
- `FATTrainConfig`: FAT training with approval, MCP, and optional Aether
  regularization.

Current FAT constructor pattern:

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

model = FunctionalAutoregressiveTransformer(cfg)
```

## 4. FAT Core

FAT changes the autoregressive object:

```text
next token -> next function coefficient vector
```

The model predicts Chebyshev coefficient trajectories:

```math
f_{\Theta_t}(x) = \sum_{n=0}^{K} \theta_n^{(t)} T_n(x)
```

Training is teacher-forced:

```text
input[:, t, b]  = true coefficient at step t
target[:, t, b] = true coefficient at step t + 1
```

Generation is autoregressive and may apply the stabilizer after each predicted
step.

## 5. Stabilizer Contract

The implemented stabilizer is:

```math
\Pi = P_p \circ P_k
```

Both projections are Chebyshev truncations in the same Hilbert space. That is
the reason the non-expansive proof applies.

Do not replace this with mixed-basis projections unless the proof and metrics
are updated.

## 6. Loss Design

FAT training combines optional terms:

```math
L =
L_{approval/MSE}
+ \lambda_{MCP} L_{MCP}
+ \lambda_{Aether} \sum_i R_i(\hat{\Theta})
```

Defaults preserve old behavior:

```julia
FATTrainConfig{Float32}(
    aether_lambda = 0f0,
    aether_spaces = (),
)
```

Enable Aether regularization explicitly:

```julia
spaces = stable_aether_spaces(Float32; K=9)

cfg = FATTrainConfig{Float32}(
    aether_lambda = 1f-3,
    aether_spaces = spaces,
)
```

## 7. AetherSpaces Design

AetherSpaces is the research identity for the mathematical space layer. The
implementation is:

- `src/Spaces/Spaces.jl`: core typed space API.
- `src/AetherSpaces/AetherSpaces.jl`: public facade and presets.

Every space should support:

```julia
to_space(space, coeffs)
from_space(state)
space_distance(space, a, b)
space_project(space, coeffs)
space_regularizer(space, coeffs)
space_metadata(space)
```

V1 stable regularization spaces:

- metric
- measure
- fractional operator

V1 diagnostic-first spaces:

- topological chart
- tropical
- ultrametric
- sheaf consistency
- categorical bridge

## 8. Zygote Constraints

Training paths use Zygote. Code inside differentiable loss computation should:

- avoid `setindex!` and `.=` mutation;
- avoid differentiating through fixed adapter metadata;
- keep regularizer helper arrays deterministic and non-trainable;
- return finite scalar losses.

Fixed Aether adapter helpers such as coefficient weights should be treated as
constants, not learned parameters.

## 9. Rollout Evaluation

One-step loss is not enough for the project goal. Canonical rollout metrics are
implemented in `src/FAT/rollout.jl`:

- one-step MSE
- rollout MSE
- final coefficient/Hilbert error
- per-step Hilbert errors
- spectral energy
- spectral energy drift
- stabilizer contraction ratio
- coefficient complexity
- error growth ratios

Aether physics benchmarks add:

- per-space distance
- per-space regularizer
- stack regularizer

## 10. Reproducibility Contract

Serious benchmark scripts should write:

- config
- command args
- git commit and dirty flag
- random seed
- losses
- rollout metrics
- Aether metrics when relevant
- rollout samples
- CSV curves
- SVG plots

The canonical Aether benchmark is:

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```

Use small overrides for smoke tests:

```bash
julia --project=. scripts/benchmark_aether_physics.jl \
  tasks=heat iters=1 K=1 seq_len=2 seed_len=1 horizon=2 \
  batch_size=1 n_embd=4 n_head=1 n_layer=0 outdir=/tmp/aether_smoke
```

## 11. What Not To Claim

Current code supports:

- stable Chebyshev rollout through the FAT stabilizer;
- opt-in Aether diagnostics and regularizers;
- reproducible physics-oriented benchmark bundles.

Current code does not yet prove:

- global invariant preservation;
- gauge covariance;
- that arbitrary models cannot drift;
- that all eight spaces should be used as training losses.
