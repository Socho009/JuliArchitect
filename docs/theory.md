# FAT Theory Note

This note states the research contract that should guide the refactoring
branch. The project is strongest when the math, implementation, and benchmark
plots all test the same claim:

> FAT is a stable autoregressive neural operator over spectral coefficient
> trajectories.

## Learning Problem

Training data is a set of function trajectories:

```math
(f_1,\ldots,f_T), \qquad f_t \in H.
```

After projection to a finite basis:

```math
\Theta_t = \operatorname{coeff}_K(f_t) \in \mathbb{R}^{K+1}.
```

The model learns a finite-context autoregressive operator:

```math
G_\psi : (\mathbb{R}^{K+1})^L \to \mathbb{R}^{K+1}.
```

Generation uses:

```math
\Theta_{t+1} = \Pi(G_\psi(\Theta_{t-L+1},\ldots,\Theta_t)).
```

## Decomposed Error Target

A useful theory target is a decomposition of rollout error into:

```math
\text{rollout error}
\le
\text{projection error}
+ \text{estimation error}
+ \text{optimization error}
+ \text{stability error}.
```

The refactoring should make each term measurable:

- projection error: compare full functions against truncated coefficients
- estimation error: train/validation gap on coefficient trajectories
- optimization error: final train loss and gradient behavior
- stability error: long-horizon growth of coefficient-space rollout error

## Stabilizer

The implemented stabilizer is Chebyshev-only:

```math
\Pi = P_p \circ P_k, \qquad 0 \le p \le k \le K.
```

Both projections are orthogonal truncations in the same coefficient/Hilbert
space, so:

```math
\|\Pi f - \Pi g\|_{H_w} \le \|f-g\|_{H_w}.
```

The Lean proof is in:

- `proofs/ArchitectProofs/Stabilizer.lean`
- `proofs/ArchitectProofs/Rollout.lean`

The rollout theorem states that if perturbations are bounded step by step, the
total generated-orbit error is bounded by the sum of those perturbations.

## Loss

The directional approval loss compares coefficient-space displacements:

```math
\Delta_t^{pred} = \Theta_{t+1}^{pred} - \Theta_t,
\qquad
\Delta_t^* = \Theta_{t+1}^* - \Theta_t.
```

```math
L_{flow}
=
1 -
\mathbb{E}_t
\left[
\left\langle
\frac{\Delta_t^{pred}}{\|\Delta_t^{pred}\|+\epsilon},
\frac{\Delta_t^*}{\|\Delta_t^*\|+\epsilon}
\right\rangle
\right].
```

Training should report both one-step loss and rollout behavior, because a
directionally aligned model can still have amplitude drift.

## Benchmark Ladder

The minimal benchmark ladder is:

1. derivative-like coefficient dynamics
2. oscillator coefficient dynamics
3. heat-equation-like spectral decay

These are implemented in `scripts/benchmark_function_rollouts.jl`.

For each task, the benchmark compares:

- FAT trained with approval loss and stabilized rollout
- the same model with unstabilized rollout
- FAT trained with approval disabled and stabilized rollout

The expected scientific outcome is not merely lower one-step MSE. The target is
controlled long-horizon error growth with the stabilizer active.

The benchmark writes the reproducibility bundle required for comparing runs:

- config and command arguments
- git commit and dirty flag
- random seed
- per-iteration losses
- aggregate metrics
- full metrics vectors
- generated rollout samples
- CSV curves
- SVG plots

The canonical metric set is:

- one-step MSE
- rollout MSE
- coefficient/Hilbert norm error
- spectral energy drift
- stabilizer contraction ratio
- coefficient complexity

## Refactoring Rule

Every future abstraction should preserve this chain:

```text
function trajectory
  -> spectral coefficients
  -> autoregressive model
  -> non-expansive stabilizer
  -> rollout metrics
  -> proof/benchmark alignment
```

If a feature does not make one of those links clearer, more measurable, or more
stable, keep it outside the core API.

## Unified Exotic Spaces

AetherSpaces is the project direction for combining stable spectral
autoregression with mathematical diagnostics over physical trajectories. The
current implementation is the `Spaces` subsystem plus the public
`Architect.AetherSpaces` facade. It extends the theory surface without changing
the default FAT path and introduces a common adapter API:

```julia
to_space(space, coeffs)
from_space(state)
space_distance(space, a, b)
space_project(space, x)
space_regularizer(space, x)
```

The v1 goal is diagnostic and theoretical: metric, measure, topological,
fractional, tropical, ultrametric, sheaf, and categorical views can all inspect
the same coefficient trajectory. They are opt-in and experimental. The
Chebyshev stabilizer remains the only proof-backed stabilizer.

## Aether Objective

The research claim for the next milestone is:

> AetherSpaces learns and evaluates continuous physical trajectories by
> combining stable spectral autoregression with multi-space mathematical
> diagnostics and regularization.

This has three aligned surfaces:

- AI: FAT is an autoregressive neural operator over spectral coefficients.
- Physics: benchmarks measure rollout stability, energy drift, damping,
  propagation, and fractional decay.
- Mathematics: Aether spaces expose Hilbert, weighted-measure, fractional,
  tropical, ultrametric, sheaf-style, and categorical residuals.

## Aether-Regularized Training

The default FAT loss remains unchanged. Opt-in Aether regularization adds a
space penalty:

```math
L =
L_{one-step/approval}
+ \lambda_{MCP} L_{MCP}
+ \lambda_{Aether} \sum_i R_i(\hat{\Theta}),
```

where each `R_i` is computed by `space_regularizer` through the selected
Aether spaces. In code this is controlled by `FATTrainConfig`:

```julia
FATTrainConfig(;
    aether_lambda = 1f-3,
    aether_spaces = stable_aether_spaces(Float32; K=K),
)
```

The default is `aether_lambda = 0` and `aether_spaces = ()`, which preserves
the existing FAT training path.

## Aether Physics Ladder

The canonical physics-first benchmark is:

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```

It evaluates harmonic oscillator, damped oscillator, heat equation spectral
decay, wave-like propagation, and fractional diffusion dynamics. For each task
it compares:

- FAT with stabilizer.
- FAT without stabilizer.
- FAT with Aether regularization and stabilizer.
- FAT with Aether regularization without stabilizer.
- ContinuousModel baseline.
- persistence baseline.
- linear oracle baseline.

The benchmark writes config, git metadata, losses, rollout metrics, Aether
per-space metrics, generated samples, CSV curves, and SVG plots under
`results/aether_physics/`.

## Proof Boundary

Aether regularizers are experimental. They can shape training and expose
failure modes, but they do not yet prove invariant preservation or gauge
covariance. The only formal non-expansive guarantee remains the corrected
Chebyshev stabilizer. Future proof work should start with metric, measure, and
fractional spaces before attempting tropical, sheaf, or categorical statements.
