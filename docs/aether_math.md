# AetherSpaces Mathematical Frame

This note gives the clean mathematical story of the project. It is not a
marketing description and not a proof file. It is the conceptual bridge between
the code, the physics benchmarks, and the future theory.

## 1. The Object Being Modeled

The basic datum is not a token and not a scalar. It is a trajectory of
functions:

```math
(f_1,\ldots,f_T), \qquad f_t \in H.
```

In the current FAT path, `H` is represented through a finite Chebyshev
coefficient space. Each function is projected to:

```math
\Theta_t = \operatorname{coeff}_K(f_t) \in \mathbb{R}^{K+1}.
```

So the learning problem becomes:

```math
(\Theta_1,\ldots,\Theta_T)
\quad \mapsto \quad
(\Theta_2,\ldots,\Theta_{T+1}).
```

The model does not learn a vocabulary distribution. It learns a finite-context
operator over coefficient trajectories:

```math
G_\psi : (\mathbb{R}^{K+1})^L \to \mathbb{R}^{K+1}.
```

Generation is:

```math
\widehat{\Theta}_{t+1}
=
\Pi\left(
G_\psi(\widehat{\Theta}_{t-L+1},\ldots,\widehat{\Theta}_t)
\right).
```

Here `Pi` is the Chebyshev stabilizer.

## 2. The Stable Core

The proof-backed stabilizer is:

```math
\Pi = P_p \circ P_k,
\qquad
0 \le p \le k \le K.
```

Both maps are truncation projections in the same Chebyshev/Hilbert coefficient
space. Therefore:

```math
\|\Pi u - \Pi v\|_{H_w}
\le
\|u - v\|_{H_w}.
```

This is the stable spine of the project:

```text
function trajectory
  -> Chebyshev coefficients
  -> autoregressive operator
  -> non-expansive stabilizer
  -> rollout metrics
```

Everything else must preserve this spine or remain opt-in.

## 3. The Aether Layer

AetherSpaces adds mathematical observers around the coefficient trajectory.
For each mathematical space `S_i`, define:

```math
\Phi_i : \mathbb{R}^{K+1} \to S_i
```

and, when meaningful:

```math
\Psi_i : S_i \to \mathbb{R}^{K+1}.
```

In code:

```julia
to_space(space, coeffs)      # Phi_i
from_space(state)            # Psi_i
space_distance(space, a, b)  # d_i
space_project(space, x)      # projection/validity map
space_regularizer(space, x)  # R_i
space_metadata(space)        # diagnostics
```

The Aether view of one coefficient vector is:

```math
\Theta
\quad \mapsto \quad
\left(
\Phi_1(\Theta),
\Phi_2(\Theta),
\ldots,
\Phi_m(\Theta)
\right).
```

The Aether view of a rollout is:

```math
(\widehat{\Theta}_1,\ldots,\widehat{\Theta}_T)
\quad \mapsto \quad
\left\{
\Phi_i(\widehat{\Theta}_t)
\right\}_{i,t}.
```

This is why AetherSpaces is better understood as a lens stack than as eight
ordinary neural layers.

## 4. The Eight Observers

The current eight spaces are:

| Space | Mathematical role | V1 status |
|---|---|---|
| `MetricSpace` | native distance in coefficient space | regularization-ready |
| `MeasureSpace` | weighted domain or coefficient importance | regularization-ready |
| `FractionalOperatorSpace` | non-integer operator scaling | regularization-ready |
| `TopologicalChartSpace` | local chart consistency | diagnostic-first |
| `TropicalSpace` | max-plus/min-plus style geometry | diagnostic-first |
| `UltrametricSpace` | hierarchical valuation geometry | diagnostic-first |
| `SheafConsistencySpace` | local-to-global gluing residual | diagnostic-first |
| `CategoricalBridgeSpace` | diagram/commutator residual | diagnostic-first |

The stable preset is:

```julia
stable_aether_spaces(Float32; K=K)
```

The full diagnostic preset is:

```julia
diagnostic_aether_spaces(Float32; K=K)
```

## 5. Residuals as Mathematical Instruments

Each space defines a residual:

```math
R_i(\Theta) \ge 0.
```

Examples:

```math
R_{\mathrm{metric}}(\Theta)
=
\|\Theta\|_W,
```

```math
R_{\mathrm{frac}}(\Theta)
=
\left\|
\left((1+n)^\alpha \theta_n\right)_{n=0}^K
\right\|^2,
```

```math
R_{\mathrm{sheaf}}(\Theta)
=
\text{gluing mismatch between local coefficient patches}.
```

These residuals are not automatically laws of nature. They are instruments.
They measure how a generated trajectory behaves when seen through a particular
mathematical structure.

## 6. Aether-Regularized Learning

The current opt-in training objective is:

```math
L
=
L_{\mathrm{approval/MSE}}
+
\lambda_{\mathrm{MCP}} L_{\mathrm{MCP}}
+
\lambda_{\mathrm{Aether}}
\sum_i R_i(\widehat{\Theta}).
```

In code:

```julia
spaces = stable_aether_spaces(Float32; K=K)

cfg = FATTrainConfig{Float32}(
    aether_lambda = 1f-3,
    aether_spaces = spaces,
)
```

When:

```julia
aether_lambda = 0
aether_spaces = ()
```

the default FAT path is unchanged.

## 7. The Commuting Diagram Goal

The long-term mathematical goal is not merely to reduce MSE. It is to make the
following diagram approximately commute:

```text
coefficient trajectory  --FAT rollout-->  predicted trajectory
        |                                      |
        | Phi_i                                | Phi_i
        v                                      v
space S_i trajectory  --native dynamics-->  predicted S_i trajectory
```

For each space `S_i`, the project asks:

```math
d_i
\left(
\Phi_i(\widehat{\Theta}_{1:T}),
\Phi_i(\Theta^*_{1:T})
\right)
\quad \text{small?}
```

and:

```math
R_i(\widehat{\Theta}_{1:T})
\quad \text{controlled?}
```

This is the mathematical meaning of "multi-space diagnostics."

## 8. Physics Interpretation

For physics trajectories, different spaces detect different failure modes:

- Hilbert error measures coefficient rollout error.
- Spectral energy drift measures artificial growth or decay.
- Measure weights express which modes or regions matter more.
- Fractional scaling tests memory/nonlocal operator behavior.
- Sheaf residuals test local-to-global consistency.
- Categorical residuals test whether chosen transformations preserve structure.

The canonical physics benchmark is:

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```

It evaluates:

- harmonic oscillator;
- damped oscillator;
- heat equation spectral decay;
- wave-like propagation;
- fractional diffusion.

## 9. What Is New Here

The project can be summarized as:

```text
FAT gives a stable autoregressive operator in spectral space.
AetherSpaces gives a typed family of mathematical observers around that operator.
The benchmark asks whether those observers improve long-horizon physical rollout.
```

The novelty is not any single space by itself. The novelty is the unified
interface:

```math
(\Phi_i, \Psi_i, d_i, P_i, R_i)_{i=1}^m
```

wrapped around a stable coefficient autoregressor.

## 10. Proof Boundary

Current formal support:

- Chebyshev stabilizer non-expansivity.
- Rollout error growth under bounded perturbations.
- Constant-function degeneration for `K = 0`.

Current experimental support:

- Aether space construction.
- deterministic projections/distances/regularizers.
- optional Aether regularization in FAT training.
- physics benchmark outputs with Aether metrics.

Not yet proven:

- global invariant preservation;
- gauge covariance;
- that Aether penalties always improve rollout;
- that all eight spaces should be used as losses;
- that generated trajectories cannot drift.

The right scientific claim today is:

> AetherSpaces turns FAT into a measurable, regularizable, multi-space
> trajectory system while keeping the proof-backed Chebyshev stabilizer as the
> stable core.

## 11. Implementation Dictionary

| Math | Code |
|---|---|
| coefficient vector `Theta` | `Array{Float32}(K+1, ...)` |
| autoregressive operator `G_psi` | `FunctionalAutoregressiveTransformer` |
| stabilizer `Pi` | `SpectralStabilizer` |
| rollout | `fat_generate`, `rollout`, `evaluate_rollout` |
| rollout metrics | `RolloutMetrics` |
| observer space `S_i` | subtype of `AbstractMathSpace` |
| map `Phi_i` | `to_space` |
| inverse/return map `Psi_i` | `from_space` |
| native metric `d_i` | `space_distance` |
| projection `P_i` | `space_project` |
| residual `R_i` | `space_regularizer` |
| space stack | `stable_aether_spaces`, `diagnostic_aether_spaces` |
| Aether loss weight | `FATTrainConfig.aether_lambda` |

## 12. The Guiding Equation

The project is guided by:

```math
\widehat{\Theta}_{t+1}
=
\Pi
\left(
G_\psi(\widehat{\Theta}_{t-L+1:t})
\right),
\qquad
\mathcal{A}(\widehat{\Theta})
=
\sum_i R_i(\Phi_i(\widehat{\Theta})).
```

The first equation generates. The second equation observes and regularizes.
Together they define the current AetherSpaces program.
