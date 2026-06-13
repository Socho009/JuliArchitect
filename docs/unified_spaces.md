# AetherSpaces Mathematical Space Subsystem

The AetherSpaces subsystem is the mathematical lens stack around FAT. In the
current package it is implemented by `src/Spaces/` and exposed through the
public `Architect.AetherSpaces` facade. It does not replace the Chebyshev/FAT
path. Instead, it gives the project a typed way to ask how a coefficient
trajectory behaves under classical, geometric, algebraic, and local-to-global
views.

## Interface

Every space implements the same small API:

```julia
to_space(space, coeffs)
from_space(state)
space_distance(space, a, b)
space_project(space, x)
space_regularizer(space, x)
space_metadata(space)
```

`SpaceState` stores native coordinates plus the space object:

```julia
SpaceState(coords, space)
```

The default FAT forward pass is unchanged. These spaces are opt-in diagnostics,
regularizers, and adapters.

## Public Facade

Normal package loading remains:

```julia
using Architect
```

The compatibility API is exported at the top level, and the Aether identity is
available as a namespace:

```julia
Architect.AetherSpaces.stable_aether_spaces(Float32; K=9)
Architect.AetherSpaces.diagnostic_aether_spaces(Float32; K=9)
```

The top-level helpers are also exported:

```julia
stable_aether_spaces(Float32; K=9)
diagnostic_aether_spaces(Float32; K=9)
default_aether_spaces(Float32; K=9)
aether_space_metrics(coeffs, target, spaces)
```

## V1 Spaces

- `MetricSpace`: Euclidean or weighted coefficient metrics. Regularization-ready.
- `MeasureSpace`: weighted norms/integration over coefficient axes. Regularization-ready.
- `FractionalOperatorSpace`: fractional-order coefficient weighting. Regularization-ready.
- `TopologicalChartSpace`: local chart projection and overlap residual. Diagnostic-first.
- `TropicalSpace`: sign-preserving log/tropical coordinate view. Diagnostic-first.
- `UltrametricSpace`: p-adic-inspired valuation coordinates and ultrametric distance. Diagnostic-first.
- `SheafConsistencySpace`: local-to-global gluing residual. Diagnostic-first.
- `CategoricalBridgeSpace`: commutator residual for structure-preservation diagrams. Diagnostic-first.

## Stack Utilities

```julia
apply_space_stack(coeffs, spaces)
space_stack_regularizer(coeffs, spaces)
space_stack_metrics(coeffs, target, spaces)
```

These utilities let experiments apply multiple spaces without changing
`FATConfig` or `fat_forward`.

## FAT Training Integration

Aether regularization is opt-in through `FATTrainConfig`:

```julia
spaces = stable_aether_spaces(Float32; K=9)

cfg = FATTrainConfig{Float32}(;
    aether_lambda = 1f-3,
    aether_spaces = spaces,
)
```

With the defaults, `aether_lambda = 0` and `aether_spaces = ()`, existing FAT
training is unchanged.

The paper-oriented physics entry point is:

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```

## Design Rule

The classical Chebyshev stabilizer remains the only proof-backed stabilizer.
Aether spaces are experimental and should be validated through diagnostics
before being treated as invariant-preserving mechanisms. Metric, measure, and
fractional spaces are the first regularization targets; tropical, ultrametric,
sheaf, and categorical spaces should remain diagnostic-first until benchmark
evidence supports using them as losses.
