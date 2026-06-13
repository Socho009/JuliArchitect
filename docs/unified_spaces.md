# Unified Exotic Mathematical Spaces

The `Spaces` subsystem is an experimental interface layer around FAT. It does
not replace the Chebyshev/FAT path. Instead, it gives the project a typed way to
ask how a coefficient trajectory behaves under non-classical mathematical
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

## V1 Spaces

- `MetricSpace`: Euclidean or weighted coefficient metrics.
- `MeasureSpace`: weighted norms/integration over coefficient axes.
- `TopologicalChartSpace`: local chart projection and overlap residual.
- `FractionalOperatorSpace`: fractional-order coefficient weighting.
- `TropicalSpace`: sign-preserving log/tropical coordinate view.
- `UltrametricSpace`: p-adic-inspired valuation coordinates and ultrametric distance.
- `SheafConsistencySpace`: local-to-global gluing residual.
- `CategoricalBridgeSpace`: commutator residual for structure-preservation diagrams.

## Stack Utilities

```julia
apply_space_stack(coeffs, spaces)
space_stack_regularizer(coeffs, spaces)
space_stack_metrics(coeffs, target, spaces)
```

These utilities let experiments apply multiple spaces without changing
`FATConfig` or `fat_forward`.

## Design Rule

The classical Chebyshev stabilizer remains the only proof-backed stabilizer.
The exotic spaces are experimental and should be validated through diagnostics
before becoming trainable layers.
