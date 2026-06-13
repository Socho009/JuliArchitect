# AetherSpaces Probe

This experiment checks the v1 mathematical-space adapters used by the
AetherSpaces subsystem.

Run:

```bash
julia --project=. scripts/probe_unified_spaces.jl
```

Small custom run:

```bash
julia --project=. scripts/probe_unified_spaces.jl K=5 steps=6 outdir=/tmp/spaces_probe
```

Default output:

```text
results/unified_spaces/space_probe.csv
```

Each row reports:

- space name;
- self-distance;
- regularizer;
- roundtrip error;
- metadata.

The probe is diagnostic only. It does not train FAT and does not alter the
default Chebyshev/FAT path.

Use `stable_aether_spaces(Float32; K=K)` for the three regularization-ready
spaces and `diagnostic_aether_spaces(Float32; K=K)` for all eight spaces.
