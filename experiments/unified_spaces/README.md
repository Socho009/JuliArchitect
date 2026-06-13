# Unified Spaces Probe

This experiment checks the v1 exotic mathematical-space adapters.

Run:

```bash
julia --project=. scripts/probe_unified_spaces.jl
```

Default output:

```text
results/unified_spaces/space_probe.csv
```

Each row reports:

- space name
- self-distance
- regularizer
- roundtrip error
- metadata

The probe is diagnostic only. It does not train FAT and does not alter the
default Chebyshev/FAT path.
