# Lean Proofs

This directory contains Lean 4 formalization for the proof-backed part of FAT.

The proofs support the narrow stability contract:

- the corrected Chebyshev stabilizer is non-expansive;
- rollout error grows at most linearly under bounded perturbations;
- the `K = 0` coefficient case degenerates to the constant-function case.

AetherSpaces regularizers are not yet formally verified.

## Main Files

- `ArchitectProofs/Stabilizer.lean`
  Proves non-expansivity of the corrected stabilizer
  `Pi = P_p o P_k` in one coefficient/Hilbert space.

- `ArchitectProofs/ErrorBound.lean`
  Proves linear error growth for non-expansive dynamics with bounded
  perturbations.

- `ArchitectProofs/Rollout.lean`
  Specializes rollout error reasoning to the corrected FAT stabilizer.

- `ArchitectProofs/FATGeneralization.lean`
  Formalizes the constant-function / `K = 0` degeneration case.

## Build

From this directory:

```bash
LEAN_CC=./leanc_wrapper.sh lake build
```

The wrapper is used for portability. Linux builds should behave like the
standard Lean toolchain; macOS builds can use the wrapper to work around linker
behavior on newer systems.

## Boundary

The Lean work does not prove that arbitrary AetherSpaces stacks preserve
invariants, prevent rollout drift, or are gauge-covariant. Those are future
research targets.
