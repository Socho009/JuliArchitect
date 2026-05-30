# Topological Folding — Design Notes for Architect.jl

This document sketches a concrete API and integration plan for "Topological Folding"
in the FAT transformer pipeline. The goal is a safe, incremental prototype that
lets us detect redundant attention heads / MLP channels, propose merges, and
exercise soft-folding before any destructive rewiring is performed.

## Goals
- Collect per-head (and per-MLP) activation / gradient statistics
- Compute similarity metrics (CKA, cosine, jacobian sketches)
- Propose clusters of equivalent components
- Provide a safe `soft` merge wrapper (gates) and a `hard` merge fallback
- Integrate minimally with FAT: `src/FAT/folding.jl` + `scripts/fold_probe.jl`

## Key APIs (prototype)

- `ProbeCollector` — accumulates activations per `(layer, head)` during forward passes.

- `wrap_attentions!(model, collector)` — replace each `Block.attention` with a
  `TrackedAttention` wrapper that forwards calls while recording pre-recombination
  per-head outputs.

- `collect_head_activations(model, loader; nbatches=20)` — run `nbatches` through
  `loader`, collect activations via `ProbeCollector` and return it.

- `compute_similarity(collector; metric=:cka)` → `(sim_matrix, keys)` — compute
  pairwise similarity between collected heads. `keys` is ordered list of `(layer, head)`.

- `propose_clusters(sim_matrix, keys; threshold=0.92)` — greedy clustering heuristic
  returning clusters of head indices for merging proposals.

- `FoldManager` (future) — orchestrates periodic analysis, soft-gating, fine-tune
  windows, and optional hard consolidation.

## Merging primitives (design)

- Soft merge (recommended first): `FoldedHead` computes weighted combination of
  original heads' outputs via learned gates `g_i` (initialized to uniform). Train
  with small entropy or L1 penalty to encourage sparsity; once gates converge,
  hard-merge by forming a single head (averaged/distilled weights) and pruning others.

- Hard merge: weight averaging, low-rank fusion, or distilled small network that
  approximates pooled behavior. After hard-merge, re-project weights to satisfy
  Cayley/spectral constraints used by `src/Layers/layers.jl` / stabilizer.

## Stability & FAT considerations

- Do not fold across components that operate on different spectral bands. Prefer
  spectral-aware similarity (per-coefficient contributions) as first-class signal.
- After any merge, run the `SpectralStabilizer` (`Π`) and numerical Lipschitz tests
  to ensure non-expansivity guarantees are preserved (or re-train a short window).

## Integration plan (short)
1. Add `src/FAT/folding.jl` prototype: `ProbeCollector`, `TrackedAttention`, similarity
   utilities, and simple clustering.
2. Add `scripts/fold_probe.jl` to generate proposals on a small FAT model and save
   similarity matrix / cluster proposals for review.
3. Iterate: implement `FoldedHead` soft wrapper + `FoldManager` for dynamic folding
   inside `fat_train!` (Phase 2).

## Usage (example)

Run a quick probe:

```bash
julia --project scripts/fold_probe.jl
```

This will create a small `ContinuousModel`, collect head activations on synthetic
functional data, compute CKA similarity, write `results/head_sim.csv`, and print
cluster proposals.

---
References and detailed merging math are in-code comments inside `src/FAT/folding.jl`.
