# Topological Folding Design Notes

Topological folding is an experimental analysis path for finding redundant
attention heads and proposing safe merges. It is separate from the proof-backed
Chebyshev stabilizer and from AetherSpaces regularization.

## Current Goal

- collect per-head activations;
- compute similarity metrics such as CKA or cosine similarity;
- propose clusters of redundant heads;
- review proposals before any destructive merge;
- validate every merge with rollout metrics.

## Current APIs

- `ProbeCollector`: stores activations by layer/head key.
- `TrackedAttention`: wraps an attention module and records head outputs.
- `wrap_attentions!(model, collector)`: instruments a model.
- `unwrap_attentions!(model)`: restores original attention modules.
- `collect_head_activations(model, loader; nbatches=20)`: collection helper.
- `compute_similarity(collector; metric=:cka)`: returns `(sim, keys)`.
- `propose_clusters(sim, keys; threshold=0.92f0)`: proposes fold groups.
- `save_similarity_csv(sim, keys, path)`: writes the matrix.
- `FoldedHead`, `apply_soft_merge`, `FoldManager`: early folding utilities.

## Usage

```julia
using Architect

collector = ProbeCollector{Float32}()
wrap_attentions!(model, collector)

# Run representative batches through the model here.

unwrap_attentions!(model)
sim, keys = compute_similarity(collector; metric=:cka)
clusters = propose_clusters(sim, keys; threshold=0.90f0)
save_similarity_csv(sim, keys, "results/head_similarity.csv")
```

## Safety Rule

Treat folding as analysis until a merge has passed:

- one-step loss check;
- rollout metric check;
- stabilizer-on and stabilizer-off comparison;
- task-specific metric review.

Do not claim folding preserves the FAT stabilizer theorem. The theorem applies
to the Chebyshev projection stabilizer, not to arbitrary architectural rewiring.
