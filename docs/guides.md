# Guides

These guides use the current codebase API. The package is loaded as
`Architect`; the research identity is AetherSpaces.

## 1. Run the Aether Physics Benchmark

Full benchmark:

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```

Fast smoke test:

```bash
julia --project=. scripts/benchmark_aether_physics.jl \
  tasks=heat iters=1 K=1 seq_len=2 seed_len=1 horizon=2 \
  batch_size=1 n_embd=4 n_head=1 n_layer=0 outdir=/tmp/aether_smoke
```

Important arguments:

- `tasks=harmonic,damped,heat,wave,fractional`
- `iters=120`
- `K=9`
- `horizon=32`
- `aether_lambda=1f-3`
- `outdir=results/aether_physics`

The output bundle contains:

- `config.json`
- `git_commit.txt`
- `summary.csv`
- per-variant losses
- rollout metrics JSON
- Aether metrics JSON
- rollout sample CSVs
- rollout error CSVs
- rollout error SVG plots

## 2. Train FAT With Optional Aether Regularization

```julia
using Architect

K = 5

model = FunctionalAutoregressiveTransformer(FATConfig{Float32}(
    K = K,
    K_anchor = K,
    P_econ = 3,
    n_embd = 16,
    n_head = 4,
    n_layer = 1,
    block_size = 8,
))

spaces = stable_aether_spaces(Float32; K=K)

cfg = FATTrainConfig{Float32}(
    learning_rate = 3f-4,
    max_iters = 20,
    warmup_iters = 10,
    aether_lambda = 1f-3,
    aether_spaces = spaces,
)

batch_fn = function ()
    x = randn(Float32, K + 1, 8, 4)
    y = 0.98f0 .* x
    return x, y
end

state = train!(model, batch_fn, cfg)
println(state.losses[end])
```

Set `aether_lambda = 0` or omit `aether_spaces` to recover default FAT
training.

## 3. Evaluate Long-Horizon Rollout

```julia
using Architect

K = 5
model = FunctionalAutoregressiveTransformer(FATConfig{Float32}(
    K = K,
    K_anchor = K,
    P_econ = 3,
    n_embd = 16,
    n_head = 4,
    n_layer = 1,
    block_size = 8,
))

truth = randn(Float32, K + 1, 20)
predicted, metrics = evaluate_rollout(
    model,
    truth;
    seed_len = 8,
    max_new = 12,
    apply_stabilizer = true,
)

println(metrics.rollout_mse)
println(metrics.spectral_energy_drift)
```

Rollout metrics are the key evaluation surface for FAT and AetherSpaces.
One-step MSE alone is not enough.

## 4. Probe the Eight Aether Spaces

```bash
julia --project=. scripts/probe_unified_spaces.jl
```

Small custom run:

```bash
julia --project=. scripts/probe_unified_spaces.jl K=5 steps=6 outdir=/tmp/spaces_probe
```

Programmatic use:

```julia
using Architect

K = 5
coeffs = randn(Float32, K + 1, 8)
target = 0.9f0 .* coeffs

spaces = diagnostic_aether_spaces(Float32; K=K)
metrics = space_stack_metrics(coeffs, target, spaces)
regularizer = space_stack_regularizer(coeffs, spaces)
```

## 5. Train the ContinuousModel Baseline

```julia
using Architect

loader = FunctionSequenceLoader{Float32}(
    task = :derivative,
    basis_dim = 8,
    seq_length = 16,
    batch_size = 16,
)

model = ContinuousModel(ContinuousConfig{Float32}(
    basis_dim = 8,
    basis_type = :monomial,
    n_embd = 32,
    n_head = 4,
    n_layer = 1,
    block_size = 16,
))

cfg = ContinuousTrainConfig{Float32}(
    learning_rate = 3f-4,
    max_iters = 20,
)

state = train!(model, loader, cfg)
```

`ContinuousModel` is useful as a transformer coefficient baseline. It does not
include the FAT stabilizer or approval loss.

## 6. Run the Existing Rollout Suite

```bash
julia --project=. scripts/benchmark_function_rollouts.jl
```

Small run:

```bash
julia --project=. scripts/benchmark_function_rollouts.jl \
  iters=1 K=1 seq_len=2 seed_len=1 horizon=2 \
  batch_size=1 n_embd=4 n_head=1 n_layer=0 outdir=/tmp/rollout_smoke
```

This suite covers derivative-like, oscillator, and heat dynamics. It compares
FAT variants against `ContinuousModel`, persistence, and oracle baselines.

## 7. Topological Folding Probe

The folding utilities are experimental. Typical flow:

```julia
using Architect

collector = ProbeCollector{Float32}()
wrap_attentions!(model, collector)

# Run representative batches through the model here.

unwrap_attentions!(model)
sim, keys = compute_similarity(collector; metric=:cka)
clusters = propose_clusters(sim, keys; threshold=0.90f0)
save_similarity_csv(sim, keys, "head_similarity.csv")
```

Use folding as analysis first. Apply merges only after checking rollout metrics.

## 8. Run Tests

Focused tests:

```bash
julia --project=. test/test_spaces.jl
julia --project=. test/test_aether_training.jl
```

Full suite:

```bash
julia --project=. test/runtests.jl
```

The Aether/FAT Zygote training path can be compilation-heavy on first run.
