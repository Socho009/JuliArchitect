# Rollout Suite

This experiment validates the core FAT evaluation rule: long-horizon
coefficient rollouts must be measured directly, not inferred from one-step
loss.

Run:

```bash
julia --project=. scripts/benchmark_function_rollouts.jl
```

Small smoke:

```bash
julia --project=. scripts/benchmark_function_rollouts.jl \
  iters=1 K=1 seq_len=2 seed_len=1 horizon=2 \
  batch_size=1 n_embd=4 n_head=1 n_layer=0 outdir=/tmp/rollout_smoke
```

Default output directory:

```text
results/rollout_suite/
```

Outputs:

- `config.json`: command arguments, benchmark config, git commit, dirty flag.
- `git_commit.txt`: commit metadata.
- `summary.csv`: one row per task and variant.
- `*_losses.csv`: training losses.
- `*_metrics.json`: rollout metrics vectors.
- `*_rollout_sample.csv`: truth and generated coefficients.
- `*_rollout_errors.csv`: rollout error curves.
- `*_rollout_errors.svg`: dependency-free plots.

Tasks:

- `derivative`: Chebyshev derivative-like coefficient dynamics.
- `oscillator`: paired coefficient rotations.
- `heat`: diagonal heat-equation-like spectral decay.

Variants:

- `fat_approval_stabilized`
- `fat_approval_unstabilized`
- `fat_mse_stabilized`
- `continuous_transformer`
- `persistence`
- `oracle_linear`

Interpretation:

- `continuous_transformer` is the non-FAT coefficient transformer baseline.
- `persistence` repeats the last seed coefficient vector.
- `oracle_linear` uses the known synthetic transition matrix and is a sanity
  lower bound, not a learned competitor.

For the AetherSpaces physics benchmark, use:

```bash
julia --project=. scripts/benchmark_aether_physics.jl
```
