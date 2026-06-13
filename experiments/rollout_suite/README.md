# Rollout Suite

This experiment validates the core FAT claim: long-horizon coefficient rollouts
should be measured directly, not inferred from one-step loss.

Run:

```bash
julia --project=. scripts/benchmark_function_rollouts.jl
```

Outputs are written to `results/rollout_suite/` by default:

- `config.json`: command arguments, benchmark config, git commit, dirty flag
- `git_commit.txt`: commit metadata
- `summary.csv`: one row per task/variant with aggregate metrics
- `*_losses.csv`: training losses per iteration
- `*_metrics.json`: full rollout metrics vectors
- `*_rollout_sample.csv`: truth and generated coefficients
- `*_rollout_errors.csv`: rollout error curves
- `*_rollout_errors.svg`: dependency-free plots

Tasks:

- `derivative`: Chebyshev derivative-like coefficient dynamics
- `oscillator`: paired coefficient rotations
- `heat`: diagonal heat-equation-like spectral decay

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
- `oracle_linear` uses the known synthetic transition matrix for the task. It is
  not a learnable competitor; it is a sanity lower bound for these controlled
  tasks.
