# AetherSpaces.jl

AetherSpaces.jl is the project goal: a Julia research system for continuous
functional AI that routes spectral trajectories through classical and exotic
mathematical spaces. The current implementation package is still named
`Architect.jl`; that package contains the Functional Autoregressive Transformer
(FAT), the experimental `Spaces` subsystem, conventional sequence-model
building blocks, training code, benchmarks, and formal proof work.

The central direction is Aether: move beyond one-step token or scalar
prediction by representing each step as a function, encoding that function by
spectral coefficients, autoregressing in coefficient space, and then evaluating
the resulting trajectory through multiple mathematical spaces.

Current status: research prototype. The core package loads as a local Julia
package, the main models and tests live under `src/` and `test/`, FAT theory is
partly machine-checked under `proofs/`, and several scripts in `scripts/`
exercise benchmarks, folding probes, and fractional collocation experiments.

## Contents

- [Core Idea](#core-idea)
- [Aether Project Goal](#aether-project-goal)
- [Research Contributions](#research-contributions)
- [Unified Mathematical Framework](#unified-mathematical-framework)
- [Mathematical Model](#mathematical-model)
- [Theoretical Positioning](#theoretical-positioning)
- [Implemented Architectures](#implemented-architectures)
- [Repository Map](#repository-map)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [FAT API](#fat-api)
- [Unified Spaces API](#unified-spaces-api)
- [Training and Data](#training-and-data)
- [Topological Folding](#topological-folding)
- [Formal Proofs](#formal-proofs)
- [Scripts](#scripts)
- [Tests](#tests)
- [Theory Note](#theory-note)
- [Refactoring Notes](#refactoring-notes)

## Core Idea

Standard autoregressive language models predict a categorical next token:

```math
p(x_{t+1} \mid x_1,\ldots,x_t) \in \Delta^{|V|}
```

FAT instead predicts a function represented in a finite spectral basis. In the
current implementation the primary basis is Chebyshev:

```math
f_{\Theta_t}(x) = \sum_{n=0}^{K} \theta_n^{(t)} T_n(x),
\qquad \Theta_t \in \mathbb{R}^{K+1}.
```

The transformer sees a sequence of coefficient vectors:

```text
(K+1, sequence_length, batch)
```

and predicts the next coefficient vector at each position. This keeps the
causal transformer backbone familiar while changing the input and output
spaces from discrete IDs to continuous function descriptors.

The discrete LM case is recovered when `K = 0`: every function is constant, so
each step is a scalar coefficient.

## Aether Project Goal

**AetherSpaces.jl** is the target identity for the project. The goal is to turn
the current FAT and `Spaces` code into a coherent system for continuous
trajectory modeling, geometric diagnostics, and eventually invariant-aware
training. In the current codebase, the Aether layer is implemented as the
experimental `src/Spaces/` subsystem inside `Architect.jl`; the repository has
not yet been mechanically renamed.

Tagline:

> A mathematical lens stack for functional trajectories and structural
> invariants.

The practical goal is to let FAT keep its stable Chebyshev coefficient path
while adding opt-in views into non-Euclidean, weighted, fractional, tropical,
ultrametric, sheaf-style, and categorical diagnostic spaces. These spaces are
not ordinary deep-learning layers in v1. They are deterministic adapters with
native distances, projections, regularizers, metadata, and stack utilities.

Technically, AetherSpaces is a multiple-dispatch interface for mapping spectral
coefficient trajectories into eight mathematical spaces, computing native
metrics or residuals, and mapping back to coefficient space where that operation
is well-defined. It is designed to support future geometric regularization and
invariant-aware training without changing the default FAT forward pass.

The public-facing intuition is an eight-lens compass for continuous AI: the
model's coefficient trajectories can be inspected through several mathematical
views instead of only through one-step Euclidean error. Today this helps measure
structure, drift, weighted error, local-to-global consistency, and algebraic
residuals. The research goal is stronger: use those diagnostics as regularizers
and eventually as proof targets for models that preserve more of the geometry
of the systems they generate.

Current boundary: AetherSpaces does not yet prove gauge covariance, does not
force invariants by itself, and does not make arbitrary models incapable of
rollout drift. The proof-backed non-expansive component remains the Chebyshev
stabilizer in the FAT path. AetherSpaces is the experimental bridge toward a
broader unified mathematical state space.

## Research Contributions

### 1. Functional Autoregressive Transformer

The central research claim is that next-token prediction is a special case of
next-function prediction. A discrete token can be viewed as a constant
function:

```math
f_i(x) = c_i, \qquad \forall x \in \mathcal{X}.
```

A standard language model predicts over a finite vocabulary:

```math
\Pr(x_{t+1} \mid x_{\le t}) \in \Delta^{|V|}.
```

FAT replaces the categorical target with a finite-dimensional representation
of a function:

```math
\Theta_{t+1} \in \mathbb{R}^{K+1}, \qquad
f_{\Theta_{t+1}}(x) = \sum_{n=0}^{K} \theta_n^{(t+1)} T_n(x).
```

This is not merely a different output head. It changes the object being
predicted from a class label to a point in a function space. The transformer
backbone then acts as a learned autoregressive operator over coefficient
trajectories.

### 2. Corrected Spectral Stabilization

The repository implements the corrected stabilizer:

```math
\Pi = P_p \circ P_k.
```

Both `P_k` and `P_p` are Chebyshev coefficient truncations in the same Hilbert
space. Because they are orthogonal projections, each is non-expansive:

```math
\|P_k f - P_k g\|_{H_w} \le \|f-g\|_{H_w},
\qquad
\|P_p f - P_p g\|_{H_w} \le \|f-g\|_{H_w}.
```

Therefore:

```math
\|\Pi f - \Pi g\|_{H_w}
= \|P_p(P_k f) - P_p(P_k g)\|_{H_w}
\le \|f-g\|_{H_w}.
```

This matters because autoregressive generation repeatedly composes the learned
transition. A non-expansive post-map prevents the stabilization stage from
amplifying coefficient-space errors at every step.

The earlier mixed Chebyshev/Fourier version is intentionally not documented as
the implemented stabilizer because projections that are orthogonal in
different Hilbert spaces do not automatically compose with operator norm at
most one.

### 3. Directional Trajectory Approval

Classical regression asks whether the predicted coefficient vector is close in
Euclidean distance. FAT also asks whether the model is moving in the correct
direction through function space:

```math
\Delta_t^{pred} = \Theta_{t+1}^{pred} - \Theta_t,
\qquad
\Delta_t^* = \Theta_{t+1}^* - \Theta_t.
```

The approval signal is cosine alignment:

```math
A_t =
\left\langle
\frac{\Delta_t^{pred}}{\|\Delta_t^{pred}\|+\epsilon},
\frac{\Delta_t^*}{\|\Delta_t^*\|+\epsilon}
\right\rangle.
```

The loss is:

```math
L_{flow} = 1 - \mathbb{E}_t[A_t].
```

In practice, training blends this with MSE:

```math
L_t =
\alpha_t L_{flow}
+ (1-\alpha_t)L_{mse}.
```

The warmup schedule lets the model first learn scale and coarse location
before the directional objective becomes dominant.

### 4. Simplicity Pressure and Folding

The project also explores two related forms of simplicity pressure:

- spectral simplicity: prefer low-complexity coefficient vectors
- architectural simplicity: identify redundant heads and merge them

The Hardin-Taylor-inspired score is:

```math
C(\Theta) = \sum_{n=0}^{K} (1+n)^\alpha |\theta_n|.
```

The folding prototype compares head activations by centered kernel alignment
(CKA):

```math
\operatorname{CKA}(X_i, X_j) =
\frac{\|(X_i^c)^\top X_j^c\|_F^2}
{\|(X_i^c)^\top X_i^c\|_F \,
 \|(X_j^c)^\top X_j^c\|_F}.
```

Heads above a similarity threshold can be grouped as candidate merge clusters.
The code currently treats this as a diagnostic/prototype compression path.

## Unified Mathematical Framework

This section gives a single notation for the theory and the code.

### Objects

Let:

- `H` be a Hilbert space of scalar functions.
- `{phi_n}` be an orthonormal basis.
- `V_K = span(phi_0,...,phi_K)` be the truncated approximation space.
- `Theta in R^(K+1)` be the coefficient vector for an element of `V_K`.

Analysis maps a function to coefficients:

```math
\operatorname{coeff}_K(f)
=
\left(
\langle f,\phi_0\rangle_H,
\ldots,
\langle f,\phi_K\rangle_H
\right)^\top.
```

Synthesis maps coefficients back to a function:

```math
\operatorname{synth}_K(\Theta)(x)
=
\sum_{n=0}^{K} \Theta_n \phi_n(x).
```

In the current FAT implementation, `phi_n = T_n`, the Chebyshev basis
functions, and `Theta` is stored as the first axis of a tensor with layout:

```text
(K+1, sequence_length, batch)
```

### Autoregressive Operator

Given context:

```math
C_t = (\Theta_{t-L+1},\ldots,\Theta_t),
```

FAT learns:

```math
\Theta_{t+1}^{raw} = G_\psi(C_t),
\qquad
G_\psi : (\mathbb{R}^{K+1})^L \to \mathbb{R}^{K+1}.
```

The stabilized generation step is:

```math
\Theta_{t+1} = \Pi(G_\psi(C_t)).
```

In code:

- `coeff_embed` maps coefficients into model width.
- `blocks` model causal dependencies between function states.
- `spectral_head` maps hidden states back to coefficients.
- `stabilizer` applies `Pi` during generation paths.

### Error Growth Intuition

If a transition map `F` is non-expansive:

```math
\|F(x)-F(y)\| \le \|x-y\|,
```

and the generated orbit receives perturbations `e_i` bounded by
`epsilon_i`, then the accumulated error grows at most linearly:

```math
\|y_n - x_n\| \le \sum_{i=1}^{n}\epsilon_i.
```

This is the practical reason the stabilizer is part of the theory. The model
itself may still make prediction errors, but the projection stage is designed
not to multiply those errors by a factor larger than one.

This theorem is formalized in `proofs/ArchitectProofs/ErrorBound.lean`.
The stabilizer-specific rollout theorem is formalized in
`proofs/ArchitectProofs/Rollout.lean`.

### Teaching Diagram

```mermaid
flowchart TD
    F[Function f(x)] -->|analysis| C[coeff_K(f)]
    C --> Enc[Linear coefficient encoder]
    Enc --> TF[Causal transformer blocks]
    TF --> Head[SpectralHead]
    Head --> Raw[Theta raw]
    Raw --> Stabilizer[Pi = P_p after P_k]
    Stabilizer --> Next[Theta t+1]
    Next -->|synthesis| Fout[Predicted function]

    subgraph Training
        Head --> Loss[Directional approval + MSE warmup]
        Loss --> Opt[AdamW + gradient clipping]
        Opt --> TF
    end
```

## Mathematical Model

### Hilbert Space

The core FAT space is a weighted Hilbert space over `[-1, 1]`:

```math
H_w = L^2([-1,1], w), \qquad
w(x) = \frac{1}{\pi\sqrt{1-x^2}}.
```

`src/FAT/phase1_hilbert.jl` implements:

- `ChebyshevBasis{K}` for polynomials `T_0,...,T_K`
- `SpectralState{T,K}` for coefficient vectors
- `evaluate` using Clenshaw recurrence
- `project`, `project_coefficient`, `inner_product`
- `hilbert_norm`, `hilbert_dist`
- `chebyshev_derivative_matrix`

### FAT Transition

For context window `L`, FAT learns:

```math
G_\psi : (\mathbb{R}^{K+1})^L \to \mathbb{R}^{K+1}
```

and applies it autoregressively:

```math
\Theta_{t+1}^{raw} = G_\psi(\Theta_{t-L+1},\ldots,\Theta_t).
```

The implemented model is:

```text
coefficient sequence
  -> Linear(K+1 -> n_embd)
  -> causal transformer blocks
  -> RMS norm
  -> SpectralHead(n_embd -> K+1)
  -> optional stabilizer during generation
```

### Spectral Stabilizer

Continuous autoregression can accumulate error if updates expand distance in
coefficient space. Architect uses a Chebyshev-only projection pipeline:

```math
\Pi = P_p \circ P_k, \qquad 0 \le p \le k \le K.
```

In code:

- `ChebyshevEconomizer{P_order}` truncates coefficient vectors.
- `SpectralStabilizer{K_anchor,P_econ}` composes the anchor projection and
  economizer.
- `verify_nonexpansive` Monte Carlo checks the non-expansive property.

The important correction is that all projections live in the same Chebyshev
coefficient/Hilbert space. Earlier mixed-basis variants involving Fourier
projection were removed because orthogonality in incompatible Hilbert spaces
does not imply a clean `L <= 1` Lipschitz bound.

### Directional Trajectory Approval Loss

`src/FAT/phase4_loss.jl` implements a loss that compares trajectory direction,
not only coefficient amplitude.

For a current state, prediction, and target:

```math
\Delta^{pred}_t = \Theta^{pred}_{t+1} - \Theta_t,
\qquad
\Delta^*_t = \Theta^*_{t+1} - \Theta_t.
```

The directional component is:

```math
L_{flow} = 1 - \mathbb{E}_t
\left[
\left\langle
\frac{\Delta^{pred}_t}{\|\Delta^{pred}_t\| + \epsilon},
\frac{\Delta^*_t}{\|\Delta^*_t\| + \epsilon}
\right\rangle
\right].
```

Training blends MSE and directional approval with
`approval_alpha_schedule`, allowing stable warmup before directional loss
dominates.

### Minimum-Complexity Prediction

`src/FAT/phase8_hardin_taylor.jl` implements a constructive approximation of
the Hardin-Taylor prediction idea. The code replaces the theorem's
non-constructive well-ordering with a spectral complexity score:

```math
C(\Theta) = \sum_{n=0}^{K} (1+n)^\alpha |\theta_n|.
```

Implemented pieces:

- `spectral_complexity`
- `mcp_loss`
- `hardin_taylor_generate`
- `ClimbEvent` records when generation increases truncation order

This is experimental, but useful as a regularizer and generation heuristic for
favoring simpler functions unless higher spectral order is needed.

## Theoretical Positioning

FAT sits near neural operator learning, but with an autoregressive sequence
modeling constraint.

| Framework | Input | Output | Autoregressive |
|-----------|-------|--------|----------------|
| Standard LM | token sequence | next token distribution | yes |
| DeepONet | sampled/input function | output function | usually no |
| Fourier Neural Operator | function/field coefficients | transformed field | usually no |
| Neural Process | context observations | function distribution | usually no |
| FAT | coefficient sequence | next coefficient vector/function | yes |

The main distinction is recurrence over function states. FAT repeatedly
applies a learned transition in coefficient space, then uses spectral
stabilization to keep that recurrence controlled.

### Supported Function Views

The codebase currently contains several function representations, with
different levels of support:

| Representation | Where | Status |
|----------------|-------|--------|
| Monomial coefficients | `FunctionSequenceLoader`, `ContinuousModel` demos | baseline data/tasks |
| Chebyshev coefficients | `src/FAT/phase1_hilbert.jl`, FAT model | primary FAT theory path |
| Fourier-style functions | `evaluate_function(...; basis=:fourier)` | utility/baseline support |
| Fractional collocation states | `src/FAT/fractional.jl` | numerical experiment |

The Lean proofs and the main stabilizer theorem are written around the
coefficient-space/Chebyshev truncation view.

## Implemented Architectures

### FunctionalAutoregressiveTransformer

Location: `src/FAT/phase5_fat_model.jl`

The research FAT model uses:

- `FATConfig`
- coefficient embedding
- transformer blocks from `src/Layers/`
- `SpectralHead`
- `SpectralStabilizer`
- `fat_forward`
- `fat_train!`
- `fat_generate`
- `adaptive_fat_generate`

### ContinuousModel

Location: `src/Models/continuous_model.jl`

This is a baseline functional transformer. It predicts coefficient vectors but
does not use the full FAT stabilizer/loss pipeline. It is used heavily by the
folding scripts and functional training utilities.

### SequenceModel

Location: `src/Models/sequence_model.jl`

Recurrent language model with `:rnn`, `:lstm`, and `:gru` cell choices.

### ContinuousModel as GPT-Style Baseline

The transformer layers are GPT-style causal blocks:

- `Embedding`
- `Attention` with RoPE and optional kernel/RFF mode
- `FeedForward`
- `Block`
- `Norm`
- `Linear`
- optional `OrthogonalLinear`

### Llama2 Inference

Location: `src/Models/llama2_inference.jl`

Pure Julia reader/executor for llama2.c-style `.bin` checkpoints:

- `Llama2Config`
- `Llama2Weights`
- `Llama2State`
- `Llama2Transformer`
- `load_llama2_checkpoint`

## Repository Map

```text
src/
  Architect.jl              package entry point and public exports
  Core/
    types.jl                abstract model/layer/type hierarchy
    config.jl               ModelConfig and GPT-style presets
    device.jl               CPU/CUDA transfer helpers
  Layers/
    layers.jl               Embedding, Norm, Linear, OrthogonalLinear
    attention.jl            causal attention, RoPE, optional kernel path
    block.jl                FeedForward and transformer Block
    recurrent.jl            RNN/LSTM/GRU cells
  Models/
    continuous_model.jl     coefficient-space transformer baseline
    sequence_model.jl       recurrent language model
    generate.jl             token sampling strategies
    llama2_inference.jl     llama2.c checkpoint inference
  Data/
    tokenizer.jl            CharTokenizer and BPETokenizer
    data_loader.jl          character data loader
    functional_data.jl      synthetic coefficient sequence tasks
  Training/
    init.jl                 initialization and spectral stats
    train.jl                classic training utilities
    functional_train.jl     Zygote AD training for ContinuousModel
  FAT/
    phase1_hilbert.jl       basis, projection, evaluation, Hilbert metrics
    phase2_spectral_head.jl hidden-to-coefficient head
    phase3_stabilizer.jl    non-expansive Chebyshev stabilizer
    phase4_loss.jl          directional trajectory approval loss
    phase5_fat_model.jl     FATConfig, model, training, generation
    phase6_data.jl          continuous-time sequence data
    phase7_mla.jl           latent attention prototype
    phase8_hardin_taylor.jl complexity regularizer and climbing generation
    folding.jl              CKA head probing and soft merge prototype
    fractional.jl           fractional collocation solver
  Spaces/
    Spaces.jl               opt-in exotic mathematical space adapters

scripts/                     experiments, probes, benchmarks
test/                        Julia test suite
tests/                       Python-side compatibility/import checks
proofs/                      Lean 4 formal verification
docs/                        short user-facing docs
results/                     benchmark/probe outputs
```

## Installation

Install Julia 1.9 or newer, then instantiate the project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

To load the package locally:

```bash
julia --project=.
```

```julia
using Architect
```

CUDA is listed in `Project.toml`; CPU fallback is supported through the local
`to_device` abstraction.

## Quick Start

### Build and Run a FAT Forward Pass

```julia
using Architect

cfg = FATConfig{Float32}(
    K = 15,
    K_anchor = 15,
    P_econ = 7,
    n_embd = 64,
    n_head = 4,
    n_layer = 2,
    block_size = 16,
    dropout = 0.0f0
)

model = FunctionalAutoregressiveTransformer(cfg)

# Shape: coefficients x sequence x batch
x = randn(Float32, cfg.K + 1, cfg.block_size, 4)
y = fat_forward(model, x)

size(y)  # (16, 16, 4)
```

### Train a Baseline ContinuousModel

```julia
using Architect

loader = FunctionSequenceLoader(
    task = :derivative,
    basis_dim = 8,
    seq_length = 16,
    batch_size = 8
)

cfg = ContinuousConfig(
    basis_dim = 8,
    basis_type = :monomial,
    n_embd = 64,
    n_head = 4,
    n_layer = 2,
    block_size = 16,
    dropout = 0.0f0
)

model = ContinuousModel(cfg)

train_cfg = Architect.Training.ContinuousTrainConfig{Float32}(
    max_iters = 100,
    learning_rate = 1f-3,
    log_interval = 20
)

state = train!(model, loader, train_cfg)
```

### Evaluate a Chebyshev State

```julia
using Architect

basis = ChebyshevBasis{4}()
theta = Float64[1.0, 0.5, -0.2, 0.0, 0.1]
state = SpectralState{Float64,4}(theta)

evaluate(basis, state, 0.25)
hilbert_norm(state)
```

## FAT API

Primary exported types and functions:

```julia
ChebyshevBasis{K}
SpectralState{T,K}
SpectralHead{T,K}
ChebyshevEconomizer{P_order}
SpectralStabilizer{K_anchor,P_econ}
FATConfig{T}
FunctionalAutoregressiveTransformer
FATTrainConfig{T}
FATTrainState{T}
RolloutMetrics{T}

evaluate
project
inner_product
hilbert_norm
hilbert_dist
verify_nonexpansive
directional_trajectory_loss
approval_alpha_schedule
fat_forward
fat_train!
fat_generate
adaptive_fat_generate
train!
rollout
evaluate_rollout
rollout_metrics
```

The main tensor convention is:

```text
features/coefficient dimension x sequence length x batch
```

This is consistent with Julia column-major layout and the existing layer
implementations.

Shared coefficient-model accessors are also exported:

```julia
coefficient_encoder(model)
coefficient_head(model)
coefficient_dimension(model)
context_length(model)
```

`ContinuousModel` and `FunctionalAutoregressiveTransformer` keep their original
fields for compatibility, but new shared code should prefer these accessors.

`RolloutMetrics` reports:

- one-step MSE
- rollout MSE
- coefficient/Hilbert norm errors
- spectral energy and spectral energy drift
- stabilizer contraction ratios
- coefficient complexity
- per-step error growth ratios

## Unified Spaces API

The AetherSpaces implementation currently lives in `src/Spaces/Spaces.jl`.
It is exposed through the public `Architect.AetherSpaces` facade and the
top-level `Spaces` compatibility API. It adds an opt-in experimental interface
for mathematical views over coefficient trajectories. It does not change the
default FAT forward path, and normal FAT training, generation, and rollout
evaluation work without passing any spaces.

The subsystem is intentionally interface-first. Each space can define how
spectral coefficients are represented, projected, measured, regularized, and
summarized. This gives experiments a common way to ask questions such as:

- How large is the trajectory error under a native metric?
- Does a projection keep coefficients inside the intended space?
- Does a local-to-global or diagram consistency residual expose structure that
  one-step MSE misses?
- Can these residuals become future training penalties without destabilizing the
  Chebyshev FAT path?

Primary exports:

```julia
AbstractMathSpace
SpaceState
MetricSpace
MeasureSpace
TopologicalChartSpace
FractionalOperatorSpace
TropicalSpace
UltrametricSpace
SheafConsistencySpace
CategoricalBridgeSpace

to_space
from_space
space_distance
space_project
space_regularizer
space_metadata
apply_space_stack
space_stack_regularizer
space_stack_metrics
stable_aether_spaces
diagnostic_aether_spaces
default_aether_spaces
aether_space_metrics
```

Implemented spaces:

- `MetricSpace`: Euclidean, weighted, and diagonal metric adapters.
- `MeasureSpace`: coefficient or node weighting for weighted norms.
- `TopologicalChartSpace`: local chart views and chart consistency residuals.
- `FractionalOperatorSpace`: deterministic fractional-operator diagnostics.
- `TropicalSpace`: max-plus/min-plus coefficient views.
- `UltrametricSpace`: p-adic-inspired hierarchical scale diagnostics.
- `SheafConsistencySpace`: local sections and gluing residuals.
- `CategoricalBridgeSpace`: map composition and commutator residuals.

Current limits:

- The spaces are deterministic adapters and diagnostics, not trainable neural
  layers.
- Space regularizers are available as scalar penalties and can be enabled in
  FAT training with `aether_lambda > 0` and `aether_spaces`.
- Invariant preservation is a research target, not a global guarantee.
- The classical Chebyshev stabilizer remains the only proof-backed stabilizer.

The detailed guide is `docs/unified_spaces.md`.

## Training and Data

### FunctionSequenceLoader

`src/Data/functional_data.jl` generates synthetic coefficient sequences for
the `ContinuousModel` baseline. Supported tasks:

- `:derivative`
- `:antiderivative`
- `:taylor`
- `:composition`

The current generator supports `basis_type = :monomial`.

### ContinuousTimeDataLoader

`src/FAT/phase6_data.jl` generates continuous-time streams for FAT benchmarks.
The benchmark script uses `task = :sine_waves` with Chebyshev coefficient
targets.

### Training Loops

The codebase contains two training paths:

- `train!` for `ContinuousModel` with Zygote AD in
  `functional_train.jl`
- `fat_train!` for `FunctionalAutoregressiveTransformer` with FAT-specific
  directional loss, AdamW state, weight decay, and gradient clipping

The canonical public style is:

```julia
train!(model, dataloader, config)
```

Function-valued batch providers are still supported for lower-level scripts.

Older classical training utilities are also present for compatibility with the
general model tests.

## Topological Folding

Location: `src/FAT/folding.jl`

Folding is a prototype compression workflow:

1. Wrap attention layers with `TrackedAttention`.
2. Collect per-head activations into `ProbeCollector`.
3. Compute head similarity with `compute_similarity`, usually linear CKA.
4. Propose clusters with `propose_clusters`.
5. Optionally apply a soft merge with `FoldedHead` / `apply_soft_merge`.

The active probe script is:

```bash
julia --project=. scripts/fold_probe_train_and_fold.jl
```

That script trains a small `ContinuousModel`, collects head activations,
computes CKA similarities, reports candidate merge clusters at multiple
thresholds, and writes a CSV under `results/`.

Folding should currently be treated as diagnostic/prototype code, not as a
fully validated compression pass.

## Formal Proofs

Lean 4 proofs live under `proofs/`.

The current proof modules cover:

- `ArchitectProofs/Stabilizer.lean`: non-expansivity of the corrected
  Chebyshev stabilizer
- `ArchitectProofs/ErrorBound.lean`: linear error growth under non-expansive
  dynamics plus bounded perturbations
- `ArchitectProofs/Rollout.lean`: rollout error growth specialized to the
  corrected FAT stabilizer
- `ArchitectProofs/FATGeneralization.lean`: `K = 0` degeneration back to the
  constant-function case

Build from the proof directory:

```bash
cd proofs
LEAN_CC=./leanc_wrapper.sh lake build
```

The wrapper is mainly for portability on macOS; Linux builds should use the
standard Lean toolchain behavior.

## Scripts

```text
scripts/benchmark_function_rollouts.jl
  Clean derivative/oscillator/heat rollout suite. Compares FAT variants against
  ContinuousModel, persistence, and oracle-linear baselines, then writes CSV
  metrics and dependency-free SVG rollout-error plots.
  Each run writes config, git commit, losses, metrics JSON, rollout samples,
  CSV curves, and plots under results/rollout_suite/ by default.

scripts/benchmark_aether_physics.jl
  Paper-oriented Aether physics suite. Compares FAT, FAT with opt-in Aether
  regularization, ContinuousModel, persistence, and oracle-linear baselines on
  harmonic, damped, heat, wave, and fractional dynamics. Writes rollout metrics,
  Aether per-space metrics, samples, CSV curves, and SVG plots under
  results/aether_physics/ by default.

scripts/ablation_stabilizer_approval.jl
  Convenience entry point for the rollout ablation suite.

scripts/probe_unified_spaces.jl
  Diagnostic probe for all eight experimental mathematical spaces. Writes a
  CSV report under results/unified_spaces/ by default.

scripts/benchmark_fat.jl
  Train ContinuousModel and FAT on continuous-time data and compare loss/time.

scripts/benchmark_full.jl
  Larger benchmark runner.

scripts/fold_probe.jl
  Folding probe utilities.

scripts/fold_probe_train_and_fold.jl
  Train a small ContinuousModel, collect head activations, compute CKA, and
  report merge candidates.

scripts/fold_probe_trained.jl
  Probe an already trained setup.

scripts/experiment_adaptive_k.jl
  Adaptive truncation-order experiment.

scripts/fractional_collocation.jl
  Fractional differential equation collocation demo.

scripts/quick_eval.jl
  Small evaluation helper.

scripts/test_regression.jl
  Regression check helper.

scripts/compare_results.jl
  Compare serialized benchmark outputs.
```

## Tests

Run the Julia tests:

```bash
julia --project=. test/runtests.jl
```

Current test files cover:

- configs
- embeddings
- layers
- attention
- model forward paths
- recurrent cells/models
- tokenizers
- training utilities
- llama2 checkpoint code
- fractional solver
- unified mathematical spaces

There is also a Python-side test under `tests/test_huggingface_import.py`.

## Theory Note

A shorter research contract lives in `docs/theory.md`. It states the learning
problem, error decomposition target, stabilizer theorem, rollout theorem, loss,
unified-space abstraction, and the benchmark ladder that should guide the
refactoring branch. The space adapter guide is `docs/unified_spaces.md`.

## Design Principles

- Keep arrays typed and layout-explicit.
- Use `features x sequence x batch` tensors for model internals.
- Prefer pure, allocation-based operations where Zygote needs to differentiate.
- Keep device movement behind `to_device`.
- Separate decay and no-decay parameters for AdamW.
- Treat FAT math, code, and proofs as a single contract: if the theory changes,
  update `src/FAT/`, `proofs/`, and this README together.

## Refactoring Notes

This branch should use the README as the baseline map of what exists today.
Useful cleanup targets visible from the current tree:

- align naming between `ContinuousModel` and `FunctionalAutoregressiveTransformer`
  where shared utilities can reasonably support both
- separate stable APIs from experimental scripts/prototypes
- keep FAT phase files coherent, but consider extracting shared tensor/model
  helpers once duplication is clear
- update `ARCHITECTURE.md` and `DESIGN.md` after source-level refactors, since
  some descriptions are older than the current file layout
- keep benchmark outputs and temporary scratch files out of the core API story

## License

MIT. See `LICENSE`.
