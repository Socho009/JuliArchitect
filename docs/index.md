# Architect.jl Complete Documentation

Welcome to the comprehensive documentation for **Architect.jl**, a Julia framework for neural language models combining classical architectures with the novel Functional Autoregressive Transformer (FAT).

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Module Reference](#module-reference)
4. [Architecture Paradigms](#architecture-paradigms)
5. [Advanced Usage](#advanced-usage)
6. [Theory & Mathematics](#theory--mathematics)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
git clone https://github.com/miniAliyah/julia-native-gpt.git
cd julia-native-gpt
julia --project -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

Verify installation:

```julia
julia> using Architect
julia> println("✓ Architect.jl loaded successfully")
```

### Your First Model

```julia
using Architect

# 1. Configure a small GPT model
config = Architect.Core.gpt_nano  # 1.3M parameters

# 2. Create and initialize
model = ContinuousModel(config)
initialize!(model; muP=true)

# 3. Make a forward pass
x = rand(Int32, 32)  # 32 token IDs
logits = model(x)    # (vocab_size, 32)
```

---

## Core Concepts

### Type System

Architect.jl uses **parametric types** for compile-time precision and device specialization:

```julia
# All layers are parametric in floating-point type P
struct Linear{P, HasBias} <: AbstractLayer{P}
    weight::AbstractMatrix{P}
    bias::Union{AbstractVector{P}, Nothing}
end

# Models are specialized for Float32, Float64, Float16
model = ContinuousModel{Float32}(config)
model_f64 = ContinuousModel{Float64}(config)
```

**Benefits**:
- Compiler generates specialized code for each precision
- No runtime type checking
- Automatic differentiation (Zygote.jl) works seamlessly

### Device Management

Transfer models and data between CPU and GPU:

```julia
using Architect.Core: to_device

# GPU (CUDA.jl)
model_gpu = to_device(model, :cuda)

# Back to CPU
model_cpu = to_device(model_gpu, :cpu)
```

---

## Module Reference

### Core Module

**Purpose**: Type definitions, configurations, device management.

**Key Types**:
- `AbstractLayer{P}` — Base type for all layers
- `AbstractModelConfig{P}` — Base type for model configurations
- `ModelConfig` — Concrete configuration struct
- `to_device(x, device)` — Device transfer abstraction

**Key Functions**:
- `initialize!(model; muP=false)` — Initialize weights
- `to_device(x, device::Symbol)` — Transfer to CPU/GPU

---

### Layers Module

**Purpose**: Basic neural network building blocks.

**Layers**:
- `Linear{P, HasBias}` — Fully connected layer
- `Embedding{P}` — Token embedding lookup
- `Norm{P}` — Layer normalization
- `Attention{P}` — Multi-head causal self-attention
- `FeedForward{P}` — Feed-forward network (SwiGLU)
- `Block{P}` — Transformer block (Attention + FFN)
- `RNNCell{P}`, `LSTMCell{P}`, `GRUCell{P}` — Recurrent cells

**Key Functions**:
- `(layer::AbstractLayer)(x)` — Forward pass (callable interface)
- `Flux.params(layer)` — Collect learnable parameters

---

### Models Module

**Purpose**: High-level model architectures.

**Model Types**:

#### ContinuousModel (GPT-2 Transformer)

```julia
config = ModelConfig(
    vocab_size=256,
    n_embd=768,
    n_head=12,
    n_layer=12,
    dropout=0.1,
    block_size=1024
)
model = ContinuousModel{Float32}(config)
```

**Forward pass**:
```julia
token_ids = randn(Int32, seq_len, batch_size)
logits = model(token_ids)  # (vocab_size, seq_len, batch_size)
```

#### SequenceModel (Recurrent LM)

```julia
config = SequenceConfig(
    vocab_size=256,
    hidden_size=512,
    n_layer=2,
    cell_type=:lstm,
    dropout=0.1
)
model = SequenceModel{Float32}(config)
```

#### Llama2Transformer (llama2.c port)

```julia
weights = load_llama2_checkpoint("model.bin")
model = Llama2Transformer(weights)
tokens = randn(Int32, seq_len, batch_size)
logits = model(tokens)
```

---

### Data Module

**Purpose**: Tokenization and batch loading.

**Tokenizers**:
- `CharTokenizer(text)` — Character-level vocabulary
- `BPETokenizer()` — Byte-pair encoding

**Data Loaders**:
- `CharDataLoader` — Batch streaming for text
- `FunctionSequenceLoader` — Synthetic functional data (FAT)

**Example**:
```julia
text = read("data.txt", String)
tokenizer = CharTokenizer(text)
loader = CharDataLoader(text, tokenizer; seq_len=256, batch_size=32)

for epoch in 1:10
    while true
        batch = next_batch!(loader)
        batch === nothing && break
        # process batch
    end
end
```

---

### Training Module

**Purpose**: Optimization and sampling.

**Configuration**:
```julia
train_config = TrainConfig(
    learning_rate=3e-4,
    max_iters=10000,
    warmup_iters=1000,
    weight_decay=0.01,
    grad_clip=1.0,
    batch_size=32
)
```

**Training Loop**:
```julia
state, losses = train!(model, make_batch_fn, train_config)
```

**Sampling**:
```julia
prompt = [1, 2, 3]  # token IDs
output = generate(
    model, prompt;
    strategy=TopKSampling(k=10, p=0.9f0),
    max_new_tokens=100
)
```

---

### FAT Module (Research Contribution)

**Purpose**: Functional Autoregressive Transformer — autoregression over function spaces.

**Key Components**:

#### Phase 1: Hilbert Space Foundations
```julia
using Architect.FAT

# Define basis
basis = ChebyshevBasis(; degree=8)

# Project coefficients
coeffs = randn(Float32, 8)
projected = project(basis, coeffs)
```

#### Phase 2: Spectral Head
```julia
spectral_head = SpectralHead{Float32}(;
    n_embd=64,
    basis_dim=8,
    basis_type=:chebyshev
)

# Forward: embeddings → coefficients
output = spectral_head(embeddings)
```

#### Phase 3: Stabilization
```julia
stabilizer = SpectralStabilizer{Float32}(;
    basis_type=:chebyshev,
    k_taylor=4,
    p_chebyshev=6
)

# Apply spectral projections
stab_coeffs = stabilizer(raw_coeffs)
verify_nonexpansive(stabilizer)  # Verify L ≤ 1
```

#### Phase 4: Directional Trajectory Loss
```julia
# Compute directional alignment
loss = directional_trajectory_loss(
    predicted_coeffs,
    target_coeffs,
    teacher_forcing=true
)
```

#### Phase 5: Full FAT Model
```julia
fat_config = FATConfig{Float32}(
    basis_dim=8,
    n_embd=64,
    n_head=4,
    n_layer=2,
    block_size=16,
    basis_type=:chebyshev
)

fat_model = FunctionalAutoregressiveTransformer(fat_config)
initialize!(fat_model)

# Forward pass
coeffs_in = randn(Float32, 8, 16, 32)  # (basis_dim, seq, batch)
coeffs_out = fat_model(coeffs_in)
```

#### Phase 9: Topological Folding
```julia
using Architect.FAT: ProbeCollector, wrap_attentions!, collect_head_activations,
                      compute_similarity, propose_clusters

# 1. Collect intermediate head activations
collector = collect_head_activations(model, loader; nbatches=20)

# 2. Compute CKA similarities between all heads
sim, keys = compute_similarity(collector; metric=:cka)

# 3. Propose head clusters to fold based on similarity threshold
clusters = propose_clusters(sim, keys; threshold=0.92f0)
```

---

## Architecture Paradigms

### 1. Transformer (GPT-2 Style)

**When to use**: Text generation, sequence classification, general-purpose language modeling.

**Strengths**:
- Proven architecture
- Large pretrained models available
- Excellent for discrete sequences

**Example**:
```julia
model = ContinuousModel(gpt_nano)
generate(model, prompt; max_new_tokens=100)
```

### 2. Recurrent (RNN/LSTM/GRU)

**When to use**: Character-level generation, sequence labeling, streaming data.

**Strengths**:
- Efficient for long sequences
- Natural for streaming
- Smaller memory footprint

**Example**:
```julia
model = SequenceModel(config)
generate(model, prompt; max_new_tokens=100)
```

### 3. Llama2 Inference

**When to use**: Using pretrained llama2.c models, production inference.

**Strengths**:
- Grouped-query attention
- Efficient inference cache
- Compatibility with llama2.c checkpoints

**Example**:
```julia
weights = load_llama2_checkpoint("model.bin")
model = Llama2Transformer(weights)
generate(model, prompt)
```

### 4. FAT (Functional Autoregressive Transformer)

**When to use**: Continuous function prediction, mathematical operators, PDEs.

**Strengths**:
- Predicts arbitrary smooth functions
- Parameter-efficient (7,000× reduction vs. LM)
- Stable autoregression (Π operator guarantees L ≤ 1)

**Example**:
```julia
# Learn polynomial differentiation
loader = FunctionSequenceLoader(:derivative, basis_dim=8)
fat_model = FunctionalAutoregressiveTransformer(fat_config)
fat_train!(fat_model, loader, train_config)

# Predict derivative
f_coeffs = [1, 0, -1, 0, 0, 0, 0, 0]  # f(x) = 1 - x²
f_prime_coeffs = fat_generate(fat_model, f_coeffs)
# → [-0, 2, 0, 0, ...]  (should be f'(x) = -2x)
```

---

## Advanced Usage

### Custom Initialization Schemes

```julia
using Architect.Training: initialize!

# Kaiming normal (for ReLU)
model = ContinuousModel(config)
initialize!(model; muP=false, init_scheme=:kaiming_normal)

# Xavier uniform (for sigmoid/tanh)
initialize!(model; init_scheme=:xavier_uniform)

# Maximal Update Parametrization
initialize!(model; muP=true)
```

### Automatic Differentiation with Zygote

```julia
using Zygote

# Define loss function
function loss(model, batch)
    x, y = batch
    logits = model(x)
    mse_loss(logits, y)
end

# Compute gradients
gs = gradient(m -> loss(m, batch), model)

# Access gradients
grad_w = gs[model.layers[1].weight]
```

### Custom Sampling Strategies

```julia
# Greedy (deterministic)
output = generate(model, prompt; strategy=GreedySampling())

# Top-K
output = generate(model, prompt; strategy=TopKSampling(k=50, p=0.95f0))

# Nucleus (Top-P)
output = generate(model, prompt; strategy=NucleusSampling(p=0.9f0))
```

### Benchmarking

```julia
using Architect

# Quick evaluation
include("scripts/quick_eval.jl")

# Full benchmark
include("scripts/benchmark_full.jl")

# Compare results
compare_results(summary1_path, summary2_path)
```

---

## Theory & Mathematics

### FAT Theoretical Foundation

**Problem**: Standard LMs discretize continuous phenomena into tokens.

**Solution**: Generalize autoregression to Hilbert spaces:

$$f_\Theta(x) = \sum_{n=0}^K \theta_n T_n(x)$$

where $\{T_n\}$ is an orthogonal basis and $\Theta \in \mathbb{R}^{K+1}$.

**Theorem (Spectral Stabilization)**: The cascade $\Pi = \mathbf{P}_F \circ \mathbf{P}_p \circ \mathbf{P}_k$ maintains Lipschitz constant $L_\Pi = 1$, ensuring linear error growth:

$$\|\hat{f}^n - f^n\| \leq \sum_{i=1}^n \epsilon_i$$

### Bases Supported

| Basis | Domain | Convergence | Best For |
|-------|--------|-------------|----------|
| Monomial | $\mathbb{R}$ | Algebraic | Polynomials, Taylor series |
| Fourier | $[0,2\pi]$ | Exponential | Periodic functions |
| Chebyshev | $[-1,1]$ | Exponential | General smooth functions |

---

## API Reference

### Core Functions

```julia
# Model creation and initialization
model = ContinuousModel(config)
initialize!(model; muP=true)

# Forward pass
logits = model(token_ids)

# Device transfer
model_gpu = to_device(model, :cuda)

# Parameter collection
params = Flux.params(model)
```

### Training Functions

```julia
# Configure training
train_config = TrainConfig(
    learning_rate=3e-4,
    max_iters=10000,
    grad_clip=1.0
)

# Run training
state, losses = train!(model, data_fn, train_config)

# Continue from checkpoint
state, losses = train!(model, data_fn, train_config; 
                       resume_state=state)
```

### Sampling Functions

```julia
# Generate with strategy
output = generate(
    model, prompt;
    strategy=TopKSampling(k=40),
    max_new_tokens=100,
    temperature=1.0f0
)

# Batch generation
outputs = [generate(model, prompt) for prompt in prompts]
```

### FAT Functions

```julia
# Create FAT model
fat_model = FunctionalAutoregressiveTransformer(fat_config)

# Train on functional tasks
fat_train!(fat_model, loader, train_config)

# Generate functional predictions
output_coeffs = fat_generate(fat_model, input_coeffs)

# Evaluate function from coefficients
y = evaluate_function(coeffs, x; basis=:chebyshev)
```

---

## Troubleshooting

### "GPU out of memory"

**Solution**: Reduce batch size or model size:
```julia
config = setproperties(config; n_layer=8, n_embd=512)
```

### "Gradient is NaN"

**Solution**: Reduce learning rate or increase grad_clip:
```julia
train_config = TrainConfig(learning_rate=1e-4, grad_clip=0.5)
```

### "Type instability" warnings

**Solution**: Ensure consistent precision:
```julia
# ✓ Correct
model = ContinuousModel{Float32}(config)
x = randn(Float32, seq_len, batch_size)

# ✗ Wrong (mixed Float32 and Float64)
x = randn(Float64, seq_len, batch_size)  # Type mismatch!
```

### Reproducibility

**Solution**: Set random seeds:
```julia
using Random
Random.seed!(42)
```

---

## Additional Resources

- **[README.md](../README.md)** — Project overview and quick start
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** — Module organization and design patterns
- **[DESIGN.md](../DESIGN.md)** — Algorithms and mathematical details
- **[examples/](../examples/)** — Runnable code examples
- **[scripts/](../scripts/)** — Benchmarking and evaluation tools

---

## Support & Contributing

- **Questions?** [GitHub Issues](https://github.com/miniAliyah/julia-native-gpt/issues)
- **Contribute**: See [README.md](../README.md#contributing--citation)
- **Citation**: See [README.md](../README.md#academic-citation)

---

**Last Updated**: May 23, 2026  
**Documentation Version**: 1.0  
**Compatible with**: Architect.jl Research Prototype (Phase 1 complete)
