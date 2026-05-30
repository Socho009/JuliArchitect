# Architect.jl Usage Guides

This document provides step-by-step guides for common tasks with Architect.jl.

---

## Table of Contents

1. [Training a Text Generation Model](#training-a-text-generation-model)
2. [Functional Prediction with FAT](#functional-prediction-with-fat)
3. [Using Pretrained Models](#using-pretrained-models)
4. [Custom Data Loaders](#custom-data-loaders)
5. [GPU Acceleration](#gpu-acceleration)
6. [Performance Optimization](#performance-optimization)
7. [Debugging & Troubleshooting](#debugging--troubleshooting)
8. [Topological Folding Analysis](#topological-folding-analysis)

---

## Training a Text Generation Model

### Step 1: Prepare Your Data

```julia
using Architect

# Load text from file
text = read("data.txt", String)

# Create character tokenizer
tokenizer = CharTokenizer(text)
println("Vocabulary size: $(tokenizer.vocab_size)")
println("Text length: $(length(text)) characters")
```

### Step 2: Configure the Model

```julia
# Use a preset configuration
config = Architect.Core.gpt_mini  # 3M parameters

# Or customize it
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=384,        # embedding dimension
    n_head=8,          # attention heads
    n_layer=8,         # transformer blocks
    block_size=256,    # context window
    dropout=0.1,
    bias=true
)
```

### Step 3: Create and Initialize the Model

```julia
using Flux

model = ContinuousModel{Float32}(config)

# Initialize with µP (Maximal Update Parametrization) for better scaling
initialize!(model; muP=true)

# Count parameters
total_params = sum(length, Flux.params(model))
println("Model parameters: $(total_params)")
```

### Step 4: Create Data Loader

```julia
loader = CharDataLoader(
    text, tokenizer;
    seq_len=config.block_size,
    batch_size=32,
    split=(0.9, 0.05, 0.05)  # train/val/test
)
```

### Step 5: Configure Training

```julia
train_config = TrainConfig(
    learning_rate=3e-4,
    max_iters=10000,
    warmup_iters=100,
    weight_decay=0.01,
    grad_clip=1.0,
    batch_size=32,
    eval_interval=100,
    eval_iters=200
)
```

### Step 6: Train the Model

```julia
using Architect.Training

# Create batch function
batch_fn = () -> next_batch!(loader)

# Train
state, losses = train!(model, batch_fn, train_config)

println("Final training loss: $(losses[end])")
```

### Step 7: Generate Text

```julia
# Create a prompt
prompt = "The quick"
prompt_ids = encode(tokenizer, prompt)

# Generate new text
output_ids = generate(
    model, prompt_ids;
    strategy=TopKSampling(k=40, p=0.9f0),
    max_new_tokens=100
)

# Decode and print
generated_text = decode(tokenizer, vec(output_ids))
println(generated_text)
```

---

## Functional Prediction with FAT

### Step 1: Create Training Data

```julia
using Architect
using Architect.FAT

# Create a loader for derivative learning
loader = FunctionSequenceLoader{Float32}(
    task=:derivative,         # learn f → f'
    basis_dim=8,              # polynomial degree
    seq_length=16,            # chain length
    batch_size=32,
    basis_type=:monomial      # or :chebyshev, :fourier
)

# View a sample batch
batch = next_batch(loader)
input_coeffs = batch.x       # (basis_dim, seq_len, batch_size)
target_coeffs = batch.y      # (basis_dim, seq_len, batch_size)
```

### Step 2: Configure FAT Model

```julia
fat_config = FATConfig{Float32}(
    basis_dim=8,
    basis_type=:monomial,
    n_embd=64,
    n_head=4,
    n_layer=2,
    block_size=16,
    spectral_cutoff=6,
    dropout=0.1
)
```

### Step 3: Create and Initialize FAT

```julia
fat_model = FunctionalAutoregressiveTransformer(fat_config)
initialize!(fat_model)
```

### Step 4: Train FAT

```julia
train_config = FATTrainConfig{Float32}(
    learning_rate=3e-4,
    max_iters=1000,
    grad_clip=0.5,
    warmup_iters=50
)

state = fat_train!(fat_model, loader, train_config)
```

### Step 5: Test Predictions

```julia
# Polynomial: f(x) = 1 - x² + 2x³
f_coeffs = Float32[1.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0]

# Predict derivative
f_prime_coeffs = fat_generate(fat_model, f_coeffs; max_new=1)

# Expected: f'(x) = -2x + 6x²  →  [0, -2, 6, 0, ...]
println("Predicted f'(x) ≈ $(f_prime_coeffs[:, 1])")

# Verify with exact operator
exact_deriv = differentiate_coeffs(f_coeffs)
println("Exact f'(x) = $(exact_deriv)")
```

### Step 6: Evaluate Functions

```julia
# Monomial basis evaluation
x = 0.5f0
y_monomial = evaluate_function(f_coeffs, x; basis=:monomial)
println("f(0.5) ≈ $(y_monomial)")

# Using Chebyshev basis
chebyshev_coeffs = Float32[1.0, 0.5, -0.25, 0.1, 0, 0, 0, 0]
y_chebyshev = evaluate_function(chebyshev_coeffs, x; basis=:chebyshev)
println("Chebyshev f(0.5) ≈ $(y_chebyshev)")
```

---

## Using Pretrained Models

### Load a Llama2 Model

```julia
using Architect

# Load weights from checkpoint
weights = load_llama2_checkpoint("model.bin")

# Create model
model = Llama2Transformer(weights)

# Generate text
prompt_ids = [1, 2, 3]  # Your token IDs
output = generate(
    model, prompt_ids;
    max_new_tokens=100,
    temperature=0.8f0
)

println(decode(tokenizer, vec(output)))
```

### Export and Import Models

```julia
using Serialization

# Save model
model_state = state_dict(model)
serialize("checkpoint.jls", model_state)

# Load model
restored_state = deserialize("checkpoint.jls")
restore_state!(model, restored_state)
```

---

## Custom Data Loaders

### Create a Custom Loader

```julia
mutable struct CustomDataLoader
    data::Matrix{Int32}
    seq_len::Int
    batch_size::Int
    current_pos::Int
end

function next_batch!(loader::CustomDataLoader)
    if loader.current_pos >= size(loader.data, 2)
        return nothing  # End of data
    end
    
    # Extract batch
    end_pos = min(loader.current_pos + loader.batch_size, size(loader.data, 2))
    batch = loader.data[:, loader.current_pos:end_pos]
    
    loader.current_pos = end_pos
    return batch
end
```

### Use in Training

```julia
loader = CustomDataLoader(data, seq_len=256, batch_size=32)

batch_fn = () -> next_batch!(loader)
state, losses = train!(model, batch_fn, train_config)
```

---

## GPU Acceleration

### Move Model to GPU

```julia
using CUDA
using Architect.Core: to_device

model = ContinuousModel{Float32}(config)
initialize!(model)

# Transfer to GPU
model_gpu = to_device(model, :cuda)

# Verify on GPU
x_gpu = CUDA.rand(Int32, 32, 16)  # (seq_len, batch_size)
logits = model_gpu(x_gpu)
println("Output on GPU: $(size(logits))")

# Transfer back to CPU
model_cpu = to_device(model_gpu, :cpu)
```

### Mixed Precision Training

```julia
# Use Float32 for model, Float16 for forward pass
model = ContinuousModel{Float32}(config)
x_fp16 = Float16.(randn(32))

# Zygote will handle conversion
loss = mean(model(x_fp16))
```

---

## Performance Optimization

### Profile Training

```julia
using Profile

# Run one training iteration
@time state, losses = train!(model, batch_fn, 
                             TrainConfig(max_iters=1))

# Profile with detailed statistics
@profile train!(model, batch_fn, TrainConfig(max_iters=10))
Profile.print()
```

### Memory Usage

```julia
using Flux: @allocated

# Check allocation for forward pass
bytes_allocated = @allocated model(x)
println("Forward pass allocation: $(bytes_allocated / 1e6) MB")

# Check gradient computation
bytes_grad = @allocated gradient(() -> mean(model(x)), Flux.params(model))
println("Gradient computation allocation: $(bytes_grad / 1e6) MB")
```

### Batch Size Tuning

```julia
# Find optimal batch size
for batch_size in [8, 16, 32, 64, 128]
    config = TrainConfig(batch_size=batch_size, max_iters=100)
    @time state, losses = train!(model, batch_fn, config)
    println("Batch size $batch_size completed")
end
```

---

## Debugging & Troubleshooting

### Check Gradient Flow

```julia
using Zygote

# Verify gradients are computed
function loss(model, x, y)
    pred = model(x)
    mean((pred .- y)^2)
end

x = randn(Float32, 256, 32)
y = randn(Float32, 256, 32)

gs = gradient(() -> loss(model, x, y), Flux.params(model))

# Check for NaN/Inf
for (p, g) in gs
    if any(isnan, g)
        println("NaN detected in gradient!")
    end
    if any(isinf, g)
        println("Inf detected in gradient!")
    end
end
```

### Verify Model Shapes

```julia
# Test forward pass with known shapes
test_batch_size = 4
test_seq_len = 32

x_test = rand(Int32, test_seq_len, test_batch_size)
logits = model(x_test)

println("Input shape: $(size(x_test))")
println("Output shape: $(size(logits))")
println("Expected: ($(config.vocab_size), $test_seq_len, $test_batch_size)")

@assert size(logits) == (config.vocab_size, test_seq_len, test_batch_size)
println("✓ Shape verification passed")
```

### Debug Diverging Loss

```julia
# Check learning rate
train_config = TrainConfig(
    learning_rate=1e-5,  # Start very small
    grad_clip=0.1,       # Tight clipping
    max_iters=10
)

state, losses = train!(model, batch_fn, train_config)
println("Losses: $(losses)")

# If loss is NaN, check:
# 1. Data quality (no Inf/NaN in data)
# 2. Model initialization (weights in reasonable range)
# 3. Learning rate (too high → divergence)
# 4. Gradient scaling (use grad_clip)
```

### Enable Logging

```julia
# Add custom logging
using Logging

# Configure logging level
logger = SimpleLogger(stdout, Logging.Debug)
with_logger(logger) do
    state, losses = train!(model, batch_fn, train_config)
end

# Or log manually
@info "Training started" config.n_layer config.n_embd
@debug "Batch loss: $loss"
@warn "Learning rate very high: $(train_config.learning_rate)"
```

---

## Example: Complete Workflow

Here's a complete example combining multiple concepts:

```julia
using Architect
using Architect.Training
using Flux

# 1. Load and prepare data
text = read("sample_text.txt", String)
tokenizer = CharTokenizer(text)
loader = CharDataLoader(text, tokenizer; 
                        seq_len=128, batch_size=16)

# 2. Create model
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=256, n_head=8, n_layer=6
)
model = ContinuousModel{Float32}(config)
initialize!(model; muP=true)

# 3. Configure training
train_config = TrainConfig(
    learning_rate=3e-4,
    max_iters=1000,
    warmup_iters=100,
    grad_clip=1.0
)

# 4. Train
batch_fn = () -> next_batch!(loader)
state, losses = train!(model, batch_fn, train_config)

# 5. Generate
prompt = "Hello"
prompt_ids = encode(tokenizer, prompt)
output = generate(model, prompt_ids; 
                  strategy=TopKSampling(k=40),
                  max_new_tokens=50)
generated = decode(tokenizer, vec(output))
println(generated)
```

---

---

## Topological Folding Analysis

This guide explains how to use the topological folding utilities to analyze and proposal merges for redundant attention heads in your model.

### Step 1: Instrument the Model with TrackedAttention
To capture head outputs without modifying your model definition, use `wrap_attentions!`. This replaces the `Attention` block of each layer with a `TrackedAttention` wrapper.

```julia
using Architect.FAT: ProbeCollector, wrap_attentions!

# 1. Initialize the collector
collector = ProbeCollector{Float32}()

# 2. Instrument the model
wrap_attentions!(model, collector)
```

### Step 2: Run Batches to Collect Activations
Run several forward passes through the model on representative inputs. The `TrackedAttention` layers will append the head outputs directly to your collector.

```julia
using Architect.Data: next_batch

# Run 16 batches to collect statistical representations
for i in 1:16
    input, target = next_batch(loader)
    # Forward pass
    _ = model(input)
end
```

### Step 3: Compute Head Similarity (CKA)
Remove the wrappers to return the model to its original state using `unwrap_attentions!`, then compute CKA (Centered Kernel Alignment) similarities between all heads:

```julia
using Architect.FAT: unwrap_attentions!, compute_similarity, save_similarity_csv

# 1. Restore original model structures
unwrap_attentions!(model)

# 2. Compute similarity matrix
sim, keys = compute_similarity(collector; metric=:cka)

# 3. Optionally save similarity to a CSV file
save_similarity_csv(sim, keys, "results/head_similarities.csv")
```

### Step 4: Propose Clusters for Folding
Group heads together that exhibit similarity above a specific threshold (typically `0.85` to `0.92` for CKA):

```julia
using Architect.FAT: propose_clusters

# Find groups of heads representing redundant subspaces
clusters = propose_clusters(sim, keys; threshold=0.90f0)

for (idx, cluster) in enumerate(clusters)
    if length(cluster) > 1
        println("Cluster $idx can be merged: $cluster")
    end
end
```

---

## Further Reading

- [Full API Reference](index.md#api-reference)
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [DESIGN.md](../DESIGN.md)
- [Examples](../examples/)

