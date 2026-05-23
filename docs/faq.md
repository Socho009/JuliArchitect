# Architect.jl FAQ

Frequently asked questions about Architect.jl and how to use it.

---

## General Questions

### What is Architect.jl?

Architect.jl is a Julia-native machine learning framework combining classical neural architectures (Transformers, RNNs, Llama2) with a novel **Functional Autoregressive Transformer (FAT)** that predicts continuous functions instead of discrete tokens.

### Why Julia?

Julia is ideal for scientific computing and ML because of:
- **Multiple dispatch**: Clean, extensible code organization
- **Type specialization**: Compile-time code generation for efficiency
- **AD-friendly**: Zygote.jl automatic differentiation works seamlessly
- **Performance**: Close to C/Fortran speed without manual optimization

### Is this production-ready?

The code is **production-ready** (parametric types, AD-compatible, no type instability). The **research contributions** (FAT algorithm) are still in empirical validation phase (Phase 2). Use Transformers/RNNs for proven architectures; use FAT for research.

### What are the key differences from PyTorch/TensorFlow implementations?

1. **No tensor libraries needed**: Pure Julia code (easier debugging)
2. **Type-driven specialization**: Parametric types replace Python's duck typing
3. **Multiple dispatch**: Single `forward()` call routes to 10+ specialized methods
4. **Functional programming**: Mutation-free code for Zygote compatibility
5. **No framework overhead**: Direct access to Julia's numerical stack

---

## Installation & Setup

### Why do I get "module not found" errors?

**Solution**: Ensure you're using the correct project environment:

```bash
# From the Architect.jl directory
julia --project -e 'using Pkg; Pkg.instantiate()'

# Verify installation
julia --project -e 'using Architect; println("OK")'
```

### How do I use this in my own project?

**Option 1**: Add to Project.toml:
```toml
[deps]
Architect = {path = "/path/to/julia-native-gpt"}
```

**Option 2**: Dynamic development mode:
```julia
using Pkg
Pkg.develop(path="/path/to/julia-native-gpt")
```

### Can I use conda/pip to install this?

Not yet—Architect.jl is a research package distributed via GitHub. Install Julia 1.9+ and clone the repository.

---

## Model Configuration

### How do I choose between gpt_nano, gpt_mini, gpt2?

| Preset | Size | Speed | Use Case |
|--------|------|-------|----------|
| `gpt_nano` | 30K | Fast | Quick experiments, testing |
| `gpt_micro` | 800K | Fast | Small projects |
| `gpt_mini` | 3M | Medium | Learning, research |
| `gpt2` | 124M | Slow | Production (CPU) |

Start with `gpt_nano` for debugging; scale up once code works.

### What's the difference between `block_size` and `seq_len`?

- **`block_size`**: Fixed context window during training (e.g., 256 tokens max)
- **`seq_len`**: Length of sequences in your data loader

They should match (or `seq_len ≥ block_size`).

### How do I use custom model sizes?

```julia
config = ModelConfig(
    vocab_size=256,
    n_embd=512,      # embedding dimension
    n_head=8,        # must divide n_embd
    n_layer=16,      # number of transformer blocks
    block_size=256,
    dropout=0.1
)
model = ContinuousModel{Float32}(config)
```

**Rule**: `n_embd % n_head == 0` (embedding dimension divisible by heads)

### What's µP (Maximal Update Parametrization)?

µP scales learning rates and initialization with model size, keeping training dynamics consistent across scales. Use `initialize!(model; muP=true)` for better transfer across sizes.

---

## Training

### Loss is NaN after a few steps—what's wrong?

**Common causes**:
1. **Learning rate too high** → Use `learning_rate=1e-4` initially
2. **Gradient explosion** → Increase `grad_clip` (e.g., `grad_clip=0.5`)
3. **Bad data** → Check data contains no Inf/NaN
4. **Type mismatch** → Ensure model and data have same precision (Float32)

**Debug**:
```julia
# Start with tiny learning rate
config = TrainConfig(learning_rate=1e-5, grad_clip=0.1, max_iters=10)
state, losses = train!(model, batch_fn, config)
println(losses)  # Should decrease steadily
```

### How do I resume training from a checkpoint?

```julia
# Save
state, losses = train!(model, batch_fn, config)
# → state contains optimizer state

# Resume
state, new_losses = train!(model, batch_fn, config; resume_state=state)
# Continues from checkpoint
```

### How long should training take?

**Rough estimates** (on CPU):
- **gpt_nano, 1 epoch**: ~5 minutes
- **gpt_micro, 1 epoch**: ~30 minutes
- **gpt_mini, 1 epoch**: ~2 hours
- **gpt2, 1 epoch**: ~1 day

GPU (CUDA) is 10–100× faster.

### Can I use mixed precision?

Partially. Use Float32 for training (most stable); float16 computation is experimental:

```julia
model = ContinuousModel{Float32}(config)
# Zygote handles Float32 ↔ Float64 conversion automatically
```

### What's the best learning rate?

Start with **3e-4** and adjust:
- **Loss diverges** → reduce to **1e-4**
- **Loss decreases too slowly** → increase to **1e-3**
- **Best practice** → use learning rate scheduler (warmup + cosine decay)

---

## Data & Tokenization

### How do I tokenize custom data?

```julia
# Character-level
tokenizer = CharTokenizer(text)
ids = encode(tokenizer, "hello")
text_back = decode(tokenizer, ids)

# Custom vocab
vocab = ["<unk>", "<pad>", "hello", "world"]
tokenizer = BPETokenizer(vocab)
```

### How do I split data into train/val/test?

```julia
loader = CharDataLoader(
    text, tokenizer;
    seq_len=256,
    batch_size=32,
    split=(0.8, 0.1, 0.1)  # train, val, test
)

# Access splits
train_data = loader.train_data
val_data = loader.val_data
test_data = loader.test_data
```

### Can I use custom data formats (CSV, JSON)?

Yes. Create a custom loader:

```julia
mutable struct CSVDataLoader
    data::Matrix
    # ... implement next_batch!
end

function next_batch!(loader::CSVDataLoader)
    # Return next batch
end
```

Then use with training:
```julia
loader = CSVDataLoader(...)
batch_fn = () -> next_batch!(loader)
train!(model, batch_fn, config)
```

---

## Inference & Generation

### How do I generate text with specific styles?

```julia
# Greedy (always pick max probability)
output = generate(model, prompt; strategy=GreedySampling())

# Top-K (sample from top K tokens)
output = generate(model, prompt; strategy=TopKSampling(k=40))

# Nucleus (sample from tokens above threshold)
output = generate(model, prompt; strategy=NucleusSampling(p=0.9f0))

# Temperature control
output = generate(model, prompt; 
                  strategy=TopKSampling(k=40),
                  temperature=0.5f0)  # Lower = more deterministic
```

### How do I generate faster?

1. **Reduce batch size**: `max_new_tokens=10` (fewer iterations)
2. **Use KV-cache**: (Already enabled for Llama2)
3. **Use smaller model**: `gpt_nano` vs `gpt2`
4. **Use GPU**: `to_device(model, :cuda)`

### How do I make generation reproducible?

```julia
using Random
Random.seed!(42)

# Now generation is deterministic
output = generate(model, prompt)
```

---

## Functional Autoregressive Transformer (FAT)

### What's FAT and when should I use it?

FAT predicts continuous functions (polynomials, smooth functions) instead of discrete tokens. Use it when:
- You need **arbitrary smooth functions** as outputs
- You have **mathematical operators** to learn (derivative, integral)
- You want **parameter efficiency** (7,000× fewer params)

### How is FAT different from standard LMs?

| Aspect | Standard LM | FAT |
|--------|------------|-----|
| Output | Probability distribution over vocab | Function coefficients |
| Basis | One-hot vectors (tokens) | Orthogonal basis (Chebyshev, Fourier) |
| Stability | May diverge in autoregression | Guaranteed non-expansive (L ≤ 1) |
| Parameters | Millions (embedding table) | Thousands (no embedding) |

### How do I train FAT?

```julia
# 1. Create data loader
loader = FunctionSequenceLoader(
    task=:derivative,    # or :antiderivative, :taylor
    basis_dim=8,
    seq_length=16,
    batch_size=32
)

# 2. Create model
fat_config = FATConfig{Float32}(
    basis_dim=8,
    n_embd=64,
    n_head=4,
    n_layer=2
)
fat_model = FunctionalAutoregressiveTransformer(fat_config)
initialize!(fat_model)

# 3. Train
train_config = FATTrainConfig(learning_rate=3e-4)
state = fat_train!(fat_model, loader, train_config)
```

### How do I evaluate a FAT model?

```julia
# Input: function coefficients
f_coeffs = Float32[1, 0, -1, 0, 0, 0, 0, 0]  # f(x) = 1 - x²

# Predict next function (e.g., derivative)
f_prime = fat_generate(fat_model, f_coeffs; max_new=1)

# Evaluate at a point
y_value = evaluate_function(f_prime[:, 1], 0.5f0; basis=:monomial)
```

### What bases are supported?

- **Monomial**: `{1, x, x², x³, ...}` — polynomials
- **Fourier**: `{1, cos(x), sin(x), ...}` — periodic functions
- **Chebyshev**: `{T₀, T₁, T₂, ...}` — optimal for smooth functions

Chebyshev is recommended (exponential convergence).

---

## GPU & Performance

### How do I enable GPU?

```julia
using CUDA
model = to_device(model, :cuda)

# Verify
println(typeof(model.layers[1].weight))  # Should be CuArray
```

### Does this work with other GPU frameworks?

Currently only CUDA (via CUDA.jl). AMD/Metal support would require additional backend integration.

### How do I benchmark my model?

```julia
using BenchmarkTools

# Forward pass timing
@time output = model(x)
@benchmark output = $model($x)

# Full iteration timing
@time loss, grads = train_step(model, batch)
```

### Why is training slow?

1. **CPU vs GPU**: CPU is 10–100× slower
2. **Model size**: Larger models → slower (O(n²) attention)
3. **Batch size**: Small batches → fewer ops/sec
4. **Precision**: Float64 slower than Float32

**Optimization**:
```julia
# Use smaller model for testing
config = ModelConfig(n_layer=2, n_embd=128)

# Use GPU
model = to_device(model, :cuda)

# Increase batch size (if memory allows)
loader = CharDataLoader(...; batch_size=64)
```

---

## Debugging & Development

### How do I check gradients?

```julia
using Zygote

gs = gradient(() -> mean(model(x)), Flux.params(model))

# Check for NaN/Inf
for (name, p) in named_params(model)
    g = gs[p]
    if any(isnan, g) || any(isinf, g)
        println("Problem in $name: gradient is $(typeof(g))")
    end
end
```

### How do I add a custom layer?

```julia
struct MyCustomLayer{P} <: AbstractLayer{P}
    weight::Matrix{P}
    bias::Vector{P}
end

# Implement forward pass
function (layer::MyCustomLayer)(x::Matrix)
    layer.weight * x .+ layer.bias
end

# Add to initialization dispatch
function initialize!(layer::MyCustomLayer; kwargs...)
    # Initialize weight, bias
    randn!(layer.weight)
end
```

### How do I profile my code?

```julia
using Profile, ProfileView

# Profile one training iteration
@profile for i in 1:10
    loss, grads = train_step(model, batch)
end

# View results
ProfileView.view()

# Or text output
Profile.print()
```

### How do I run tests?

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

---

## Common Errors

### Error: "LoadError: UndefVarError: ContinuousModel not defined"

**Fix**: Make sure module is loaded:
```julia
using Architect
```

### Error: "DimensionMismatch"

**Cause**: Input/output size mismatch
**Fix**: Check model expects shape `(seq_len, batch_size)` not `(batch_size, seq_len)`

### Error: "GPU out of memory"

**Fix**: Reduce batch size or model size:
```julia
config = setproperties(config; n_layer=4, n_embd=256)
train_config = setproperties(train_config; batch_size=8)
```

### Error: "type UndefVarError"

**Cause**: Type instability (mixing Float32 and Float64)
**Fix**: Ensure consistent precision:
```julia
model = ContinuousModel{Float32}(config)
x = Float32.(x)  # Convert input to Float32
```

---

## Contributing & Support

### How do I report a bug?

1. Create minimal reproducible example
2. Open GitHub Issue with:
   - Julia version
   - Error message
   - Minimal code to reproduce
   - Expected vs actual behavior

### How do I contribute code?

1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit Pull Request

### Where can I ask questions?

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas
- **DM [@miniAliyah](https://github.com/miniAliyah)**: Research discussions

---

## Citation

If you use Architect.jl in research, please cite:

```bibtex
@software{architect2026,
  title={Architect.jl: A Julia Framework for Functional Autoregressive Transformers},
  author={Boularak, Souheyb},
  year={2026},
  url={https://github.com/miniAliyah/julia-native-gpt}
}
```

For the FAT algorithm:

```bibtex
@article{boularak2026fat,
  title={Functional Autoregressive Transformers: Rethinking Next-Token Prediction 
         through Continuous Functional Analysis},
  author={Boularak, Souheyb},
  year={2026},
  note={Submitted to arXiv}
}
```

---

**Last Updated**: May 23, 2026
