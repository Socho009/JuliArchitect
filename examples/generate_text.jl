# ─── Text Generation Example ──────────────────────────────────────────────────
#
# Demonstrates text generation with all sampling strategies in JuliaNativeGPT:
# - GreedySampling: deterministic, always picks highest-probability token
# - TopKSampling: filters to top-k tokens, then samples with temperature
# - NucleusSampling: filters by cumulative probability threshold, then samples
#
# Usage:
#   julia --project examples/generate_text.jl

using JuliaNativeGPT
using Random

Random.seed!(123)

# ─── 1. Create and initialize a model ────────────────────────────────────────

# Use a small model for demonstration
vocab_size = 65  # Approximate character-level vocabulary
block_size = 64

config = GPTConfig{Float32}(
    model_type = :gpt_nano,
    vocab_size = vocab_size,
    block_size = block_size,
    embd_pdrop = 0.0,
    resid_pdrop = 0.0,
    attn_pdrop = 0.0,
)

model = GPTModel(config)
initialize!(model)
println("Model: $model")
println("Config: $config")
println()

# ─── 2. Create a prompt (random tokens for demo) ─────────────────────────────

# In a real scenario, you'd encode text with a tokenizer.
# Here we use random tokens to demonstrate the generation API.
prompt = reshape([1, 5, 10, 15, 20], :, 1)  # (seq_len=5, batch=1)
println("Input prompt token IDs: $(vec(prompt))")
println()

# ─── 3. Greedy Sampling ──────────────────────────────────────────────────────

println("=" ^ 60)
println("  GREEDY SAMPLING")
println("=" ^ 60)
println()
println("Greedy sampling always selects the token with the highest logit.")
println("It is deterministic: same input → same output every time.")
println()

output1 = generate(model, prompt; strategy=GreedySampling(), max_new_tokens=20)
output2 = generate(model, prompt; strategy=GreedySampling(), max_new_tokens=20)

println("  Run 1: $(vec(output1))")
println("  Run 2: $(vec(output2))")
println("  Identical? $(output1 == output2)")
println()

# ─── 4. Top-K Sampling ───────────────────────────────────────────────────────

println("=" ^ 60)
println("  TOP-K SAMPLING")
println("=" ^ 60)
println()
println("Top-k sampling keeps only the k highest-probability tokens,")
println("applies temperature scaling, then samples from the distribution.")
println()

# Low temperature (more focused)
println("─── k=5, temperature=0.5 (focused) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = TopKSampling(5, 0.5f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# Medium temperature
println("─── k=10, temperature=1.0 (balanced) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = TopKSampling(10, 1.0f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# High temperature (more random)
println("─── k=20, temperature=1.5 (creative) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = TopKSampling(20, 1.5f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# Large k (approaches full distribution sampling)
println("─── k=50, temperature=1.0 (wide) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = TopKSampling(50, 1.0f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# ─── 5. Nucleus (Top-P) Sampling ─────────────────────────────────────────────

println("=" ^ 60)
println("  NUCLEUS (TOP-P) SAMPLING")
println("=" ^ 60)
println()
println("Nucleus sampling keeps the smallest set of tokens whose cumulative")
println("probability exceeds p, then samples from that set with temperature.")
println()

# Tight nucleus (very focused)
println("─── p=0.5, temperature=0.8 (tight nucleus) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = NucleusSampling(0.5f0, 0.8f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# Standard nucleus
println("─── p=0.9, temperature=1.0 (standard) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = NucleusSampling(0.9f0, 1.0f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# Wide nucleus with high temperature
println("─── p=0.95, temperature=1.2 (creative) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = NucleusSampling(0.95f0, 1.2f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# Full distribution (p=1.0)
println("─── p=1.0, temperature=1.0 (full distribution) ───")
for i in 1:3
    output = generate(model, prompt;
        strategy = NucleusSampling(1.0f0, 1.0f0),
        max_new_tokens = 20
    )
    println("  Run $i: $(vec(output))")
end
println()

# ─── 6. Verify prefix preservation ───────────────────────────────────────────

println("=" ^ 60)
println("  PREFIX PRESERVATION CHECK")
println("=" ^ 60)
println()
println("All strategies preserve the original prompt tokens in the output.")
println()

original_prompt = vec(prompt)
for (name, strategy) in [
    ("Greedy", GreedySampling()),
    ("TopK(k=10)", TopKSampling(10, 1.0f0)),
    ("Nucleus(p=0.9)", NucleusSampling(0.9f0, 1.0f0)),
]
    output = generate(model, prompt; strategy=strategy, max_new_tokens=15)
    output_vec = vec(output)
    prefix_preserved = output_vec[1:length(original_prompt)] == original_prompt
    println("  $name: prefix preserved = $prefix_preserved")
end
println()

# ─── 7. Batch generation ─────────────────────────────────────────────────────

println("=" ^ 60)
println("  BATCH GENERATION")
println("=" ^ 60)
println()
println("Generate multiple sequences in parallel (batch dimension).")
println()

batch_prompt = rand(1:vocab_size, 5, 3)  # (seq_len=5, batch=3)
println("Batch prompt shape: $(size(batch_prompt))")

output_batch = generate(model, batch_prompt;
    strategy = TopKSampling(10, 1.0f0),
    max_new_tokens = 10
)
println("Output shape: $(size(output_batch)) (seq_len=15, batch=3)")
for b in 1:3
    println("  Sequence $b: $(output_batch[:, b])")
end
println()

println("Done! All sampling strategies demonstrated successfully.")
