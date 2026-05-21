# ─── Character-Level Training Example ─────────────────────────────────────────
#
# Demonstrates end-to-end character-level training with JuliaNativeGPT:
# 1. Build a CharTokenizer from sample text
# 2. Encode text into token IDs
# 3. Create a GPTConfig (gpt_nano for fast demo)
# 4. Build and initialize a GPTModel
# 5. Train the model on random batches from the text
# 6. Generate text from the trained model
#
# Usage:
#   julia --project examples/train_char_native.jl
#   julia --project examples/train_char_native.jl path/to/textfile.txt

using JuliaNativeGPT
using Random

Random.seed!(42)

# ─── 1. Load or define sample text ───────────────────────────────────────────

if length(ARGS) >= 1 && isfile(ARGS[1])
    println("Loading text from: $(ARGS[1])")
    text = read(ARGS[1], String)
else
    # Inline sample text for quick demo
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles,
    And by opposing end them. To die: to sleep;
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to, 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep: perchance to dream: ay, there's the rub;
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil,
    Must give us pause: there's the respect
    That makes calamity of so long life.
    """
    println("Using inline Shakespeare sample text ($(length(text)) chars)")
end

# ─── 2. Create CharTokenizer and encode text ─────────────────────────────────

tokenizer = CharTokenizer(text)
println("Vocabulary size: $(tokenizer.vocab_size)")
println("Characters: $(String(tokenizer.idx_to_char))")

# Encode the full text into token IDs
data = encode(tokenizer, text)
println("Encoded length: $(length(data)) tokens")

# ─── 3. Create GPTConfig (gpt_nano for fast demo) ────────────────────────────

block_size = 32  # Short context window for fast training
config = GPTConfig{Float32}(
    model_type = :gpt_nano,
    vocab_size = tokenizer.vocab_size,
    block_size = block_size,
    embd_pdrop = 0.0,
    resid_pdrop = 0.0,
    attn_pdrop = 0.0,
)
println("\nModel config: $config")

# ─── 4. Build GPTModel and initialize weights ────────────────────────────────

model = GPTModel(config)
initialize!(model)
println("Model: $model")

# ─── 5. Create dataset function returning random batches ──────────────────────

batch_size = 4

"""
    make_batch()

Create a random batch from the encoded text data.
Returns (input, target) matrices of shape (block_size, batch_size)
where target is the input shifted by one position.
"""
function make_batch()
    max_start = length(data) - block_size - 1
    input = Matrix{Int}(undef, block_size, batch_size)
    target = Matrix{Int}(undef, block_size, batch_size)

    for b in 1:batch_size
        start_idx = rand(1:max_start)
        input[:, b] = data[start_idx:start_idx+block_size-1]
        target[:, b] = data[start_idx+1:start_idx+block_size]
    end

    return input, target
end

# ─── 6. Train the model ──────────────────────────────────────────────────────

train_config = TrainConfig{Float32}(
    learning_rate = 1f-3,
    weight_decay = 0.0f0,
    max_iters = 200,
    grad_clip = 1.0f0,
    batch_size = batch_size,
    betas = (0.9f0, 0.99f0),
)
println("\nTraining config: $train_config")
println("Starting training...")

state = train!(model, make_batch, train_config)

println("\nTraining complete!")
println("  Initial loss: $(state.losses[1])")
println("  Final loss:   $(state.losses[end])")
println("  Iterations:   $(state.iter)")

# ─── 7. Generate text from the trained model ──────────────────────────────────

# Start with a short prompt
prompt_text = "To be"
prompt_ids = encode(tokenizer, prompt_text)
prompt_matrix = reshape(prompt_ids, :, 1)  # (seq_len, batch=1)

println("\n─── Text Generation ───")
println("Prompt: \"$prompt_text\"")

# Generate with greedy sampling
output_greedy = generate(model, prompt_matrix;
    strategy = GreedySampling(),
    max_new_tokens = 100
)
generated_text = decode(tokenizer, vec(output_greedy))
println("\n[Greedy] Generated:")
println("  \"$generated_text\"")

# Generate with top-k sampling
output_topk = generate(model, prompt_matrix;
    strategy = TopKSampling(10, 0.8f0),
    max_new_tokens = 100
)
generated_text_topk = decode(tokenizer, vec(output_topk))
println("\n[Top-K (k=10, T=0.8)] Generated:")
println("  \"$generated_text_topk\"")

# Generate with nucleus sampling
output_nucleus = generate(model, prompt_matrix;
    strategy = NucleusSampling(0.9f0, 0.9f0),
    max_new_tokens = 100
)
generated_text_nucleus = decode(tokenizer, vec(output_nucleus))
println("\n[Nucleus (p=0.9, T=0.9)] Generated:")
println("  \"$generated_text_nucleus\"")

println("\nDone!")
