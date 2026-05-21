#!/usr/bin/env julia
# ─── Character-Level RNN/LSTM/GRU Training ────────────────────────────────────
#
# Julia-native reimagining of Karpathy's char-rnn.
# Trains a recurrent language model on character-level text data,
# then generates sample text using the same SamplingStrategy dispatch
# as the GPT transformer model.
#
# Usage:
#   julia --project examples/train_charrnn.jl                    # LSTM on inline Shakespeare
#   julia --project examples/train_charrnn.jl path/to/text.txt   # custom text file
#   julia --project examples/train_charrnn.jl --cell=gru         # GRU instead of LSTM
#   julia --project examples/train_charrnn.jl --cell=rnn --hidden=256

using JuliaNativeGPT
using Random

Random.seed!(42)

# ─── Parse arguments ──────────────────────────────────────────────────────────

cell_type = :lstm
hidden_size = 128
n_layers = 2
seq_length = 50
batch_size = 32
max_iters = 200
learning_rate = 2f-3
text_file = nothing

for arg in ARGS
    if startswith(arg, "--cell=")
        cell_type = Symbol(split(arg, "=")[2])
    elseif startswith(arg, "--hidden=")
        hidden_size = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--layers=")
        n_layers = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--seq=")
        seq_length = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--batch=")
        batch_size = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--iters=")
        max_iters = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--lr=")
        learning_rate = parse(Float32, split(arg, "=")[2])
    elseif !startswith(arg, "--")
        text_file = arg
    end
end

# ─── Load text data ──────────────────────────────────────────────────────────

if text_file !== nothing && isfile(text_file)
    println("Loading text from: $text_file")
    text = read(text_file, String)
else
    # Try tinyshakespeare from char-rnn data
    shakespeare_path = joinpath(@__DIR__, "..", "char-rnn", "data", "tinyshakespeare", "input.txt")
    if isfile(shakespeare_path)
        println("Loading tinyshakespeare from char-rnn/data/...")
        text = read(shakespeare_path, String)
    else
        println("Using inline Shakespeare excerpt...")
        text = """
        First Citizen:
        Before we proceed any further, hear me speak.

        All:
        Speak, speak.

        First Citizen:
        You are all resolved rather to die than to famish?

        All:
        Resolved. resolved.

        First Citizen:
        First, you know Caius Marcius is chief enemy to the people.

        All:
        We know't, we know't.

        First Citizen:
        Let us kill him, and we'll have corn at our own price.
        Is't a verdict?

        All:
        No more talking on't; let it be done: away, away!

        Second Citizen:
        One word, good citizens.

        First Citizen:
        We are accounted poor citizens, the patricians good.
        What authority surfeits on would relieve us: if they
        would yield us but the superfluity, while it were
        wholesome, we might guess they relieved us humanely;
        but they think we are too dear: the leanness that
        afflicts us, the object of our misery, is as an
        inventory to particularise their abundance; our
        sufferance is a gain to them Let us revenge this with
        our pikes, ere we become rakes: for the gods know I
        speak this in hunger for bread, not in thirst for revenge.
        """
    end
end

println("Text length: $(length(text)) characters")
println()

# ─── Build data loader ───────────────────────────────────────────────────────

loader = CharDataLoader(text; batch_size=batch_size, seq_length=seq_length)
println("Data loader: $loader")
println("Vocabulary: $(loader.tokenizer.vocab_size) unique characters")
println()

# ─── Build model ─────────────────────────────────────────────────────────────

config = RecurrentConfig{Float32}(
    vocab_size = loader.tokenizer.vocab_size,
    seq_length = seq_length,
    n_layer = n_layers,
    hidden_size = hidden_size,
    cell_type = cell_type,
    dropout = 0.0,
)

model = RecurrentLM(config)
initialize!(model)

println("Model: $model")
println()

# ─── Spectral diagnostics (pre-training) ─────────────────────────────────────

println("─── Pre-training spectral stats ───")
stats = compute_recurrent_spectral_stats(model)
for (k, v) in sort(collect(stats))
    println("  $k = $(round(v; digits=4))")
end
println()

# ─── Train ───────────────────────────────────────────────────────────────────

train_config = TrainConfig{Float32}(
    learning_rate = learning_rate,
    weight_decay = 0.0f0,
    max_iters = max_iters,
    grad_clip = 5.0f0,
    batch_size = batch_size,
)

dataset = make_batch_fn(loader, :train)

println("Training $(config.cell_type) for $(train_config.max_iters) iterations...")
println("  lr=$(train_config.learning_rate), grad_clip=$(train_config.grad_clip)")
println()

t_start = time()
state = train!(model, dataset, train_config)
elapsed = time() - t_start

println()
println("Training complete in $(round(elapsed; digits=1))s")
println("  First loss:  $(round(state.losses[1]; digits=4))")
println("  Final loss:  $(round(state.losses[end]; digits=4))")
println("  Loss ratio:  $(round(state.losses[end] / state.losses[1]; digits=3))")
println()

# ─── Spectral diagnostics (post-training) ────────────────────────────────────

println("─── Post-training spectral stats ───")
stats = compute_recurrent_spectral_stats(model)
for (k, v) in sort(collect(stats))
    println("  $k = $(round(v; digits=4))")
end
println()

# ─── Generate sample text ────────────────────────────────────────────────────

println("═" ^ 60)
println("GENERATED TEXT SAMPLES")
println("═" ^ 60)

# Seed with a short prompt
seed_text = "First"
seed_ids = encode(loader.tokenizer, seed_text)
prompt = reshape(seed_ids, :, 1)  # (seq_len, 1)

strategies = [
    ("Greedy", GreedySampling()),
    ("Top-K (k=10, T=0.8)", TopKSampling(10, 0.8f0)),
    ("Nucleus (p=0.9, T=0.9)", NucleusSampling(0.9f0, 0.9f0)),
]

for (name, strategy) in strategies
    output_ids = generate(model, prompt; strategy=strategy, max_new_tokens=200)
    output_text = decode(loader.tokenizer, vec(output_ids))
    println()
    println("── $name ──")
    println(output_text)
end

println()
println("═" ^ 60)
