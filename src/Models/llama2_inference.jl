# ─── Llama 2 Inference ────────────────────────────────────────────────────────
#
# Julia translation of Karpathy's llama2.c/run.c
# "Inference for Llama-2 Transformer model" — faithful to the C original
# but integrated into Architect's type system and sampling dispatch.
#
# Loads pre-trained .bin checkpoints from the llama2.c ecosystem.

# ─── Config (maps to C's Config struct) ───────────────────────────────────────

"""
    Llama2Config

Llama-2 model configuration, read directly from checkpoint header.
Maps 1:1 to the 7-int Config struct in llama2.c/run.c.
"""
struct Llama2Config
    dim::Int32            # transformer dimension
    hidden_dim::Int32     # FFN hidden dimension
    n_layers::Int32       # number of layers
    n_heads::Int32        # number of query heads
    n_kv_heads::Int32     # number of key/value heads (GQA)
    vocab_size::Int32     # vocabulary size
    seq_len::Int32        # max sequence length
end

function Base.show(io::IO, c::Llama2Config)
    print(io, "Llama2Config(dim=$(c.dim), layers=$(c.n_layers), heads=$(c.n_heads), kv_heads=$(c.n_kv_heads), vocab=$(c.vocab_size), seq=$(c.seq_len))")
end

# ─── Weights (maps to C's TransformerWeights) ─────────────────────────────────

"""
    Llama2Weights{P}

All weight tensors for a Llama-2 model, memory-mapped from a .bin checkpoint.
Layout matches llama2.c's `memory_map_weights` exactly.
"""
struct Llama2Weights{P<:AbstractFloat}
    token_embedding_table::Matrix{P}  # (dim, vocab_size)
    rms_att_weight::Matrix{P}         # (dim, n_layers)
    wq::Array{P,3}                    # (dim, n_heads*head_size, n_layers)
    wk::Array{P,3}                    # (dim, n_kv_heads*head_size, n_layers)
    wv::Array{P,3}                    # (dim, n_kv_heads*head_size, n_layers)
    wo::Array{P,3}                    # (n_heads*head_size, dim, n_layers)
    rms_ffn_weight::Matrix{P}         # (dim, n_layers)
    w1::Array{P,3}                    # (dim, hidden_dim, n_layers)
    w2::Array{P,3}                    # (hidden_dim, dim, n_layers)
    w3::Array{P,3}                    # (dim, hidden_dim, n_layers)
    rms_final_weight::Vector{P}       # (dim,)
    wcls::Matrix{P}                   # (dim, vocab_size) — classifier
end

# ─── RunState (maps to C's RunState) ──────────────────────────────────────────

"""
    Llama2State{P}

Mutable activation buffers for the forward pass, matching run.c's RunState.
Pre-allocated once, reused across timesteps.
"""
mutable struct Llama2State{P<:AbstractFloat}
    x::Vector{P}          # activation at current timestep (dim,)
    xb::Vector{P}         # residual branch buffer (dim,)
    xb2::Vector{P}        # additional buffer (dim,)
    hb::Vector{P}         # FFN hidden buffer (hidden_dim,)
    hb2::Vector{P}        # FFN hidden buffer 2 (hidden_dim,)
    q::Vector{P}          # query (dim,)
    att::Matrix{P}        # attention scores (seq_len, n_heads)
    logits::Vector{P}     # output logits (vocab_size,)
    key_cache::Array{P,3}   # (kv_dim, seq_len, n_layers)
    value_cache::Array{P,3} # (kv_dim, seq_len, n_layers)
end

function Llama2State{P}(c::Llama2Config) where P
    kv_dim = (c.dim * c.n_kv_heads) ÷ c.n_heads
    Llama2State{P}(
        zeros(P, c.dim),
        zeros(P, c.dim),
        zeros(P, c.dim),
        zeros(P, c.hidden_dim),
        zeros(P, c.hidden_dim),
        zeros(P, c.dim),
        zeros(P, c.seq_len, c.n_heads),
        zeros(P, c.vocab_size),
        zeros(P, kv_dim, c.seq_len, c.n_layers),
        zeros(P, kv_dim, c.seq_len, c.n_layers),
    )
end

# ─── Transformer (bundles config + weights + state) ───────────────────────────

"""
    Llama2Transformer{P}

Complete Llama-2 transformer for inference, mirroring llama2.c's Transformer struct.
Load from a .bin checkpoint with `load_llama2_checkpoint`.
"""
struct Llama2Transformer{P<:AbstractFloat}
    config::Llama2Config
    weights::Llama2Weights{P}
    state::Llama2State{P}
end

function Base.show(io::IO, t::Llama2Transformer{P}) where P
    c = t.config
    n_params = c.vocab_size * c.dim + c.n_layers * (
        c.dim + c.dim*c.dim + 2*c.dim*(c.dim*c.n_kv_heads÷c.n_heads) +
        c.dim*c.dim + c.dim + 2*c.dim*c.hidden_dim + c.hidden_dim*c.dim
    ) + c.dim + c.vocab_size*c.dim
    print(io, "Llama2Transformer{$P}(dim=$(c.dim), layers=$(c.n_layers), heads=$(c.n_heads), params≈$(round(n_params/1e6; digits=1))M)")
end

# ─── Checkpoint Loading ───────────────────────────────────────────────────────

"""
    load_llama2_checkpoint(filepath::String; P=Float32)

Load a llama2.c .bin checkpoint file into a `Llama2Transformer`.

The binary format is:
  [7 × Int32 config header] [flat Float32 weights in known order]

Negative vocab_size signals unshared classifier weights.
"""
function load_llama2_checkpoint(filepath::String; P::Type{<:AbstractFloat}=Float32)
    data = read(filepath)
    buf = IOBuffer(data)

    # Read config header (7 × Int32)
    dim = read(buf, Int32)
    hidden_dim = read(buf, Int32)
    n_layers = read(buf, Int32)
    n_heads = read(buf, Int32)
    n_kv_heads = read(buf, Int32)
    vocab_size_raw = read(buf, Int32)
    seq_len = read(buf, Int32)

    shared_weights = vocab_size_raw > 0
    vocab_size = abs(vocab_size_raw)
    config = Llama2Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)

    head_size = dim ÷ n_heads
    kv_dim = (dim * n_kv_heads) ÷ n_heads

    # Helper: read N floats and convert to P
    function read_weights(n::Int)
        raw = Vector{Float32}(undef, n)
        read!(buf, raw)
        return P == Float32 ? raw : P.(raw)
    end

    # Read weights in llama2.c's exact order
    token_embedding_table = reshape(read_weights(vocab_size * dim), dim, vocab_size)
    rms_att_weight = reshape(read_weights(n_layers * dim), dim, n_layers)
    wq = reshape(read_weights(n_layers * dim * n_heads * head_size), dim, n_heads * head_size, n_layers)
    wk = reshape(read_weights(n_layers * dim * kv_dim), dim, kv_dim, n_layers)
    wv = reshape(read_weights(n_layers * dim * kv_dim), dim, kv_dim, n_layers)
    wo = reshape(read_weights(n_layers * n_heads * head_size * dim), n_heads * head_size, dim, n_layers)
    rms_ffn_weight = reshape(read_weights(n_layers * dim), dim, n_layers)
    w1 = reshape(read_weights(n_layers * dim * hidden_dim), dim, hidden_dim, n_layers)
    w2 = reshape(read_weights(n_layers * hidden_dim * dim), hidden_dim, dim, n_layers)
    w3 = reshape(read_weights(n_layers * dim * hidden_dim), dim, hidden_dim, n_layers)
    rms_final_weight = read_weights(dim)

    # Skip freq_cis (RoPE precomputed in run.c, we compute on the fly)
    skip_bytes = seq_len * head_size ÷ 2 * 2 * sizeof(Float32)
    skip(buf, skip_bytes)

    wcls = shared_weights ? copy(token_embedding_table) : reshape(read_weights(vocab_size * dim), dim, vocab_size)

    weights = Llama2Weights{P}(
        token_embedding_table, rms_att_weight,
        wq, wk, wv, wo,
        rms_ffn_weight, w1, w2, w3,
        rms_final_weight, wcls
    )

    state = Llama2State{P}(config)
    return Llama2Transformer{P}(config, weights, state)
end

# ─── Neural Net Blocks (faithful to run.c) ────────────────────────────────────

"""
    _llama_rmsnorm!(o, x, weight, size)

Norm: o = weight .* (x / √(mean(x²) + ε))
Matches run.c's `rmsnorm` exactly.
"""
function _llama_rmsnorm!(o::Vector{P}, x::Vector{P}, weight::AbstractVector{P}) where P
    ss = sum(abs2, x) / length(x) + P(1e-5)
    ss = one(P) / sqrt(ss)
    @inbounds for j in eachindex(o)
        o[j] = weight[j] * (ss * x[j])
    end
end

"""
    _llama_forward!(transformer, token, pos) → logits

Single-token forward pass, matching run.c's `forward` function exactly.
`token` is 1-indexed, `pos` is 0-indexed (matching C convention internally).
"""
function _llama_forward!(t::Llama2Transformer{P}, token::Int, pos::Int) where P
    c = t.config
    w = t.weights
    s = t.state
    dim = Int(c.dim)
    kv_dim = (dim * Int(c.n_kv_heads)) ÷ Int(c.n_heads)
    kv_mul = Int(c.n_heads) ÷ Int(c.n_kv_heads)
    hidden_dim = Int(c.hidden_dim)
    head_size = dim ÷ Int(c.n_heads)
    n_heads = Int(c.n_heads)
    n_layers = Int(c.n_layers)
    seq_len = Int(c.seq_len)

    # Copy token embedding into x (1-indexed token)
    s.x .= @view w.token_embedding_table[:, token]

    # Forward all layers
    for l in 1:n_layers

        # Attention rmsnorm
        _llama_rmsnorm!(s.xb, s.x, @view w.rms_att_weight[:, l])

        # QKV matmuls for this position
        # q = wq[:,:,l]' * xb  (run.c: matmul(q, xb, wq + l*dim*dim, dim, dim))
        mul!(s.q, (@view w.wq[:, :, l])', s.xb)

        # k and v go directly into the cache at this position
        pos1 = pos + 1  # 1-indexed position for Julia arrays
        k_view = @view s.key_cache[:, pos1, l]
        v_view = @view s.value_cache[:, pos1, l]
        mul!(k_view, (@view w.wk[:, :, l])', s.xb)
        mul!(v_view, (@view w.wv[:, :, l])', s.xb)

        # RoPE: rotate q and k in each head
        @inbounds for i in 0:2:(dim-1)
            head_dim = i % head_size
            freq = P(1.0) / P(10000.0)^(P(head_dim) / P(head_size))
            val = P(pos) * freq
            fcr = cos(val)
            fci = sin(val)
            # Rotate q
            v0 = s.q[i+1]
            v1 = s.q[i+2]
            s.q[i+1] = v0 * fcr - v1 * fci
            s.q[i+2] = v0 * fci + v1 * fcr
            # Rotate k (only for kv_dim positions)
            if i < kv_dim
                v0 = k_view[i+1]
                v1 = k_view[i+2]
                k_view[i+1] = v0 * fcr - v1 * fci
                k_view[i+2] = v0 * fci + v1 * fcr
            end
        end

        # Multihead attention
        for h in 0:(n_heads-1)
            q_offset = h * head_size
            q_head = @view s.q[q_offset+1 : q_offset+head_size]

            # Compute attention scores for all positions up to pos
            for t_pos in 0:pos
                kv_head = h ÷ kv_mul
                k_offset = kv_head * head_size
                k_head = @view s.key_cache[k_offset+1 : k_offset+head_size, t_pos+1, l]
                score = dot(q_head, k_head) / sqrt(P(head_size))
                s.att[t_pos+1, h+1] = score
            end

            # Softmax over 0..pos
            att_slice = @view s.att[1:pos+1, h+1]
            max_val = maximum(att_slice)
            att_slice .= exp.(att_slice .- max_val)
            att_sum = sum(att_slice)
            att_slice ./= att_sum

            # Weighted sum of values → xb
            xb_head = @view s.xb[q_offset+1 : q_offset+head_size]
            xb_head .= zero(P)
            for t_pos in 0:pos
                kv_head = h ÷ kv_mul
                v_offset = kv_head * head_size
                v_head = @view s.value_cache[v_offset+1 : v_offset+head_size, t_pos+1, l]
                a = s.att[t_pos+1, h+1]
                @inbounds for i in eachindex(xb_head)
                    xb_head[i] += a * v_head[i]
                end
            end
        end

        # Output projection: xb2 = wo[:,:,l]' * xb
        mul!(s.xb2, (@view w.wo[:, :, l])', s.xb)

        # Residual connection
        s.x .+= s.xb2

        # FFN rmsnorm
        _llama_rmsnorm!(s.xb, s.x, @view w.rms_ffn_weight[:, l])

        # FFN: FeedForward — self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul!(s.hb, (@view w.w1[:, :, l])', s.xb)
        mul!(s.hb2, (@view w.w3[:, :, l])', s.xb)

        # SiLU activation + gate
        @inbounds for i in eachindex(s.hb)
            val = s.hb[i]
            val *= one(P) / (one(P) + exp(-val))  # silu(x) = x * σ(x)
            s.hb[i] = val * s.hb2[i]              # gate with w3
        end

        # Down projection: xb = w2[:,:,l]' * hb
        mul!(s.xb, (@view w.w2[:, :, l])', s.hb)

        # Residual connection
        s.x .+= s.xb
    end

    # Final rmsnorm
    _llama_rmsnorm!(s.x, s.x, w.rms_final_weight)

    # Classifier into logits
    mul!(s.logits, w.wcls', s.x)

    return s.logits
end

# ─── Generation (integrates with existing SamplingStrategy dispatch) ──────────

"""
    generate(transformer::Llama2Transformer, prompt_tokens::Vector{Int};
             strategy=GreedySampling(), max_tokens=256)

Autoregressive generation from a Llama-2 transformer.
Uses the same `SamplingStrategy` dispatch as SequenceModel and SequenceModel.

`prompt_tokens` should be 1-indexed token IDs.
"""
function generate(
    transformer::Llama2Transformer{P},
    prompt_tokens::Vector{Int};
    strategy::SamplingStrategy = GreedySampling(),
    max_tokens::Int = 256
) where P
    seq_len = Int(transformer.config.seq_len)
    steps = min(max_tokens, seq_len)

    token = prompt_tokens[1]
    result = Int[token]

    for pos in 0:(steps-1)
        logits = _llama_forward!(transformer, token, pos)

        if pos < length(prompt_tokens) - 1
            # Still consuming prompt — force next token
            next = prompt_tokens[pos + 2]
        else
            # Sample from logits using existing dispatch
            next = sample_next_token(logits, strategy)
        end

        push!(result, next)
        token = next

        # Stop on BOS token (id=1 in Llama's vocab, but 2 in 1-indexed)
        next == 2 && break
    end

    return result
end

# ─── Llama2 Tokenizer (.bin format) ───────────────────────────────────────────

"""
    Llama2Tokenizer

SentencePiece-compatible tokenizer loaded from llama2.c's tokenizer.bin format.
"""
struct Llama2Tokenizer
    vocab::Vector{String}
    vocab_scores::Vector{Float32}
    vocab_size::Int
    max_token_length::Int
end

"""
    load_llama2_tokenizer(filepath::String, vocab_size::Int)

Load a tokenizer.bin file from the llama2.c ecosystem.
"""
function load_llama2_tokenizer(filepath::String, vocab_size::Int)
    buf = IOBuffer(read(filepath))

    max_token_length = read(buf, Int32)
    vocab = Vector{String}(undef, vocab_size)
    scores = Vector{Float32}(undef, vocab_size)

    for i in 1:vocab_size
        scores[i] = read(buf, Float32)
        len = read(buf, Int32)
        bytes = Vector{UInt8}(undef, len)
        read!(buf, bytes)
        vocab[i] = String(bytes)
    end

    Llama2Tokenizer(vocab, scores, vocab_size, Int(max_token_length))
end

"""
    decode(tok::Llama2Tokenizer, tokens::Vector{Int}) → String

Decode a sequence of 1-indexed token IDs back to text.
"""
function decode(tok::Llama2Tokenizer, tokens::Vector{Int})
    pieces = String[]
    prev = 0
    for (i, t) in enumerate(tokens)
        1 <= t <= tok.vocab_size || continue
        piece = tok.vocab[t]
        # After BOS token (1), strip leading whitespace
        if prev == 1 && startswith(piece, ' ')
            piece = piece[2:end]
        end
        # Handle raw byte tokens like <0x0A>
        m = match(r"^<0x([0-9A-Fa-f]{2})>$", piece)
        if m !== nothing
            piece = String([parse(UInt8, m.captures[1]; base=16)])
        end
        push!(pieces, piece)
        prev = t
    end
    return join(pieces)
end

# Device transfer for Llama2 structures
import ..Core

function Core.to_device(w::Llama2Weights{P}, device::Symbol) where P
    return Llama2Weights{P}(
        to_device(w.token_embedding_table, device),
        to_device(w.rms_att_weight, device),
        to_device(w.wq, device),
        to_device(w.wk, device),
        to_device(w.wv, device),
        to_device(w.wo, device),
        to_device(w.rms_ffn_weight, device),
        to_device(w.w1, device),
        to_device(w.w2, device),
        to_device(w.w3, device),
        to_device(w.rms_final_weight, device),
        to_device(w.wcls, device)
    )
end

function Core.to_device(s::Llama2State{P}, device::Symbol) where P
    return Llama2State{P}(
        to_device(s.x, device), to_device(s.xb, device), to_device(s.xb2, device),
        to_device(s.hb, device), to_device(s.hb2, device), to_device(s.q, device),
        to_device(s.att, device), to_device(s.logits, device),
        to_device(s.key_cache, device), to_device(s.value_cache, device)
    )
end

function Core.to_device(t::Llama2Transformer{P}, device::Symbol) where P
    cfg = t.config
    weights_new = to_device(t.weights, device)
    state_new = to_device(t.state, device)
    return Llama2Transformer{P}(cfg, weights_new, state_new)
end

"""
    encode(tok::Llama2Tokenizer, text::String; bos=true, eos=false) → Vector{Int}

Encode text into 1-indexed token IDs using BPE merging.
Matches llama2.c's `encode` function.
"""
function encode(tok::Llama2Tokenizer, text::String; bos::Bool=true, eos::Bool=false)
    tokens = Int[]

    # Build sorted vocab lookup
    sorted_pairs = sort([(tok.vocab[i], i) for i in 1:tok.vocab_size]; by=first)
    function str_lookup(s::String)
        idx = searchsortedfirst(sorted_pairs, (s,); by=first)
        if idx <= length(sorted_pairs) && sorted_pairs[idx][1] == s
            return sorted_pairs[idx][2]
        end
        return -1
    end

    bos && push!(tokens, 1)  # BOS token

    # Add dummy prefix (sentencepiece convention)
    if !isempty(text)
        dummy = str_lookup(" ")
        dummy != -1 && push!(tokens, dummy)
    end

    # Encode each character/byte
    for ch in text
        s = string(ch)
        id = str_lookup(s)
        if id != -1
            push!(tokens, id)
        else
            # Byte fallback: encode each byte as token (offset by 3)
            for b in Vector{UInt8}(s)
                push!(tokens, Int(b) + 3)
            end
        end
    end

    # BPE merge loop
    while true
        best_score = -1f10
        best_id = -1
        best_idx = -1

        for i in 1:(length(tokens)-1)
            merged = tok.vocab[tokens[i]] * tok.vocab[tokens[i+1]]
            id = str_lookup(merged)
            if id != -1 && tok.vocab_scores[id] > best_score
                best_score = tok.vocab_scores[id]
                best_id = id
                best_idx = i
            end
        end

        best_idx == -1 && break

        tokens[best_idx] = best_id
        deleteat!(tokens, best_idx + 1)
    end

    eos && push!(tokens, 2)  # EOS token
    return tokens
end
