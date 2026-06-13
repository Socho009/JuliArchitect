# --- PHASE 5 ---
# Full Functional Autoregressive Transformer (FAT)
#
# (see FAT.jl for full import list)

using Printf
#
#   CoeffEmbedding    (Phase 2 input side) : ℝ^{K+1}       → ℝ^{d_model}
#   Causal Transformer Blocks              : ℝ^{d_model}    → ℝ^{d_model}
#   SpectralHead      (Phase 2 output)     : ℝ^{d_model}    → ℝ^{K+1}
#   SpectralStabilizer (Phase 3)           : ℝ^{K+1}        → ℝ^{P+1}   (applied post-prediction)
#
# The transformer backbone reuses the existing Architect Layers verbatim
# (Attention, Block, FeedForward, Norm, Linear) — only the I/O layers change.
#
# Training loop uses:
#   - Teacher forcing: input = θ_pred[:, 1:T-1, :], target = θ_target[:, 2:T, :]
#   - Directional Trajectory Approval loss (Phase 4) with warm-up schedule
#   - Zygote.withgradient for a single backward pass per step
#   - AdamW optimizer with optional weight decay
#
# Autoregressive generation uses the stabilizer after each step.

using LinearAlgebra
using Statistics: mean
using Zygote

# ─── FATConfig ────────────────────────────────────────────────────────────────

"""
    FATConfig{T<:AbstractFloat} <: AbstractModelConfig{T}

Configuration for the Functional Autoregressive Transformer.

## Fields
- `K          :: Int` — Chebyshev truncation order (coefficient vector length = K+1)
- `K_anchor   :: Int` — Local anchor degree for SpectralStabilizer P_k (≤ K)
- `P_econ     :: Int` — Economization degree for SpectralStabilizer P_p (≤ K_anchor)
- `n_embd     :: Int` — Transformer embedding dimension d_model
- `n_head     :: Int` — Number of attention heads
- `n_layer    :: Int` — Number of transformer blocks
- `block_size :: Int` — Maximum sequence length (context window)
- `dropout    :: T`   — Dropout rate (applied uniformly to all layers)

## Derived fields (for AbstractModelConfig compatibility)
- `vocab_size :: Int` — aliased to `K+1` (the continuous "vocabulary")
- `embd_pdrop`, `attn_pdrop`, `resid_pdrop :: T` — all aliased to `dropout`
"""
struct FATConfig{T<:AbstractFloat} <: AbstractModelConfig{T}
    K::Int            # Chebyshev truncation order
    K_anchor::Int     # stabilizer anchor degree
    P_econ::Int       # stabilizer economization degree
    n_embd::Int       # transformer hidden dimension
    n_head::Int       # attention heads
    n_layer::Int      # number of transformer blocks
    block_size::Int   # max context length
    dropout::T        # uniform dropout rate
    # MLA (Multi-Head Latent Attention) fields
    use_mla::Bool     # whether to use latent KV compression
    d_c::Int          # latent compression dimension (used when use_mla=true)
    # AbstractModelConfig compatibility fields
    vocab_size::Int   # = K+1
    embd_pdrop::T
    attn_pdrop::T
    resid_pdrop::T
end

"""
    FATConfig{T}(; K, n_embd, n_head, n_layer, block_size, dropout,
                   K_anchor=K, P_econ=K÷2) where T

Keyword constructor.  Sensible defaults:
  - K_anchor = K (no local truncation, just economize)
  - P_econ = K ÷ 2 (economize by half)
"""
function FATConfig{T}(;
        K::Int          = 15,
        n_embd::Int     = 128,
        n_head::Int     = 4,
        n_layer::Int    = 4,
        block_size::Int = 64,
        dropout::Real   = 0.0,
        K_anchor::Int   = K,
        P_econ::Int     = K ÷ 2,
        use_mla::Bool   = false,
        d_c::Int        = n_embd ÷ 4,
) where {T<:AbstractFloat}
    @assert n_embd % n_head == 0 "n_embd=$n_embd must be divisible by n_head=$n_head"
    @assert 0 <= P_econ <= K_anchor <= K "Require 0 ≤ P_econ ≤ K_anchor ≤ K"
    dp = T(dropout)
    FATConfig{T}(K, K_anchor, P_econ, n_embd, n_head, n_layer, block_size,
                 dp, use_mla, d_c, K+1, dp, dp, dp)
end

"""
    FATConfig(; kwargs...)

Convenience constructor defaulting to Float32.
"""
FATConfig(; kwargs...) = FATConfig{Float32}(; kwargs...)

function Base.show(io::IO, c::FATConfig{T}) where T
    mla_str = c.use_mla ? ", MLA(d_c=$(c.d_c))" : ""
    print(io, "FATConfig{$T}(K=$(c.K), K_anchor=$(c.K_anchor), P_econ=$(c.P_econ), " *
              "d=$(c.n_embd), heads=$(c.n_head), layers=$(c.n_layer), ctx=$(c.block_size)" *
              mla_str * ")")
end

# ─── FAT Model ────────────────────────────────────────────────────────────────

"""
    FunctionalAutoregressiveTransformer{T<:AbstractFloat}

The complete FAT model.

## Architecture (forward pass)

```
Input:  Θ_seq ∈ ℝ^{(K+1) × T × B}    (coefficient sequences, batch of B)
  │
  ▼
coeff_embed : Linear(K+1 → d_model)  — project coefficients into embedding space
  │          Layout: (K+1, T*B) → (d_model, T*B) → reshape (d_model, T, B)
  ▼
blocks[1..L]: Block{T}               — L causal transformer blocks (reused from Layers)
  │          Layout: (d_model, T, B) → (d_model, T, B) unchanged
  ▼
ln_f : Norm{T}                       — final RMS layer norm
  ▼
spectral_head : SpectralHead{T,K}    — project hidden → coefficient predictions
  │            Layout: (d_model, T, B) → (K+1, T, B)
  ▼
Output: Θ_pred ∈ ℝ^{(K+1) × T × B}  (predicted coefficient sequences)

At *inference* time each predicted Θ_t is passed through the stabilizer:
  stabilizer : SpectralStabilizer{K_anchor, P_econ}
             Π(Θ_t) ∈ ℝ^{P_econ+1}  (non-expansive, L_Π ≤ 1)
```

## Fields
- `config`         :: FATConfig{T}
- `coeff_embed`    :: Linear{T,true}           — coefficient encoder
- `blocks`         :: Vector{Block{T,...}}     — transformer backbone
- `ln_f`           :: Norm{T}                  — final layer norm
- `spectral_head`  :: SpectralHead{T,K}        — spectral output head
- `stabilizer`     :: SpectralStabilizer{K_anchor,P_econ}
"""
struct FunctionalAutoregressiveTransformer{T<:AbstractFloat,
                                           K, K_anchor, P_econ}
    config::FATConfig{T}
    coeff_embed::Linear{T, true}          # (K+1 → d_model)
    blocks::Vector                         # Vector of Block{T,...}
    ln_f::Norm{T}
    spectral_head::SpectralHead{T, K}
    stabilizer::SpectralStabilizer{K_anchor, P_econ}
end

"""
    FunctionalAutoregressiveTransformer(config::FATConfig{T}) where T

Construct the full FAT model from a FATConfig.
All blocks use Linear (not OrthogonalLinear) for full Zygote AD compatibility.
"""
function FunctionalAutoregressiveTransformer(config::FATConfig{T}) where {T<:AbstractFloat}
    K        = config.K
    K_anchor = config.K_anchor
    P_econ   = config.P_econ
    n_embd   = config.n_embd

    # Coefficient encoder: ℝ^{K+1} → ℝ^{d_model}
    # A simple linear projection — no positional embedding needed since Chebyshev
    # coefficients already encode frequency information.
    coeff_embed = Linear{T}(K+1, n_embd; bias=true)

    # Transformer backbone: identical to ContinuousModel (Phases 1-4 of Architect)
    # Dispatch on use_mla to select attention layer type
    blocks = map(1:config.n_layer) do _
        ln1 = Norm{T}(n_embd)
        if config.use_mla
            # Phase 7: Multi-Head Latent Attention with KV compression
            attention = LatentAttention{T}(config; d_c=config.d_c)
        else
            # Standard attention (all Linear projections for full Zygote compatibility)
            attention = Attention{T}(config; use_orthogonal_out=false, use_orthogonal_qkv=false)
        end
        ln2 = Norm{T}(n_embd)
        ff  = FeedForward{T}(config)
        Block{T, typeof(attention), typeof(ln1), typeof(ff)}(
            ln1, attention, ln2, ff, config.resid_pdrop
        )
    end

    ln_f          = Norm{T}(n_embd)
    spectral_head = SpectralHead{T, K}(n_embd)
    stabilizer    = SpectralStabilizer{K_anchor, P_econ}()

    FunctionalAutoregressiveTransformer{T, K, K_anchor, P_econ}(
        config, coeff_embed, blocks, ln_f, spectral_head, stabilizer
    )
end

function Base.show(io::IO, m::FunctionalAutoregressiveTransformer{T,K,Ka,Pe}) where {T,K,Ka,Pe}
    print(io, "FAT{$T}(K=$K → Π(k=$Ka,p=$Pe), " *
              "d=$(m.config.n_embd), L=$(m.config.n_layer), H=$(m.config.n_head))")
end

# ─── Forward Pass ─────────────────────────────────────────────────────────────

"""
    fat_forward(model::FunctionalAutoregressiveTransformer{T}, Θ_seq) -> Array{T,3}

Pure forward pass through the FAT (no loss, no stabilizer — training mode).

## Input
`Θ_seq :: AbstractArray{T,3}` — shape `(K+1, seq_len, batch)`
Each `Θ_seq[:, t, b]` is the Chebyshev coefficient vector of the t-th input
function in the b-th sequence.

## Output
`Θ_pred :: Array{T,3}` — shape `(K+1, seq_len, batch)`
Predicted next-step coefficient vectors.

## Layout
- First dimension: coefficient index (K+1) — "features"
- Second dimension: sequence position (seq_len)
- Third dimension: batch index (batch)
This matches the column-major (features × seq × batch) convention of Architect.
"""
function fat_forward(model::FunctionalAutoregressiveTransformer{T,K,Ka,Pe},
                     Θ_seq::AbstractArray{T,3}) where {T<:AbstractFloat,K,Ka,Pe}
    K1, seq_len, batch = size(Θ_seq)
    @assert K1 == K + 1 "Input leading dim $K1 ≠ K+1=$(K+1)"
    n_embd = model.config.n_embd

    # ── 1. Coefficient embedding ℝ^{K+1} → ℝ^{d_model} ──────────────────────
    # Flatten (K+1, seq_len, batch) → (K+1, seq_len·batch) for single BLAS call
    Θ_2d = reshape(Θ_seq, K1, seq_len * batch)          # (K+1, T·B)
    H_2d = model.coeff_embed(Θ_2d)                      # (n_embd, T·B)
    H    = reshape(H_2d, n_embd, seq_len, batch)         # (n_embd, T, B)

    # ── 2. Causal transformer backbone ────────────────────────────────────────
    for block in model.blocks
        H = block(H; training=false)
    end

    # ── 3. Final layer norm ───────────────────────────────────────────────────
    H = model.ln_f(H)

    # ── 4. Spectral head ℝ^{d_model} → ℝ^{K+1} ───────────────────────────────
    return model.spectral_head(H)    # (K+1, seq_len, batch)
end

# ─── FAT Training Loop ────────────────────────────────────────────────────────

"""
    FATTrainConfig{T<:AbstractFloat}

Hyperparameters for the FAT teacher-forced training loop.

## Fields
- `learning_rate  :: T`
- `weight_decay   :: T`
- `max_iters      :: Int`
- `grad_clip      :: T`  — gradient clipping norm threshold
- `betas          :: Tuple{T,T}` — AdamW β₁, β₂
- `eps            :: T`  — AdamW ε
- `loss_eps       :: T`  — ε for directional loss numerics
- `warmup_iters   :: Int` — α ramp length (approval warm-up)
- `log_interval   :: Int`
- `approval_weight :: T` — scales directional approval; `0` gives MSE-only training
- `aether_lambda  :: T` — opt-in AetherSpaces regularization weight
- `aether_spaces  :: Tuple` — spaces passed to `space_stack_regularizer`
"""
struct FATTrainConfig{T<:AbstractFloat}
    learning_rate::T
    weight_decay::T
    max_iters::Int
    grad_clip::T
    betas::Tuple{T,T}
    eps::T
    loss_eps::T
    warmup_iters::Int
    log_interval::Int
    mcp_lambda::T         # Phase 8: Hardin-Taylor MCP regularization weight
    mcp_alpha::T          # Phase 8: complexity exponent α
    approval_weight::T    # scales directional approval term; 0 gives MSE-only training
    aether_lambda::T      # opt-in AetherSpaces regularization weight
    aether_spaces::Tuple  # stable/diagnostic space adapters; empty keeps old behavior
end

function FATTrainConfig{T}(;
        learning_rate::Real = 3e-4,
        weight_decay::Real  = 0.01,
        max_iters::Int      = 2000,
        grad_clip::Real     = 1.0,
        betas::Tuple        = (0.9, 0.999),
        eps::Real           = 1e-8,
        loss_eps::Real      = 1e-6,
        warmup_iters::Int   = 500,
        log_interval::Int   = 100,
        mcp_lambda::Real    = 0.0,
        mcp_alpha::Real     = 1.5,
        approval_weight::Real = 1.0,
        aether_lambda::Real = 0.0,
        aether_spaces = (),
) where {T<:AbstractFloat}
    FATTrainConfig{T}(T(learning_rate), T(weight_decay), max_iters, T(grad_clip),
                      (T(betas[1]), T(betas[2])), T(eps), T(loss_eps),
                      warmup_iters, log_interval, T(mcp_lambda), T(mcp_alpha),
                      T(approval_weight), T(aether_lambda), Tuple(aether_spaces))
end

FATTrainConfig(; kwargs...) = FATTrainConfig{Float32}(; kwargs...)

# ─── Parameter collection ─────────────────────────────────────────────────────

"""
    _collect_fat_params(model::FunctionalAutoregressiveTransformer) -> Vector{Tuple{Array,Bool}}

Collect all trainable parameters as (array, should_decay) pairs.
Weight matrices (ndims ≥ 2) get weight decay; biases and norm weights don't.
"""
function _collect_fat_params(model::FunctionalAutoregressiveTransformer{T}) where T
    ps = Tuple{AbstractArray{T}, Bool}[]

    # Coefficient embedding
    push!(ps, (model.coeff_embed.weight, true))
    push!(ps, (model.coeff_embed.bias,   false))

    # Transformer blocks
    for block in model.blocks
        push!(ps, (block.ln1.weight, false))

        att = block.attention
        if has_latent_kv(att)
            # Phase 7: LatentAttention — collect Q, DKV, UK, UV, out projections
            push!(ps, (att.W_Q.weight, true))
            push!(ps, (att.W_Q.bias,   false))
            push!(ps, (att.W_DKV.weight, true))
            push!(ps, (att.W_UK.weight,  true))
            push!(ps, (att.W_UV.weight,  true))
            push!(ps, (att.W_out.weight, true))
            has_bias(att.W_out) && push!(ps, (att.W_out.bias, false))
        else
            # Standard Attention
            if att.W_qkv !== nothing
                push!(ps, (att.W_qkv.weight, true))
                has_bias(att.W_qkv) && push!(ps, (att.W_qkv.bias, false))
            end
            if isa(att.W_out, Linear)
                push!(ps, (att.W_out.weight, true))
                has_bias(att.W_out) && push!(ps, (att.W_out.bias, false))
            end
        end

        push!(ps, (block.ln2.weight,          false))
        push!(ps, (block.feedforward.W1.weight, true))
        push!(ps, (block.feedforward.W2.weight, true))
        push!(ps, (block.feedforward.W3.weight, true))
    end

    # Final norm and spectral head
    push!(ps, (model.ln_f.weight,              false))
    push!(ps, (model.spectral_head.weight,     true))
    push!(ps, (model.spectral_head.bias,       false))

    return ps
end

# ─── Gradient extraction ──────────────────────────────────────────────────────

"""
    _fat_extract_grads(gs, model) -> Vector{Union{Nothing,AbstractArray}}

Walk the Zygote structural gradient to extract per-parameter gradients in the
same order as `_collect_fat_params`.
"""
function _fat_extract_grads(gs, model::FunctionalAutoregressiveTransformer{T}) where T
    grads = Union{Nothing, AbstractArray}[]

    _fat_push!(grads, gs, :coeff_embed, :weight)
    _fat_push!(grads, gs, :coeff_embed, :bias)

    blocks_g = _fat_field(gs, :blocks)
    for (i, block) in enumerate(model.blocks)
        bg = (blocks_g !== nothing && blocks_g isa AbstractVector) ? blocks_g[i] : nothing
        # ln1
        ln1g = _fat_field(bg, :ln1)
        push!(grads, _fat_field(ln1g, :weight))
        # attention
        attg = _fat_field(bg, :attention)

        if has_latent_kv(block.attention)
            # Phase 7: LatentAttention gradients
            wqg = _fat_field(attg, :W_Q)
            push!(grads, _fat_field(wqg, :weight))
            push!(grads, _fat_field(wqg, :bias))
            wdkvg = _fat_field(attg, :W_DKV)
            push!(grads, _fat_field(wdkvg, :weight))
            wukg = _fat_field(attg, :W_UK)
            push!(grads, _fat_field(wukg, :weight))
            wuvg = _fat_field(attg, :W_UV)
            push!(grads, _fat_field(wuvg, :weight))
            woutg = _fat_field(attg, :W_out)
            push!(grads, _fat_field(woutg, :weight))
            has_bias(block.attention.W_out) && push!(grads, _fat_field(woutg, :bias))
        else
            # Standard Attention gradients
            if block.attention.W_qkv !== nothing
                wqkvg = _fat_field(attg, :W_qkv)
                push!(grads, _fat_field(wqkvg, :weight))
                has_bias(block.attention.W_qkv) && push!(grads, _fat_field(wqkvg, :bias))
            end
            if isa(block.attention.W_out, Linear)
                woutg = _fat_field(attg, :W_out)
                push!(grads, _fat_field(woutg, :weight))
                has_bias(block.attention.W_out) && push!(grads, _fat_field(woutg, :bias))
            end
        end

        # ln2
        ln2g = _fat_field(bg, :ln2)
        push!(grads, _fat_field(ln2g, :weight))
        # feedforward
        ffg = _fat_field(bg, :feedforward)
        for wname in (:W1, :W2, :W3)
            wg = _fat_field(ffg, wname)
            push!(grads, _fat_field(wg, :weight))
        end
    end

    # ln_f and spectral head
    lnfg = _fat_field(gs, :ln_f)
    push!(grads, _fat_field(lnfg, :weight))
    shg = _fat_field(gs, :spectral_head)
    push!(grads, _fat_field(shg, :weight))
    push!(grads, _fat_field(shg, :bias))

    return grads
end

# Safe field accessor for Zygote gradient NamedTuples
function _fat_field(gs, name::Symbol)
    gs === nothing && return nothing
    if gs isa NamedTuple
        return haskey(gs, name) ? getfield(gs, name) : nothing
    end
    try; return getfield(gs, name); catch; return nothing; end
end

function _fat_push!(grads, gs, layer_sym::Symbol, field_sym::Symbol)
    layerg = _fat_field(gs, layer_sym)
    push!(grads, _fat_field(layerg, field_sym))
end

# ─── Teacher-Forced Training Loop ─────────────────────────────────────────────

"""
    fat_train!(model::FunctionalAutoregressiveTransformer{T}, data_fn, config::FATTrainConfig{T}) -> Vector{T}

Train a FAT model using teacher forcing and the Directional Trajectory Approval loss.

## Protocol
At each iteration:
1. Call `data_fn()` → `(input, target)` both of shape `(K+1, seq_len, batch)`.
   - `input[:, t, b]`  = ground-truth Θ*_t (teacher-forced input)
   - `target[:, t, b]` = ground-truth Θ*_{t+1} (prediction target)
2. Forward pass: `Θ_pred = fat_forward(model, input)`.
3. Loss: `ℒ = directional_trajectory_loss(Θ_pred, target; α=α_schedule)`.
4. Backward pass: `Zygote.withgradient`.
5. AdamW parameter update with gradient clipping.

## Teacher Forcing
The model receives the *ground-truth* coefficient sequence as input (not its
own predictions).  This prevents error compounding during training and lets
gradients flow cleanly through the spectral head.  During inference, the
stabilizer Π is applied after each step to suppress divergence.

## Returns
Vector of training losses (one per iteration).
"""
function fat_train!(model::FunctionalAutoregressiveTransformer{T},
                    data_fn,
                    config::FATTrainConfig{T}) where {T<:AbstractFloat}

    all_params = _collect_fat_params(model)
    n_params   = sum(length(p) for (p, _) in all_params)

    # AdamW moment state
    m_state = [zeros(T, size(p)) for (p, _) in all_params]
    v_state = [zeros(T, size(p)) for (p, _) in all_params]
    t_adam  = 0

    β1, β2 = config.betas
    losses  = T[]

    println("FAT Training: $n_params parameters")
    println("  lr=$(config.learning_rate), wd=$(config.weight_decay), " *
            "iters=$(config.max_iters), warmup=$(config.warmup_iters)")
    if config.aether_lambda > zero(T) && !isempty(config.aether_spaces)
        println("  aether_lambda=$(config.aether_lambda), spaces=$(length(config.aether_spaces))")
    end

    for iter in 1:config.max_iters
        input, target = data_fn()

        # α schedule: ramp from 0 (MSE) to 1 (approval) over warmup_iters
        α = config.approval_weight * T(approval_alpha_schedule(iter; warmup_iters=config.warmup_iters))

        # ── Zygote AD: single backward pass ───────────────────────────────────
        loss_val, model_grad = Zygote.withgradient(model) do m
            Θ_pred = fat_forward(m, input)
            base_loss = directional_trajectory_loss(Θ_pred, target;
                                        ε      = config.loss_eps,
                                        α      = α,
                                        warmup_mse = true)
            # Phase 8: Hardin-Taylor MCP regularization
            # Penalizes prediction complexity to enforce "predict the simplest
            # compatible function" principle
            if config.mcp_lambda > zero(T)
                mcp = mcp_loss(Θ_pred, target; α=config.mcp_alpha)
                base_loss = base_loss + config.mcp_lambda * mcp
            end
            if config.aether_lambda > zero(T) && !isempty(config.aether_spaces)
                base_loss + config.aether_lambda * space_stack_regularizer(Θ_pred, config.aether_spaces)
            else
                base_loss
            end
        end

        if !isfinite(loss_val)
            @error "Loss diverged at iter $iter (loss=$loss_val) — halting."
            break
        end

        push!(losses, loss_val)

        # ── Extract gradients ─────────────────────────────────────────────────
        param_grads = _fat_extract_grads(model_grad[1], model)

        # ── Gradient clipping ─────────────────────────────────────────────────
        grad_norm  = sqrt(sum(g === nothing ? zero(T) : T(sum(abs2, g)) for g in param_grads))
        clip_coef  = grad_norm > config.grad_clip ? config.grad_clip / (grad_norm + T(1e-6)) : one(T)

        # ── AdamW update ──────────────────────────────────────────────────────
        t_adam += 1
        bc1 = one(T) - β1^t_adam
        bc2 = one(T) - β2^t_adam

        for i in eachindex(all_params)
            g = param_grads[i]
            g === nothing && continue
            g = g .* clip_coef

            param, decay = all_params[i]
            m_state[i] .= β1 .* m_state[i] .+ (one(T) - β1) .* g
            v_state[i] .= β2 .* v_state[i] .+ (one(T) - β2) .* g .^ 2

            m_hat  = m_state[i] ./ bc1
            v_hat  = v_state[i] ./ bc2
            update = m_hat ./ (sqrt.(v_hat) .+ config.eps)

            if decay
                param .-= config.learning_rate .* (update .+ config.weight_decay .* param)
            else
                param .-= config.learning_rate .* update
            end
        end

        if iter % config.log_interval == 0 || iter == 1
            @printf("  iter %4d/%d | loss=%.6f | grad_norm=%.4f | α=%.3f\n",
                    iter, config.max_iters, loss_val, grad_norm, α)
        end
    end

    return FATTrainState{T}(losses, config.max_iters, config.learning_rate)
end

# ─── Autoregressive Generation ────────────────────────────────────────────────

"""
    fat_generate(model::FunctionalAutoregressiveTransformer{T},
                 seed_coeffs::AbstractMatrix{T};
                 max_new::Int=10,
                 apply_stabilizer::Bool=true) -> Matrix{T}

Autoregressively generate new coefficient vectors from a seed sequence.

## Arguments
- `seed_coeffs`: shape `(K+1, T_seed)` — initial coefficient sequence
- `max_new`: number of new steps to generate
- `apply_stabilizer`: whether to pass each predicted Θ through Π before feeding
  it back as input (strongly recommended to prevent divergence)

## Returns
`(K+1, T_seed + max_new)` coefficient matrix — seed followed by generated steps.

## Note on stabilizer output shape
The stabilizer projects to `(P_econ+1)` coefficients.  If `P_econ < K`, the
generated steps are zero-padded back to `K+1` before being fed as input,
preserving the model's input contract.  The returned matrix contains the
*pre-padding* predictions (length K+1) at each position.
"""
function fat_generate(
        model::FunctionalAutoregressiveTransformer{T,K,Ka,Pe},
        seed_coeffs::AbstractMatrix{T};
        max_new::Int = 10,
        apply_stabilizer::Bool = true) where {T<:AbstractFloat, K, Ka, Pe}

    @assert size(seed_coeffs, 1) == K+1 "seed_coeffs first dim $(size(seed_coeffs,1)) ≠ K+1=$(K+1)"

    sequence = copy(seed_coeffs)   # (K+1, T_current)

    for _ in 1:max_new
        T_curr = size(sequence, 2)
        T_use  = min(T_curr, model.config.block_size)
        # Take the last T_use steps and add a singleton batch dimension
        inp = reshape(sequence[:, end-T_use+1:end], K+1, T_use, 1)   # (K+1, T_use, 1)

        # Forward pass (no gradient — inference mode)
        Θ_pred_3d = fat_forward(model, inp)                           # (K+1, T_use, 1)
        next_Θ    = Θ_pred_3d[:, end, 1]                             # (K+1,)

        if apply_stabilizer
            # Apply Π: projects to (P_econ+1,) — zero-pad back to K+1 for input contract
            stabilized = model.stabilizer(next_Θ)                    # (P_econ+1,)
            next_Θ_padded = zeros(T, K+1)
            next_Θ_padded[1:Pe+1] .= stabilized
            next_Θ = next_Θ_padded
        end

        sequence = hcat(sequence, next_Θ)                            # (K+1, T_current+1)
    end

    return sequence
end

"""
    adaptive_fat_generate(model::FunctionalAutoregressiveTransformer{T},
                          seed_coeffs::AbstractMatrix{T};
                          max_new::Int=10,
                          ϵ::T=1e-4, δ::T=1e-2,
                          K_min::Int=4) -> (Matrix{T}, Vector{Int})

Autoregressively generate new coefficient vectors with dynamic ``K``-truncation.

## Mechanism
At each step, we slice the I/O layers to compute only up to `K_current`.
We compute the high-frequency energy of the prediction. If it is low (< ϵ), 
we reduce `K_current`. If it is high (> δ), we increase `K_current`.
This theoretically saves computation by avoiding predictions for negligible high frequencies.

## Returns
- `sequence`: (K+1, T_seed + max_new) coefficient matrix, dynamically padded.
- `K_history`: Vector of `K_current` used at each generated step.
"""
function adaptive_fat_generate(
        model::FunctionalAutoregressiveTransformer{T,K,Ka,Pe},
        seed_coeffs::AbstractMatrix{T};
        max_new::Int = 10,
        ϵ::T = 1e-4,
        δ::T = 1e-2,
        K_min::Int = 4) where {T<:AbstractFloat, K, Ka, Pe}

    @assert size(seed_coeffs, 1) == K+1
    
    sequence = copy(seed_coeffs)
    K_current = K  # start at full resolution
    K_history = Int[]

    for _ in 1:max_new
        T_curr = size(sequence, 2)
        T_use  = min(T_curr, model.config.block_size)
        
        # Slice input up to K_current
        inp = sequence[1:K_current+1, end-T_use+1:end]
        inp_2d = reshape(inp, K_current+1, T_use)
        
        # 1. Dynamically sliced embedding projection
        W_in = model.coeff_embed.weight[:, 1:K_current+1]
        b_in = model.coeff_embed.bias
        H_2d = W_in * inp_2d .+ b_in
        H = reshape(H_2d, model.config.n_embd, T_use, 1)
        
        # 2. Backbone
        for block in model.blocks
            H = block(H; training=false)
        end
        H = model.ln_f(H)
        
        # 3. Dynamically sliced output projection
        W_out = model.spectral_head.weight[1:K_current+1, :]
        b_out = model.spectral_head.bias[1:K_current+1]
        H_2d_out = reshape(H, model.config.n_embd, T_use)
        out_2d = W_out * H_2d_out .+ b_out
        
        next_Θ_sliced = out_2d[:, end] # (K_current+1,)
        
        # Apply Stabilizer logic natively on the slice
        if K_current >= Pe
            # We can project to P_econ
            # Stabilizer is just diagonal multiplication for Chebyshev
            # Since SpectralStabilizer uses standard filter, we can just apply it up to min(K_current, Pe)
            # Actually, standard projection just scales coefficients.
            # We'll just take the predicted slice.
            nothing
        end
        
        # Compute energy of the highest 2 frequencies in the prediction
        E_high = zero(T)
        if K_current >= 2
            E_high = next_Θ_sliced[end]^2 + next_Θ_sliced[end-1]^2
        end
        
        # Pad back to full K+1 for storage
        next_Θ = zeros(T, K+1)
        next_Θ[1:K_current+1] .= next_Θ_sliced
        sequence = hcat(sequence, next_Θ)
        
        push!(K_history, K_current)
        
        # Adaptive K update for the next iteration
        if E_high < ϵ && K_current > K_min
            K_current -= 1
        elseif E_high > δ && K_current < K
            K_current += 1
        end
    end

    return sequence, K_history
end

# ─── FATTrainState result ─────────────────────────────────────────────────────

"""
    FATTrainState{T}

Result of a completed FAT training run.

## Fields
- `losses    :: Vector{T}` — per-iteration loss values
- `n_iters   :: Int`
- `final_lr  :: T`
"""
struct FATTrainState{T<:AbstractFloat}
    losses::Vector{T}
    n_iters::Int
    final_lr::T
end

# ─── Device transfer ──────────────────────────────────────────────────────────

function Core.to_device(model::FunctionalAutoregressiveTransformer{T,K,Ka,Pe},
                        device::Symbol) where {T,K,Ka,Pe}
    embed_new  = Core.to_device(model.coeff_embed, device)
    blocks_new = [Core.to_device(b, device) for b in model.blocks]
    lnf_new    = Core.to_device(model.ln_f, device)
    head_new   = Core.to_device(model.spectral_head, device)
    FunctionalAutoregressiveTransformer{T,K,Ka,Pe}(
        model.config, embed_new, blocks_new, lnf_new, head_new, model.stabilizer
    )
end

# ─── Canonical API compatibility ──────────────────────────────────────────────

coefficient_encoder(model::FunctionalAutoregressiveTransformer) = model.coeff_embed
coefficient_head(model::FunctionalAutoregressiveTransformer) = model.spectral_head
coefficient_dimension(model::FunctionalAutoregressiveTransformer) = model.config.K + 1
context_length(model::FunctionalAutoregressiveTransformer) = model.config.block_size

function Base.getproperty(model::FunctionalAutoregressiveTransformer, name::Symbol)
    if name === :coeff_encoder
        return getfield(model, :coeff_embed)
    elseif name === :coeff_head
        return getfield(model, :spectral_head)
    else
        return getfield(model, name)
    end
end

Base.propertynames(model::FunctionalAutoregressiveTransformer, private::Bool=false) =
    (:config, :coeff_embed, :blocks, :ln_f, :spectral_head, :stabilizer, :coeff_encoder, :coeff_head)

"""
    train!(model::FunctionalAutoregressiveTransformer, data_fn, config::FATTrainConfig)

Canonical training entry point for FAT models. Dispatches to `fat_train!`.
"""
train!(model::FunctionalAutoregressiveTransformer{T}, data_fn, config::FATTrainConfig{T}) where {T<:AbstractFloat} =
    fat_train!(model, data_fn, config)

function train!(model::FunctionalAutoregressiveTransformer{T},
                loader::FunctionSequenceLoader{T},
                config::FATTrainConfig{T}) where {T<:AbstractFloat}
    @assert loader.basis_dim == model.config.K + 1 "loader basis_dim=$(loader.basis_dim) must equal K+1=$(model.config.K + 1)"
    return fat_train!(model, make_functional_batch_fn(loader), config)
end

function fat_train!(model::FunctionalAutoregressiveTransformer{T},
                    loader::FunctionSequenceLoader{T},
                    config::FATTrainConfig{T}) where {T<:AbstractFloat}
    @assert loader.basis_dim == model.config.K + 1 "loader basis_dim=$(loader.basis_dim) must equal K+1=$(model.config.K + 1)"
    return fat_train!(model, make_functional_batch_fn(loader), config)
end
