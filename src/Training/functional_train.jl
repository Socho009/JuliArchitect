# ─── AD-Based Training for ContinuousModel ────────────────────────────────────
#
# Uses Zygote automatic differentiation for O(1) backward pass.
# Replaces O(P) finite-difference forward passes per training step.

using Zygote

"""
    _is_decay_param(param) → Bool

Weight matrices (2D+) get weight decay; biases and LN params (1D) don't.
"""
_is_decay_param(p::AbstractArray) = ndims(p) >= 2

# ─── AD Training Config ──────────────────────────────────────────────────────

"""
    ContinuousTrainConfig{P}

Training configuration for ContinuousModel with AD gradients.
"""
struct ContinuousTrainConfig{P<:AbstractFloat}
    learning_rate::P
    weight_decay::P
    max_iters::Int
    grad_clip::P
    betas::Tuple{P, P}
    eps::P
    log_interval::Int
end

function ContinuousTrainConfig{P}(;
    learning_rate::Real = 1e-3,
    weight_decay::Real = 0.01,
    max_iters::Int = 1000,
    grad_clip::Real = 1.0,
    betas::Tuple = (0.9, 0.999),
    eps::Real = 1e-8,
    log_interval::Int = 50,
) where P
    ContinuousTrainConfig{P}(P(learning_rate), P(weight_decay), max_iters,
                              P(grad_clip), (P(betas[1]), P(betas[2])), P(eps), log_interval)
end

ContinuousTrainConfig(; kwargs...) = ContinuousTrainConfig{Float32}(; kwargs...)

# ─── Collect params with decay info ──────────────────────────────────────────

"""
    _collect_functional_params(model::ContinuousModel{P}) → Vector{Tuple{Array, Bool}}

Collect (param_array, should_decay) pairs from ContinuousModel.
"""
function _collect_functional_params(model::ContinuousModel{P}) where P
    params = Tuple{AbstractArray{P}, Bool}[]

    push!(params, (model.coeff_embed.weight, true))
    push!(params, (model.coeff_embed.bias, false))


    for block in model.blocks
        push!(params, (block.ln1.weight, false))
        if block.attention.W_qkv !== nothing
            push!(params, (block.attention.W_qkv.weight, true))
            if has_bias(block.attention.W_qkv)
                push!(params, (block.attention.W_qkv.bias, false))
            end
        end
        if isa(block.attention.W_out, Linear)
            push!(params, (block.attention.W_out.weight, true))
            if has_bias(block.attention.W_out)
                push!(params, (block.attention.W_out.bias, false))
            end
        end
        push!(params, (block.ln2.weight, false))
        push!(params, (block.feedforward.W1.weight, true))
        push!(params, (block.feedforward.W2.weight, true))
        push!(params, (block.feedforward.W3.weight, true))
    end

    push!(params, (model.ln_f.weight, false))
    push!(params, (model.spectral_head.weight, true))
    push!(params, (model.spectral_head.bias, false))

    return params
end

# ─── AD Training Loop ─────────────────────────────────────────────────────────

"""
    train!(model::ContinuousModel{P}, data_fn, config::ContinuousTrainConfig{P}) where P

Train a ContinuousModel using Zygote AD + AdamW.

Uses `Zygote.withgradient` on the model struct for a single backward pass.
This is O(1) backward passes per step instead of O(num_params) forward passes.

`data_fn` should return `(input, target)` both `Array{P, 3}` of shape
`(basis_dim, seq_length, batch_size)`.
"""
function train!(model::ContinuousModel{P}, data_fn, config::ContinuousTrainConfig{P}) where P
    all_params = _collect_functional_params(model)
    n_params = sum(length(p) for (p, _) in all_params)

    # AdamW state
    m_state = [zeros(P, size(p)) for (p, _) in all_params]
    v_state = [zeros(P, size(p)) for (p, _) in all_params]
    t = 0

    β1, β2 = config.betas
    losses = P[]

    println("Training ContinuousModel ($n_params params) with Zygote AD")
    println("  lr=$(config.learning_rate), wd=$(config.weight_decay), iters=$(config.max_iters)")

    for iter in 1:config.max_iters
        input, target = data_fn()

        # ── Zygote AD: single backward pass ──
        loss_val, model_grad = Zygote.withgradient(model) do m
            _, loss = forward(m, input; targets=target, training=false)
            loss
        end

        if !isfinite(loss_val)
            error("Training halted at iter $iter: loss=$loss_val")
        end

        push!(losses, loss_val)

        # Extract per-parameter gradients from the structural gradient
        param_grads = _extract_grads(model_grad[1], model)

        # AdamW step
        t += 1
        bc1 = one(P) - β1^t
        bc2 = one(P) - β2^t

        # Gradient clipping
        grad_norm = sqrt(sum(g === nothing ? zero(P) : P(sum(abs2, g)) for g in param_grads))
        clip_coef = grad_norm > config.grad_clip ? config.grad_clip / (grad_norm + P(1e-6)) : one(P)

        for i in eachindex(all_params)
            g = param_grads[i]
            g === nothing && continue
            g = g .* clip_coef
            param, decay = all_params[i]

            m_state[i] .= β1 .* m_state[i] .+ (one(P) - β1) .* g
            v_state[i] .= β2 .* v_state[i] .+ (one(P) - β2) .* g .^ 2

            m_hat = m_state[i] ./ bc1
            v_hat = v_state[i] ./ bc2
            update = m_hat ./ (sqrt.(v_hat) .+ config.eps)

            if decay
                param .-= config.learning_rate .* (update .+ config.weight_decay .* param)
            else
                param .-= config.learning_rate .* update
            end
        end

        if iter % config.log_interval == 0 || iter == 1
            println("  iter $iter/$(config.max_iters) | loss=$(round(loss_val; digits=6)) | grad_norm=$(round(grad_norm; digits=4))")
        end
    end

    return TrainState{P}(losses, config.max_iters)
end

"""
    train!(model::ContinuousModel, loader::FunctionSequenceLoader, config::ContinuousTrainConfig)

Canonical dataloader-based training entry point for `ContinuousModel`.
"""
function train!(model::ContinuousModel{P},
                loader::FunctionSequenceLoader{P},
                config::ContinuousTrainConfig{P}) where {P<:AbstractFloat}
    return train!(model, make_functional_batch_fn(loader), config)
end

# ─── Gradient Extraction from Zygote Structural Gradients ─────────────────────

"""
    _extract_grads(grad_struct, model::ContinuousModel) → Vector{Union{Nothing, Array}}

Walk the Zygote gradient structure to extract per-parameter gradients
in the same order as `_collect_functional_params`.
"""
function _extract_grads(gs, model::ContinuousModel{P}) where P
    grads = Union{Nothing, AbstractArray}[]

    _push_grad!(grads, gs, :coeff_embed, :weight)
    _push_grad!(grads, gs, :coeff_embed, :bias)


    blocks_g = _field(gs, :blocks)
    for (i, block) in enumerate(model.blocks)
        bg = blocks_g !== nothing && blocks_g isa AbstractVector ? blocks_g[i] : nothing
        _push_block_grad!(grads, bg, block)
    end

    _push_grad!(grads, gs, :ln_f, :weight)
    _push_grad!(grads, gs, :spectral_head, :weight)
    _push_grad!(grads, gs, :spectral_head, :bias)

    return grads
end

function _push_grad!(grads, gs, layer_name::Symbol, field_name::Symbol)
    layer_g = _field(gs, layer_name)
    val = _field(layer_g, field_name)
    push!(grads, val)
end

function _push_block_grad!(grads, bg, block)
    # ln1
    ln1_g = _field(bg, :ln1)
    push!(grads, _field(ln1_g, :weight))

    # attention
    att_g = _field(bg, :attention)
    if block.attention.W_qkv !== nothing
        wqkv_g = _field(att_g, :W_qkv)
        push!(grads, _field(wqkv_g, :weight))
        if has_bias(block.attention.W_qkv)
            push!(grads, _field(wqkv_g, :bias))
        end
    end
    if isa(block.attention.W_out, Linear)
        wout_g = _field(att_g, :W_out)
        push!(grads, _field(wout_g, :weight))
        if has_bias(block.attention.W_out)
            push!(grads, _field(wout_g, :bias))
        end
    end

    # ln2
    ln2_g = _field(bg, :ln2)
    push!(grads, _field(ln2_g, :weight))

    # feedforward
    ff_g = _field(bg, :feedforward)
    w1_g = _field(ff_g, :W1)
    push!(grads, _field(w1_g, :weight))
    w2_g = _field(ff_g, :W2)
    push!(grads, _field(w2_g, :weight))
    w3_g = _field(ff_g, :W3)
    push!(grads, _field(w3_g, :weight))
end

"""
Safely access a field from a Zygote gradient NamedTuple/struct.
Returns nothing if the gradient is nothing or the field doesn't exist.
"""
function _field(gs, name::Symbol)
    gs === nothing && return nothing
    if gs isa NamedTuple
        haskey(gs, name) ? getfield(gs, name) : nothing
    else
        try
            getfield(gs, name)
        catch
            nothing
        end
    end
end
