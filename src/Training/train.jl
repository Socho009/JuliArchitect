# ─── Training Loop ─────────────────────────────────────────────────────────────
#
# Implements AdamW optimizer with weight decay separation via multiple dispatch,
# gradient norm clipping, and NaN/Inf loss halting.
# Uses Julia's mutating convention (!) for in-place model updates.
# Gradients computed via numerical finite differences.

# ─── Training Configuration ───────────────────────────────────────────────────

"""
    TrainConfig{P<:AbstractFloat}

Training configuration struct parameterized on floating-point precision type `P`.
All floating-point hyperparameters are stored as type `P` for compile-time precision control.

# Fields
- `learning_rate::P` — Learning rate for AdamW optimizer
- `weight_decay::P` — Weight decay coefficient (applied only to decay group)
- `max_iters::Int` — Maximum number of training iterations
- `grad_clip::P` — Maximum gradient norm for clipping
- `batch_size::Int` — Batch size for training
- `betas::Tuple{P, P}` — Adam β1 and β2 momentum coefficients
- `eps::P` — Adam epsilon for numerical stability
"""
struct TrainConfig{P<:AbstractFloat}
    learning_rate::P
    weight_decay::P
    max_iters::Int
    grad_clip::P
    batch_size::Int
    betas::Tuple{P, P}
    eps::P
    spectral_lambda::P
    spectral_power_iters::Int
end

"""
    TrainConfig{P}(; kwargs...) where {P<:AbstractFloat}

Construct a TrainConfig with keyword arguments and sensible defaults.

# Keyword Arguments
- `learning_rate::Real = 3e-4` — Learning rate
- `weight_decay::Real = 0.1` — Weight decay coefficient
- `max_iters::Int = 1000` — Maximum training iterations
- `grad_clip::Real = 1.0` — Gradient norm clip value
- `batch_size::Int = 64` — Batch size
- `betas::Tuple = (0.9, 0.95)` — Adam β1, β2
- `eps::Real = 1e-8` — Adam epsilon

# Examples
```julia
config = TrainConfig{Float32}(learning_rate=3f-4, max_iters=5000)
```
"""
function TrainConfig{P}(;
    learning_rate::Real = 3e-4,
    weight_decay::Real = 0.1,
    max_iters::Int = 1000,
    grad_clip::Real = 1.0,
    batch_size::Int = 64,
    betas::Tuple{<:Real, <:Real} = (0.9, 0.95),
    eps::Real = 1e-8,
    spectral_lambda::Real = 0.0,
    spectral_power_iters::Int = 8,
) where {P<:AbstractFloat}
    TrainConfig{P}(
        P(learning_rate), P(weight_decay), max_iters, P(grad_clip),
        batch_size, (P(betas[1]), P(betas[2])), P(eps), P(spectral_lambda), spectral_power_iters
    )
end

"""
    TrainConfig(; kwargs...)

Convenience constructor defaulting to Float32 precision.
"""
TrainConfig(; kwargs...) = TrainConfig{Float32}(; kwargs...)

function Base.show(io::IO, c::TrainConfig{P}) where P
    print(io, "TrainConfig{$P}(lr=$(c.learning_rate), wd=$(c.weight_decay), ")
    print(io, "max_iters=$(c.max_iters), grad_clip=$(c.grad_clip), ")
    print(io, "batch_size=$(c.batch_size), betas=$(c.betas), eps=$(c.eps))")
end


# ─── Weight Decay Separation via Multiple Dispatch ────────────────────────────

"""
    abstract type ParamGroup end

Abstract type for parameter grouping in weight decay separation.
Uses Julia's type dispatch to determine which parameters receive weight decay.
"""
abstract type ParamGroup end

"""
    DecayGroup <: ParamGroup

Parameter group that receives weight decay regularization.
Applied to weight matrices of Linear layers (2D+ parameters).
"""
struct DecayGroup <: ParamGroup end

"""
    NoDecayGroup <: ParamGroup

Parameter group that does NOT receive weight decay regularization.
Applied to biases, LayerNorm parameters, and embedding weights.
"""
struct NoDecayGroup <: ParamGroup end

"""
    param_group(::Embedding)

Embedding weights do not receive weight decay.
"""
param_group(::Embedding) = NoDecayGroup()

"""
    param_group(::Norm)

LayerNorm parameters (weight and bias) do not receive weight decay.
"""
param_group(::Norm) = NoDecayGroup()

"""
    param_group(::Linear)

Linear layer weight matrices receive weight decay.
Note: biases are handled separately and never receive decay.
"""
param_group(::Linear) = DecayGroup()

# Treat OrthogonalLinear parameterization as a decay group (its A matrix)
if isdefined(Main, :OrthogonalLinear)
    param_group(::OrthogonalLinear) = DecayGroup()
end

# ─── Parameter Collection ─────────────────────────────────────────────────────

"""
    ParamInfo

Holds a reference to a parameter array along with metadata about whether
it should receive weight decay.
"""
struct ParamInfo{P<:AbstractFloat}
    param::AbstractArray{P}
    decay::Bool
end

"""
    collect_params(model) where P

Collect all trainable parameters from the model, separated into decay and no-decay groups.
Returns a vector of ParamInfo structs.

Weight decay rules:
- Decay: weight matrices of Linear layers (2D parameters)
- No decay: biases, LayerNorm weights/biases, Embedding weights
"""
function collect_params(model::ContinuousModel{P}) where P
    param_infos = ParamInfo{P}[]

    push!(param_infos, ParamInfo{P}(model.coeff_embed.weight, true))
    if has_bias(model.coeff_embed)
        push!(param_infos, ParamInfo{P}(model.coeff_embed.bias, false))
    end

    # Transformer blocks
    for block in model.blocks
        # Norm 1 — no decay
        push!(param_infos, ParamInfo{P}(block.ln1.weight, false))

        # Attention projections: support combined Linear or separate OrthogonalLinear
        if block.attention.W_qkv !== nothing
            push!(param_infos, ParamInfo{P}(block.attention.W_qkv.weight, true))
            if has_bias(block.attention.W_qkv)
                push!(param_infos, ParamInfo{P}(block.attention.W_qkv.bias, false))
            end
        else
            if block.attention.W_q !== nothing
                push!(param_infos, ParamInfo{P}(block.attention.W_q.A, true))
                push!(param_infos, ParamInfo{P}(block.attention.W_q.bias, false))
            end
            if block.attention.W_k !== nothing
                push!(param_infos, ParamInfo{P}(block.attention.W_k.A, true))
                push!(param_infos, ParamInfo{P}(block.attention.W_k.bias, false))
            end
            if block.attention.W_v !== nothing
                push!(param_infos, ParamInfo{P}(block.attention.W_v.A, true))
                push!(param_infos, ParamInfo{P}(block.attention.W_v.bias, false))
            end
        end

        # Attention W_out — can be Linear or OrthogonalLinear
        if isa(block.attention.W_out, Linear)
            push!(param_infos, ParamInfo{P}(block.attention.W_out.weight, true))
            if has_bias(block.attention.W_out)
                push!(param_infos, ParamInfo{P}(block.attention.W_out.bias, false))
            end
        elseif isa(block.attention.W_out, OrthogonalLinear)
            push!(param_infos, ParamInfo{P}(block.attention.W_out.A, true))
            push!(param_infos, ParamInfo{P}(block.attention.W_out.bias, false))
        end

        # Norm 2 — no decay
        push!(param_infos, ParamInfo{P}(block.ln2.weight, false))

        # FeedForward W1, W2, W3 — weight decays, no bias
        push!(param_infos, ParamInfo{P}(block.feedforward.W1.weight, true))
        push!(param_infos, ParamInfo{P}(block.feedforward.W2.weight, true))
        push!(param_infos, ParamInfo{P}(block.feedforward.W3.weight, true))
    end

    # Final Norm — no decay
    push!(param_infos, ParamInfo{P}(model.ln_f.weight, false))

    # Coefficient head
    push!(param_infos, ParamInfo{P}(model.spectral_head.weight, true))
    if has_bias(model.spectral_head)
        push!(param_infos, ParamInfo{P}(model.spectral_head.bias, false))
    end

    return param_infos
end

# ─── AdamW Optimizer State ────────────────────────────────────────────────────

"""
    AdamWState{P}

Mutable state for the AdamW optimizer, storing first and second moment estimates
for each parameter.

# Fields
- `m::Vector{Array{P}}` — First moment estimates (mean of gradients)
- `v::Vector{Array{P}}` — Second moment estimates (mean of squared gradients)
- `t::Int` — Current timestep (for bias correction)
"""
mutable struct AdamWState{P<:AbstractFloat}
    m::Vector{Array{P}}   # First moments (same shape as params)
    v::Vector{Array{P}}   # Second moments (same shape as params)
    t::Int                # Timestep
end

"""
    AdamWState(param_infos::Vector{ParamInfo{P}}) where P

Initialize AdamW state with zero moments for all parameters.
"""
function AdamWState(param_infos::Vector{ParamInfo{P}}) where P
    m = [zeros(P, size(pi.param)) for pi in param_infos]
    v = [zeros(P, size(pi.param)) for pi in param_infos]
    AdamWState{P}(m, v, 0)
end

# ─── Gradient Norm Clipping ───────────────────────────────────────────────────

"""
    clip_grad_norm!(grads::Vector, max_norm::P) where P

Clip the global gradient norm to `max_norm`. Modifies gradients in-place.
Returns the original (unclipped) global norm.

Global norm = sqrt(sum of squared norms of all gradient tensors)
If global_norm > max_norm, scale all gradients by (max_norm / global_norm).
"""
function clip_grad_norm!(grads::Vector, max_norm::P) where {P<:AbstractFloat}
    # Compute global gradient norm
    total_norm_sq = zero(P)
    for g in grads
        if g !== nothing
            total_norm_sq += sum(abs2, g)
        end
    end
    global_norm = sqrt(total_norm_sq)

    # Clip if necessary
    if global_norm > max_norm
        clip_coef = max_norm / (global_norm + P(1e-6))
        for i in eachindex(grads)
            if grads[i] !== nothing
                grads[i] = grads[i] .* clip_coef
            end
        end
    end

    return global_norm
end

# ─── AdamW Update Step ────────────────────────────────────────────────────────

"""
    adamw_step!(param_infos, grads, state, config)

Perform a single AdamW optimization step.

AdamW decouples weight decay from the gradient update:
- For decay parameters: θ = θ - lr * (m̂/(√v̂ + ε) + weight_decay * θ)
- For no-decay parameters: θ = θ - lr * m̂/(√v̂ + ε)

Where m̂ and v̂ are bias-corrected first and second moment estimates.
"""
function adamw_step!(
    param_infos::Vector{ParamInfo{P}},
    grads::Vector,
    state::AdamWState{P},
    config::TrainConfig{P}
) where P
    state.t += 1
    β1, β2 = config.betas
    lr = config.learning_rate
    ε = config.eps
    wd = config.weight_decay

    # Bias correction factors
    bc1 = one(P) - β1^state.t
    bc2 = one(P) - β2^state.t

    for i in eachindex(param_infos)
        g = grads[i]
        if g === nothing
            continue
        end

        pi = param_infos[i]

        # Update biased first moment estimate: m = β1*m + (1-β1)*g
        state.m[i] .= β1 .* state.m[i] .+ (one(P) - β1) .* g

        # Update biased second moment estimate: v = β2*v + (1-β2)*g²
        state.v[i] .= β2 .* state.v[i] .+ (one(P) - β2) .* (g .^ 2)

        # Bias-corrected estimates
        m_hat = state.m[i] ./ bc1
        v_hat = state.v[i] ./ bc2

        # Adam update: param -= lr * m_hat / (sqrt(v_hat) + eps)
        update = m_hat ./ (sqrt.(v_hat) .+ ε)

        # AdamW: decoupled weight decay (only for decay group)
        if pi.decay
            pi.param .-= lr .* (update .+ wd .* pi.param)
        else
            pi.param .-= lr .* update
        end
    end
end

# ─── Training Loop ────────────────────────────────────────────────────────────

"""
    TrainState{P}

Return value from `train!` containing training history and final state.

# Fields
- `losses::Vector{P}` — Loss value at each iteration
- `iter::Int` — Total iterations completed
"""
struct TrainState{P<:AbstractFloat}
    losses::Vector{P}
    iter::Int
end

"""
    train!(model, dataset, config::TrainConfig{P}) where P

Train the GPT model in-place using AdamW optimization.

The `dataset` should be a callable (function or functor) that returns
`(input_batch, target_batch)` tuples when called, where:
- `input_batch` is a matrix of token indices (seq_len, batch_size)
- `target_batch` is a matrix of target token indices (seq_len, batch_size)

# Arguments
- `model` — SequenceModel to train (mutated in-place)
- `dataset` — Callable returning (x, y) batches
- `config` — TrainConfig with optimizer hyperparameters

# Returns
- `TrainState{P}` with loss history and iteration count

# Throws
- `ErrorException` if NaN or Inf loss is encountered

# Examples
```julia
config = TrainConfig{Float32}(learning_rate=3f-4, max_iters=100)
state = train!(model, dataset, config)
println("Final loss: \$(state.losses[end])")
```
"""
function train!(model, dataset, config::TrainConfig{P}) where P
    # Collect parameters with decay metadata
    param_infos = collect_params(model)

    # Initialize optimizer state
    opt_state = AdamWState(param_infos)

    # Loss history
    losses = P[]

    for iter in 1:config.max_iters
        # Get batch from dataset
        x, y = dataset()

        # Compute loss and gradients via numerical differentiation (includes spectral regularizer)
        loss_val, grads = _compute_loss_and_grads(model, param_infos, x, y; spectral_lambda=config.spectral_lambda, power_iters=config.spectral_power_iters)

        # Check for NaN/Inf loss
        if !isfinite(loss_val)
            error("Training halted at iteration $iter: loss is $(loss_val) (NaN or Inf detected)")
        end

        push!(losses, loss_val)

        # Clip gradient norms
        clip_grad_norm!(grads, config.grad_clip)

        # Apply AdamW update
        adamw_step!(param_infos, grads, opt_state, config)
    end

    return TrainState{P}(losses, config.max_iters)
end


# ─── Spectral Diagnostics & Regularizer ──────────────────────────────────────

"""
    power_iteration(W; num_iters=10)

Estimate the spectral norm (largest singular value) of matrix `W` using
power iteration. Returns the estimated top singular value.
"""
function power_iteration(W::AbstractMatrix{P}; num_iters::Int=10) where P<:AbstractFloat
    m, n = size(W)
    # start with random vector of size n (columns)
    v = rand(P, n)
    nv = sqrt(sum(abs2, v)) + eps(eltype(v))
    v ./= nv

    for i in 1:num_iters
        wv = W * v                 # (m,)
        if all(iszero, wv)
            return zero(P)
        end
        v = W' * wv               # (n,)
        nv = sqrt(sum(abs2, v))
        if nv < eps(eltype(v))
            break
        end
        v ./= nv
    end
    sigma = sqrt(sum(abs2, W * v))
    return sigma
end


"""
    compute_spectral_stats(model; power_iters=8)

Compute spectral diagnostics across the model. Returns a Dict where keys
are descriptive names and values are scalars (spectral norm or orthogonality error).
"""
function compute_spectral_stats(model; power_iters::Int=8) where P<:AbstractFloat
    stats = Dict{String,Any}()
    for (i, block) in enumerate(model.blocks)
        att = block.attention
        n_embd = model.config.n_embd
        if att.W_qkv !== nothing
            W = att.W_qkv.weight
            Wq = W[1:n_embd, :]
            Wk = W[n_embd+1:2*n_embd, :]
            Wv = W[2*n_embd+1:3*n_embd, :]
            stats["block_$i.Wq_sigma"] = power_iteration(Wq; num_iters=power_iters)
            stats["block_$i.Wk_sigma"] = power_iteration(Wk; num_iters=power_iters)
            stats["block_$i.Wv_sigma"] = power_iteration(Wv; num_iters=power_iters)
        else
            if att.W_q !== nothing
                Wq = orthogonal_weight(att.W_q)
                Wk = orthogonal_weight(att.W_k)
                Wv = orthogonal_weight(att.W_v)
                stats["block_$i.Wq_ort_err"] = norm(Wq' * Wq - Matrix{P}(I, size(Wq,1), size(Wq,1)))
                stats["block_$i.Wk_ort_err"] = norm(Wk' * Wk - Matrix{P}(I, size(Wk,1), size(Wk,1)))
                stats["block_$i.Wv_ort_err"] = norm(Wv' * Wv - Matrix{P}(I, size(Wv,1), size(Wv,1)))
            end
        end

        # W_out
        if isa(att.W_out, Linear)
            stats["block_$i.Wout_sigma"] = power_iteration(att.W_out.weight; num_iters=power_iters)
        elseif isa(att.W_out, OrthogonalLinear)
            Wout = orthogonal_weight(att.W_out)
            stats["block_$i.Wout_ort_err"] = norm(Wout' * Wout - Matrix{P}(I, size(Wout,1), size(Wout,1)))
        end
    end

    # Head
    head = hasproperty(model, :spectral_head) ? model.spectral_head : (hasproperty(model, :decoder) ? model.decoder : (hasproperty(model, :lm_head) ? model.lm_head : nothing))
    if head !== nothing
        if isa(head, Linear)
            stats["head_sigma"] = power_iteration(head.weight; num_iters=power_iters)
        elseif isa(head, OrthogonalLinear)
            Wlm = orthogonal_weight(head)
            stats["head_ort_err"] = norm(Wlm' * Wlm - Matrix{P}(I, size(Wlm,1), size(Wlm,1)))
        end
    end

    return stats
end


"""
    spectral_regularizer(model; λ=1e-3, power_iters=8)

Compute a lightweight spectral regularization penalty for the model.
Returns `(penalty, details)` where `penalty` is a scalar and `details` is a Dict
with per-layer diagnostics used to compute the penalty.
"""
function spectral_regularizer(model; λ::Real=1e-3, power_iters::Int=8) where P<:AbstractFloat
    pen = zero(P)
    details = Dict{String,Any}()
    for (i, block) in enumerate(model.blocks)
        att = block.attention
        n_embd = model.config.n_embd
        if att.W_qkv !== nothing
            W = att.W_qkv.weight
            Wq = W[1:n_embd, :]
            Wk = W[n_embd+1:2*n_embd, :]
            Wv = W[2*n_embd+1:3*n_embd, :]
            s_q = power_iteration(Wq; num_iters=power_iters)
            s_k = power_iteration(Wk; num_iters=power_iters)
            s_v = power_iteration(Wv; num_iters=power_iters)
            pen += P(λ) * ((s_q - one(P))^2 + (s_k - one(P))^2 + (s_v - one(P))^2)
            details["block_$i.Wq_sigma"] = s_q
            details["block_$i.Wk_sigma"] = s_k
            details["block_$i.Wv_sigma"] = s_v
        else
            if att.W_q !== nothing
                Wq = orthogonal_weight(att.W_q)
                Wk = orthogonal_weight(att.W_k)
                Wv = orthogonal_weight(att.W_v)
                ort_q = norm(Wq' * Wq - Matrix{P}(I, size(Wq,1), size(Wq,1)))
                ort_k = norm(Wk' * Wk - Matrix{P}(I, size(Wk,1), size(Wk,1)))
                ort_v = norm(Wv' * Wv - Matrix{P}(I, size(Wv,1), size(Wv,1)))
                pen += P(λ) * (ort_q^2 + ort_k^2 + ort_v^2)
                details["block_$i.Wq_ort_err"] = ort_q
                details["block_$i.Wk_ort_err"] = ort_k
                details["block_$i.Wv_ort_err"] = ort_v
            end
        end

        # W_out
        if isa(att.W_out, Linear)
            s_out = power_iteration(att.W_out.weight; num_iters=power_iters)
            pen += P(λ) * ((s_out - one(P))^2)
            details["block_$i.Wout_sigma"] = s_out
        elseif isa(att.W_out, OrthogonalLinear)
            Wout = orthogonal_weight(att.W_out)
            ort_out = norm(Wout' * Wout - Matrix{P}(I, size(Wout,1), size(Wout,1)))
            pen += P(λ) * (ort_out^2)
            details["block_$i.Wout_ort_err"] = ort_out
        end
    end

    # Head
    head = hasproperty(model, :spectral_head) ? model.spectral_head : (hasproperty(model, :decoder) ? model.decoder : (hasproperty(model, :lm_head) ? model.lm_head : nothing))
    if head !== nothing
        if isa(head, Linear)
            s_lm = power_iteration(head.weight; num_iters=power_iters)
            pen += P(λ) * ((s_lm - one(P))^2)
            details["head_sigma"] = s_lm
        elseif isa(head, OrthogonalLinear)
            Wlm = orthogonal_weight(head)
            ort_lm = norm(Wlm' * Wlm - Matrix{P}(I, size(Wlm,1), size(Wlm,1)))
            pen += P(λ) * (ort_lm^2)
            details["head_ort_err"] = ort_lm
        end
    end

    return pen, details
end


"""
    print_spectral_stats(stats)

Pretty-print spectral diagnostics dictionary produced by `compute_spectral_stats`.
"""
function print_spectral_stats(stats::Dict)
    for (k, v) in sort(collect(stats); by=first)
        println("$k = $v")
    end
end

# ─── Gradient Computation ─────────────────────────────────────────────────────

"""
    _compute_loss_and_grads(model, param_infos, x, y)

Compute the forward pass loss and gradients with respect to all parameters.
Uses numerical finite differences for gradient computation.

Returns (loss_value, gradients_vector) where gradients_vector is ordered
to match param_infos.
"""
function _compute_loss_and_grads(
    model, param_infos::Vector{ParamInfo{P}},
    x::AbstractMatrix{<:Integer}, y::AbstractMatrix{<:Integer};
    spectral_lambda::Real=0.0, power_iters::Int=8
) where P
    # Forward pass to get base loss value
    _, base_loss = forward(model, x; targets=y, training=false)

    base_pen = zero(P)
    if spectral_lambda != 0.0
        base_pen, _ = spectral_regularizer(model; λ=spectral_lambda, power_iters=power_iters)
    end
    base_loss_total = base_loss + base_pen

    # Compute gradients using forward finite differences (includes penalty)
    ε = sqrt(eps(P))  # Optimal step size for forward differences
    grads = Vector{Any}(undef, length(param_infos))

    for i in eachindex(param_infos)
        param = param_infos[i].param
        grad = similar(param)

        for j in eachindex(param)
            orig = param[j]
            param[j] = orig + ε
            _, loss_plus = forward(model, x; targets=y, training=false)

            pen_plus = zero(P)
            if spectral_lambda != 0.0
                pen_plus, _ = spectral_regularizer(model; λ=spectral_lambda, power_iters=power_iters)
            end

            param[j] = orig
            grad[j] = (loss_plus + pen_plus - base_loss_total) / ε
        end

        grads[i] = grad
    end

    return base_loss_total, grads
end




# ─── Recurrent Model Parameter Collection ────────────────────────────────────

"""
    collect_params(model::SequenceModel{P}) where P

Collect all trainable parameters from a SequenceModel, separated into
decay and no-decay groups.

Weight decay rules (matching GPT convention):
- Decay: weight matrices of Linear layers (W_ih, W_hh, decoder)
- No decay: biases, LayerNorm weights/biases, Embedding weights
"""
function collect_params(model::SequenceModel{P}) where P
    param_infos = ParamInfo{P}[]

    # Embedding — no decay
    push!(param_infos, ParamInfo{P}(model.embedding.weight, false))

    # Recurrent cells — weights decay, biases don't
    for cell in model.cells
        _collect_cell_params!(param_infos, cell)
    end

    # Final LayerNorm — no decay
    push!(param_infos, ParamInfo{P}(model.ln_f.weight, false))
    push!(param_infos, ParamInfo{P}(model.ln_f.bias, false))

    # Decoder — weight decays, bias doesn't
    push!(param_infos, ParamInfo{P}(model.decoder.weight, true))
    if has_bias(model.decoder)
        push!(param_infos, ParamInfo{P}(model.decoder.bias, false))
    end

    return param_infos
end

function _collect_cell_params!(param_infos::Vector{ParamInfo{P}}, cell::RNNCell{P}) where P
    push!(param_infos, ParamInfo{P}(cell.W_ih.weight, true))
    push!(param_infos, ParamInfo{P}(cell.W_ih.bias, false))
    push!(param_infos, ParamInfo{P}(cell.W_hh.weight, true))
    push!(param_infos, ParamInfo{P}(cell.W_hh.bias, false))
end

function _collect_cell_params!(param_infos::Vector{ParamInfo{P}}, cell::LSTMCell{P}) where P
    push!(param_infos, ParamInfo{P}(cell.W_ih.weight, true))
    push!(param_infos, ParamInfo{P}(cell.W_ih.bias, false))
    push!(param_infos, ParamInfo{P}(cell.W_hh.weight, true))
    push!(param_infos, ParamInfo{P}(cell.W_hh.bias, false))
end

function _collect_cell_params!(param_infos::Vector{ParamInfo{P}}, cell::GRUCell{P}) where P
    for W in (cell.W_iz, cell.W_hz, cell.W_ir, cell.W_hr, cell.W_in, cell.W_hn)
        push!(param_infos, ParamInfo{P}(W.weight, true))
        push!(param_infos, ParamInfo{P}(W.bias, false))
    end
end

# ─── BPTT Training for SequenceModel ───────────────────────────────────────────

"""
    train!(model::SequenceModel{P}, dataset, config::TrainConfig{P}) where P

Train a recurrent language model with Backpropagation Through Time (BPTT).

Key differences from SequenceModel training:
- Hidden states are carried over between batches (state continuity)
- Gradients flow through the full BPTT unroll window (seq_length)

The `dataset` should be callable returning `(x, y)` where both are
`(seq_length, batch_size)` integer matrices.
"""
function train!(model::SequenceModel{P}, dataset, config::TrainConfig{P}) where P
    param_infos = collect_params(model)
    opt_state = AdamWState(param_infos)
    losses = P[]

    # Persistent BPTT state (carried across batches)
    bptt_states = nothing

    for iter in 1:config.max_iters
        x, y = dataset()
        B = size(x, 2)

        # Initialize states on first iter or if batch size changed
        if bptt_states === nothing
            bptt_states = [init_state(cell, B) for cell in model.cells]
        end

        # Compute loss and gradients (with state carry-over)
        loss_val, grads = _compute_recurrent_loss_and_grads(
            model, param_infos, x, y, bptt_states;
            spectral_lambda=config.spectral_lambda,
            power_iters=config.spectral_power_iters
        )

        # Update carried-over states (detach from computation graph)
        _, _, bptt_states = forward(model, x; training=false, init_states=bptt_states)

        if !isfinite(loss_val)
            error("Training halted at iteration $iter: loss is $(loss_val)")
        end

        push!(losses, loss_val)
        clip_grad_norm!(grads, config.grad_clip)
        adamw_step!(param_infos, grads, opt_state, config)
    end

    return TrainState{P}(losses, config.max_iters)
end

"""
    _compute_recurrent_loss_and_grads(model, param_infos, x, y, states; kwargs...)

Compute loss and gradients for a SequenceModel using finite differences.
Includes state carry-over for BPTT continuity.
"""
function _compute_recurrent_loss_and_grads(
    model::SequenceModel{P}, param_infos::Vector{ParamInfo{P}},
    x::AbstractMatrix{<:Integer}, y::AbstractMatrix{<:Integer},
    states::Vector;
    spectral_lambda::Real=0.0, power_iters::Int=8
) where P
    # Base loss with state carry-over
    _, base_loss, _ = forward(model, x; targets=y, training=false, init_states=states)

    base_pen = zero(P)
    if spectral_lambda != 0.0
        base_pen = _recurrent_spectral_penalty(model; λ=spectral_lambda, power_iters=power_iters)
    end
    base_loss_total = base_loss + base_pen

    # Finite-difference gradients
    ε = sqrt(eps(P))
    grads = Vector{Any}(undef, length(param_infos))

    for i in eachindex(param_infos)
        param = param_infos[i].param
        grad = similar(param)

        for j in eachindex(param)
            orig = param[j]
            param[j] = orig + ε
            _, loss_plus, _ = forward(model, x; targets=y, training=false, init_states=states)

            pen_plus = zero(P)
            if spectral_lambda != 0.0
                pen_plus = _recurrent_spectral_penalty(model; λ=spectral_lambda, power_iters=power_iters)
            end

            param[j] = orig
            grad[j] = (loss_plus + pen_plus - base_loss_total) / ε
        end

        grads[i] = grad
    end

    return base_loss_total, grads
end

# ─── Spectral Diagnostics for Recurrent Weights ─────────────────────────────

"""
    _recurrent_spectral_penalty(model::SequenceModel{P}; λ, power_iters) where P

Compute spectral regularization penalty for recurrent weight matrices.
Penalizes deviation of σ_max(W_hh) from 1.0 — critical for RNN stability.
When σ_max(W_hh) ≈ 1.0, the network operates at the "edge of chaos"
where gradients neither vanish nor explode.
"""
function _recurrent_spectral_penalty(model::SequenceModel{P}; λ::Real=1e-3, power_iters::Int=8) where P
    pen = zero(P)
    for cell in model.cells
        pen += _cell_spectral_penalty(cell, P(λ), power_iters)
    end
    return pen
end

function _cell_spectral_penalty(cell::RNNCell{P}, λ::P, power_iters::Int) where P
    σ = power_iteration(cell.W_hh.weight; num_iters=power_iters)
    return λ * (σ - one(P))^2
end

function _cell_spectral_penalty(cell::LSTMCell{P}, λ::P, power_iters::Int) where P
    σ = power_iteration(cell.W_hh.weight; num_iters=power_iters)
    return λ * (σ - one(P))^2
end

function _cell_spectral_penalty(cell::GRUCell{P}, λ::P, power_iters::Int) where P
    pen = zero(P)
    for W in (cell.W_hz, cell.W_hr, cell.W_hn)
        σ = power_iteration(W.weight; num_iters=power_iters)
        pen += λ * (σ - one(P))^2
    end
    return pen
end

"""
    compute_recurrent_spectral_stats(model::SequenceModel{P}; power_iters=8) where P

Compute spectral norm diagnostics for all recurrent weight matrices.
Returns a Dict with per-layer σ_max(W_hh) values.
"""
function compute_recurrent_spectral_stats(model::SequenceModel{P}; power_iters::Int=8) where P
    stats = Dict{String,Any}()
    for (i, cell) in enumerate(model.cells)
        _cell_spectral_stats!(stats, cell, i, power_iters)
    end
    return stats
end

function _cell_spectral_stats!(stats, cell::RNNCell{P}, idx::Int, power_iters::Int) where P
    stats["cell_$idx.W_ih_sigma"] = power_iteration(cell.W_ih.weight; num_iters=power_iters)
    stats["cell_$idx.W_hh_sigma"] = power_iteration(cell.W_hh.weight; num_iters=power_iters)
end

function _cell_spectral_stats!(stats, cell::LSTMCell{P}, idx::Int, power_iters::Int) where P
    H = cell.hidden_size
    W = cell.W_hh.weight
    stats["cell_$idx.W_hh_i_sigma"] = power_iteration(W[1:H, :]; num_iters=power_iters)
    stats["cell_$idx.W_hh_f_sigma"] = power_iteration(W[H+1:2H, :]; num_iters=power_iters)
    stats["cell_$idx.W_hh_o_sigma"] = power_iteration(W[2H+1:3H, :]; num_iters=power_iters)
    stats["cell_$idx.W_hh_g_sigma"] = power_iteration(W[3H+1:4H, :]; num_iters=power_iters)
end

function _cell_spectral_stats!(stats, cell::GRUCell{P}, idx::Int, power_iters::Int) where P
    stats["cell_$idx.W_hz_sigma"] = power_iteration(cell.W_hz.weight; num_iters=power_iters)
    stats["cell_$idx.W_hr_sigma"] = power_iteration(cell.W_hr.weight; num_iters=power_iters)
    stats["cell_$idx.W_hn_sigma"] = power_iteration(cell.W_hn.weight; num_iters=power_iters)
end
