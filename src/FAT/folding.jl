# ──── Topological Folding for Transformer Compression ────────────────────────
#
# Phase 9: Prototype topological folding to detect and merge redundant
# attention heads / MLP blocks while maintaining representational power.
#
# Included from FAT.jl — uses sibling imports (..Layers, etc.) just like
# the other phase files.

using Printf

# ─── Exports ──────────────────────────────────────────────────────────────────

export ProbeCollector, TrackedAttention
export wrap_attentions!, collect_head_activations
export compute_similarity, propose_clusters, save_similarity_csv
export FoldedHead, apply_soft_merge
export FoldManager, should_fold, propose_and_apply_fold!

# ─── ProbeCollector ───────────────────────────────────────────────────────────

"""
    ProbeCollector

Accumulates per-(layer, head) activation tensors during forward passes.
Stored shape per batch: (head_dim, seq_len, batch)
"""
mutable struct ProbeCollector{P<:AbstractFloat}
    data::Dict{Tuple{Int,Int}, Vector{Array{P,3}}}
end
ProbeCollector{P}() where P = ProbeCollector{P}(Dict{Tuple{Int,Int}, Vector{Array{P,3}}}())
ProbeCollector() = ProbeCollector{Float32}()

function Base.empty!(c::ProbeCollector)
    empty!(c.data)
    return c
end

# ─── TrackedAttention ─────────────────────────────────────────────────────────

"""TrackedAttention wrapper

Wraps an existing `Attention` instance and records pre-recombination
per-head outputs into a `ProbeCollector` during forward. Subtypes
`AbstractAttention{P}` so it can be placed inside a `Block`.
"""
struct TrackedAttention{P<:AbstractFloat, Mask} <: AbstractAttention{P}
    attn::Attention{P, Mask}
    layer_idx::Int
    collector::ProbeCollector{P}
end

function TrackedAttention(attn::Attention{P, Mask}, layer_idx::Int, c::ProbeCollector{P}) where {P, Mask}
    return TrackedAttention{P, Mask}(attn, layer_idx, c)
end

function Base.show(io::IO, ta::TrackedAttention{P, M}) where {P, M}
    print(io, "TrackedAttention(layer=$(ta.layer_idx), heads=$(ta.attn.n_head))")
end

# Delegate field access that Block / other code may need
Base.getproperty(ta::TrackedAttention, s::Symbol) = begin
    if s in (:attn, :layer_idx, :collector)
        return getfield(ta, s)
    else
        # Delegate to the inner attention (n_head, head_dim, mask, etc.)
        return getproperty(getfield(ta, :attn), s)
    end
end

"""Forward pass for TrackedAttention.

This mirrors `Attention` forward logic but captures the per-head
outputs (shape: head_dim × T × B) and appends them into the collector.
"""
function (ta::TrackedAttention{P, Mask})(x::AbstractArray{P,3}; training::Bool=true) where {P, Mask<:AbstractMatrix{Bool}}
    attn = ta.attn
    n_embd, T, B = size(x)
    n_head = attn.n_head
    head_dim = attn.head_dim

    # Project to Q,K,V
    if attn.W_q !== nothing
        Q = attn.W_q(x)
        K = attn.W_k(x)
        V = attn.W_v(x)
    else
        qkv = attn.W_qkv(x)  # (3*n_embd, T, B)
        Q = qkv[1:n_embd, :, :]
        K = qkv[n_embd+1:2*n_embd, :, :]
        V = qkv[2*n_embd+1:3*n_embd, :, :]
    end

    # Reshape to (head_dim, T, n_head, B)
    Q = reshape(Q, head_dim, T, n_head, B)
    K = reshape(K, head_dim, T, n_head, B)
    V = reshape(V, head_dim, T, n_head, B)

    # RoPE
    if head_dim % 2 == 0
        Q = apply_position_encoding!(Q)
        K = apply_position_encoding!(K)
    end

    # Scores and weights
    scale = P(1.0 / sqrt(head_dim))
    att = _batched_attention_scores(Q, K, scale)
    causal_mask = @view attn.mask[1:T, 1:T]
    att = _apply_causal_mask(att, causal_mask, P)
    att = _softmax(att; dims=2)
    att = _dropout(att, attn.attn_pdrop; training=training)

    # Per-head outputs (head_dim, T, n_head, B)
    out_heads = _batched_attention_output(att, V)

    # Record outputs in collector
    for h in 1:n_head
        key = (ta.layer_idx, h)
        # view and copy to break references
        out_h = copy(@view out_heads[:, :, h, :])
        if !haskey(ta.collector.data, key)
            ta.collector.data[key] = Vector{Array{P,3}}()
        end
        push!(ta.collector.data[key], out_h)
    end

    # Recombine heads and apply output projection + dropout
    out = reshape(out_heads, n_embd, T, B)
    out = attn.W_out(out)
    out = _dropout(out, attn.resid_pdrop; training=training)
    return out
end

# ─── Wrapping ─────────────────────────────────────────────────────────────────

"""Replace `Block.attention` with `TrackedAttention` instances.

Because `Block` is immutable we construct new `Block` objects and assign
them back into `model.blocks`.  Works with any model that stores blocks
in a `Vector` field named `blocks`.
"""
function wrap_attentions!(model, collector::ProbeCollector{P}) where {P}
    for (i, block) in enumerate(model.blocks)
        attn = block.attention
        # Skip if already tracked
        attn isa TrackedAttention && continue

        ta = TrackedAttention(attn, i, collector)
        new_block = Block{P, typeof(ta), typeof(block.ln1), typeof(block.feedforward)}(
            block.ln1, ta, block.ln2, block.feedforward, block.resid_pdrop
        )
        model.blocks[i] = new_block
    end
    return collector
end

"""Unwrap `TrackedAttention` back to original `Attention` instances."""
function unwrap_attentions!(model)
    for (i, block) in enumerate(model.blocks)
        attn = block.attention
        if attn isa TrackedAttention
            P = eltype(block.ln1.weight)
            orig_attn = attn.attn
            new_block = Block{P, typeof(orig_attn), typeof(block.ln1), typeof(block.feedforward)}(
                block.ln1, orig_attn, block.ln2, block.feedforward, block.resid_pdrop
            )
            model.blocks[i] = new_block
        end
    end
end

# ─── Activation Collection ────────────────────────────────────────────────────

"""Collect activations by running `nbatches` from `loader` through `model`.

Returns the `ProbeCollector` filled with captured activations.
Uses `wrap_attentions!` to instrument the model, runs forward passes,
then restores the original attention layers.

`loader` must support `next_batch(loader) -> (input, target)`.
"""
function collect_head_activations(model, loader, next_batch_fn::Function; nbatches::Int=20)
    P = eltype(model.ln_f.weight)
    collector = ProbeCollector{P}()

    # Instrument the model
    wrap_attentions!(model, collector)

    try
        for i in 1:nbatches
            input, target = next_batch_fn(loader)
            basis_dim, T, B = size(input)

            # Run the standard forward pass — TrackedAttention records per-head outputs
            coeffs_2d = reshape(input, basis_dim, T * B)
            x_2d = model.coeff_encoder(coeffs_2d)  # (n_embd, T*B)
            n_embd = size(x_2d, 1)
            x = reshape(x_2d, n_embd, T, B)

            for block in model.blocks
                x = block(x; training=false)
            end

            # Final norm + head (not needed for collection, but keeps pipeline complete)
            x = model.ln_f(x)
            x_2d_out = reshape(x, size(x,1), T * B)
            _ = model.coeff_head(x_2d_out)
        end
    finally
        # Always restore original attention layers
        unwrap_attentions!(model)
    end

    return collector
end

# Convenience overload: auto-detect next_batch function
function collect_head_activations(model, loader; nbatches::Int=20)
    # The loader is from Data module — use its next_batch method
    # We find the right method by trying the module hierarchy
    batch_fn = _resolve_next_batch(loader)
    return collect_head_activations(model, loader, batch_fn; nbatches=nbatches)
end

"""Resolve the appropriate `next_batch` function for a loader."""
function _resolve_next_batch(loader)
    # Try to find next_batch in the module that defines the loader type
    loader_mod = parentmodule(typeof(loader))
    if isdefined(loader_mod, :next_batch)
        return getfield(loader_mod, :next_batch)
    end
    error("Cannot find `next_batch` method for loader type $(typeof(loader)). " *
          "Pass the function explicitly: collect_head_activations(model, loader, next_batch_fn)")
end

# ─── Similarity Computation ──────────────────────────────────────────────────

"""Assemble a 2D matrix from collected tensors for a single head.

Output shape: (features, samples) where features = head_dim * seq_len
and samples = sum(batch_sizes) across collected minibatches.
"""
function _assemble_matrix(collected::Vector{Array{P,3}}) where P
    mats = Vector{Array{P,2}}(undef, length(collected))
    for (i, a) in enumerate(collected)
        h, T, B = size(a)
        mats[i] = reshape(a, h * T, B)
    end
    return hcat(mats...)
end

"""Compute linear CKA between two feature matrices X and Y.
X, Y: (features, samples)
"""
function _linear_cka(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T<:AbstractFloat
    # center features (feature-wise mean across samples)
    Xc = X .- mean(X, dims=2)
    Yc = Y .- mean(Y, dims=2)
    K = Xc' * Yc
    num = sum(abs2, K)
    Kxx = Xc' * Xc; Kyy = Yc' * Yc
    denom = sqrt(sum(abs2, Kxx) * sum(abs2, Kyy))
    if denom == zero(T)
        return zero(T)
    end
    return num / denom
end

"""Compute pairwise similarity for all collected heads.

Returns `(sim_matrix, keys)` where `keys` is vector of `(layer, head)` in
the same order as rows/columns of `sim_matrix`.
"""
function compute_similarity(collector::ProbeCollector; metric::Symbol = :cka)
    klist = sort(collect(keys(collector.data)))
    n = length(klist)
    sim = zeros(Float32, n, n)

    mats = Vector{Array{Float32,2}}(undef, n)
    for (i, k) in enumerate(klist)
        mats[i] = Float32.(_assemble_matrix(collector.data[k]))
    end

    for i in 1:n
        for j in i:n
            val = if metric == :cka
                Float32(_linear_cka(mats[i], mats[j]))
            elseif metric == :cosine
                xi = mean(mats[i]; dims=2); yj = mean(mats[j]; dims=2)
                num = vec(xi)' * vec(yj)
                den = norm(vec(xi)) * norm(vec(yj))
                den == 0f0 ? 0f0 : Float32(num / den)
            else
                error("unknown metric: $metric")
            end
            sim[i, j] = val
            sim[j, i] = val
        end
    end
    return sim, klist
end

"""Greedy clustering: group heads whose similarity exceeds `threshold`.

Return: Vector of clusters, each cluster a Vector of `(layer, head)` keys.
"""
function propose_clusters(sim::AbstractMatrix{T}, keys::Vector{K}; threshold::T=0.92) where {T<:AbstractFloat, K}
    n = size(sim, 1)
    assigned = falses(n)
    clusters = Vector{Vector{K}}()
    for i in 1:n
        if assigned[i]
            continue
        end
        members = [i]
        assigned[i] = true
        for j in (i+1):n
            if !assigned[j] && sim[i, j] ≥ threshold
                push!(members, j); assigned[j] = true
            end
        end
        push!(clusters, [keys[m] for m in members])
    end
    return clusters
end

"""Save similarity matrix to CSV with header keys.
"""
function save_similarity_csv(sim::AbstractMatrix{T}, keys::Vector{Tuple{Int,Int}}, outpath::AbstractString) where T
    open(outpath, "w") do io
        # header
        print(io, "key")
        for k in keys
            print(io, ",", "L$(k[1])_H$(k[2])")
        end
        println(io)
        for (i, k) in enumerate(keys)
            print(io, "L$(k[1])_H$(k[2])")
            for j in 1:size(sim, 2)
                @printf(io, ",%.6f", sim[i, j])
            end
            println(io)
        end
    end
end

# ───── Soft Merge: FoldedHead ─────────────────────────────────────────────────

"""
    FoldedHead{P}

Wraps multiple equivalent heads with learned gate weights.
During training, gates can be sparsified (entropy/L1 penalty).
Once converged, hard-merge by constructing a single head.
"""
struct FoldedHead{P<:AbstractFloat}
    gate_weights::Vector{P}  # (n_heads,)
    n_heads::Int
end

function FoldedHead{P}(n_heads::Int) where P
    gates = fill(P(1.0 / n_heads), n_heads)
    FoldedHead{P}(gates, n_heads)
end

"""
    apply_soft_merge(out_heads, gates) → (head_dim, T, B)

Weighted combination of per-head outputs using learned gates.
Mutation-free for Zygote AD compatibility.
"""
function apply_soft_merge(out_heads::AbstractArray{P,4}, gates::Vector{P}) where P
    head_dim, T, n_heads, B = size(out_heads)
    g_norm = gates ./ sum(gates)
    # Build output without mutation: sum over weighted heads
    out = g_norm[1] .* @view out_heads[:, :, 1, :]
    for h in 2:n_heads
        out = out .+ g_norm[h] .* @view out_heads[:, :, h, :]
    end
    return out
end

# ───── FoldManager: Dynamic Folding Orchestrator ──────────────────────────────

"""
    FoldManager{P}

Manages periodic similarity analysis, folding proposals, and soft-merge application.
"""
mutable struct FoldManager{P<:AbstractFloat}
    enabled::Bool
    fold_interval::Int
    similarity_threshold::P
    ft_steps::Int
    metric::Symbol
    fold_history::Vector{Dict}
    current_fold_id::Dict{Tuple{Int,Int}, Int}
    soft_merged_heads::Dict{Int, Vector{Tuple{Int,Int}}}
end

function FoldManager{P}(; enabled::Bool=true, fold_interval::Int=500,
                         similarity_threshold::Real=0.92, ft_steps::Int=50,
                         metric::Symbol=:cka) where P
    FoldManager{P}(
        enabled, fold_interval, P(similarity_threshold), ft_steps, metric,
        Vector{Dict}(),
        Dict{Tuple{Int,Int}, Int}(),
        Dict{Int, Vector{Tuple{Int,Int}}}()
    )
end

FoldManager(; kwargs...) = FoldManager{Float32}(; kwargs...)

"""
    should_fold(mgr::FoldManager, step::Int) → Bool

Determine if folding should be triggered at this training step.
"""
function should_fold(mgr::FoldManager, step::Int)::Bool
    return mgr.enabled && (step > 0) && (step % mgr.fold_interval == 0)
end

"""
    propose_and_apply_fold!(mgr, collector, model)

Analyze similarity, propose clusters, record fold assignments.
"""
function propose_and_apply_fold!(mgr::FoldManager, collector::ProbeCollector, model)
    sim, klist = compute_similarity(collector; metric=mgr.metric)
    clusters = propose_clusters(sim, klist; threshold=mgr.similarity_threshold)
    
    fold_proposals = Vector{Tuple{Int, Vector}}()
    for (fold_id, cluster) in enumerate(clusters)
        if length(cluster) > 1
            for head_key in cluster
                mgr.current_fold_id[head_key] = fold_id
            end
            mgr.soft_merged_heads[fold_id] = cluster
            push!(fold_proposals, (fold_id, cluster))
        end
    end
    push!(mgr.fold_history, copy(mgr.current_fold_id))
    return fold_proposals
end
