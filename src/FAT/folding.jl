# ──── Topological Folding for Transformer Compression ────────────────────────
#
# Phase 0 (new): Prototype topological folding to detect and merge redundant
# attention heads / MLP blocks while maintaining representational power.

module FATFolding

using LinearAlgebra
using Statistics
using Printf
import ..Architect

const Layers = Architect.Layers
const Data = Architect.Data

export ProbeCollector, TrackedAttention
export wrap_attentions!, collect_head_activations
export compute_similarity, propose_clusters, save_similarity_csv
export FoldedHead, apply_soft_merge
export FoldManager, should_fold, propose_and_apply_fold!

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

"""TrackedAttention wrapper

Wraps an existing `Layers.Attention` instance and records pre-recombination
per-head outputs into a `ProbeCollector` during forward. Safe to install by
replacing transformer `Block` objects (immutable) with a fresh `Block` that
holds a `TrackedAttention` instance.
"""
struct TrackedAttention{P, Mask}
    attn::Any
    layer_idx::Int
    collector::ProbeCollector{P}
end

function TrackedAttention(attn, layer_idx::Int, c::ProbeCollector{P}) where {P}
    return TrackedAttention{P, typeof(attn.mask)}(attn, layer_idx, c)
end

function Base.show(io::IO, ta::TrackedAttention{P, M}) where {P, M}
    print(io, "TrackedAttention(layer=$(ta.layer_idx), heads=$(ta.attn.n_head))")
end

"""Forward pass for TrackedAttention.

This mirrors `Layers.Attention` forward logic but captures the per-head
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
        Q = Layers.apply_position_encoding!(Q)
        K = Layers.apply_position_encoding!(K)
    end

    # Scores and weights
    scale = P(1.0 / sqrt(head_dim))
    att = Layers._batched_attention_scores(Q, K, scale)
    causal_mask = @view attn.mask[1:T, 1:T]
    att = Layers._apply_causal_mask(att, causal_mask, P)
    att = Layers._softmax(att; dims=2)
    att = Layers._dropout(att, attn.attn_pdrop; training=training)

    # Per-head outputs (head_dim, T, n_head, B)
    out_heads = Layers._batched_attention_output(att, V)

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
    out = Layers._dropout(out, attn.resid_pdrop; training=training)
    return out
end

"""Replace `Block.attention` with `TrackedAttention` instances in-place.

Because `Block` is immutable we construct new `Block` objects and assign
them back into `model.blocks`.
"""
function wrap_attentions!(model::Any, collector::ProbeCollector)
    for (i, block) in enumerate(model.blocks)
        # infer precision from ln1 weight
        P = eltype(block.ln1.weight)
        attn = block.attention
        # build tracked attention
        ta = TrackedAttention(attn, i, collector)
        # create new Block with the tracked attention
        new_block = Layers.Block{P, typeof(ta), typeof(block.ln1), typeof(block.feedforward)}(
            block.ln1, ta, block.ln2, block.feedforward, block.resid_pdrop
        )
        model.blocks[i] = new_block
    end
    return collector
end

"""Collect activations by running `nbatches` from `loader` through `model`.

Returns the `ProbeCollector` filled with captured activations.
`loader` must implement `next_batch(loader) -> (input, target)` as in
`FunctionSequenceLoader`.
"""
function collect_head_activations(model::Any, loader; nbatches::Int=20)
    P = eltype(model.ln_f.weight)
    collector = ProbeCollector{P}()

    for i in 1:nbatches
        # Attempt to call the loader's batch function robustly. First try a plain
        # `next_batch(loader)` (in case it's in scope), then `Data.next_batch`,
        # then look up `next_batch` from the module where the loader type is defined.
        input = nothing; target = nothing
        tried_errs = Vector{Any}()
        try
            input, target = next_batch(loader)
        catch e1
            push!(tried_errs, e1)
            try
                input, target = Data.next_batch(loader)
            catch e2
                push!(tried_errs, e2)
                try
                    m = parentmodule(typeof(loader))
                    bf = getfield(m, :next_batch)
                    input, target = bf(loader)
                catch e3
                    push!(tried_errs, e3)
                    rethrow(e1)
                end
            end
        end
        basis_dim, T, B = size(input)

        # 1. Encode coefficients → embeddings
        coeffs_2d = reshape(input, basis_dim, T * B)
        x_2d = model.coeff_encoder(coeffs_2d)  # (n_embd, T*B)
        n_embd = size(x_2d, 1)
        x = reshape(x_2d, n_embd, T, B)

        # 2. Instrumented transformer blocks: for each block compute per-head outputs
        for (li, block) in enumerate(model.blocks)
            # compute attention inputs (pre-norm)
            x_norm = block.ln1(x)
            attn = block.attention

            # Project input to Q,K,V via combined or separate projections
            if attn.W_q !== nothing
                Q = attn.W_q(x_norm)
                K = attn.W_k(x_norm)
                V = attn.W_v(x_norm)
            else
                qkv = attn.W_qkv(x_norm)
                Q = qkv[1:size(qkv,1)÷3, :, :]
                K = qkv[size(qkv,1)÷3+1:2*(size(qkv,1)÷3), :, :]
                V = qkv[2*(size(qkv,1)÷3)+1:end, :, :]
            end

            head_dim = attn.head_dim
            n_head = attn.n_head

            # reshape
            Q = reshape(Q, head_dim, T, n_head, B)
            K = reshape(K, head_dim, T, n_head, B)
            V = reshape(V, head_dim, T, n_head, B)

            # apply RoPE if needed
            if head_dim % 2 == 0
                Q = Layers.apply_position_encoding!(Q)
                K = Layers.apply_position_encoding!(K)
            end

            # compute scores and per-head outputs
            scale = P(1.0 / sqrt(head_dim))
            att = Layers._batched_attention_scores(Q, K, scale)
            causal_mask = @view attn.mask[1:T, 1:T]
            att = Layers._apply_causal_mask(att, causal_mask, P)
            att = Layers._softmax(att; dims=2)
            att = Layers._dropout(att, attn.attn_pdrop; training=false)
            out_heads = Layers._batched_attention_output(att, V)

            # record per-head outputs
            for h in 1:n_head
                key = (li, h)
                out_h = copy(@view out_heads[:, :, h, :])
                if !haskey(collector.data, key)
                    collector.data[key] = Vector{Array{P,3}}()
                end
                push!(collector.data[key], out_h)
            end

            # now apply block forward to keep model state consistent
            attn_out = block.attention(x_norm; training=false)
            x = x .+ attn_out
            mlp_out = block.feedforward(block.ln2(x); training=false)
            x = x .+ mlp_out
        end

        # final norm and head (not needed for collection)
        x = model.ln_f(x)
        x_2d_out = reshape(x, size(x,1), T * B)
        _ = model.coeff_head(x_2d_out)
    end

    return collector
end

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
"""
function apply_soft_merge(out_heads::AbstractArray{P,4}, gates::Vector{P}) where P
    head_dim, T, n_heads, B = size(out_heads)
    g_norm = gates ./ sum(gates)
    out = zeros(P, head_dim, T, B)
    for h in 1:n_heads
        out .+= g_norm[h] .* @view out_heads[:, :, h, :]
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
function propose_and_apply_fold!(mgr::FoldManager, collector::ProbeCollector, model::Any)
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

end # module FATFolding
