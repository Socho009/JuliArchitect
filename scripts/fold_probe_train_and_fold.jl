#!/usr/bin/env julia
# Train a small FAT model and probe for head merging opportunities
# This gives us a realistic test of topological folding with actual training

include(joinpath(@__DIR__, "..", "src", "Architect.jl"))

using .Architect
using .Architect.FAT: collect_head_activations, compute_similarity,
                      propose_clusters, save_similarity_csv
using Statistics: mean, median, std
import Zygote

function main()
    println("\n" * "="^70)
    println("TRAIN SMALL FAT MODEL & ANALYZE HEAD FOLDING OPPORTUNITIES")
    println("="^70)
    
    # Configuration for quick training
    basis_dim = 8
    n_embd = 64
    n_head = 4
    n_layer = 2
    block_size = 16
    batch_size = 8
    train_iters = 100  # Quick training for probe
    learning_rate = 5.0f-4
    
    println("\n[MODEL CONFIG]")
    println("  Basis: Chebyshev($basis_dim)")
    println("  Embedding dim: $n_embd")
    println("  Heads: $n_head per layer")
    println("  Layers: $n_layer")
    println("  Block size: $block_size, Batch size: $batch_size")
    
    # Build model
    println("\n[BUILDING MODEL]")
    try
        cfg = ContinuousConfig(
            basis_dim=basis_dim,
            basis_type=:chebyshev,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            dropout=0.0f0
        )
        model = ContinuousModel(cfg)
        
        # Initialize parameters
        try
            Architect.initialize!(model; muP=false)
            println("✓ Model initialized")
        catch
            println("✓ Model created (initialization skipped)")
        end
        
        # Get precision
        P = eltype(model.ln_f.weight)
        println("✓ Model type: $(typeof(model)), precision: $P")
        
        # Create data loader
        println("\n[DATA LOADER]")
        loader = FunctionSequenceLoader(
            basis_dim=basis_dim,
            seq_length=block_size,
            batch_size=batch_size,
            task=:derivative
        )
        println("✓ FunctionSequenceLoader created")
        
        # ── Collect trainable parameters ──────────────────────────────────
        params = Tuple{AbstractArray, Bool}[]   # (array, should_decay)
        # coeff_encoder
        push!(params, (model.coeff_encoder.weight, true))
        push!(params, (model.coeff_encoder.bias,   false))
        # blocks
        for block in model.blocks
            push!(params, (block.ln1.weight, false))
            att = block.attention
            if att.W_qkv !== nothing
                push!(params, (att.W_qkv.weight, true))
                Architect.Layers.has_bias(att.W_qkv) && push!(params, (att.W_qkv.bias, false))
            end
            if att.W_q !== nothing
                push!(params, (att.W_q.weight, true))
                Architect.Layers.has_bias(att.W_q) && push!(params, (att.W_q.bias, false))
                push!(params, (att.W_k.weight, true))
                Architect.Layers.has_bias(att.W_k) && push!(params, (att.W_k.bias, false))
                push!(params, (att.W_v.weight, true))
                Architect.Layers.has_bias(att.W_v) && push!(params, (att.W_v.bias, false))
            end
            if isa(att.W_out, Architect.Layers.Linear)
                push!(params, (att.W_out.weight, true))
                Architect.Layers.has_bias(att.W_out) && push!(params, (att.W_out.bias, false))
            end
            push!(params, (block.ln2.weight, false))
            push!(params, (block.feedforward.W1.weight, true))
            push!(params, (block.feedforward.W2.weight, true))
            push!(params, (block.feedforward.W3.weight, true))
        end
        # ln_f and coeff_head
        push!(params, (model.ln_f.weight, false))
        push!(params, (model.coeff_head.weight, true))
        push!(params, (model.coeff_head.bias,   false))

        n_params = sum(length(p) for (p, _) in params)
        println("  Trainable parameters: $n_params")

        # AdamW state
        m_state = [zeros(P, size(p)) for (p, _) in params]
        v_state = [zeros(P, size(p)) for (p, _) in params]
        t_adam = 0
        β1, β2 = P(0.9), P(0.999)
        adam_eps = P(1e-8)
        wd = P(0.01)

        # ── Training loop ─────────────────────────────────────────────────
        println("\n[TRAINING ($train_iters iterations with AdamW)]")
        losses = Float64[]
        
        for iter in 1:train_iters
            input, target = Architect.Data.next_batch(loader)
            target_2d = reshape(target, size(target,1), size(target,2) * size(target,3))
            
            # Forward + backward with Zygote
            loss_val, model_grad = Zygote.withgradient(model) do m
                bd, T, B = size(input)
                coeffs_2d = reshape(input, bd, T * B)
                x_2d = m.coeff_encoder(coeffs_2d)
                ne = size(x_2d, 1)
                x = reshape(x_2d, ne, T, B)
                
                for block in m.blocks
                    x = block(x; training=true)
                end
                
                x = m.ln_f(x)
                x_2d_out = reshape(x, size(x, 1), T * B)
                pred = m.coeff_head(x_2d_out)
                
                # MSE loss
                sum(abs2, pred .- target_2d) / P(size(pred, 1) * T * B)
            end
            
            push!(losses, loss_val)
            
            if !isfinite(loss_val)
                println("  ⚠ Loss diverged at iter $iter — halting training")
                break
            end
            
            # Extract gradients using the same field-walking approach as fat_train!
            gs = model_grad[1]
            param_grads = _extract_grads_continuous(gs, model)
            
            # Gradient clipping
            grad_norm = sqrt(sum(g === nothing ? 0.0 : Float64(sum(abs2, g)) for g in param_grads))
            clip_coef = grad_norm > 1.0 ? P(1.0 / (grad_norm + 1e-6)) : one(P)
            
            # AdamW update
            t_adam += 1
            bc1 = one(P) - β1^t_adam
            bc2 = one(P) - β2^t_adam
            
            for i in eachindex(params)
                g = param_grads[i]
                g === nothing && continue
                g = g .* clip_coef
                
                param, decay = params[i]
                m_state[i] .= β1 .* m_state[i] .+ (one(P) - β1) .* g
                v_state[i] .= β2 .* v_state[i] .+ (one(P) - β2) .* g .^ 2
                
                m_hat  = m_state[i] ./ bc1
                v_hat  = v_state[i] ./ bc2
                update = m_hat ./ (sqrt.(v_hat) .+ adam_eps)
                
                if decay
                    param .-= learning_rate .* (update .+ wd .* param)
                else
                    param .-= learning_rate .* update
                end
            end
            
            if iter % 20 == 0 || iter == 1
                println("  Iter $iter/$train_iters, Loss: $(round(loss_val, digits=6)), Grad norm: $(round(grad_norm, digits=4))")
            end
        end
        
        println("✓ Training complete. Final loss: $(round(losses[end], digits=6))")
        println("  Loss reduction: $(round(losses[1], digits=4)) → $(round(losses[end], digits=4))")
        
        # Now run the folding probe on the trained model
        println("\n" * "="^70)
        println("FOLDING ANALYSIS ON TRAINED MODEL")
        println("="^70)
        
        println("\n[COLLECTING ACTIVATIONS (20 batches)]")
        collector = collect_head_activations(model, loader, Architect.Data.next_batch; nbatches=20)
        println("✓ Collected $(length(collector.data)) head activations")
        
        # Compute similarity
        println("\n[COMPUTING SIMILARITY MATRIX (CKA)]")
        sim, keys = compute_similarity(collector; metric=:cka)
        println("✓ Similarity matrix: $(size(sim, 1))×$(size(sim, 2))")
        
        # Analyze similarity distribution
        upper_tri = [sim[i,j] for i=1:size(sim,1) for j=i+1:size(sim,2)]
        println("\nSimilarity Statistics:")
        println("  Min:  $(round(minimum(upper_tri), digits=4))")
        println("  Mean: $(round(mean(upper_tri), digits=4))")
        println("  Median: $(round(median(upper_tri), digits=4))")
        println("  Max:  $(round(maximum(upper_tri), digits=4))")
        println("  Std:  $(round(std(upper_tri), digits=4))")
        
        # Test clustering at different thresholds
        println("\n" * "-"^70)
        println("HEAD MERGING OPPORTUNITIES AT DIFFERENT THRESHOLDS")
        println("-"^70)
        
        thresholds = [0.70f0, 0.75f0, 0.80f0, 0.85f0, 0.90f0, 0.92f0]
        best_thresh = nothing
        best_merge_count = 0
        
        for thresh in thresholds
            clusters = propose_clusters(sim, keys; threshold=thresh)
            merged_clusters = [c for c in clusters if length(c) > 1]
            n_merged = length(merged_clusters)
            total_merged_heads = n_merged > 0 ? sum(length(c) for c in merged_clusters) : 0
            param_reduction = total_merged_heads > 0 ? 
                round(100.0 * total_merged_heads / length(keys), digits=1) : 0.0
            
            println("\nThreshold: $thresh")
            println("  Clusters: $(length(clusters)) total, $n_merged with size > 1")
            println("  Merged heads: $total_merged_heads / $(length(keys)) ($param_reduction% reduction)")
            
            if n_merged > 0
                println("  Merged groups:")
                for c in merged_clusters
                    sims = [sim[findfirst(==(c[i]), keys), findfirst(==(c[j]), keys)] 
                           for i in 1:length(c) for j in i+1:length(c)]
                    avg_sim = length(sims) > 0 ? mean(sims) : 0.0
                    println("    $c → avg similarity: $(round(avg_sim, digits=3))")
                end
                
                if n_merged > best_merge_count && param_reduction > 5.0
                    best_thresh = thresh
                    best_merge_count = n_merged
                end
            end
        end
        
        # Save similarity matrix
        results_dir = joinpath(@__DIR__, "..", "results")
        csv_path = joinpath(results_dir, "head_sim_trained_small.csv")
        save_similarity_csv(sim, keys, csv_path)
        
        println("\n" * "="^70)
        println("SUMMARY")
        println("="^70)
        println("Model: $(n_layer)-layer, $(n_head)-head FAT (trained $train_iters iterations)")
        println("Total heads: $(length(keys))")
        println("Similarity metric: CKA (Centered Kernel Alignment)")
        if best_thresh !== nothing
            println("Best folding threshold: $best_thresh (merges $best_merge_count pairs)")
        else
            println("Recommendation: Lower thresholds (0.70-0.80) for meaningful merging")
        end
        println("Similarity CSV: $csv_path")
        println("="^70 * "\n")
        
    catch e
        println("❌ Error: $e")
        println("\nStacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

# ─── Gradient extraction for ContinuousModel ─────────────────────────────────
# Mirrors the pattern from phase5_fat_model.jl's _fat_extract_grads

function _safe_field(gs, name::Symbol)
    gs === nothing && return nothing
    if gs isa NamedTuple
        return haskey(gs, name) ? getfield(gs, name) : nothing
    end
    try; return getfield(gs, name); catch; return nothing; end
end

function _extract_grads_continuous(gs, model)
    grads = Union{Nothing, AbstractArray}[]
    
    # coeff_encoder
    enc_g = _safe_field(gs, :coeff_encoder)
    push!(grads, _safe_field(enc_g, :weight))
    push!(grads, _safe_field(enc_g, :bias))
    
    # blocks
    blocks_g = _safe_field(gs, :blocks)
    for (i, block) in enumerate(model.blocks)
        bg = (blocks_g !== nothing && blocks_g isa AbstractVector) ? blocks_g[i] : nothing
        
        # ln1
        ln1g = _safe_field(bg, :ln1)
        push!(grads, _safe_field(ln1g, :weight))
        
        # attention
        attg = _safe_field(bg, :attention)
        att = block.attention
        
        if att.W_qkv !== nothing
            wqkvg = _safe_field(attg, :W_qkv)
            push!(grads, _safe_field(wqkvg, :weight))
            Architect.Layers.has_bias(att.W_qkv) && push!(grads, _safe_field(wqkvg, :bias))
        end
        if att.W_q !== nothing
            wqg = _safe_field(attg, :W_q)
            push!(grads, _safe_field(wqg, :weight))
            Architect.Layers.has_bias(att.W_q) && push!(grads, _safe_field(wqg, :bias))
            wkg = _safe_field(attg, :W_k)
            push!(grads, _safe_field(wkg, :weight))
            Architect.Layers.has_bias(att.W_k) && push!(grads, _safe_field(wkg, :bias))
            wvg = _safe_field(attg, :W_v)
            push!(grads, _safe_field(wvg, :weight))
            Architect.Layers.has_bias(att.W_v) && push!(grads, _safe_field(wvg, :bias))
        end
        if isa(att.W_out, Architect.Layers.Linear)
            woutg = _safe_field(attg, :W_out)
            push!(grads, _safe_field(woutg, :weight))
            Architect.Layers.has_bias(att.W_out) && push!(grads, _safe_field(woutg, :bias))
        end
        
        # ln2
        ln2g = _safe_field(bg, :ln2)
        push!(grads, _safe_field(ln2g, :weight))
        
        # feedforward
        ffg = _safe_field(bg, :feedforward)
        for wname in (:W1, :W2, :W3)
            wg = _safe_field(ffg, wname)
            push!(grads, _safe_field(wg, :weight))
        end
    end
    
    # ln_f
    lnfg = _safe_field(gs, :ln_f)
    push!(grads, _safe_field(lnfg, :weight))
    
    # coeff_head
    headg = _safe_field(gs, :coeff_head)
    push!(grads, _safe_field(headg, :weight))
    push!(grads, _safe_field(headg, :bias))
    
    return grads
end

main()
