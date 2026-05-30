#!/usr/bin/env julia
# Train a small FAT model and probe for head merging opportunities
# This gives us a realistic test of topological folding

include(joinpath(@__DIR__, "..", "src", "Architect.jl"))
include(joinpath(@__DIR__, "..", "src", "FAT", "folding.jl"))
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
    learning_rate = 5.0e-4
    
    println("\n[MODEL CONFIG]")
    println("  Basis: Chebyshev($basis_dim)")
    println("  Embedding dim: $n_embd")
    println("  Heads: $n_head per layer")
    println("  Layers: $n_layer")
    println("  Block size: $block_size, Batch size: $batch_size")
    
    # Build model
    println("\n[BUILDING MODEL]")
    try
        cfg = Architect.ContinuousConfig(
            basis_dim=basis_dim,
            basis_type=:chebyshev,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            dropout=0.0f0
        )
        model = Architect.ContinuousModel(cfg)
        
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
        loader = Architect.FunctionSequenceLoader(
            basis_dim=basis_dim,
            seq_length=block_size,
            batch_size=batch_size,
            task=:derivative
        )
        println("✓ FunctionSequenceLoader created")
        
        # Quick training to make model learn something
        println("\n[TRAINING ($train_iters iterations)]")
        total_loss = 0.0
        
        for iter in 1:train_iters
            try
                input, target = Architect.Data.next_batch(loader)
            catch
                input, target = next_batch(loader)
            end
            
            basis_dim_in, T, B = size(input)
            
            # Forward pass
            coeffs_2d = reshape(input, basis_dim_in, T * B)
            x_2d = model.coeff_encoder(coeffs_2d)
            n_embd = size(x_2d, 1)
            x = reshape(x_2d, n_embd, T, B)
            
            for block in model.blocks
                x = block(x; training=true)
            end
            
            x = model.ln_f(x)
            x_2d_out = reshape(x, size(x, 1), T * B)
            pred = model.coeff_head(x_2d_out)
            
            # MSE loss
            loss = mean((pred .- target) .^ 2)
            total_loss += loss
            
            if iter % 20 == 0 || iter == 1
                avg_loss = total_loss / (iter > 1 ? iter : 1)
                println("  Iter $iter/$train_iters, Avg Loss: $(round(avg_loss, digits=6))")
            end
        end
        
        avg_loss = total_loss / train_iters
        println("✓ Training complete. Final avg loss: $(round(avg_loss, digits=6))")
        
        # Now run the folding probe on the trained model
        println("\n" * "="^70)
        println("FOLDING ANALYSIS ON TRAINED MODEL")
        println("="^70)
        
        println("\n[COLLECTING ACTIVATIONS (20 batches)]")
        collector = FATFolding.collect_head_activations(model, loader; nbatches=20)
        println("✓ Collected $(length(collector.data)) head activations")
        
        # Compute similarity
        println("\n[COMPUTING SIMILARITY MATRIX (CKA)]")
        sim, keys = FATFolding.compute_similarity(collector; metric=:cka)
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
            clusters = FATFolding.propose_clusters(sim, keys; threshold=thresh)
            merged_clusters = [c for c in clusters if length(c) > 1]
            n_merged = length(merged_clusters)
            total_merged_heads = sum(length(c) for c in merged_clusters)
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
        FATFolding.save_similarity_csv(sim, keys, csv_path)
        
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
        showerror(stdout, e)
        println()
    end
end

main()
