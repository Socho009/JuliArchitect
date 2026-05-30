#!/usr/bin/env julia
# Probe script: analyze head similarity on a fully-trained FAT model
# Tests topological folding with a lower threshold to see meaningful head merging

include(joinpath(@__DIR__, "..", "src", "Architect.jl"))

using .Architect
using .Architect.FAT: collect_head_activations, compute_similarity,
                      propose_clusters, save_similarity_csv
using Statistics: mean, std
using Serialization

function load_serialized_model(path::String)
    """Load a model from .ser file"""
    if !isfile(path)
        error("Model file not found: $path")
    end
    try
        return deserialize(path)
    catch e
        error("Failed to deserialize model from $path: $e")
    end
end

function main()
    println("\n" * "="^70)
    println("TOPOLOGICAL FOLDING PROBE - TRAINED MODEL ANALYSIS")
    println("="^70)
    
    # Determine which model to load
    results_dir = joinpath(@__DIR__, "..", "results")
    model_paths = Dict(
        "small" => joinpath(results_dir, "bench_small.ser"),
        "medium" => joinpath(results_dir, "bench_medium.ser"),
        "large" => joinpath(results_dir, "bench_large.ser"),
        "current" => joinpath(results_dir, "bench_current.ser")
    )
    
    # Try to load the current/largest available model
    chosen_model = nothing
    chosen_name = nothing
    for (name, path) in [("current", model_paths["current"]), 
                         ("large", model_paths["large"]),
                         ("medium", model_paths["medium"]),
                         ("small", model_paths["small"])]
        if isfile(path)
            println("\n📁 Found model: $name ($path)")
            try
                chosen_model = load_serialized_model(path)
                chosen_name = name
                println("✓ Successfully loaded $name model")
                break
            catch e
                println("⚠ Failed to load $name: $e")
                continue
            end
        end
    end
    
    if chosen_model === nothing
        error("❌ No trained models found. Run benchmark_fat.jl first.")
    end
    
    println("\n" * "-"^70)
    println("MODEL STRUCTURE")
    println("-"^70)
    println("Model type: $(typeof(chosen_model))")
    if hasfield(typeof(chosen_model), :blocks)
        println("Number of blocks (layers): $(length(chosen_model.blocks))")
        if !isempty(chosen_model.blocks)
            block = chosen_model.blocks[1]
            if hasfield(typeof(block), :attention)
                attn = block.attention
                println("Heads per layer: $(attn.n_head)")
                println("Head dimension: $(attn.head_dim)")
            end
        end
    end
    
    # Create a data loader for collecting activations
    println("\n" * "-"^70)
    println("ACTIVATION COLLECTION")
    println("-"^70)
    
    loader = nothing
    
    try
        loader = FunctionSequenceLoader(
            basis_dim=8, 
            seq_length=16, 
            batch_size=8, 
            task=:derivative
        )
        println("Created FunctionSequenceLoader")
    catch e
        println("⚠ FunctionSequenceLoader construction failed: $e")
        println("Attempting alternative loader setup...")
    end
    
    if loader !== nothing
        println("\n🔍 Collecting per-head activations (16 batches)...")
        try
            @time collector = collect_head_activations(
                chosen_model, loader, Architect.Data.next_batch; nbatches=16
            )
            
            if isempty(collector.data)
                error("No activations collected!")
            end
            
            n_collected_heads = length(collector.data)
            println("✓ Collected activations for $n_collected_heads heads")
            
            # Compute similarity matrix
            println("\n" * "-"^70)
            println("SIMILARITY ANALYSIS (CKA Metric)")
            println("-"^70)
            
            @time sim, keys = compute_similarity(collector; metric=:cka)
            
            println("✓ Computed $(size(sim, 1))×$(size(sim, 2)) similarity matrix")
            println("\nSimilarity statistics:")
            
            # Upper triangle (excluding diagonal)
            upper_tri = [sim[i,j] for i=1:size(sim,1) for j=i+1:size(sim,2)]
            if !isempty(upper_tri)
                println("  Min off-diagonal: $(minimum(upper_tri))")
                println("  Mean off-diagonal: $(mean(upper_tri))")
                println("  Max off-diagonal: $(maximum(upper_tri))")
                println("  Std off-diagonal: $(std(upper_tri))")
            end
            
            # Test multiple thresholds
            thresholds = [0.70f0, 0.75f0, 0.80f0, 0.85f0, 0.90f0, 0.92f0]
            println("\n" * "-"^70)
            println("CLUSTERING PROPOSALS AT DIFFERENT THRESHOLDS")
            println("-"^70)
            
            for thresh in thresholds
                clusters = propose_clusters(sim, keys; threshold=thresh)
                n_merged_pairs = sum(length(c) > 1 ? 1 : 0 for c in clusters)
                total_heads_in_merged = sum(length(c) > 1 ? length(c) : 0 for c in clusters)
                
                println("\nThreshold: $(thresh)")
                println("  Total clusters: $(length(clusters))")
                println("  Merged clusters (size > 1): $n_merged_pairs")
                if n_merged_pairs > 0
                    println("  Total heads in merged clusters: $total_heads_in_merged")
                    for (i, c) in enumerate(clusters)
                        if length(c) > 1
                            println("    Cluster $i: $c")
                        end
                    end
                end
            end
            
            # Save detailed similarity matrix at best threshold
            csv_path = joinpath(results_dir, "head_sim_trained_$(chosen_name).csv")
            save_similarity_csv(sim, keys, csv_path)
            println("\n✓ Wrote detailed similarity matrix to: $csv_path")
            
            # Summary statistics
            println("\n" * "="^70)
            println("SUMMARY")
            println("="^70)
            println("Model: $chosen_name")
            println("Total heads analyzed: $n_collected_heads")
            println("Similarity metric: CKA (Centered Kernel Alignment)")
            println("Recommended folding threshold: 0.85-0.90")
            println("CSV output: $csv_path")
            println("="^70 * "\n")
            
        catch e
            println("❌ Activation collection failed: $e")
            println("\nStacktrace:")
            showerror(stdout, e)
            println()
        end
    else
        println("⚠️ Skipping activation collection - loader could not be initialized")
        println("The FunctionSequenceLoader requires specific task data that may not be available.")
    end
end

main()
