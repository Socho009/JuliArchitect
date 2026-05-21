using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Architect
using Architect.FAT
using Architect.Data
using LinearAlgebra
using Printf
using Statistics: mean

function main()
    println("=== Adaptive K-Truncation Experiment ===")
    
    # 1. Setup Model and Data
    K = 15
    # Small model for fast execution
    config = FATConfig(K=K, n_embd=64, n_head=2, n_layer=2, block_size=16)
    model = FunctionalAutoregressiveTransformer(config)
    
    # Use a mixed signal so that high frequencies occasionally appear
    loader = ContinuousTimeDataLoader(; task=:sine_waves, K=K, seq_length=16, batch_size=1)
    batch_fn = make_continuous_batch_fn(loader)
    
    train_config = FATTrainConfig(learning_rate=0.005, max_iters=5, log_interval=5, warmup_iters=5)
    println("Training FAT model briefly to get non-zero weights...")
    fat_train!(model, batch_fn, train_config)
    
    # 2. Get a seed sequence
    input, _ = batch_fn()
    seed = input[:, 1:4, 1]  # Take first 4 steps as seed
    
    max_new = 50
    
    # 3. Standard Generation (Fixed K)
    println("\nRunning Standard Generation (Fixed K=$K)...")
    # Precompile to ensure fair timing
    fat_generate(model, seed; max_new=2, apply_stabilizer=true)
    
    time_fixed = @elapsed begin
        seq_fixed = fat_generate(model, seed; max_new=max_new, apply_stabilizer=true)
    end
    
    # 4. Adaptive Generation (Dynamic K)
    println("Running Adaptive Generation...")
    # Precompile
    adaptive_fat_generate(model, seed; max_new=2, ϵ=1f-1, δ=5f-1, K_min=4)
    
    # Try multiple generation runs to measure time more accurately
    time_adapt = @elapsed begin
        seq_adapt, k_history = adaptive_fat_generate(model, seed; max_new=max_new, ϵ=1f-1, δ=5f-1, K_min=4)
    end
    
    # 5. Compare Results
    println("\n=== Experiment Results ===")
    @printf("Fixed Generation Time    : %.4f seconds\n", time_fixed)
    @printf("Adaptive Generation Time : %.4f seconds\n", time_adapt)
    
    if time_fixed > 0
        speedup = (time_fixed - time_adapt) / time_fixed * 100
        @printf("Time Saved               : %.2f%%\n", speedup)
    end
    
    avg_k = mean(k_history)
    @printf("Average K used (Adaptive): %.2f (Max: %d)\n", avg_k, K)
    
    # Measure Mean L2 distance between the coefficients
    diff = norm(seq_fixed .- seq_adapt) / length(seq_fixed)
    @printf("Mean Coefficient Diff    : %.6f\n", diff)
    
    println("\nK History over $max_new steps:")
    println(k_history)
end

main()
