# scripts/benchmark_fat.jl
#
# Benchmarks the Functional Autoregressive Transformer (FAT) against
# a baseline ContinuousModel (which lacks spectral stabilization and directional loss).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Architect
using Printf

function main()
    println("=== Benchmarking FAT vs Baseline ContinuousModel ===")
    
    # 1. Setup Data Loader
    K = 15
    basis_dim = K + 1
    seq_length = 32
    batch_size = 16
    
    loader = ContinuousTimeDataLoader(;
        task = :sine_waves,
        K = K,
        seq_length = seq_length,
        batch_size = batch_size
    )
    
    # Data generator for training
    batch_fn = make_continuous_batch_fn(loader)
    
    # Common Hyperparameters
    n_embd = 64
    n_head = 4
    n_layer = 3
    dropout = 0.0f0
    max_iters = 200
    lr = 5e-4
    
    println("\n[Dataset] Sine waves mapped to Chebyshev K=$K (d=$basis_dim), seq_len=$seq_length")
    
    # -------------------------------------------------------------------------
    # 2. Train Baseline ContinuousModel
    # -------------------------------------------------------------------------
    println("\n--- Baseline ContinuousModel ---")
    c_config = ContinuousConfig{Float32}(;
        basis_dim = basis_dim,
        basis_type = :chebyshev,
        n_embd = n_embd,
        n_head = n_head,
        n_layer = n_layer,
        block_size = seq_length,
        dropout = dropout
    )
    
    c_model = ContinuousModel(c_config)
    initialize!(c_model)
    
    c_train_config = Architect.Training.ContinuousTrainConfig{Float32}(;
        learning_rate = lr,
        weight_decay = 0.01,
        max_iters = max_iters,
        log_interval = 50
    )
    
    time_baseline = @elapsed begin
        c_state = Architect.Training.train!(c_model, batch_fn, c_train_config)
    end
    
    # -------------------------------------------------------------------------
    # 3. Train Functional Autoregressive Transformer
    # -------------------------------------------------------------------------
    println("\n--- Functional Autoregressive Transformer (FAT) ---")
    f_config = FATConfig{Float32}(;
        K = K,
        n_embd = n_embd,
        n_head = n_head,
        n_layer = n_layer,
        block_size = seq_length,
        dropout = dropout,
        K_anchor = K,    # fully stabilized
        P_econ = 3
    )
    
    f_model = FunctionalAutoregressiveTransformer(f_config)
    
    f_train_config = FATTrainConfig{Float32}(;
        learning_rate = lr,
        weight_decay = 0.01,
        max_iters = max_iters,
        log_interval = 50
    )
    
    time_fat = @elapsed begin
        f_state = fat_train!(f_model, batch_fn, f_train_config)
    end
    
    # -------------------------------------------------------------------------
    # 4. Results
    # -------------------------------------------------------------------------
    println("\n=== Benchmark Results ===")
    @printf("Baseline ContinuousModel : Final Loss = %.6f | Time = %.2fs\n", c_state.losses[end], time_baseline)
    @printf("FAT Model                : Final Loss = %.6f | Time = %.2fs\n", f_state.losses[end], time_fat)
    
    if f_state.losses[end] < c_state.losses[end]
        println("=> FAT achieves lower loss thanks to trajectory approval and spectral stabilization!")
    end
end

main()
