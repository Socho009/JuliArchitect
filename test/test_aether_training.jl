using Test

include(joinpath(@__DIR__, "..", "src", "Architect.jl"))
using .Architect
using Random

function tiny_aether_batch(K::Int, seq_len::Int, batch_size::Int)
    input = Array{Float32}(undef, K + 1, seq_len, batch_size)
    target = similar(input)
    for b in 1:batch_size, t in 1:seq_len, n in 0:K
        x = Float32(sin(0.11 * t + 0.07 * b + 0.03 * n) / (1 + n))
        input[n+1, t, b] = x
        target[n+1, t, b] = 0.97f0 * x
    end
    return input, target
end

@testset "Aether-regularized FAT training" begin
    Random.seed!(123)
    K = 1
    seq_len = 2
    batch_size = 1

    default_cfg = FATTrainConfig{Float32}(max_iters=1)
    @test default_cfg.aether_lambda == 0f0
    @test isempty(default_cfg.aether_spaces)

    model_cfg = FATConfig{Float32}(
        K = K,
        K_anchor = K,
        P_econ = 1,
        n_embd = 4,
        n_head = 1,
        n_layer = 0,
        block_size = seq_len,
        dropout = 0.0f0,
    )
    model = FunctionalAutoregressiveTransformer(model_cfg)
    spaces = stable_aether_spaces(Float32; K=K)
    train_cfg = FATTrainConfig{Float32}(
        learning_rate = 1f-4,
        max_iters = 1,
        warmup_iters = 1,
        log_interval = 1,
        aether_lambda = 1f-4,
        aether_spaces = spaces,
    )
    state = train!(model, () -> tiny_aether_batch(K, seq_len, batch_size), train_cfg)

    @test length(state.losses) == 1
    @test isfinite(state.losses[1])
end
