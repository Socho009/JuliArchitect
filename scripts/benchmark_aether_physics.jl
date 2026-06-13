#!/usr/bin/env julia

# Paper-oriented AetherSpaces physics benchmark.
#
# This script compares stabilized FAT, unstabilized FAT, Aether-regularized FAT,
# ContinuousModel, persistence, and linear-oracle baselines on physics-inspired
# coefficient dynamics. Outputs are written under results/aether_physics/.

include(joinpath(@__DIR__, "benchmark_function_rollouts.jl"))

Base.@kwdef struct AetherBenchConfig
    K::Int = 9
    seq_len::Int = 16
    seed_len::Int = 8
    horizon::Int = 32
    batch_size::Int = 16
    n_embd::Int = 48
    n_head::Int = 4
    n_layer::Int = 2
    iters::Int = 120
    lr::Float32 = 6f-4
    seed::Int = 42
    aether_lambda::Float32 = 1f-3
    tasks::String = "harmonic,damped,heat,wave,fractional"
    outdir::String = joinpath(@__DIR__, "..", "results", "aether_physics")
end

function config_dict(cfg::AetherBenchConfig)
    return Dict(String(name) => getfield(cfg, name) for name in fieldnames(AetherBenchConfig))
end

function parse_aether_config(args)
    cfg = AetherBenchConfig()
    values = Dict{Symbol,Any}(name => getfield(cfg, name) for name in fieldnames(AetherBenchConfig))
    for arg in args
        occursin("=", arg) || continue
        key, value = split(arg, "=", limit=2)
        sym = Symbol(key)
        haskey(values, sym) || continue
        old = values[sym]
        values[sym] = old isa Int ? parse(Int, value) :
                      old isa Float32 ? Float32(parse(Float64, value)) :
                      value
    end
    return AetherBenchConfig(; values...)
end

function write_aether_run_metadata(cfg::AetherBenchConfig, args)
    open(joinpath(cfg.outdir, "config.json"), "w") do io
        JSON.print(io, Dict(
            "benchmark" => "aether_physics",
            "config" => config_dict(cfg),
            "args" => collect(args),
            "git" => git_metadata(),
        ), 2)
        println(io)
    end
    open(joinpath(cfg.outdir, "git_commit.txt"), "w") do io
        meta = git_metadata()
        println(io, meta["commit"])
        println(io, "dirty=$(meta["dirty"])")
    end
end

function aether_task_matrix(task::Symbol, K::Int, ::Type{T}) where {T<:AbstractFloat}
    d = K + 1
    if task == :harmonic
        M = Matrix{T}(I, d, d)
        for i in 1:2:d-1
            omega = T(0.08 + 0.012 * i)
            c = cos(omega)
            s = sin(omega)
            M[i, i] = c
            M[i, i+1] = -s
            M[i+1, i] = s
            M[i+1, i+1] = c
        end
        return M
    elseif task == :damped
        return T(0.975) .* aether_task_matrix(:harmonic, K, T)
    elseif task == :heat
        rates = T[exp(-T(0.055) * T(n)^2) for n in 0:K]
        return Diagonal(rates)
    elseif task == :wave
        M = Matrix{T}(I, d, d)
        for i in 2:2:d-1
            omega = T(0.04 * sqrt(i))
            c = cos(omega)
            s = sin(omega)
            M[i, i] = c
            M[i, i+1] = -s
            M[i+1, i] = s
            M[i+1, i+1] = c
        end
        return T(0.998) .* M
    elseif task == :fractional
        rates = T[exp(-T(0.035) * T(n + 1)^T(1.35)) for n in 0:K]
        return Diagonal(rates)
    else
        error("unknown Aether physics task: $task")
    end
end

function aether_make_batch_fn(task::Symbol, cfg::AetherBenchConfig)
    M = aether_task_matrix(task, cfg.K, Float32)
    return function ()
        all = Array{Float32}(undef, cfg.K + 1, cfg.seq_len + 1, cfg.batch_size)
        for b in 1:cfg.batch_size
            all[:, :, b] .= make_sequence(M, cfg.K, cfg.seq_len + 1, Float32)
        end
        return all[:, 1:cfg.seq_len, :], all[:, 2:cfg.seq_len+1, :]
    end
end

function train_aether_fat(task::Symbol, cfg::AetherBenchConfig; aether::Bool)
    model_cfg = FATConfig{Float32}(
        K = cfg.K,
        K_anchor = cfg.K,
        P_econ = max(1, cfg.K ÷ 2),
        n_embd = cfg.n_embd,
        n_head = cfg.n_head,
        n_layer = cfg.n_layer,
        block_size = cfg.seq_len,
        dropout = 0.0f0,
    )
    model = FunctionalAutoregressiveTransformer(model_cfg)
    spaces = aether ? stable_aether_spaces(Float32; K=cfg.K) : ()
    train_cfg = FATTrainConfig{Float32}(
        learning_rate = cfg.lr,
        max_iters = cfg.iters,
        warmup_iters = max(1, cfg.iters ÷ 2),
        log_interval = max(1, cfg.iters ÷ 4),
        approval_weight = 1.0f0,
        aether_lambda = aether ? cfg.aether_lambda : 0.0f0,
        aether_spaces = spaces,
    )
    state = train!(model, aether_make_batch_fn(task, cfg), train_cfg)
    return model, state
end

function train_aether_continuous(task::Symbol, cfg::AetherBenchConfig)
    model_cfg = ContinuousConfig{Float32}(
        basis_dim = cfg.K + 1,
        basis_type = :chebyshev,
        n_embd = cfg.n_embd,
        n_head = cfg.n_head,
        n_layer = cfg.n_layer,
        block_size = cfg.seq_len,
        dropout = 0.0f0,
    )
    model = ContinuousModel(model_cfg)
    train_cfg = ContinuousTrainConfig{Float32}(
        learning_rate = cfg.lr,
        max_iters = cfg.iters,
        log_interval = max(1, cfg.iters ÷ 4),
    )
    state = train!(model, aether_make_batch_fn(task, cfg), train_cfg)
    return model, state
end

json_value(x::Real) = Float64(x)
json_value(x::AbstractString) = x
json_value(x::Bool) = x
json_value(x::AbstractArray) = [json_value(v) for v in x]
json_value(x::AbstractDict) = Dict(string(k) => json_value(v) for (k, v) in x)
json_value(x) = x

function aether_metric_summary(predicted, truth, seed_len::Int, spaces)
    pred_future = predicted[:, seed_len+1:end]
    truth_future = truth[:, seed_len+1:size(predicted, 2)]
    metrics = space_stack_metrics(pred_future, truth_future, spaces)
    distances = Float64[]
    regularizers = Float64[]
    for (key, value) in metrics
        endswith(key, "_distance") && push!(distances, Float64(value))
        endswith(key, "_regularizer") && push!(regularizers, Float64(value))
    end
    return Dict(
        "mean_distance" => isempty(distances) ? NaN : mean(distances),
        "mean_regularizer" => isempty(regularizers) ? NaN : mean(regularizers),
        "stack_regularizer" => Float64(space_stack_regularizer(pred_future, spaces)),
        "per_space" => json_value(metrics),
    )
end

function write_aether_metrics_json(path, predicted, truth, seed_len::Int, spaces)
    open(path, "w") do io
        JSON.print(io, json_value(aether_metric_summary(predicted, truth, seed_len, spaces)), 2)
        println(io)
    end
end

function write_aether_summary_header(path)
    open(path, "w") do io
        println(io, "task,variant,one_step_mse,rollout_mse,final_error,final_hilbert_error,mean_energy,mean_energy_drift,mean_contraction,mean_complexity,mean_aether_distance,aether_regularizer,last_loss,seed,git_commit,git_dirty")
    end
end

function append_aether_summary(path, cfg, task, variant, predicted, truth, metrics, last_loss, spaces)
    meta = git_metadata()
    mean_contraction = isempty(metrics.stabilizer_contraction_ratios) ? NaN : mean(metrics.stabilizer_contraction_ratios)
    aether = aether_metric_summary(predicted, truth, cfg.seed_len, spaces)
    open(path, "a") do io
        @printf(io, "%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%s,%s\n",
                task, variant, metrics.one_step_mse, metrics.rollout_mse,
                metrics.final_error, metrics.final_hilbert_error,
                mean(metrics.spectral_energy), mean(metrics.spectral_energy_drift),
                mean_contraction, mean(metrics.complexity),
                aether["mean_distance"], aether["stack_regularizer"], last_loss,
                cfg.seed, meta["commit"], string(meta["dirty"]))
    end
end

function run_aether_task(task::Symbol, cfg::AetherBenchConfig, summary_path)
    println("\n=== Aether physics task: $task ===")
    task_seed = cfg.seed + abs(hash(task) % 10_000)

    Random.seed!(task_seed + 101)
    fat_model, fat_state = train_aether_fat(task, cfg; aether=false)
    Random.seed!(task_seed + 202)
    aether_model, aether_state = train_aether_fat(task, cfg; aether=true)
    Random.seed!(task_seed + 303)
    continuous_model, continuous_state = train_aether_continuous(task, cfg)

    write_losses_csv(joinpath(cfg.outdir, "$(task)_fat_losses.csv"), fat_state.losses)
    write_losses_csv(joinpath(cfg.outdir, "$(task)_fat_aether_losses.csv"), aether_state.losses)
    write_losses_csv(joinpath(cfg.outdir, "$(task)_continuous_losses.csv"), continuous_state.losses)

    Random.seed!(task_seed + 404)
    M = aether_task_matrix(task, cfg.K, Float32)
    truth = make_sequence(M, cfg.K, cfg.seed_len + cfg.horizon, Float32)
    seed = truth[:, 1:cfg.seed_len]

    p_fat_stab, m_fat_stab = evaluate_rollout(fat_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=true)
    p_fat_raw, m_fat_raw = evaluate_rollout(fat_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=false)
    p_aether_stab, m_aether_stab = evaluate_rollout(aether_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=true)
    p_aether_raw, m_aether_raw = evaluate_rollout(aether_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=false)
    p_continuous, m_continuous = evaluate_rollout(continuous_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=false)
    p_persistence = persistence_rollout(seed; max_new=cfg.horizon)
    m_persistence = rollout_metrics(p_persistence, truth; seed_len=cfg.seed_len)
    p_oracle = linear_oracle_rollout(M, seed; max_new=cfg.horizon)
    m_oracle = rollout_metrics(p_oracle, truth; seed_len=cfg.seed_len)

    metric_spaces = diagnostic_aether_spaces(Float32; K=cfg.K)
    variants = Dict(
        "fat_stabilized" => (p_fat_stab, m_fat_stab, fat_state.losses[end]),
        "fat_unstabilized" => (p_fat_raw, m_fat_raw, fat_state.losses[end]),
        "fat_aether_stabilized" => (p_aether_stab, m_aether_stab, aether_state.losses[end]),
        "fat_aether_unstabilized" => (p_aether_raw, m_aether_raw, aether_state.losses[end]),
        "continuous_transformer" => (p_continuous, m_continuous, continuous_state.losses[end]),
        "persistence" => (p_persistence, m_persistence, NaN),
        "oracle_linear" => (p_oracle, m_oracle, NaN),
    )

    for (variant, (predicted, metrics, last_loss)) in variants
        append_aether_summary(summary_path, cfg, task, variant, predicted, truth, metrics, last_loss, metric_spaces)
        write_rollout_sample_csv(joinpath(cfg.outdir, "$(task)_$(variant)_rollout_sample.csv"),
                                 predicted, truth[:, 1:size(predicted, 2)])
        write_metrics_json(joinpath(cfg.outdir, "$(task)_$(variant)_metrics.json"), metrics)
        write_aether_metrics_json(joinpath(cfg.outdir, "$(task)_$(variant)_aether_metrics.json"),
                                  predicted, truth, cfg.seed_len, metric_spaces)
    end

    series = Dict(
        "FAT + stabilizer" => m_fat_stab.step_errors,
        "FAT no stabilizer" => m_fat_raw.step_errors,
        "Aether + stabilizer" => m_aether_stab.step_errors,
        "Aether no stabilizer" => m_aether_raw.step_errors,
        "continuous" => m_continuous.step_errors,
        "persistence" => m_persistence.step_errors,
        "oracle" => m_oracle.step_errors,
    )
    write_error_csv(joinpath(cfg.outdir, "$(task)_rollout_errors.csv"), series)
    write_svg_plot(joinpath(cfg.outdir, "$(task)_rollout_errors.svg"),
                   "Aether physics rollout error: $task", series)

    @printf("  FAT stabilized final error:       %.6g\n", m_fat_stab.final_error)
    @printf("  FAT unstabilized final error:     %.6g\n", m_fat_raw.final_error)
    @printf("  Aether stabilized final error:    %.6g\n", m_aether_stab.final_error)
    @printf("  Aether unstabilized final error:  %.6g\n", m_aether_raw.final_error)
    @printf("  continuous final error:           %.6g\n", m_continuous.final_error)
    @printf("  persistence final error:          %.6g\n", m_persistence.final_error)
    @printf("  oracle final error:               %.6g\n", m_oracle.final_error)
end

function main(args=ARGS)
    cfg = parse_aether_config(args)
    mkpath(cfg.outdir)
    write_aether_run_metadata(cfg, args)
    summary_path = joinpath(cfg.outdir, "summary.csv")
    write_aether_summary_header(summary_path)

    tasks = Symbol.(strip.(split(cfg.tasks, ",")))
    for task in tasks
        run_aether_task(task, cfg, summary_path)
    end

    println("\nWrote Aether physics results to $(cfg.outdir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
