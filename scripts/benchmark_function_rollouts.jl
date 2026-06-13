#!/usr/bin/env julia

# Clean rollout benchmark for FAT.
#
# Tasks:
#   derivative  - truncated Chebyshev derivative-like coefficient dynamics
#   oscillator  - paired coefficient rotations
#   heat        - diagonal heat-equation-like spectral decay
#
# Outputs:
#   results/rollout_suite/summary.csv
#   results/rollout_suite/<task>_rollout_errors.svg
#
# Compared variants:
#   FAT + approval loss + stabilizer
#   FAT + approval loss without stabilizer
#   FAT + MSE-only training + stabilizer
#   ContinuousModel baseline
#   persistence baseline
#   oracle linear dynamics baseline

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Architect
using JSON
using LinearAlgebra
using Printf
using Random
using Statistics

Base.@kwdef struct BenchConfig
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
    outdir::String = joinpath(@__DIR__, "..", "results", "rollout_suite")
end

function config_dict(cfg::BenchConfig)
    return Dict(String(name) => getfield(cfg, name) for name in fieldnames(BenchConfig))
end

function git_metadata()
    commit = try
        readchomp(`git rev-parse HEAD`)
    catch
        "unknown"
    end
    dirty = try
        !isempty(readchomp(`git status --short`))
    catch
        true
    end
    return Dict("commit" => commit, "dirty" => dirty)
end

function write_run_metadata(cfg::BenchConfig, args)
    open(joinpath(cfg.outdir, "config.json"), "w") do io
        JSON.print(io, Dict(
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

function parse_config(args)
    cfg = BenchConfig()
    values = Dict{Symbol,Any}(name => getfield(cfg, name) for name in fieldnames(BenchConfig))
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
    return BenchConfig(; values...)
end

function task_matrix(task::Symbol, K::Int, ::Type{T}) where {T<:AbstractFloat}
    d = K + 1
    if task == :derivative
        basis = ChebyshevBasis{K}()
        Dsmall = chebyshev_derivative_matrix(basis, T)
        D = zeros(T, d, d)
        D[1:K, :] .= Dsmall
        return T(0.92) .* Matrix{T}(I, d, d) .+ T(0.03) .* D
    elseif task == :oscillator
        M = Matrix{T}(I, d, d)
        for i in 1:2:d-1
            omega = T(0.10 + 0.015 * i)
            c = cos(omega)
            s = sin(omega)
            M[i, i] = c
            M[i, i+1] = -s
            M[i+1, i] = s
            M[i+1, i+1] = c
        end
        return T(0.995) .* M
    elseif task == :heat
        rates = T[exp(-T(0.055) * T(n)^2) for n in 0:K]
        return Diagonal(rates)
    else
        error("unknown task: $task")
    end
end

function random_state(K::Int, ::Type{T}) where {T<:AbstractFloat}
    coeffs = randn(T, K + 1)
    for n in 0:K
        coeffs[n+1] /= T(1 + n)^T(1.35)
    end
    return coeffs
end

function make_sequence(M, K::Int, total_steps::Int, ::Type{T}) where {T<:AbstractFloat}
    seq = zeros(T, K + 1, total_steps)
    seq[:, 1] .= random_state(K, T)
    for t in 2:total_steps
        seq[:, t] .= M * seq[:, t-1]
    end
    return seq
end

function make_batch_fn(task::Symbol, cfg::BenchConfig)
    M = task_matrix(task, cfg.K, Float32)
    return function ()
        all = Array{Float32}(undef, cfg.K + 1, cfg.seq_len + 1, cfg.batch_size)
        for b in 1:cfg.batch_size
            all[:, :, b] .= make_sequence(M, cfg.K, cfg.seq_len + 1, Float32)
        end
        return all[:, 1:cfg.seq_len, :], all[:, 2:cfg.seq_len+1, :]
    end
end

function train_fat(task::Symbol, cfg::BenchConfig; approval_weight::Float32)
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
    train_cfg = FATTrainConfig{Float32}(
        learning_rate = cfg.lr,
        max_iters = cfg.iters,
        warmup_iters = max(1, cfg.iters ÷ 2),
        log_interval = max(1, cfg.iters ÷ 4),
        approval_weight = approval_weight,
    )
    state = train!(model, make_batch_fn(task, cfg), train_cfg)
    return model, state
end

function train_continuous(task::Symbol, cfg::BenchConfig)
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
    state = train!(model, make_batch_fn(task, cfg), train_cfg)
    return model, state
end

function persistence_rollout(seed::AbstractMatrix{T}; max_new::Int) where {T<:AbstractFloat}
    sequence = copy(seed)
    last_state = seed[:, end]
    for _ in 1:max_new
        sequence = hcat(sequence, last_state)
    end
    return sequence
end

function linear_oracle_rollout(M, seed::AbstractMatrix{T}; max_new::Int) where {T<:AbstractFloat}
    sequence = copy(seed)
    current = seed[:, end]
    for _ in 1:max_new
        current = M * current
        sequence = hcat(sequence, current)
    end
    return sequence
end

function write_summary_header(path)
    open(path, "w") do io
        println(io, "task,variant,one_step_mse,rollout_mse,final_error,final_hilbert_error,mean_energy,mean_energy_drift,mean_contraction,mean_complexity,last_loss,seed,git_commit,git_dirty")
    end
end

function append_summary(path, cfg, task, variant, metrics, last_loss)
    meta = git_metadata()
    mean_contraction = isempty(metrics.stabilizer_contraction_ratios) ? NaN : mean(metrics.stabilizer_contraction_ratios)
    open(path, "a") do io
        @printf(io, "%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%s,%s\n",
                task, variant, metrics.one_step_mse, metrics.rollout_mse,
                metrics.final_error, metrics.final_hilbert_error,
                mean(metrics.spectral_energy), mean(metrics.spectral_energy_drift),
                mean_contraction, mean(metrics.complexity), last_loss,
                cfg.seed, meta["commit"], string(meta["dirty"]))
    end
end

function write_losses_csv(path, losses)
    open(path, "w") do io
        println(io, "iter,loss")
        for (i, loss) in enumerate(losses)
            @printf(io, "%d,%.9g\n", i, loss)
        end
    end
end

function write_rollout_sample_csv(path, predicted, truth)
    coeffs, steps = size(predicted)
    open(path, "w") do io
        truth_cols = ["truth_c$(i-1)" for i in 1:coeffs]
        pred_cols = ["pred_c$(i-1)" for i in 1:coeffs]
        println(io, "step," * join(vcat(truth_cols, pred_cols), ","))
        for t in 1:steps
            vals = vcat(truth[:, t], predicted[:, t])
            println(io, string(t) * "," * join(string.(vals), ","))
        end
    end
end

function write_metrics_json(path, metrics)
    open(path, "w") do io
        JSON.print(io, Dict(
            "one_step_mse" => metrics.one_step_mse,
            "rollout_mse" => metrics.rollout_mse,
            "final_error" => metrics.final_error,
            "final_hilbert_error" => metrics.final_hilbert_error,
            "step_errors" => metrics.step_errors,
            "hilbert_errors" => metrics.hilbert_errors,
            "spectral_energy" => metrics.spectral_energy,
            "spectral_energy_drift" => metrics.spectral_energy_drift,
            "stabilizer_contraction_ratios" => metrics.stabilizer_contraction_ratios,
            "complexity" => metrics.complexity,
            "error_growth_ratios" => metrics.error_growth_ratios,
        ), 2)
        println(io)
    end
end

function write_error_csv(path, series::Dict{String,Vector{Float32}})
    labels = sort(collect(keys(series)))
    maxlen = maximum(length(series[label]) for label in labels)
    open(path, "w") do io
        println(io, "step," * join(labels, ","))
        for i in 1:maxlen
            vals = String[]
            for label in labels
                v = series[label]
                push!(vals, i <= length(v) ? string(v[i]) : "")
            end
            println(io, string(i) * "," * join(vals, ","))
        end
    end
end

function svg_polyline(values, xmin, xmax, ymin, ymax, width, height, color)
    n = length(values)
    n == 0 && return ""
    pts = String[]
    for (i, y) in enumerate(values)
        xpix = 50 + (i - xmin) / max(1, xmax - xmin) * (width - 80)
        ypix = 20 + (ymax - y) / max(eps(Float64), ymax - ymin) * (height - 60)
        push!(pts, @sprintf("%.2f,%.2f", xpix, ypix))
    end
    return "<polyline fill=\"none\" stroke=\"$color\" stroke-width=\"2\" points=\"" * join(pts, " ") * "\" />"
end

function write_svg_plot(path, title, series::Dict{String,Vector{Float32}})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#111111", "#17becf"]
    width, height = 760, 420
    allvals = reduce(vcat, values(series))
    ymin = 0.0
    ymax = max(maximum(Float64.(allvals)), 1e-8)
    xmax = maximum(length(v) for v in values(series))
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
        println(io, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>")
        println(io, "<text x=\"50\" y=\"18\" font-family=\"sans-serif\" font-size=\"15\">$title</text>")
        println(io, "<line x1=\"50\" y1=\"360\" x2=\"730\" y2=\"360\" stroke=\"#333\"/>")
        println(io, "<line x1=\"50\" y1=\"20\" x2=\"50\" y2=\"360\" stroke=\"#333\"/>")
        for (idx, label) in enumerate(sort(collect(keys(series))))
            println(io, svg_polyline(Float64.(series[label]), 1, xmax, ymin, ymax, width, height, colors[idx]))
            yleg = 38 + 18 * idx
            println(io, "<line x1=\"560\" y1=\"$yleg\" x2=\"590\" y2=\"$yleg\" stroke=\"$(colors[idx])\" stroke-width=\"2\"/>")
            println(io, "<text x=\"598\" y=\"$(yleg + 4)\" font-family=\"sans-serif\" font-size=\"12\">$label</text>")
        end
        println(io, "<text x=\"350\" y=\"398\" font-family=\"sans-serif\" font-size=\"12\">rollout step</text>")
        println(io, "<text x=\"6\" y=\"205\" transform=\"rotate(-90 12,205)\" font-family=\"sans-serif\" font-size=\"12\">coefficient error</text>")
        println(io, "</svg>")
    end
end

function run_task(task::Symbol, cfg::BenchConfig, summary_path)
    println("\n=== task: $task ===")
    task_seed = cfg.seed + abs(hash(task) % 10_000)

    Random.seed!(task_seed + 101)
    full_model, full_state = train_fat(task, cfg; approval_weight=1.0f0)
    Random.seed!(task_seed + 202)
    mse_model, mse_state = train_fat(task, cfg; approval_weight=0.0f0)
    Random.seed!(task_seed + 303)
    continuous_model, continuous_state = train_continuous(task, cfg)

    write_losses_csv(joinpath(cfg.outdir, "$(task)_fat_approval_losses.csv"), full_state.losses)
    write_losses_csv(joinpath(cfg.outdir, "$(task)_fat_mse_losses.csv"), mse_state.losses)
    write_losses_csv(joinpath(cfg.outdir, "$(task)_continuous_losses.csv"), continuous_state.losses)

    Random.seed!(task_seed + 404)
    M = task_matrix(task, cfg.K, Float32)
    truth = make_sequence(M, cfg.K, cfg.seed_len + cfg.horizon, Float32)
    seed = truth[:, 1:cfg.seed_len]

    p_full_stab, m_full_stab = evaluate_rollout(full_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=true)
    p_full_raw, m_full_raw = evaluate_rollout(full_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=false)
    p_mse_stab, m_mse_stab = evaluate_rollout(mse_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=true)
    p_continuous, m_continuous = evaluate_rollout(continuous_model, truth; seed_len=cfg.seed_len, max_new=cfg.horizon, apply_stabilizer=false)
    p_persistence = persistence_rollout(seed; max_new=cfg.horizon)
    m_persistence = rollout_metrics(p_persistence, truth; seed_len=cfg.seed_len)
    p_oracle = linear_oracle_rollout(M, seed; max_new=cfg.horizon)
    m_oracle = rollout_metrics(p_oracle, truth; seed_len=cfg.seed_len)

    append_summary(summary_path, cfg, task, "fat_approval_stabilized", m_full_stab, full_state.losses[end])
    append_summary(summary_path, cfg, task, "fat_approval_unstabilized", m_full_raw, full_state.losses[end])
    append_summary(summary_path, cfg, task, "fat_mse_stabilized", m_mse_stab, mse_state.losses[end])
    append_summary(summary_path, cfg, task, "continuous_transformer", m_continuous, continuous_state.losses[end])
    append_summary(summary_path, cfg, task, "persistence", m_persistence, NaN)
    append_summary(summary_path, cfg, task, "oracle_linear", m_oracle, NaN)

    variants = Dict(
        "fat_approval_stabilized" => (p_full_stab, m_full_stab),
        "fat_approval_unstabilized" => (p_full_raw, m_full_raw),
        "fat_mse_stabilized" => (p_mse_stab, m_mse_stab),
        "continuous_transformer" => (p_continuous, m_continuous),
        "persistence" => (p_persistence, m_persistence),
        "oracle_linear" => (p_oracle, m_oracle),
    )
    for (variant, (predicted, metrics)) in variants
        write_rollout_sample_csv(joinpath(cfg.outdir, "$(task)_$(variant)_rollout_sample.csv"),
                                 predicted, truth[:, 1:size(predicted, 2)])
        write_metrics_json(joinpath(cfg.outdir, "$(task)_$(variant)_metrics.json"), metrics)
    end

    series = Dict(
        "approval + stabilizer" => m_full_stab.step_errors,
        "approval no stabilizer" => m_full_raw.step_errors,
        "mse + stabilizer" => m_mse_stab.step_errors,
        "continuous transformer" => m_continuous.step_errors,
        "persistence" => m_persistence.step_errors,
        "oracle linear" => m_oracle.step_errors,
    )
    write_error_csv(joinpath(cfg.outdir, "$(task)_rollout_errors.csv"), series)
    write_svg_plot(joinpath(cfg.outdir, "$(task)_rollout_errors.svg"),
                   "Rollout error: $task", series)

    @printf("  stabilized final error:   %.6g\n", m_full_stab.final_error)
    @printf("  unstabilized final error: %.6g\n", m_full_raw.final_error)
    @printf("  mse-only final error:     %.6g\n", m_mse_stab.final_error)
    @printf("  continuous final error:   %.6g\n", m_continuous.final_error)
    @printf("  persistence final error:  %.6g\n", m_persistence.final_error)
    @printf("  oracle final error:       %.6g\n", m_oracle.final_error)
end

function main(args=ARGS)
    cfg = parse_config(args)
    mkpath(cfg.outdir)
    write_run_metadata(cfg, args)
    summary_path = joinpath(cfg.outdir, "summary.csv")
    write_summary_header(summary_path)

    for task in (:derivative, :oscillator, :heat)
        run_task(task, cfg, summary_path)
    end

    println("\nWrote rollout suite results to $(cfg.outdir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
