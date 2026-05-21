#!/usr/bin/env julia
"""
Compare two benchmark JSON result files and report percent changes.

Usage: julia --project scripts/compare_results.jl base.json head.json
"""

using Printf

if length(ARGS) < 2
    println("Usage: julia --project scripts/compare_results.jl base.summary head.summary")
    exit(1)
end

function read_summary(path)
    d = Dict{String,Float64}()
    for line in readlines(path)
        if isempty(strip(line))
            continue
        end
        k, v = split(line, ':', limit=2)
        k = strip(k)
        v = strip(v)
        try
            d[k] = parse(Float64, v)
        catch
            # not numeric, skip
        end
    end
    return d
end

base = read_summary(ARGS[1])
head = read_summary(ARGS[2])

function pct_change(old, new)
    if old == 0
        return new == 0 ? 0.0 : Inf
    end
    return (new - old) / abs(old) * 100.0
end

println("Comparison")
println("---------")
println(@sprintf("params: %d -> %d  (change: %+.2f%%)", base["n_params"], head["n_params"], pct_change(base["n_params"], head["n_params"])))
println(@sprintf("train time (s): %.4f -> %.4f  (change: %+.2f%%)", base["train_time_s"], head["train_time_s"], pct_change(base["train_time_s"], head["train_time_s"])))
println(@sprintf("train time / iter (s): %.6f -> %.6f  (change: %+.2f%%)", base["train_time_per_iter_s"], head["train_time_per_iter_s"], pct_change(base["train_time_per_iter_s"], head["train_time_per_iter_s"])))
println(@sprintf("inference mean (ms): %.4f -> %.4f  (change: %+.2f%%)", base["inference_mean_ms"], head["inference_mean_ms"], pct_change(base["inference_mean_ms"], head["inference_mean_ms"])))
println(@sprintf("inference median (ms): %.4f -> %.4f  (change: %+.2f%%)", base["inference_median_ms"], head["inference_median_ms"], pct_change(base["inference_median_ms"], head["inference_median_ms"])))
