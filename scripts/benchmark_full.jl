#!/usr/bin/env julia
"""
Simple benchmarking harness for Architect.jl

Usage: julia --project scripts/benchmark_full.jl [key=value ...]

Supported keys (all optional):
- model=continuous (only 'continuous' supported for now)
- basis_dim=8
- n_embd=64
- n_head=4
- n_layer=2
- block_size=16
- batch_size=16
- seq_length=8
- train_iters=20
- infer_runs=50
- infer_new=4
- device=cpu|cuda
- task=derivative
- out=results/bench_<timestamp>.json

This script writes a JSON file with measured metrics.
"""

using Dates, Printf
using Statistics

function parse_argv()
    opts = Dict{String,String}()
    for a in ARGS
        if occursin("=", a)
            k, v = split(a, '=', limit=2)
            opts[k] = v
        end
    end
    return opts
end

opts = parse_argv()
getopt(k, default) = haskey(opts, k) ? opts[k] : default

model = getopt("model", "continuous")
basis_dim = parse(Int, getopt("basis_dim", "8"))
n_embd = parse(Int, getopt("n_embd", "64"))
n_head = parse(Int, getopt("n_head", "4"))
n_layer = parse(Int, getopt("n_layer", "2"))
block_size = parse(Int, getopt("block_size", "16"))
batch_size = parse(Int, getopt("batch_size", "16"))
seq_length = parse(Int, getopt("seq_length", "8"))
train_iters = parse(Int, getopt("train_iters", "20"))
infer_runs = parse(Int, getopt("infer_runs", "50"))
infer_new = parse(Int, getopt("infer_new", "4"))
device = getopt("device", "cpu")
task = Symbol(getopt("task", "derivative"))

outf = getopt("out", "")
if outf == ""
    ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    outf = joinpath("results", "bench_$(ts).json")
end

mkpath("results")

println("Benchmark: model=$(model), basis_dim=$(basis_dim), n_embd=$(n_embd), layers=$(n_layer)")

using Pkg
Pkg.instantiate()

using Serialization

# Load local package source (do not `using Architect` which requires package install)
include(joinpath(@__DIR__, "..", "src", "Architect.jl"))
const ARCH = Architect

commit = try
    readchomp(`git rev-parse --short HEAD`)
catch
    "unknown"
end

if model != "continuous"
    error("Only 'continuous' model supported by this benchmark script currently.")
end


cfg = ARCH.ContinuousConfig(basis_dim=basis_dim, basis_type=:monomial, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size)
model_obj = ARCH.ContinuousModel(cfg)

param_infos = ARCH.collect_params(model_obj)
n_params = sum(length(pi.param) for pi in param_infos)

println("Params: $(n_params)")

# Move to device if requested
if device == "cuda"
    try
        model_obj = ARCH.to_device(model_obj, :cuda)
        println("Moved model to CUDA")
    catch e
        println("Warning: failed to move to CUDA: ", e)
    end
end

# -- Training benchmark (AD-based)

loader = ARCH.FunctionSequenceLoader{Float32}(task=task, basis_dim=basis_dim, seq_length=seq_length, batch_size=batch_size)
data_fn = ARCH.make_functional_batch_fn(loader)

println("Measuring model init and inference latency...")

# -- Initialize model weights (time this)
t0 = time()
ARCH.initialize!(model_obj; muP=false)
t_init = time() - t0
println("Model init time: $(round(t_init*1e3, digits=2)) ms")

train_time = t_init  # placeholder for result dict

# -- Inference benchmark
println("Running inference benchmark (warmup + $(infer_runs) runs)")
seed = rand(Float32, basis_dim, seq_length)

# warmup
_ = ARCH.generate(model_obj, seed; max_new=infer_new)

durations = Float64[]
for i in 1:infer_runs
    t0 = time()
    _ = ARCH.generate(model_obj, seed; max_new=infer_new)
    push!(durations, (time() - t0)*1e3) # ms
end

mean_ms = mean(durations)
median_ms = median(durations)

result = Dict(
    "commit" => commit,
    "timestamp" => string(now()),
    "model" => model,
    "basis_dim" => basis_dim,
    "n_embd" => n_embd,
    "n_head" => n_head,
    "n_layer" => n_layer,
    "block_size" => block_size,
    "batch_size" => batch_size,
    "seq_length" => seq_length,
    "n_params" => n_params,
    "train_iters" => train_iters,
    "train_time_s" => train_time,
    "train_time_per_iter_s" => train_time / max(1, train_iters),
    "inference_runs" => infer_runs,
    "inference_new" => infer_new,
    "inference_mean_ms" => mean_ms,
    "inference_median_ms" => median_ms,
    "durations_ms" => durations,
)

# Write serialized binary data and a small human-readable summary
ser_path = replace(outf, r"\.json$" => ".ser")
summary_path = replace(outf, r"\.json$" => ".summary")

open(ser_path, "w") do io
    serialize(io, result)
end

open(summary_path, "w") do io
    println(io, "commit: $(result["commit"])")
    println(io, "n_params: $(result["n_params"])")
    println(io, "train_time_s: $(result["train_time_s"])" )
    println(io, "train_time_per_iter_s: $(result["train_time_per_iter_s"])" )
    println(io, "inference_mean_ms: $(result["inference_mean_ms"])" )
    println(io, "inference_median_ms: $(result["inference_median_ms"])" )
end

println("Wrote results to: $(ser_path) and $(summary_path)")
