#!/usr/bin/env julia
"""
Performance regression test.
Fails if model parameters or inference latency exceed acceptable thresholds.

Usage: julia --project scripts/test_regression.jl [fail_on_regression=true]

Exit code:
  0 — all metrics within thresholds
  1 — one or more metrics exceeded
"""

using Serialization

# Thresholds (from documented baselines in DESIGN.md)
THRESHOLDS = Dict(
    :small_params => 20_000,         # small model: 16.9K
    :default_params => 150_000,      # default: 133K
    :small_infer_p50_ms => 1.0,      # small: 0.63 ms
    :init_time_ms => 1200.0,         # all models: ~600-856 ms
)

fail_on_regression = length(ARGS) > 0 ? parse(Bool, ARGS[1]) : true

# Load minimal modules (avoid full Architect which requires Zygote)
using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "Core", "Core.jl"))
include(joinpath(@__DIR__, "..", "src", "Layers", "Layers.jl"))
include(joinpath(@__DIR__, "..", "src", "Models", "Models.jl"))
include(joinpath(@__DIR__, "..", "src", "Data", "Data.jl"))

failed = false

# Test 1: Small model (16.9K params)
cfg_small = Models.ContinuousConfig(basis_dim=4, n_embd=32, n_layer=1)
model_small = Models.ContinuousModel(cfg_small)
params_small = sum(length(pi.param) for pi in Models.collect_params(model_small))

if params_small > THRESHOLDS[:small_params]
    println("✗ FAIL: Small model params $(params_small) > $(THRESHOLDS[:small_params])")
    failed = true
else
    println("✓ PASS: Small model params $(params_small) ≤ $(THRESHOLDS[:small_params])")
end

# Test 2: Default model (133K params)
cfg_default = Models.ContinuousConfig(basis_dim=8, n_embd=64, n_layer=2)
model_default = Models.ContinuousModel(cfg_default)
params_default = sum(length(pi.param) for pi in Models.collect_params(model_default))

if params_default > THRESHOLDS[:default_params]
    println("✗ FAIL: Default model params $(params_default) > $(THRESHOLDS[:default_params])")
    failed = true
else
    println("✓ PASS: Default model params $(params_default) ≤ $(THRESHOLDS[:default_params])")
end

# Test 3: Small model inference latency
loader_small = Data.FunctionSequenceLoader{Float32}(basis_dim=4, seq_length=6, batch_size=8)
seed_small = rand(Float32, 4, 6)

# Warmup
_ = Models.generate(model_small, seed_small; max_new=2)

# Measure p50 (median)
durations = Float64[]
for _ in 1:20
    t0 = time()
    _ = Models.generate(model_small, seed_small; max_new=2)
    push!(durations, (time() - t0) * 1e3)  # ms
end
p50_latency = median(durations)

if p50_latency > THRESHOLDS[:small_infer_p50_ms]
    println("✗ FAIL: Small model p50 latency $(round(p50_latency, digits=3)) ms > $(THRESHOLDS[:small_infer_p50_ms]) ms")
    failed = true
else
    println("✓ PASS: Small model p50 latency $(round(p50_latency, digits=3)) ms ≤ $(THRESHOLDS[:small_infer_p50_ms]) ms")
end

# Test 4: Init time (both models)
# Note: skipped in this lightweight version to avoid Loading Training module
# See full benchmark_full.jl for detailed timing measurements

println()
if failed
    if fail_on_regression
        println("Performance regression detected. Exiting with status 1.")
        exit(1)
    else
        println("Performance regression detected, but fail_on_regression=false.")
    end
else
    println("All performance tests passed ✓")
    exit(0)
end
