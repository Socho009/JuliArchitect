#!/usr/bin/env julia

# Focused ablation entry point.
# This delegates to the clean rollout suite with smaller defaults that are
# convenient for quick iteration from the command line.

include(joinpath(@__DIR__, "benchmark_function_rollouts.jl"))

main(ARGS)
