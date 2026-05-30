#!/usr/bin/env julia
# Simple probe script to collect per-head activations and produce similarity proposals

# Load project source directly (avoid requiring Pkg.develop)
include(joinpath(@__DIR__, "..", "src", "Architect.jl"))

using .Architect
using .Architect.FAT: ProbeCollector, TrackedAttention, collect_head_activations,
                      compute_similarity, propose_clusters, save_similarity_csv

println("Constructing small FAT model for probe...")
cfg = ContinuousConfig(basis_dim=8, n_embd=64, n_head=4, n_layer=2, block_size=16)
model = ContinuousModel(cfg)

# Optional: initialize parameters consistently
if isdefined(Architect, :initialize!)
    Architect.initialize!(model; muP=false)
end

loader = FunctionSequenceLoader(basis_dim=cfg.basis_dim, seq_length=cfg.block_size, batch_size=8, task=:derivative)

println("Collecting activations (8 batches)")
collector = collect_head_activations(model, loader, Architect.Data.next_batch; nbatches=8)
println("✓ Collected $(length(collector.data)) head activations")

println("Computing similarity (CKA)")
sim, keys = compute_similarity(collector; metric=:cka)

results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
    mkpath(results_dir)
end
outcsv = joinpath(results_dir, "head_sim.csv")
save_similarity_csv(sim, keys, outcsv)

clusters = propose_clusters(sim, keys; threshold=0.92f0)
println("Proposed clusters (layer,head):")
for c in clusters
    println(c)
end

println("Wrote similarity CSV to: ", outcsv)
