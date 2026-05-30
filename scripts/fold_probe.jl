#!/usr/bin/env julia
# Simple probe script to collect per-head activations and produce similarity proposals

# Load project source directly (avoid requiring Pkg.develop)
include(joinpath(@__DIR__, "..", "src", "Architect.jl"))

# Load the folding prototype
include(joinpath(@__DIR__, "..", "src", "FAT", "folding.jl"))

println("Constructing small FAT model for probe...")
cfg = Architect.ContinuousConfig(basis_dim=8, n_embd=64, n_head=4, n_layer=2, block_size=16)
model = Architect.ContinuousModel(cfg)

# Optional: initialize parameters consistently
if isdefined(Architect, :initialize!)
    Architect.initialize!(model; muP=false)
end

loader = Architect.FunctionSequenceLoader(basis_dim=cfg.basis_dim, seq_length=cfg.block_size, batch_size=8, task=:derivative)

println("Collecting activations (8 batches)")
println("Loader type: ", typeof(loader))
println("Methods for Architect.Data.next_batch:")
println(methods(Architect.Data.next_batch))

collector = FATFolding.collect_head_activations(model, loader; nbatches=8)

println("Computing similarity (CKA)")
sim, keys = FATFolding.compute_similarity(collector; metric=:cka)

results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
    mkpath(results_dir)
end
outcsv = joinpath(results_dir, "head_sim.csv")
FATFolding.save_similarity_csv(sim, keys, outcsv)

clusters = FATFolding.propose_clusters(sim, keys; threshold=0.92f0)
println("Proposed clusters (layer,head):")
for c in clusters
    println(c)
end

println("Wrote similarity CSV to: ", outcsv)
