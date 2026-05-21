module Training

using ..Core
using ..Layers
using ..Models
using ..Data
using LinearAlgebra
using Random
using Statistics

# ─── Exports ────────────────────────────────────────────────────────────────
export TrainConfig, train!
export ParamGroup, DecayGroup, NoDecayGroup, param_group
export collect_params, TrainState
export compute_recurrent_spectral_stats
export initialize!

# ─── Includes ───────────────────────────────────────────────────────────────
include("train.jl")
include("functional_train.jl")
include("init.jl")

end # module Training
