module Models

using ..Core
using ..Layers
using LinearAlgebra

# ─── Exports ────────────────────────────────────────────────────────────────
export ContinuousConfig, ContinuousModel, evaluate_function, forward
export SequenceConfig, SequenceModel
export Llama2Config, Llama2Weights, Llama2State, Llama2Transformer, load_llama2_checkpoint
export SamplingStrategy, GreedySampling, TopKSampling, NucleusSampling
export generate, sample_next_token

# ─── Includes ───────────────────────────────────────────────────────────────
include("generate.jl")
include("continuous_model.jl")
include("sequence_model.jl")
include("llama2_inference.jl")

end # module Models
