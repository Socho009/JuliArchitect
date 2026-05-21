module Core

using LinearAlgebra

# ─── Exports ────────────────────────────────────────────────────────────────
export AbstractModelConfig, AbstractLayer, AbstractAttention, AbstractBlock
export AbstractRecurrentCell, AbstractRecurrentState
export ModelConfig, MODEL_PRESETS
export to_device

# ─── Includes ───────────────────────────────────────────────────────────────
include("types.jl")
include("config.jl")
include("device.jl")

end # module Core
