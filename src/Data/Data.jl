module Data

# ─── Exports ────────────────────────────────────────────────────────────────
export CharTokenizer, BPETokenizer, encode, decode, bpe_merge
export CharDataLoader, next_batch!, make_batch_fn
export FunctionSequenceLoader, next_batch, make_functional_batch_fn
export differentiate_coeffs, antidifferentiate_coeffs, taylor_coefficients

# ─── Includes ───────────────────────────────────────────────────────────────
include("tokenizer.jl")
include("data_loader.jl")
include("functional_data.jl")

end # module Data
