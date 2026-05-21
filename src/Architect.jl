module Architect

# ─── Submodule includes ───────────────────────────────────────────────────────
# Each include() defines the submodule before its `using .X` can resolve.
# Dependency order: Core → Layers → Models/Data → Training → FAT
include("Core/Core.jl")
include("Layers/Layers.jl")
include("Models/Models.jl")
include("Data/Data.jl")
include("Training/Training.jl")
include("FAT/FAT.jl")

# ─── Submodule usings ────────────────────────────────────────────────────────
using .Core
using .Layers
using .Models
using .Data
using .Training
using .FAT

# ─── Global Exports ───────────────────────────────────────────────────────────

# Re-exporting from Core
export AbstractModelConfig, AbstractLayer, AbstractAttention, AbstractBlock
export AbstractRecurrentCell, AbstractRecurrentState
export ModelConfig, MODEL_PRESETS
export to_device

# Re-exporting from Layers
export Embedding, Norm, Linear, OrthogonalLinear, has_bias, orthogonal_weight
export Attention
export FeedForward, Block
export RNNCell, LSTMCell, GRUCell

# Re-exporting from Models
export ContinuousConfig, ContinuousModel, evaluate_function
export SequenceConfig, SequenceModel
export Llama2Config, Llama2Weights, Llama2State, Llama2Transformer, load_llama2_checkpoint

# Re-exporting from Data
export CharTokenizer, BPETokenizer, encode, decode, bpe_merge
export CharDataLoader, next_batch!, make_batch_fn
export FunctionSequenceLoader, next_batch, make_functional_batch_fn
export differentiate_coeffs, antidifferentiate_coeffs, taylor_coefficients

# Re-exporting from Training
export TrainConfig, train!
export ParamGroup, DecayGroup, NoDecayGroup, param_group
export collect_params, TrainState
export initialize!
export SamplingStrategy, GreedySampling, TopKSampling, NucleusSampling
export generate, sample_next_token
export compute_recurrent_spectral_stats

# Re-exporting from FAT
# Phase 1 — Hilbert space abstractions
export AbstractFunctionalBasis, ChebyshevBasis, basis_dim
export SpectralState, evaluate, project, project_coefficient
export inner_product, inner_product_exact
export gauss_chebyshev_nodes_weights, chebyshev_derivative_matrix
export hilbert_norm, hilbert_dist
# Phase 2 — Spectral head
export SpectralHead, extract_state
# Phase 3 — Stabilization pipeline
export ChebyshevEconomizer, SpectralStabilizer, verify_nonexpansive
# Phase 4 — Directional Trajectory Approval loss
export directional_trajectory_loss, compute_approval_per_step
export approval_alpha_schedule, check_loss_gradient
# Phase 5 — Full FAT model
export FATConfig, FunctionalAutoregressiveTransformer
export fat_forward, fat_train!, fat_generate, adaptive_fat_generate, FATTrainConfig, FATTrainState
# Phase 6 — Continuous Data Streams
export ContinuousTimeDataLoader, make_continuous_batch_fn

end # module Architect
