"""
    module FAT

Functional Autoregressive Transformer — a Julia-native implementation of the
FAT architecture from Boularak (2026), "Rethinking Next-Token Prediction
through Continuous Functional Analysis".

This module provides:

**Phase 1 — Core Hilbert Space Abstractions** (`phase1_hilbert.jl`)
  - `AbstractFunctionalBasis`, `ChebyshevBasis{K}`
  - `SpectralState{T,K}` — element of V_K = span{T₀,…,T_K} ⊂ L²([-1,1],w)
  - `evaluate` (Clenshaw's algorithm), `project`, `inner_product`, `hilbert_norm`
  - `chebyshev_derivative_matrix` — exact symbolic differentiation on coefficients

**Phase 2 — Spectral Head** (`phase2_spectral_head.jl`)
  - `SpectralHead{T,K}` — linear map ℝ^{d_model} → ℝ^{K+1}
    replacing the discrete-vocabulary softmax head

**Phase 3 — Stabilization Pipeline** (`phase3_stabilizer.jl`)
  - `ChebyshevEconomizer{P}` — orthogonal truncation P_p in L²([-1,1],w)
  - `SpectralStabilizer{K_anchor,P_econ}` — corrected Π = P_p ∘ P_k
    with inline proof that L_Π ≤ 1 within a single Hilbert space

**Phase 4 — Directional Trajectory Approval Loss** (`phase4_loss.jl`)
  - `directional_trajectory_loss` — cosine-similarity loss in coefficient space
  - `approval_alpha_schedule` — warm-up schedule blending MSE → approval loss
  - `check_loss_gradient` — Zygote gradient sanity check

**Phase 5 — Full FAT Model** (`phase5_fat_model.jl`)
  - `FATConfig{T}` — unified configuration
  - `FunctionalAutoregressiveTransformer` — complete assembled model
  - `fat_forward` — pure differentiable forward pass
  - `fat_train!` — teacher-forced AdamW training loop with approval loss
  - `fat_generate` — autoregressive generation with stabilizer
"""
module FAT

using LinearAlgebra
using Statistics
using Printf
using Zygote

# Import from sibling submodules using explicit paths to avoid the Julia 1.12
# ambiguity between our Architect.Core and Julia's built-in Core module.
import ..Core
import ..Core: AbstractModelConfig
import ..Layers: AbstractLayer, AbstractAttention, Linear, Norm, Attention, Block, FeedForward
import ..Layers: has_bias, _batched_attention_scores, _apply_causal_mask, _softmax, _dropout
import ..Layers: _batched_attention_output, apply_position_encoding!

# ─── Exports ─────────────────────────────────────────────────────────────────

# Phase 1
export AbstractFunctionalBasis, ChebyshevBasis, basis_dim
export SpectralState, evaluate, project, project_coefficient
export inner_product, inner_product_exact
export gauss_chebyshev_nodes_weights, chebyshev_derivative_matrix
export hilbert_norm, hilbert_dist

# Phase 2
export SpectralHead, extract_state, n_params

# Phase 3
export ChebyshevEconomizer, SpectralStabilizer, verify_nonexpansive

# Phase 4
export directional_trajectory_loss, compute_approval_per_step
export approval_alpha_schedule, check_loss_gradient

# Phase 5 — Full FAT model
export FATConfig, FunctionalAutoregressiveTransformer
export fat_forward, fat_train!, fat_generate, adaptive_fat_generate, FATTrainConfig, FATTrainState
# Phase 6 — Continuous Data Streams
export ContinuousTimeDataLoader, next_batch, make_continuous_batch_fn
# Phase 7 — Multi-Head Latent Attention
export LatentAttention, has_latent_kv
# Phase 8 — Hardin-Taylor Minimum-Complexity Prediction
export spectral_complexity, mcp_loss, ClimbEvent, hardin_taylor_generate

# ─── Includes (in dependency order) ─────────────────────────────────────────
include("phase1_hilbert.jl")
include("phase2_spectral_head.jl")
include("phase3_stabilizer.jl")
include("phase4_loss.jl")
include("phase5_fat_model.jl")
include("phase6_data.jl")
include("phase7_mla.jl")
include("phase8_hardin_taylor.jl")

end # module FAT
