# ─── Weight Initialization via Multiple Dispatch ──────────────────────────────
#
# Uses Julia's multiple dispatch to select the correct initialization for each
# layer type — replacing Python's isinstance() pattern with the type system.
# The `!` convention indicates mutation (modifying weights in-place).

"""
    initialize!(layer::Embedding{P}; muP::Bool=false) where P

Initialize an Embedding layer. By default uses Normal(0, 0.02).
If `muP=true` use µP-style initialization with std = 1/sqrt(n_embd).
"""
function initialize!(layer::Embedding{P}; muP::Bool=false) where P
    randn!(layer.weight)
    if muP
        n_embd = size(layer.weight, 1)
        layer.weight .*= P(1.0 / sqrt(n_embd))
    else
        layer.weight .*= P(0.02)
    end
    return layer
end

"""
    initialize!(layer::Norm{P}) where P

Initialize a Norm (RMSNorm) layer: weights = 1.0.
"""
function initialize!(layer::Norm{P}) where P
    layer.weight .= one(P)
    return layer
end

"""
    initialize!(layer::Linear{P, HB}; std::Real=0.02) where {P, HB}

Initialize a Linear layer with Normal(0, std) weights and zero biases.
The `std` keyword allows scaled initialization for output projections.
"""
function initialize!(layer::Linear{P, HB}; std::Real=0.02, muP::Bool=false) where {P, HB}
    randn!(layer.weight)
    if muP
        in_features = size(layer.weight, 2)
        layer.weight .*= P(1.0 / sqrt(in_features))
    else
        layer.weight .*= P(std)
    end
    layer.bias .= zero(P)
    return layer
end


"""
    initialize!(layer::OrthogonalLinear{P, HB}; init_scale=1e-3, muP::Bool=false)

Initialize a OrthogonalLinear layer's skew-symmetric parameter. For µP-style
initialization the scale is set to `1/sqrt(n)` where `n` is the square dim.
"""
function initialize!(layer::OrthogonalLinear{P, HB}; init_scale::Real=1e-3, muP::Bool=false) where {P, HB}
    n = size(layer.A, 1)
    R = randn(P, n, n)
    S = 0.5 .* (R .- R')
    if muP
        layer.A .= S .* P(1.0 / sqrt(n))
    else
        layer.A .= S .* P(init_scale)
    end
    layer.bias .= zero(P)
    return layer
end

"""
    initialize!(layer::Attention{P}; n_layers::Int) where P

Initialize a Attention layer:
- W_qkv (Q/K/V projection): Normal(0, 0.02), biases = 0
- W_out (output projection): Normal(0, 0.02/sqrt(2*n_layers)), biases = 0

The output projection uses scaled initialization to prevent residual stream growth
as depth increases (per GPT-2 paper).
"""
function initialize!(layer::Attention{P}; n_layers::Int, muP::Bool=false) where P
    if muP
        # µP-style for Q/K/V projection (combined or separate)
        if layer.W_qkv !== nothing
            initialize!(layer.W_qkv; muP=true)
        else
            initialize!(layer.W_q; muP=true)
            initialize!(layer.W_k; muP=true)
            initialize!(layer.W_v; muP=true)
        end

        # scaled output projection std based on fan-in and depth
        # handle Linear and OrthogonalLinear differently
        if isa(layer.W_out, Linear)
            fan_in = size(layer.W_out.weight, 2)
            scaled_std = 1.0 / sqrt(fan_in) / sqrt(2 * n_layers)
            initialize!(layer.W_out; std=scaled_std, muP=false)
        elseif isa(layer.W_out, OrthogonalLinear)
            n = size(layer.W_out.A, 1)
            scaled_init = 1.0 / sqrt(n) / sqrt(2 * n_layers)
            initialize!(layer.W_out; init_scale=scaled_init, muP=false)
        else
            try
                initialize!(layer.W_out; muP=false)
            catch
            end
        end
    else
        # Standard init for Q,K,V projection (combined or separate)
        if layer.W_qkv !== nothing
            initialize!(layer.W_qkv; std=0.02)
        else
            initialize!(layer.W_q; std=0.02)
            initialize!(layer.W_k; std=0.02)
            initialize!(layer.W_v; std=0.02)
        end

        # Scaled init for output projection
        if isa(layer.W_out, Linear)
            scaled_std = 0.02 / sqrt(2 * n_layers)
            initialize!(layer.W_out; std=scaled_std)
        elseif isa(layer.W_out, OrthogonalLinear)
            init_scale = 0.02 / sqrt(2 * n_layers)
            initialize!(layer.W_out; init_scale=init_scale)
        else
            try
                initialize!(layer.W_out; std=0.02)
            catch
            end
        end
    end

    return layer
end

"""
    initialize!(layer::FeedForward{P}; n_layers::Int) where P

Initialize a FeedForward (SwiGLU) layer:
- W1, W3 (gate/up): Normal(0, 0.02), no bias
- W2 (down/output projection): Normal(0, 0.02/sqrt(2*n_layers)), no bias
"""
function initialize!(layer::FeedForward{P}; n_layers::Int, muP::Bool=false) where P
    if muP
        fan_in1 = size(layer.W1.weight, 2)
        layer.W1.weight .*= P(1.0 / sqrt(fan_in1))
        layer.W3.weight .*= P(1.0 / sqrt(fan_in1))
        fan_in2 = size(layer.W2.weight, 2)
        scaled_std = 1.0 / sqrt(fan_in2) / sqrt(2 * n_layers)
        randn!(layer.W2.weight)
        layer.W2.weight .*= P(scaled_std)
    else
        randn!(layer.W1.weight); layer.W1.weight .*= P(0.02)
        randn!(layer.W3.weight); layer.W3.weight .*= P(0.02)
        scaled_std = 0.02 / sqrt(2 * n_layers)
        randn!(layer.W2.weight); layer.W2.weight .*= P(scaled_std)
    end
    return layer
end

"""
    initialize!(block::Block{P}; n_layers::Int) where P

Initialize a Block by dispatching to each sub-layer:
- Norm layers: weights = 1.0
- Attention: Q/K/V Normal(0, 0.02), output scaled by 1/sqrt(2*n_layers)
- FeedForward: W1/W3 Normal(0, 0.02), W2 scaled by 1/sqrt(2*n_layers)
"""
function initialize!(block::Block{P}; n_layers::Int, muP::Bool=false) where P
    initialize!(block.ln1)
    initialize!(block.attention; n_layers=n_layers, muP=muP)
    initialize!(block.ln2)
    initialize!(block.feedforward; n_layers=n_layers, muP=muP)
    return block
end


# ─── Recurrent Cell Initialization ────────────────────────────────────────────

"""
    initialize!(cell::RNNCell{P}; muP::Bool=false) where P

Initialize an RNN cell: Normal(0, 0.02) for weights, zero biases.
If `muP=true`, use 1/√(fan_in) scaling.
"""
function initialize!(cell::RNNCell{P}; muP::Bool=false) where P
    initialize!(cell.W_ih; muP=muP)
    initialize!(cell.W_hh; muP=muP)
    return cell
end

"""
    initialize!(cell::LSTMCell{P}; muP::Bool=false) where P

Initialize an LSTM cell with forget gate bias = 1.0 (Jozefowicz et al., 2015).
This encourages the network to remember by default in early training.

Gate layout in fused projection: [input; forget; output; candidate]
Only the W_ih forget bias is set to 1.0, matching char-rnn's convention.
"""
function initialize!(cell::LSTMCell{P}; muP::Bool=false) where P
    initialize!(cell.W_ih; muP=muP)
    initialize!(cell.W_hh; muP=muP)

    # Forget gate bias = 1.0 (second block of 4 in fused gates)
    H = cell.hidden_size
    cell.W_ih.bias[H+1:2H] .= one(P)

    return cell
end

"""
    initialize!(cell::GRUCell{P}; muP::Bool=false) where P

Initialize a GRU cell: Normal(0, 0.02) for all gate weights, zero biases.
"""
function initialize!(cell::GRUCell{P}; muP::Bool=false) where P
    initialize!(cell.W_iz; muP=muP)
    initialize!(cell.W_hz; muP=muP)
    initialize!(cell.W_ir; muP=muP)
    initialize!(cell.W_hr; muP=muP)
    initialize!(cell.W_in; muP=muP)
    initialize!(cell.W_hn; muP=muP)
    return cell
end

"""
    initialize!(model::SequenceModel; muP::Bool=false)

Initialize all weights in a SequenceModel by recursively dispatching
to each layer's initialization method.
"""
function initialize!(model::SequenceModel; muP::Bool=false)
    # Embedding
    initialize!(model.embedding; muP=muP)

    # Recurrent cells
    for cell in model.cells
        initialize!(cell; muP=muP)
    end

    # Final layer norm
    initialize!(model.ln_f)

    # Decoder
    if muP
        initialize!(model.decoder; muP=true)
    else
        initialize!(model.decoder; std=0.02)
    end

    return model
end

"""
    initialize!(model::ContinuousModel; muP::Bool=false)

Initialize all weights in a ContinuousModel by recursively dispatching
to each layer's initialization method.
"""
function initialize!(model::ContinuousModel; muP::Bool=false)
    initialize!(model.coeff_embed; std=0.02, muP=muP)
    
    n_layers = length(model.blocks)
    for block in model.blocks
        initialize!(block; n_layers=n_layers, muP=muP)
    end
    
    initialize!(model.ln_f)
    initialize!(model.spectral_head; std=0.02, muP=muP)
    
    return model
end
