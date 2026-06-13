# Canonical coefficient-model accessors.
#
# Shared coefficient-model accessors. The canonical concrete field vocabulary
# is `coeff_embed` / `spectral_head` for both ContinuousModel and FAT.

"""
    coefficient_encoder(model)

Return the layer that maps coefficient vectors into model width.
"""
coefficient_encoder(model::ContinuousModel) = model.coeff_embed

"""
    coefficient_head(model)

Return the layer that maps hidden states back to coefficient vectors.
"""
coefficient_head(model::ContinuousModel) = model.spectral_head

"""
    coefficient_dimension(model)

Return the number of coefficients predicted at each sequence step.
"""
coefficient_dimension(model::ContinuousModel) = model.config.basis_dim

"""
    context_length(model)

Return the model's autoregressive context length.
"""
context_length(model::ContinuousModel) = model.config.block_size

# Compatibility aliases for old ContinuousModel field names.
function Base.getproperty(model::ContinuousModel, name::Symbol)
    if name === :coeff_encoder
        return getfield(model, :coeff_embed)
    elseif name === :coeff_head
        return getfield(model, :spectral_head)
    else
        return getfield(model, name)
    end
end

Base.propertynames(model::ContinuousModel, private::Bool=false) =
    (:config, :coeff_embed, :blocks, :ln_f, :spectral_head, :dropout_rate, :coeff_encoder, :coeff_head)
