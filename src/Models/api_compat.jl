# Canonical coefficient-model accessors.
#
# ContinuousModel predates the FAT-specific naming and uses `coeff_encoder` /
# `coeff_head`.  New shared utilities should call these accessors instead of
# reaching into concrete field names.

"""
    coefficient_encoder(model)

Return the layer that maps coefficient vectors into model width.
"""
coefficient_encoder(model::ContinuousModel) = model.coeff_encoder

"""
    coefficient_head(model)

Return the layer that maps hidden states back to coefficient vectors.
"""
coefficient_head(model::ContinuousModel) = model.coeff_head

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

# Compatibility property aliases. These keep older and newer naming styles
# usable while refactors move scripts toward the accessor functions above.
function Base.getproperty(model::ContinuousModel, name::Symbol)
    if name === :coeff_embed
        return getfield(model, :coeff_encoder)
    elseif name === :spectral_head
        return getfield(model, :coeff_head)
    else
        return getfield(model, name)
    end
end
