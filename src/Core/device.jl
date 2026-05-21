# ─── Device Transfer and Dispatch ──────────────────────────────────────────────

"""
    _cuda_available() -> Bool

Check if CUDA.jl is loaded and a GPU is available.
Returns false if CUDA.jl is not installed or no GPU is detected.
"""
function _cuda_available()
    try
        # Check if CUDA module is loaded and functional
        if isdefined(Main, :CUDA) || isdefined(@__MODULE__, :CUDA)
            cuda_mod = isdefined(@__MODULE__, :CUDA) ? (@__MODULE__).CUDA : Main.CUDA
            return cuda_mod.functional()
        end
        return false
    catch
        return false
    end
end

"""
    _to_device_array(x::AbstractArray, device::Symbol)

Transfer a single array to the specified device.
- :cpu → converts to standard Julia Array
- :cuda → converts to CuArray (falls back to CPU if CUDA unavailable)
- :metal → placeholder, falls back to CPU with a warning

Returns the array on the target device.
"""
function _to_device_array(x::AbstractArray, device::Symbol)
    if device == :cpu
        return Array(x)
    elseif device == :cuda
        if _cuda_available()
            cuda_mod = isdefined(@__MODULE__, :CUDA) ? (@__MODULE__).CUDA : Main.CUDA
            return cuda_mod.CuArray(x)
        else
            @warn "CUDA not available, falling back to CPU" maxlog=1
            return Array(x)
        end
    elseif device == :metal
        @warn "Metal backend not yet supported, falling back to CPU" maxlog=1
        return Array(x)
    else
        error("Unknown device: $device. Supported devices: :cpu, :cuda, :metal")
    end
end

"""
    _to_device_array(x::AbstractMatrix{Bool}, device::Symbol)

Transfer a boolean matrix (e.g., causal mask) to the specified device.
"""
function _to_device_array(x::AbstractMatrix{Bool}, device::Symbol)
    if device == :cpu
        return BitMatrix(x)
    elseif device == :cuda
        if _cuda_available()
            cuda_mod = isdefined(@__MODULE__, :CUDA) ? (@__MODULE__).CUDA : Main.CUDA
            return cuda_mod.CuArray(x)
        else
            @warn "CUDA not available, falling back to CPU" maxlog=1
            return BitMatrix(x)
        end
    elseif device == :metal
        @warn "Metal backend not yet supported, falling back to CPU" maxlog=1
        return BitMatrix(x)
    else
        error("Unknown device: $device. Supported devices: :cpu, :cuda, :metal")
    end
end

"""
    to_device(x::AbstractArray, device::Symbol)

Generic device-transfer entrypoint for arrays. Submodules should add
specialized `Core.to_device(::Type, device::Symbol)` methods for their
custom types (Embedding, Linear, Block, models, etc.).
"""
function to_device(x::AbstractArray, device::Symbol)
    return _to_device_array(x, device)
end

"""
    to_device(x, device::Symbol)

Fallback stub: if a specific type does not implement `to_device`, this
error will remind the developer to add a specialized method in the
appropriate submodule. This is intentionally strict to avoid silent
in-place mutations across devices.
"""
function to_device(x, device::Symbol)
    error("to_device not implemented for type $(typeof(x)). Define `Core.to_device(::YourType, device::Symbol)` in the corresponding submodule.")
end

