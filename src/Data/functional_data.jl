# ─── Function Sequence Data ───────────────────────────────────────────────────
#
# Generates training data for the Functional Autoregressive Transformer.
# Each sample is a sequence of related functions represented as coefficient vectors.
#
# Tasks:
#   :derivative    — f, f', f'', f''', ... (learn differentiation)
#   :antiderivative — f''', f'', f', f     (learn integration)
#   :taylor        — successive Taylor approximations of a target function
#   :composition   — f, g∘f, h∘g∘f, ...   (learn function composition)

"""
    FunctionSequenceLoader{P}

Generates batches of function sequences for training ContinuousModel.
Each batch: (input_coeffs, target_coeffs) where target is input shifted by 1.
"""
struct FunctionSequenceLoader{P<:AbstractFloat}
    task::Symbol            # :derivative, :antiderivative, :taylor
    basis_dim::Int          # polynomial degree + 1
    seq_length::Int         # number of functions per sequence
    batch_size::Int
    basis_type::Symbol      # :monomial
end

function FunctionSequenceLoader{P}(;
    task::Symbol = :derivative,
    basis_dim::Int = 8,
    seq_length::Int = 8,
    batch_size::Int = 32,
    basis_type::Symbol = :monomial,
) where P
    @assert task in (:derivative, :antiderivative, :taylor, :composition)
    @assert basis_type == :monomial "Only :monomial basis supported for data generation"
    FunctionSequenceLoader{P}(task, basis_dim, seq_length, batch_size, basis_type)
end

FunctionSequenceLoader(; kwargs...) = FunctionSequenceLoader{Float32}(; kwargs...)

function Base.show(io::IO, l::FunctionSequenceLoader{P}) where P
    print(io, "FunctionSequenceLoader{$P}(task=$(l.task), basis_dim=$(l.basis_dim), seq=$(l.seq_length), batch=$(l.batch_size))")
end

# ─── Derivative operator on monomial coefficients ─────────────────────────────

"""
    differentiate_coeffs(c::Vector{P}) where P

Compute the derivative of a polynomial in monomial basis.
f(x) = c₁ + c₂x + c₃x² + c₄x³ + ...
f'(x) = c₂ + 2c₃x + 3c₄x² + ...

Returns coefficient vector of the same length (padded with zero).
"""
function differentiate_coeffs(c::AbstractVector{P}) where P
    d = length(c)
    dc = zeros(P, d)
    for j in 2:d
        dc[j-1] = P(j-1) * c[j]  # power rule: d/dx(x^n) = n*x^(n-1)
    end
    return dc
end

"""
    antidifferentiate_coeffs(c::Vector{P}; C=0) where P

Compute the antiderivative (integral) of a polynomial.
f(x) = c₁ + c₂x + c₃x² + ...
∫f(x)dx = C + c₁x + (c₂/2)x² + (c₃/3)x³ + ...

Returns coefficient vector of the same length (truncates highest term).
"""
function antidifferentiate_coeffs(c::AbstractVector{P}; C::P = zero(P)) where P
    d = length(c)
    ic = zeros(P, d)
    ic[1] = C
    for j in 1:d-1
        ic[j+1] = c[j] / P(j)  # integral rule: ∫x^n dx = x^(n+1)/(n+1)
    end
    return ic
end

# ─── Taylor series coefficients ───────────────────────────────────────────────

"""
    taylor_coefficients(f_symbol::Symbol, degree::Int, P::Type)

Compute Taylor series coefficients for standard functions.
Returns vector of length degree+1: [f(0)/0!, f'(0)/1!, f''(0)/2!, ...]
"""
function taylor_coefficients(f_sym::Symbol, degree::Int, ::Type{P}) where P
    c = zeros(P, degree + 1)
    if f_sym == :exp
        for k in 0:degree
            c[k+1] = one(P) / P(factorial(big(k)))
        end
    elseif f_sym == :sin
        for k in 0:degree
            if isodd(k)
                c[k+1] = P((-1)^((k-1)÷2)) / P(factorial(big(k)))
            end
        end
    elseif f_sym == :cos
        for k in 0:degree
            if iseven(k)
                c[k+1] = P((-1)^(k÷2)) / P(factorial(big(k)))
            end
        end
    elseif f_sym == :log1p  # log(1+x)
        for k in 1:degree
            c[k+1] = P((-1)^(k+1)) / P(k)
        end
    end
    return c
end

# ─── Batch Generation ─────────────────────────────────────────────────────────

"""
    next_batch(loader::FunctionSequenceLoader{P}) → (input, target)

Generate a batch of function sequences.
Returns (input, target) each of shape (basis_dim, seq_length, batch_size).
Input is positions 1..T, target is positions 2..T+1 (next-function prediction).
"""
function next_batch(loader::FunctionSequenceLoader{P}) where P
    d = loader.basis_dim
    T = loader.seq_length
    B = loader.batch_size

    # Generate T+1 functions per sample (T input + 1 target at end)
    all_coeffs = Array{P}(undef, d, T + 1, B)

    for b in 1:B
        _generate_sequence!(view(all_coeffs, :, :, b), loader)
    end

    input  = all_coeffs[:, 1:T, :]
    target = all_coeffs[:, 2:T+1, :]
    return input, target
end

function _generate_sequence!(out::AbstractMatrix{P}, loader::FunctionSequenceLoader{P}) where P
    d = loader.basis_dim
    T_total = size(out, 2)  # seq_length + 1

    if loader.task == :derivative
        # Start with a random polynomial, apply derivative repeatedly
        c = randn(P, d) * P(0.5)
        for t in 1:T_total
            out[:, t] .= c
            c = differentiate_coeffs(c)
        end

    elseif loader.task == :antiderivative
        # Start with a simple polynomial, generate its antiderivatives
        c = zeros(P, d)
        c[1] = randn(P) * P(0.5)  # start with a constant
        c[2] = randn(P) * P(0.5)  # and linear term
        # Build sequence backwards: generate derivatives then reverse
        derivs = Vector{Vector{P}}(undef, T_total)
        derivs[T_total] = copy(c)
        for t in (T_total-1):-1:1
            derivs[t] = antidifferentiate_coeffs(derivs[t+1]; C=randn(P)*P(0.1))
        end
        for t in 1:T_total
            out[:, t] .= derivs[t]
        end

    elseif loader.task == :taylor
        # Successive Taylor approximations of a random target function
        target_fn = rand([:exp, :sin, :cos, :log1p])
        full_coeffs = taylor_coefficients(target_fn, d - 1, P)
        # Scale randomly
        scale = P(0.5) + rand(P) * P(2.0)
        full_coeffs .*= scale
        # Each position t reveals one more term
        for t in 1:T_total
            c = zeros(P, d)
            terms = min(t, d)
            c[1:terms] .= full_coeffs[1:terms]
            out[:, t] .= c
        end

    elseif loader.task == :composition
        # f₁, f₂=g∘f₁, f₃=h∘g∘f₁ where g,h are simple affine transforms
        c = randn(P, d) * P(0.3)
        c[1] = randn(P) * P(0.5)
        c[2] = one(P) + randn(P) * P(0.1)  # near-identity linear term
        for t in 1:T_total
            out[:, t] .= c
            # Apply a simple transformation: scale and shift coefficients
            a = one(P) + randn(P) * P(0.1)
            b_shift = randn(P) * P(0.1)
            c_new = zeros(P, d)
            c_new[1] = c[1] * a + b_shift
            for j in 2:d
                c_new[j] = c[j] * a^P(j)
            end
            c = c_new
        end
    end
end

"""
    make_batch_fn(loader::FunctionSequenceLoader)

Return a callable that generates batches, matching the training loop interface.
"""
make_functional_batch_fn(loader::FunctionSequenceLoader) = () -> next_batch(loader)
