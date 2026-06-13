# --- PHASE 6 ---
# Continuous Data Streams
#
# Generates synthetic continuous functions projected into the Chebyshev basis.
# Used for benchmarking the Functional Autoregressive Transformer.

export ContinuousTimeDataLoader, next_batch, make_continuous_batch_fn

"""
    ContinuousTimeDataLoader{P, K}

Generates batches of functional sequences (represented in Chebyshev basis).
Each batch: (input_coeffs, target_coeffs) where target is input shifted by 1.
"""
struct ContinuousTimeDataLoader{P<:AbstractFloat, K}
    task::Symbol            # :sine_waves, :step_functions
    seq_length::Int         # number of functions per sequence
    batch_size::Int
    basis::ChebyshevBasis{K}
end

function ContinuousTimeDataLoader{P}(;
    task::Symbol = :sine_waves,
    K::Int = 7,
    seq_length::Int = 16,
    batch_size::Int = 32
) where P
    @assert task in (:sine_waves, :step_functions, :lorentz, :fractional_oscillator)
    basis = ChebyshevBasis{K}()
    ContinuousTimeDataLoader{P, K}(task, seq_length, batch_size, basis)
end

ContinuousTimeDataLoader(; kwargs...) = ContinuousTimeDataLoader{Float32}(; kwargs...)

function Base.show(io::IO, l::ContinuousTimeDataLoader{P,K}) where {P,K}
    print(io, "ContinuousTimeDataLoader{\$P, K=\$K}(task=\$(l.task), seq=\$(l.seq_length), batch=\$(l.batch_size))")
end

"""
    next_batch(loader::ContinuousTimeDataLoader{P, K}) → (input, target)

Returns (input, target) each of shape (K+1, seq_length, batch_size).
Input is positions 1..T, target is positions 2..T+1 (next-function prediction).
"""
function next_batch(loader::ContinuousTimeDataLoader{P, K}) where {P, K}
    d = K + 1
    T = loader.seq_length
    B = loader.batch_size

    # Generate T+1 functions per sample
    all_coeffs = Array{P}(undef, d, T + 1, B)

    for b in 1:B
        _generate_continuous_sequence!(view(all_coeffs, :, :, b), loader)
    end

    input  = all_coeffs[:, 1:T, :]
    target = all_coeffs[:, 2:T+1, :]
    return input, target
end

function _generate_continuous_sequence!(out::AbstractMatrix{P}, loader::ContinuousTimeDataLoader{P, K}) where {P, K}
    T_total = size(out, 2)
    N_nodes = K + 10  # higher degree for accurate quadrature

    if loader.task == :sine_waves
        # Base frequency between 1 and 4
        base_freq = P(1.0) + rand(P) * P(3.0)
        phase = rand(P) * P(2π)
        for t in 1:T_total
            # Smoothly evolving function: phase drifts over time
            f_x = x -> sin(base_freq * x + phase + P(t) * P(0.3))
            
            # Project onto Chebyshev basis
            state = project(loader.basis, f_x, P; N=N_nodes)
            out[:, t] .= state.coeffs
        end

    elseif loader.task == :step_functions
        for t in 1:T_total
            # Step location moves back and forth
            step_loc = P(0.8) * sin(P(t) * P(0.4) + rand(P)*P(0.1))
            f_x = x -> x > step_loc ? one(P) : -one(P)
            
            state = project(loader.basis, f_x, P; N=N_nodes + 20) # step needs more nodes
            out[:, t] .= state.coeffs
        end

    elseif loader.task == :lorentz
        # Chaotic dynamics approximated as polynomials
        dt = P(0.05)
        x_c, y_c, z_c = P(1.0), P(1.0), P(1.0)
        sigma, rho, beta = P(10.0), P(28.0), P(8.0/3.0)
        
        # Burn-in
        for _ in 1:50
            dx = sigma * (y_c - x_c)
            dy = x_c * (rho - z_c) - y_c
            dz = x_c * y_c - beta * z_c
            x_c += dx * dt; y_c += dy * dt; z_c += dz * dt
        end

        for t in 1:T_total
            # Advance Lorentz
            dx = sigma * (y_c - x_c)
            dy = x_c * (rho - z_c) - y_c
            dz = x_c * y_c - beta * z_c
            x_c += dx * dt; y_c += dy * dt; z_c += dz * dt
            
            # Create a spatial function from the chaotic state
            f_x = x -> P(0.1) * (x_c * x + y_c * x^2 + z_c * x^3)
            
            state = project(loader.basis, f_x, P; N=N_nodes)
            out[:, t] .= state.coeffs
        end

    elseif loader.task == :fractional_oscillator
        # Solve fractional mass-spring-damper for a sequence of alphas between 1.1 and 2.0
        m = P(0.8) + rand(P) * P(0.7)
        c = P(0.5) + rand(P) * P(1.0)
        k = P(2.0) + rand(P) * P(3.0)
        u0 = randn(P) * P(0.5)
        v0 = randn(P) * P(0.2)
        Tf = P(5.0)
        
        # Random forcing function
        F_amp = randn(P) * P(0.5)
        F_freq = P(1.0) + rand(P) * P(2.0)
        F_func(t_val) = F_amp * sin(F_freq * t_val)
        
        # Range of alphas for sequence length T_total
        alphas = range(P(1.1), P(2.0), length=T_total)
        
        # Pre-build shifted monomial Vandermonde inverse for interpolation
        x_cgl = [cos(i * π / K) for i in 0:K]
        V_shifted = zeros(P, K+1, K+1)
        for i in 1:(K+1)
            for j in 1:(K+1)
                V_shifted[i, j] = (x_cgl[i] + 1.0)^(j - 1)
            end
        end
        V_inv = inv(V_shifted)
        
        for t in 1:T_total
            alpha = alphas[t]
            
            # Solve using the fractional solver
            _, U_values = solve_fractional_msd(K, alpha, m, c, k, u0, v0, Tf, F_func)
            
            # Compute shifted monomial coefficients
            a = V_inv * U_values
            
            # Continuous interpolating function on x ∈ [-1, 1]
            f_x = x_val -> sum(a[p+1] * (x_val + 1.0)^p for p in 0:K)
            
            # Project onto Chebyshev basis using exact DCT-I (N = K)
            state = project(loader.basis, f_x, P; N=K+4)
            out[:, t] .= state.coeffs
        end
    end
end

"""
    make_continuous_batch_fn(loader::ContinuousTimeDataLoader)

Return a callable that generates batches, matching the training loop interface.
"""
make_continuous_batch_fn(loader::ContinuousTimeDataLoader) = () -> next_batch(loader)
