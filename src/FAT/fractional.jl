using LinearAlgebra
using SpecialFunctions: gamma

"""
    chebyshev_nodes(N::Int)

Compute Chebyshev-Gauss-Lobatto (CGL) nodes on [-1, 1].
Returns nodes sorted from 1 down to -1.
"""
function chebyshev_nodes(N::Int)
    return [cos(i * π / N) for i in 0:N]
end

"""
    fractional_diff_matrices(N::Int, α::Real)

Construct first-order and fractional-order differentiation matrices on CGL nodes
using shifted monomials (x + 1)^k.
"""
function fractional_diff_matrices(N::Int, α::Real)
    x = chebyshev_nodes(N)
    n = N + 1
    
    # 1. Construct Vandermonde matrix V for (x + 1)^k
    V = zeros(typeof(α), n, n)
    for i in 1:n
        for k in 1:n
            power = k - 1
            V[i, k] = (x[i] + 1.0)^power
        end
    end
    
    # 2. Construct derivative Vandermonde matrix V_1
    V_1 = zeros(typeof(α), n, n)
    for i in 1:n
        for k in 1:n
            power = k - 1
            if power >= 1
                V_1[i, k] = power * (x[i] + 1.0)^(power - 1)
            end
        end
    end
    
    # 3. Construct fractional derivative Vandermonde matrix V_α
    V_α = zeros(typeof(α), n, n)
    m = ceil(Int, α) # should be 2 for 1 < α <= 2
    for i in 1:n
        for k in 1:n
            power = k - 1
            if power >= m
                # Caputo derivative of (x+1)^power
                coeff = gamma(power + 1) / gamma(power + 1 - α)
                V_α[i, k] = coeff * (x[i] + 1.0)^(power - α)
            end
        end
    end
    
    # 4. Compute collocation matrices
    # D_1 = V_1 * V^-1
    # D_α = V_α * V^-1
    V_inv = inv(V)
    D_1 = V_1 * V_inv
    D_α = V_α * V_inv
    
    return D_1, D_α, x
end

"""
    solve_fractional_msd(N::Int, α::Real, m::Real, c::Real, k::Real, u0::Real, v0::Real, Tf::Real, F_func::Function)

Solve the Fractional Mass-Spring-Damper system:
    m D_t^α u(t) + c D_t^1 u(t) + k u(t) = F(t)
on t ∈ [0, Tf] with u(0) = u0, u'(0) = v0.
"""
function solve_fractional_msd(N::Int, α::Real, m::Real, c::Real, k::Real, u0::Real, v0::Real, Tf::Real, F_func::Function)
    # 1. Get differentiation matrices on [-1, 1]
    D_1_x, D_α_x, x = fractional_diff_matrices(N, α)
    
    # 2. Scale matrices to time domain [0, Tf]
    scale1 = 2.0 / Tf
    scaleα = (2.0 / Tf)^α
    
    D_1 = scale1 .* D_1_x
    D_α = scaleα .* D_α_x
    
    # Map CGL nodes to time
    t = (Tf / 2.0) .* (x .+ 1.0)
    
    # 3. Assemble collocation matrix
    n = N + 1
    A = m .* D_α .+ c .* D_1 .+ k .* I(n)
    
    # 4. Build RHS vector
    RHS = [F_func(ti) for ti in t]
    
    # 5. Enforce Initial Conditions at t = 0
    # x[N+1] = -1 corresponding to t = 0.
    A[N, :] .= D_1[N+1, :]
    RHS[N] = v0
    
    A[N+1, :] .= 0.0
    A[N+1, N+1] = 1.0
    RHS[N+1] = u0
    
    # 6. Solve system
    U = A \ RHS
    return t, U
end
