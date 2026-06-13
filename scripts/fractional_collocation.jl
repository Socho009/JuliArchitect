using LinearAlgebra
using SpecialFunctions: gamma
using Printf

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
    V = zeros(n, n)
    for i in 1:n
        for k in 1:n
            power = k - 1
            V[i, k] = (x[i] + 1.0)^power
        end
    end
    
    # 2. Construct derivative Vandermonde matrix V_1
    V_1 = zeros(n, n)
    for i in 1:n
        for k in 1:n
            power = k - 1
            if power >= 1
                V_1[i, k] = power * (x[i] + 1.0)^(power - 1)
            end
        end
    end
    
    # 3. Construct fractional derivative Vandermonde matrix V_α
    V_α = zeros(n, n)
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
    # D_t^1 = (2/Tf) * D_x^1
    # D_t^α = (2/Tf)^α * D_x^α
    scale1 = 2.0 / Tf
    scaleα = (2.0 / Tf)^α
    
    D_1 = scale1 .* D_1_x
    D_α = scaleα .* D_α_x
    
    # Map CGL nodes to time
    t = (Tf / 2.0) .* (x .+ 1.0)
    
    # 3. Assemble collocation matrix
    # A = m * D_α + c * D_1 + k * I
    n = N + 1
    A = m .* D_α .+ c .* D_1 .+ k .* I(n)
    
    # 4. Build RHS vector
    RHS = [F_func(ti) for ti in t]
    
    # 5. Enforce Initial Conditions at t = 0
    # In CGL nodes sorted from 1 down to -1:
    # x[N+1] = -1 corresponding to t = 0.
    # We replace:
    # - Row N (second-to-last): velocity condition u'(0) = v0
    # - Row N+1 (last): displacement condition u(0) = u0
    
    # Row N: velocity condition
    # u'(0) = sum(D_1[N+1, j] * u[j]) = v0
    A[N, :] .= D_1[N+1, :]
    RHS[N] = v0
    
    # Row N+1: displacement condition
    # u(0) = u[N+1] = u0
    A[N+1, :] .= 0.0
    A[N+1, N+1] = 1.0
    RHS[N+1] = u0
    
    # 6. Solve system
    U = A \ RHS
    return t, U
end

# --- Exact Analytical Solutions for α = 2 ---

function exact_solution_classical(t::Real, m::Real, c::Real, k::Real, u0::Real, v0::Real)
    # Characteristic equation: m r^2 + c r + k = 0
    # Discriminant: d = c^2 - 4*m*k
    d = c^2 - 4.0 * m * k
    
    if d > 1e-9 # Overdamped
        r1 = (-c + sqrt(d)) / (2.0 * m)
        r2 = (-c - sqrt(d)) / (2.0 * m)
        # u(t) = C1*e^(r1*t) + C2*e^(r2*t)
        # u(0) = C1 + C2 = u0
        # u'(0) = C1*r1 + C2*r2 = v0
        # C2 = u0 - C1 => C1*r1 + (u0 - C1)*r2 = v0 => C1*(r1 - r2) = v0 - u0*r2
        C1 = (v0 - u0 * r2) / (r1 - r2)
        C2 = u0 - C1
        return C1 * exp(r1 * t) + C2 * exp(r2 * t)
        
    elseif abs(d) <= 1e-9 # Critically damped
        r = -c / (2.0 * m)
        # u(t) = (C1 + C2*t)*e^(r*t)
        # u(0) = C1 = u0
        # u'(0) = C2 + C1*r = v0 => C2 = v0 - u0*r
        C1 = u0
        C2 = v0 - u0 * r
        return (C1 + C2 * t) * exp(r * t)
        
    else # Underdamped (d < 0)
        λ = -c / (2.0 * m)
        μ = sqrt(4.0 * m * k - c^2) / (2.0 * m)
        # u(t) = e^(λ*t) * (C1*cos(μ*t) + C2*sin(μ*t))
        # u(0) = C1 = u0
        # u'(0) = λ*C1 + μ*C2 = v0 => C2 = (v0 - λ*u0) / μ
        C1 = u0
        C2 = (v0 - λ * u0) / μ
        return exp(λ * t) * (C1 * cos(μ * t) + C2 * sin(μ * t))
    end
end

# --- RUN VERIFICATION CASES ---

function run_verification()
    println("=========================================================")
    println("   Chebyshev Fractional Collocation Solver Verification   ")
    println("=========================================================")
    
    # Parameters
    Tf = 5.0
    u0 = 1.0
    v0 = 0.0
    F_func(t) = 0.0 # Free vibration
    N = 16 # Number of collocation points
    
    # Test cases: (m, c, k, label)
    cases = [
        (1.0, 3.0, 2.0, "Overdamped (c^2 > 4mk)"),
        (1.0, 2.0, 1.0, "Critically Damped (c^2 = 4mk)"),
        (1.0, 1.0, 5.0, "Underdamped (c^2 < 4mk)"),
        (1.0, 0.0, 4.0, "Pure Harmonic (c = 0)")
    ]
    
    for (m, c, k, label) in cases
        println("\n--- Case: $label (m=$m, c=$c, k=$k) ---")
        
        # 1. Solve numerically using α = 2.0
        t, U_num = solve_fractional_msd(N, 2.0, m, c, k, u0, v0, Tf, F_func)
        
        # 2. Compute exact solutions
        U_exact = [exact_solution_classical(ti, m, c, k, u0, v0) for ti in t]
        
        # 3. Print comparison table
        println(@sprintf("%-8s | %-12s | %-12s | %-12s", "t", "Numerical", "Exact", "Abs Error"))
        println("-"^52)
        
        max_err = 0.0
        # Print a subset of points (e.g. every 2nd point) for brevity
        for i in 1:length(t)
            err = abs(U_num[i] - U_exact[i])
            if err > max_err
                max_err = err
            end
            if i % 2 == 1 || i == length(t)
                println(@sprintf("%8.4f | %12.8f | %12.8f | %12.4e", t[i], U_num[i], U_exact[i], err))
            end
        end
        println("-"^52)
        println(@sprintf("Max Absolute Error: %e", max_err))
    end
    
    # --- Fractional Order Case (1 < α < 2) ---
    println("\n=========================================================")
    println("   Fractional Order Solver Run (Memory & Viscoelasticity) ")
    println("=========================================================")
    m, c, k = 1.0, 1.0, 4.0
    u0, v0 = 1.0, 0.0
    
    println("Parameters: m=$m, c=$c, k=$k, u(0)=$u0, u'(0)=$v0, Tf=$Tf")
    println("\nComparing displacement u(t) for different values of α:")
    println("-"^60)
    
    # Solve for different alpha
    t_1_2, U_1_2 = solve_fractional_msd(N, 1.2, m, c, k, u0, v0, Tf, F_func)
    t_1_5, U_1_5 = solve_fractional_msd(N, 1.5, m, c, k, u0, v0, Tf, F_func)
    t_1_8, U_1_8 = solve_fractional_msd(N, 1.8, m, c, k, u0, v0, Tf, F_func)
    t_2_0, U_2_0 = solve_fractional_msd(N, 2.0, m, c, k, u0, v0, Tf, F_func)
    
    println(@sprintf("%-8s | %-11s | %-11s | %-11s | %-11s", "t", "α = 1.2", "α = 1.5", "α = 1.8", "α = 2.0"))
    println("-"^60)
    for i in length(t_1_5):-2:1
        println(@sprintf("%8.4f | %11.6f | %11.6f | %11.6f | %11.6f", 
            t_1_5[i], U_1_2[i], U_1_5[i], U_1_8[i], U_2_0[i]))
    end
    println("-"^60)
    println("Observation: Lower values of α show a much slower, non-exponential decay curve.")
    println("This matches Mittag-Leffler behavior (fractional memory/hereditary effect).")
    println("=========================================================")
end

run_verification()
