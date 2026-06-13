using Test
using Architect

@testset "Fractional Collocation Solver Tests" begin

    @testset "Caputo Fractional Derivative Matrix (D_α)" begin
        N = 8
        α = 1.5
        D_1, D_α, x = fractional_diff_matrices(N, α)
        
        # Test function: u(x) = (x + 1)^2
        # Analytic derivative: D^1.5 u(x) = Γ(3)/Γ(1.5) * (x + 1)^0.5 = (4 / √π) * √(x + 1)
        u = [(xi + 1.0)^2 for xi in x]
        
        # Numerical fractional derivative
        du_num = D_α * u
        
        # Analytic fractional derivative
        du_analytic = [(4.0 / sqrt(π)) * sqrt(xi + 1.0) for xi in x]
        
        # Check that they match with high accuracy
        for i in 1:(N+1)
            @test isapprox(du_num[i], du_analytic[i], atol=1e-5)
        end
    end

    @testset "First-Order Derivative Matrix (D_1)" begin
        N = 8
        D_1, _, x = fractional_diff_matrices(N, 1.5)
        
        # Test function: u(x) = x^2
        # Analytic derivative: u'(x) = 2x
        u = [xi^2 for xi in x]
        du_num = D_1 * u
        du_analytic = [2.0 * xi for xi in x]
        
        for i in 1:(N+1)
            @test isapprox(du_num[i], du_analytic[i], atol=1e-5)
        end
    end

    @testset "Convergence to Classical Case (α → 2)" begin
        N = 16
        m, c, k = 1.0, 1.0, 4.0
        u0, v0 = 1.0, 0.0
        Tf = 5.0
        F_func(t) = 0.0
        
        # Solve with α = 1.999
        t_frac, U_frac = solve_fractional_msd(N, 1.999, m, c, k, u0, v0, Tf, F_func)
        
        # Solve with α = 2.0 (classical limit)
        t_class, U_class = solve_fractional_msd(N, 2.0, m, c, k, u0, v0, Tf, F_func)
        
        # Verify they are extremely close
        for i in 1:(N+1)
            @test isapprox(U_frac[i], U_class[i], atol=5e-3)
        end
    end

    @testset "ContinuousTimeDataLoader Integration (:fractional_oscillator)" begin
        K = 7
        seq_length = 8
        batch_size = 4
        loader = ContinuousTimeDataLoader{Float64}(
            task=:fractional_oscillator, 
            K=K, 
            seq_length=seq_length, 
            batch_size=batch_size
        )
        
        input, target = Architect.FAT.next_batch(loader)
        
        # Dimensions should be: (K + 1, seq_length, batch_size)
        @test size(input) == (K + 1, seq_length, batch_size)
        @test size(target) == (K + 1, seq_length, batch_size)
        
        # Check that there are no NaNs or Infs
        @test all(!isnan, input)
        @test all(!isnan, target)
        @test all(!isinf, input)
        @test all(!isinf, target)
    end
end
