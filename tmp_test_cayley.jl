using LinearAlgebra, Random
include("src/types.jl")
include("src/layers.jl")

c = CayleyLinear(16)
x = randn(Float32,16,4)
y = c(x)
println("y size: ", size(y))
W = cayley_weight(c)
err = norm(W' * W - Matrix{Float32}(I,16,16))
println("orthogonality error (Fro): ", err)
a_sym_err = maximum(abs.(c.A + c.A'))
println("max(A + A') (should be ~0): ", a_sym_err)
