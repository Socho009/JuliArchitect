using LinearAlgebra, Random
include("src/types.jl")
include("src/layers.jl")
include("src/attention.jl")
include("src/block.jl")
include("src/model.jl")
include("src/init.jl")

c = CayleyLinear(32)
initialize!(c; muP=true)
W = cayley_weight(c)
println("norm W'*W - I = ", norm(W' * W - Matrix{Float32}(I,32,32)))
println("max(A + A') = ", maximum(abs.(c.A + c.A')))
