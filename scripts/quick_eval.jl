#!/usr/bin/env julia
# Quick evaluation script for lightweight functional helpers
# - Uses src/Data/functional_data.jl (differentiate, antidifferentiate, taylor)

root = abspath(joinpath(@__DIR__, ".."))
include(joinpath(root, "src", "Data", "functional_data.jl"))

println("== Quick Eval: functional_data helpers ==")

c = Float32[2.0, -1.0, 3.0, 0.5]
println("coeffs: ", c)

dc = differentiate_coeffs(c)
println("derivative coeffs: ", dc)

ic = antidifferentiate_coeffs(dc; C=2.0f0)
println("antiderivative (C=2.0): ", ic)

c_exp = taylor_coefficients(:exp, 7, Float32)
println("taylor_coeffs(:exp,7): ", c_exp)

println("-- generate a small batch (FunctionSequenceLoader) --")
loader = FunctionSequenceLoader{Float32}(task=:derivative, basis_dim=8, seq_length=4, batch_size=4)
input, target = next_batch(loader)
println("input size: ", size(input), ", target size: ", size(target))

println("-- microbenchmark differentiate_coeffs --")
N = 200_000
t0 = time()
for i in 1:N
    differentiate_coeffs(randn(Float32, 8))
end
t1 = time()
dur = t1 - t0
println("N=", N, ", duration=", round(dur, digits=4), "s, per_call=", round(dur/N*1e6, digits=2), " μs")

println("== Done ==")
