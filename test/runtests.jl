using Test

const TEST_FILES = [
    "test_config.jl",
    "test_embedding.jl",
    "test_layers.jl",
    "test_attention.jl",
    "test_model.jl",
    "test_recurrent.jl",
    "test_tokenizer.jl",
    "test_train.jl",
    "test_llama2.jl",
    "test_fractional.jl"
]

@testset "Architect.jl Test Suite" begin
    for f in TEST_FILES
        @testset "$f" begin
            cmd = `julia --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, f))`
            @test success(cmd)
        end
    end
end
