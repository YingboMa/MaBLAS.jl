using SafeTestsets

@time begin
    @time @safetestset "Julia BLAS" begin include("juliablas_tests.jl") end
end
