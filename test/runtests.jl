using SafeTestsets

@time begin
    @time @safetestset "BLAS" begin include("gemm_tests.jl") end
end
