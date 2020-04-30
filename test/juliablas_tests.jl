using Test
using SmallLinearAlgebra
using LinearAlgebra

@testset "Size check" begin
    C = randn(12, 11); A = rand(2, 3); B = rand(2, 3)
    @test_throws DimensionMismatch SmallLinearAlgebra.mul!(C, A, B)
end

@testset "Kernel sized matmul" begin
    _m, _k, _n = 8*3, 8, 6*5
    # lower cache_params to check multiple of cache sizes more easily
    cache_params = (cache_m=_m, cache_k=_k, cache_n=_n)
    m, k, n = _m*2, _k*3, _n*5
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    for α in (1, 2, 3, false, true), β in (1, 2, 3, false, true), packing in (true, false)
        @test SmallLinearAlgebra.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=packing) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
    end
end

@testset "Clean up loop tests" begin
    _m, _k, _n = 8*3, 8, 6*5
    # lower cache_params to check clean up loops more easily
    cache_params = (cache_m=_m, cache_k=_k, cache_n=_n)
    m, k, n = 73, 131, 257 # all prime numbers
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    for α in (1, 2, 3, false, true), β in (1, 2, 3, false, true), packing in (true, false)
        @test SmallLinearAlgebra.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=packing) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
    end
end

@testset "View of matrix" begin
    V = randn(200, 200)
    A = @view V[1:100,  1:100]
    B = @view V[101:end, 1:100]
    C = @view V[1:100, 101:end]
    _m, _k, _n = 8*3, 8, 6*5
    cache_params = (cache_m=_m, cache_k=_k, cache_n=_n)
    @test LinearAlgebra.mul!((copy(C)), A, B) ≈ SmallLinearAlgebra.mul!(C, A, B; cache_params=cache_params)
    @test LinearAlgebra.mul!((copy(C)), A, B) ≈ SmallLinearAlgebra.mul!(C, A, B; cache_params=cache_params, packing=true)

    A = @view V[1:2:end, 1:100]
    @test_throws ArgumentError SmallLinearAlgebra.mul!(C, A, B)
    @test LinearAlgebra.mul!((copy(C)), A, B) ≈ SmallLinearAlgebra.mul!(C, A, B; cache_params=cache_params, packing=true)
end
