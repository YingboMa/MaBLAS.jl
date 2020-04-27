using Test
using SmallLinearAlgebra
using LinearAlgebra

@testset "Size check" begin
    C = randn(12, 11); A = rand(2, 3); B = rand(2, 3)
    @test_throws DimensionMismatch SmallLinearAlgebra.mul!(C, A, B)
end

@testset "Kernel sized matmul" begin
    m, k, n = 8, 83, 30
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    for α in (1, 2, 3, false, true), β in (1, 2, 3, false, true)
        @test SmallLinearAlgebra.mul!((copy(C)), A, B, 2, 3) ≈ LinearAlgebra.mul!((copy(C)), A, B, 2, 3)
    end
end
