using Test
using SmallLinearAlgbra

@testset "Size check" begin
    C = randn(12, 11); A = rand(2, 3); B = rand(2, 3)
    @test_throws DimensionMismatch SmallLinearAlgbra.mul!(C, A, B)
end
