using Test
using MaBLAS
using LinearAlgebra
using TimerOutputs

noreturn_mul!(args...; kwargs...) = (MaBLAS.mul!(args...; kwargs...); nothing)

@testset "Size check" begin
    C = randn(12, 11); A = rand(2, 3); B = rand(2, 3)
    @test_throws DimensionMismatch MaBLAS.mul!(C, A, B)
end

@testset "Kernel sized matmul" begin
    _m, _k, _n = 8*3, 8, 6*5
    # lower cache_params to check multiple of cache sizes more easily
    cache_params = (cache_m=_m, cache_k=_k, cache_n=_n)
    m, k, n = _m*2, _k*3, _n*5
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    for α in (1, 2, 3, false, true), β in (1, 2, 3, false, true), packa in (true, false), packb in (true, false)
        for kernel_params in [(Val(8), Val(6)), (Val(12), Val(4)), (Val(4), Val(3))]
            @test MaBLAS.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(packa, packb)) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
        end
    end
    packa, packb = true, false
    α, β = randn(2)
    @test MaBLAS.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(packa, packb)) == MaBLAS.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(Val(packa), Val(packb)))
end

@testset "Block size tests" begin
    @test MaBLAS.partition_k(2000, 532) == 500
    @test MaBLAS.partition_k(401, 400) == 201
    @test MaBLAS.partition_k(900, 400) == 300
end

@testset "Clean up loop tests" begin
    _m, _k, _n = 8*3, 8, 6*5
    # lower cache_params to check clean up loops more easily
    cache_params = (cache_m=_m, cache_k=_k, cache_n=_n)
    α, β = randn(2)
    for T in (Float64, Float32)
        for packa in (true, false), packb in (true, false), (m, k, n) in [(73, 131, 257), (101, 103, 107), (109, 113, 127), (131, 137, 139), (149, 151, 157), (163, 167, 173), (179, 181, 191), (193, 197, 199)] # all primes
            for kernel_params in [(Val(8), Val(6)), (Val(12), Val(4)), (Val(4), Val(3))]
                C = rand(T, m, n)
                A = rand(T, m, k)
                B = rand(T, k, n)
                @test MaBLAS.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(packa, packb)) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
                @allocated noreturn_mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(packa, packb))
                @allocated(noreturn_mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(packa, packb))) <= (packa + packb) * 176 # minor allocation when packing
            end
        end
    end
end

@testset "View of matrix" begin
    V = randn(200, 200)
    A = @view V[1:100,  1:100]
    B = @view V[101:end, 1:100]
    C = @view V[1:100, 101:end]
    _m, _k, _n = 8*3, 8, 6*5
    cache_params = (cache_m=_m, cache_k=_k, cache_n=_n)
    @test LinearAlgebra.mul!((copy(C)), A, B) ≈ MaBLAS.mul!(C, A, B; cache_params=cache_params)
    for packa in (true, false), packb in (true, false)
        @test LinearAlgebra.mul!((copy(C)), A, B) ≈ MaBLAS.mul!(C, A, B; cache_params=cache_params, packing=(packa, packb))
    end

    A = @view V[1:2:end, 1:100]
    @test_throws ArgumentError MaBLAS.mul!(C, A, B)
    @test_throws ArgumentError MaBLAS.mul!(C, A, B; packing=(false, true))
    for packb in (true, false)
        @test LinearAlgebra.mul!((copy(C)), A, B) ≈ MaBLAS.mul!(C, A, B; cache_params=cache_params, packing=(true, packb))
    end

    for packa in (true, false), packb in (true, false)
        @test LinearAlgebra.mul!((copy(C)), B, A) ≈ MaBLAS.mul!(C, B, A; cache_params=cache_params, packing=(packa, packb))
    end

    C = @view V[1:2:end, 101:end]
    for packa in (true, false), packb in (true, false)
        @test_throws ArgumentError MaBLAS.mul!(C, A, B; cache_params=cache_params, packing=(packa, packb))
    end
end

@testset "Timer & Transposes/Adjoints" begin
    m, k, n = 73, 131, 257 # all prime numbers
    C = rand(m, n)
    timer = MaBLAS.get_timer()
    MaBLAS.enable_timer()
    MaBLAS.reset_timer!()
    for α in (1, 2, 3.0, false, true), β in (1, 2, 3.0, false, true), A in (rand(m, k), rand(k, m)'), B in (rand(k, n), rand(n, k)'), packa in (true, false), packb in (true, false)
        !packa && A isa Adjoint && continue
        AB = MaBLAS.mul!((copy(C)), A, B, α, β; packing=(packa, packb))
        @test AB ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
    end
    tim, alloc = TimerOutputs.totmeasured(timer)
    @test tim > 0
    @test alloc == 0
    display(timer)

    MaBLAS.reset_timer!()
    MaBLAS.disable_timer()

    α, β = rand(2)
    for A in (rand(m, k), rand(k, m)', PermutedDimsArray(rand(k, m), (2, 1))), B in (rand(k, n), rand(n, k)', PermutedDimsArray(rand(n, k), (2, 1)))
        MaBLAS.mul!((copy(C)), A, B, α, β; packing=(true, true)) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
    end
    display(timer)
    @test all(iszero, TimerOutputs.totmeasured(timer))
end
