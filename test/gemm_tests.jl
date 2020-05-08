using Test
using MaBLAS
using LinearAlgebra
using TimerOutputs

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
        for kernel_params in [(Val(8), Val(6)), (Val(12), Val(4))]
            @test MaBLAS.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(packa, packb)) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
        end
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
    for α in (1, 2, 3, false, true), β in (1, 2, 3, false, true), packa in (true, false), packb in (true, false)
        for kernel_params in [(Val(8), Val(6)), (Val(12), Val(4))]
            @test MaBLAS.mul!((copy(C)), A, B, α, β; cache_params=cache_params, packing=(packa, packb)) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
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
    for α in (1, 2, 3, false, true), β in (1, 2, 3, false, true), A in (rand(m, k), rand(k, m)'), B in (rand(k, n), rand(n, k)'), packa in (true, false), packb in (true, false)
        try
            AB = MaBLAS.mul!((copy(C)), A, B, α, β; packing=(packa, packb))
            @test AB ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
        catch
            @test_skip AB ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β) # tiling doesn't support nonunit leading striding
        end
    end
    tim, alloc = TimerOutputs.totmeasured(timer)
    @test tim > 0
    @test alloc == 0
    display(timer)

    MaBLAS.reset_timer!()
    MaBLAS.disable_timer()
    for α in (1, 2, 3, false, true), β in (1, 2, 3, false, true), A in (rand(m, k), rand(k, m)'), B in (rand(k, n), rand(n, k)')
        MaBLAS.mul!((copy(C)), A, B, α, β; packing=(true, true)) ≈ LinearAlgebra.mul!((copy(C)), A, B, α, β)
    end
    display(timer)
    @test all(iszero, TimerOutputs.totmeasured(timer))
end
