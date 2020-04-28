using SIMD
using LinearAlgebra: Transpose, Adjoint
using Base.Cartesian: @nexprs

# General direction: we want to avoid pointer arithmetic as much as possible,
# also, don't strict type the container type

# I'll try to implement everything without defining a type, and all variables
# should stays local if possible. We can always clean up the code later when we
# find a good direction/structure.

###
### BLAS parameters and convenience aliases
###

###
### User-level API
###

function mul!(C, A, B, α=true, β=false)
    iszeroα = iszero(α)
    if iszero(β) && iszeroα
        return fill!(C, zero(eltype(C)))
    elseif isone(β) && iszeroα
        return C
    elseif iszeroα
        # TODO: optimize this
        for ii in eachindex(C)
            Cii = @inbounds C[ii]
            Cii = β * Cii
            @inbounds C[ii] = Cii
        end
        return C
    end
    # assume Float64 for now
    cache_params = (cache_m=72, cache_k=256, cache_n=4080)
    kernel_params = (Val(8), Val(6))
    return tiling_mul!(C, A, B, α, β, cache_params, kernel_params)
end

###
### Lower-level `_mul!`
###

function packing_mul!(C, A, B, α=true, β=false)

end


function tiling_mul!(C, A, B, α, β, (cache_m, cache_k, cache_n), kernel_params::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}
    m, k, n = checkmulsize(C, A, B)
    # tiling
    for cachejstart in 1:cache_n:n
        cachejend = min(cachejstart + cache_n - 1, n)
        for cachepstart in 1:cache_k:k
            cachepend = min(cachepstart + cache_k - 1, k)
            for cacheistart in 1:cache_m:m
                cacheiend = min(cacheistart + cache_m - 1, m)
                # macrokernel
                for microjstart in cachejstart:micro_n:cachejend, microistart in cacheistart:micro_m:cacheiend
                    # (micro_m × micro_n) = (micro_m × ks) * (ks × micro_n)
                    # C[i:(i+micro_m-1)] = A[i:(i+micro_m-1), ps] * B[ps, j:(j+micro_n-1)]
                    tiling_microkernel!(C, A, B, α, β, microistart, microjstart, cachepstart, cachepend, kernel_params)
                end
            end # cachei
        end # cachep
    end # cachej
    return C
end

###
### Packing
###

function packAbuffer!

end

function packBbuffer!

end

###
### Micro kernel
###

@noinline function tiling_microkernel!(C::SIMD.FastContiguousArray{Float64}, A::SIMD.FastContiguousArray{Float64}, B::SIMD.FastContiguousArray{Float64}, α, β,
                                     i, j, pstart, pend, ::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}
    T = Float64
    N = 4
    V = Vec{N, T}
    st = sizeof(T)
    sv = sizeof(V)
    incA = stride(A, 2)*st
    punroll, pleft = divrem(pend - pstart + 1, 8)

    ptrĈ = getptr(C, i, j) # pointer with offset
    Ĉ11 = zero(V)
    Ĉ21 = zero(V)
    Ĉ12 = zero(V)
    Ĉ22 = zero(V)
    Ĉ13 = zero(V)
    Ĉ23 = zero(V)
    Ĉ14 = zero(V)
    Ĉ24 = zero(V)
    Ĉ15 = zero(V)
    Ĉ25 = zero(V)
    Ĉ16 = zero(V)
    Ĉ26 = zero(V)

    # rank-1 updates
    p = pstart
    ptrÂ = getptr(A, i, p)
    for _ in 1:punroll
        @nexprs 8 u -> begin
            # iteration u
            Â1 = vecload(V, A, ptrÂ + (u - 1) * incA, 1, 1)
            Â2 = vecload(V, A, ptrÂ + (u - 1) * incA, 2, 1)
            B1 = V(@inbounds B[p + (u - 1), j])
            Ĉ11 = fma(Â1, B1, Ĉ11)
            Ĉ21 = fma(Â2, B1, Ĉ21)
            B2 = V(@inbounds B[p + (u - 1), j + 1])
            Ĉ12 = fma(Â1, B2, Ĉ12)
            Ĉ22 = fma(Â2, B2, Ĉ22)
            B3 = V(@inbounds B[p + (u - 1), j + 2])
            Ĉ13 = fma(Â1, B3, Ĉ13)
            Ĉ23 = fma(Â2, B3, Ĉ23)
            B4 = V(@inbounds B[p + (u - 1), j + 3])
            Ĉ14 = fma(Â1, B4, Ĉ14)
            Ĉ24 = fma(Â2, B4, Ĉ24)
            B5 = V(@inbounds B[p + (u - 1), j + 4])
            Ĉ15 = fma(Â1, B5, Ĉ15)
            Ĉ25 = fma(Â2, B5, Ĉ25)
            B6 = V(@inbounds B[p + (u - 1), j + 5])
            Ĉ16 = fma(Â1, B6, Ĉ16)
            Ĉ26 = fma(Â2, B6, Ĉ26)
        end
        p += 8
        ptrÂ += 8incA
    end

    for _ in 1:pleft
        Â1 = vecload(V, A, ptrÂ, 1, 1)
        Â2 = vecload(V, A, ptrÂ, 2, 1)
        B1 = V(@inbounds B[p, j])
        Ĉ11 = fma(Â1, B1, Ĉ11)
        Ĉ21 = fma(Â2, B1, Ĉ21)
        B2 = V(@inbounds B[p, j+1])
        Ĉ12 = fma(Â1, B2, Ĉ12)
        Ĉ22 = fma(Â2, B2, Ĉ22)
        B3 = V(@inbounds B[p, j+2])
        Ĉ13 = fma(Â1, B3, Ĉ13)
        Ĉ23 = fma(Â2, B3, Ĉ23)
        B4 = V(@inbounds B[p, j+3])
        Ĉ14 = fma(Â1, B4, Ĉ14)
        Ĉ24 = fma(Â2, B4, Ĉ24)
        B5 = V(@inbounds B[p, j+4])
        Ĉ15 = fma(Â1, B5, Ĉ15)
        Ĉ25 = fma(Â2, B5, Ĉ25)
        B6 = V(@inbounds B[p, j+5])
        Ĉ16 = fma(Â1, B6, Ĉ16)
        Ĉ26 = fma(Â2, B6, Ĉ26)
        p += 1
        ptrÂ += incA
    end

    if β === false
        vecstore(α * Ĉ11, C, ptrĈ, 1, 1)
        vecstore(α * Ĉ21, C, ptrĈ, 2, 1)
        vecstore(α * Ĉ12, C, ptrĈ, 1, 2)
        vecstore(α * Ĉ22, C, ptrĈ, 2, 2)
        vecstore(α * Ĉ13, C, ptrĈ, 1, 3)
        vecstore(α * Ĉ23, C, ptrĈ, 2, 3)
        vecstore(α * Ĉ14, C, ptrĈ, 1, 4)
        vecstore(α * Ĉ24, C, ptrĈ, 2, 4)
        vecstore(α * Ĉ15, C, ptrĈ, 1, 5)
        vecstore(α * Ĉ25, C, ptrĈ, 2, 5)
        vecstore(α * Ĉ16, C, ptrĈ, 1, 6)
        vecstore(α * Ĉ26, C, ptrĈ, 2, 6)
    else
        vecstore(fma(α, Ĉ11, β * vecload(V, C, ptrĈ, 1, 1)), C, ptrĈ, 1, 1)
        vecstore(fma(α, Ĉ21, β * vecload(V, C, ptrĈ, 2, 1)), C, ptrĈ, 2, 1)
        vecstore(fma(α, Ĉ12, β * vecload(V, C, ptrĈ, 1, 2)), C, ptrĈ, 1, 2)
        vecstore(fma(α, Ĉ22, β * vecload(V, C, ptrĈ, 2, 2)), C, ptrĈ, 2, 2)
        vecstore(fma(α, Ĉ13, β * vecload(V, C, ptrĈ, 1, 3)), C, ptrĈ, 1, 3)
        vecstore(fma(α, Ĉ23, β * vecload(V, C, ptrĈ, 2, 3)), C, ptrĈ, 2, 3)
        vecstore(fma(α, Ĉ14, β * vecload(V, C, ptrĈ, 1, 4)), C, ptrĈ, 1, 4)
        vecstore(fma(α, Ĉ24, β * vecload(V, C, ptrĈ, 2, 4)), C, ptrĈ, 2, 4)
        vecstore(fma(α, Ĉ15, β * vecload(V, C, ptrĈ, 1, 5)), C, ptrĈ, 1, 5)
        vecstore(fma(α, Ĉ25, β * vecload(V, C, ptrĈ, 2, 5)), C, ptrĈ, 2, 5)
        vecstore(fma(α, Ĉ16, β * vecload(V, C, ptrĈ, 1, 6)), C, ptrĈ, 1, 6)
        vecstore(fma(α, Ĉ26, β * vecload(V, C, ptrĈ, 2, 6)), C, ptrĈ, 2, 6)
    end

    return nothing
end
#TODO: (A'B), AB', and A'B'

function microkernel!
end

###
### Utilities
###

function checkmulsize(C, A, B)
    cm, cn = size(C)
    am, ak = size(A)
    bk, bn = size(B)
    (cm == am && ak == bk && cn == bn) || throw(DimensionMismatch("C has dimensions ($cm, $cn), A has dimensions ($am, $ak), but B has dimensions ($bk, $bn)"))
    return cm, ak, bn
end

@inline getptr(A::SIMD.FastContiguousArray, i, j) = pointer(A, (j - 1) * stride(A, 2) + i)

@inline function vecload(V::Type{Vec{N,T}}, A::SIMD.FastContiguousArray, ptr::Ptr{T}, i, j)::V where {N,T}
    ii = (j - 1) * stride(A, 2) + (i - 1) * N
    return vload(V, ptr + ii*sizeof(T))
end
@inline function vecstore(v::Vec{N,T}, A::SIMD.FastContiguousArray, ptr::Ptr{T}, i, j) where {N,T}
    ii = (j - 1) * stride(A, 2) + (i - 1) * N
    vstore(v, ptr + ii*sizeof(T))
    return nothing
end
