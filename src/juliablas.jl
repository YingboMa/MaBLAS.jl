using SIMD
using LinearAlgebra: Transpose, Adjoint
using Base.Cartesian: @nexprs
using StaticArrays

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

function mul!(C, A, B, α=true, β=false; cache_params=(cache_m=72, cache_k=256, cache_n=4080), packing=false)
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
    kernel_params = (Val(8), Val(6))
    if packing
        packing_mul!(C, A, B, α, β, cache_params, kernel_params)
    else
        tiling_mul!(C, A, B, α, β, cache_params, kernel_params)
    end
    return C
end

###
### Lower-level `_mul!`
###

function packing_mul!(C, A, B, α, β, (cache_m, cache_k, cache_n), kernel_params::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}
    Abuffer = zeros(cache_m * cache_k)
    Bbuffer = zeros(cache_k * cache_n)
    m, k, n = checkmulsize(C, A, B)
    for cachej₁ in 1:cache_n:n; cachej₂ = min(cachej₁ + cache_n - 1, n)
        for cachep₁ in 1:cache_k:k; cachep₂ = min(cachep₁ + cache_k - 1, k)
            _β = cachep₁ == 1 ? β : one(β)
            packBbuffer!(Bbuffer, B, cachep₁, cachep₂, cachej₁, cachej₂, micro_n)
            for cachei₁ in 1:cache_m:m; cachei₂ = min(cachei₁ + cache_m - 1, m)
                packAbuffer!(Abuffer, A, cachei₁, cachei₂, cachep₁, cachep₂, micro_m)
                # macrokernel
                for microj₁ in cachej₁:micro_n:cachej₂; nleft = min(cachej₂ - microj₁ + 1, micro_n)
                    for microi₁ in cachei₁:micro_m:cachei₂; mleft = min(cachei₂ - microi₁ + 1, micro_m)
                        if nleft == micro_n && mleft == micro_m
                            # (micro_m × ks) * (ks × micro_n)
                            # Ĉ = A[i:(i+micro_m-1), ps] * B[ps, j:(j+micro_n-1)]
                            packing_microkernel!(C, Abuffer, Bbuffer, α, _β, microi₁, microj₁, cachei₁, cachej₁, cachep₁, cachep₂, kernel_params)
                        else
                            # (mleft × ks) * (ks × nleft)
                            # Ĉ = A[i:(i+mleft-1), ps] * B[ps, j:(j+nleft-1)]
                            #cleanup_packing_microkernel!(C, A, B, α, _β, microi₁, microj₁, cachep₁, cachep₂, mleft, nleft)
                        end
                    end # microi
                end # microj
            end # cachei
        end # cachep
    end # cachej
    return C
end

function tiling_mul!(C, A, B, α, β, (cache_m, cache_k, cache_n), kernel_params::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}
    cs1 = stride(C, 1)
    as1 = stride(A, 1)
    bs1 = stride(B, 1)
    if !(cs1 == as1 == bs1 == 1)
        throw(ArgumentError("tiling_mul! doesn't support nonunit leading stride matrices. Got stride(C, 1) = $cs1, stride(A, 1) = $as1, and stride(B, 1) = $bs1."))
    end
    m, k, n = checkmulsize(C, A, B)
    for cachej₁ in 1:cache_n:n; cachej₂ = min(cachej₁ + cache_n - 1, n)
        for cachep₁ in 1:cache_k:k; cachep₂ = min(cachep₁ + cache_k - 1, k)
            _β = cachep₁ == 1 ? β : one(β)
            for cachei₁ in 1:cache_m:m; cachei₂ = min(cachei₁ + cache_m - 1, m)
                # macrokernel
                for microj₁ in cachej₁:micro_n:cachej₂; nleft = min(cachej₂ - microj₁ + 1, micro_n)
                    for microi₁ in cachei₁:micro_m:cachei₂; mleft = min(cachei₂ - microi₁ + 1, micro_m)
                        if nleft == micro_n && mleft == micro_m
                            # (micro_m × ks) * (ks × micro_n)
                            # Ĉ = A[i:(i+micro_m-1), ps] * B[ps, j:(j+micro_n-1)]
                            tiling_microkernel!(C, A, B, α, _β, microi₁, microj₁, cachep₁, cachep₂, kernel_params)
                        else
                            # (mleft × ks) * (ks × nleft)
                            # Ĉ = A[i:(i+mleft-1), ps] * B[ps, j:(j+nleft-1)]
                            cleanup_tiling_microkernel!(C, A, B, α, _β, microi₁, microj₁, cachep₁, cachep₂, mleft, nleft)
                        end
                    end # microi
                end # microj
            end # cachei
        end # cachep
    end # cachej
    return C
end

###
### Packing
###

function packAbuffer!(Abuffer, A, cachei₁, cachei₂, cachep₁, cachep₂, micro_m)
    # terminology:
    # a `micro_m × 1` vector is a panel
    # a `micro_m × cache_k` matrix is a block
    l = 0 # write position in packing buffer
    for i₁ in cachei₁:micro_m:cachei₂ # iterate over blocks
        i₂ = i₁ + micro_m - 1
        if i₂ <= cachei₂ # full panel
            # copy a panel to the buffer contiguously
            for p in cachep₁:cachep₂, i in i₁:i₂
                @inbounds Abuffer[l += 1] = A[i, p]
            end
        else # a panel is not full
            for p in cachep₁:cachep₂ # iterate through panel columns
                for i in i₁:cachei₂  # iterate through live panel rows
                    @inbounds Abuffer[l += 1] = A[i, p]
                end
                for i in (i₂ + 1):i₂ # pad the rest of the panel with zero
                    @inbounds Abuffer[l += 1] = zero(eltype(Abuffer))
                end
            end
        end
    end
    return nothing
end

function packBbuffer!(Bbuffer, B, cachep₁, cachep₂, cachej₁, cachej₂, micro_n)
    # terminology:
    # a `1 × micro_n` vector is a panel
    # a `cache_k × micro_n` matrix is a block
    l = 0 # write position in packing buffer
    for j₁ in cachej₁:micro_n:cachej₂ # iterate over blocks
        j₂ = j₁ + micro_n - 1
        if j₂ <= cachej₂ # full panel
            # copy a panel to the buffer contiguously
            for p in cachep₁:cachep₂, j in j₁:j₂
                @inbounds Bbuffer[l += 1] = B[p, j]
            end
        else # a panel is not full
            for p in cachep₁:cachep₂ # iterate through panel columns
                for j in j₁:cachej₂  # iterate through live panel rows
                    @inbounds Bbuffer[l += 1] = B[i, p]
                end
                for j in (j₂ + 1):j₂ # pad the rest of the panel with zero
                    @inbounds Bbuffer[l += 1] = zero(eltype(Bbuffer))
                end
            end
        end
    end
    return nothing
end

###
### Micro kernel
###

@noinline function packing_microkernel!(C::StridedMatrix{Float64}, Abuffer::StridedVector{Float64}, Bbuffer::StridedVector{Float64}, α, β,
                                        microi₁, microj₁, cachei₁, cachej₁, cachep₁, cachep₂, ::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}
    T = Float64
    N = 4
    V = Vec{N, T}
    st = sizeof(T)
    sv = sizeof(V)
    sc2 = stride(C, 2)*st
    # we unroll 8 times
    piters = cachep₂ - cachep₁ + 1
    punroll, pleft = divrem(piters, 8)

    ptrĈ = getptr(C, microi₁, microj₁) # pointer with offset
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

    Aoffset = (microi₁ - cachei₁) * piters * st
    Boffset = (microj₁ - cachej₁) * piters * st
    ptrÂ = pointer(Abuffer) + Aoffset
    ptrB̂ = pointer(Bbuffer) + Boffset

    # prefetch C matrix
    prefetcht0(ptrĈ)
    prefetcht0(ptrĈ +  sc2)
    prefetcht0(ptrĈ + 2sc2)
    prefetcht0(ptrĈ + 3sc2)
    prefetcht0(ptrĈ + 4sc2)
    prefetcht0(ptrĈ + 5sc2)
    # rank-1 updates
    for _ in 1:punroll
        prefetcht0(ptrÂ + 8 * st * micro_m)
        @nexprs 8 u -> begin
            # iteration u
            Â1 = vload(V, ptrÂ + (u - 1) * st * micro_m)
            Â2 = vload(V, ptrÂ + (u - 1) * st * micro_m + sv)
            #B1..6 = V(@inbounds B[p + (u - 1), j + 0..5])
            B1 = V(unsafe_load(ptrB̂ + (u - 1) * st * micro_n + 0))
            Ĉ11 = fma(Â1, B1, Ĉ11)
            Ĉ21 = fma(Â2, B1, Ĉ21)
            B2 = V(unsafe_load(ptrB̂ + (u - 1) * st * micro_n + st))
            Ĉ12 = fma(Â1, B2, Ĉ12)
            Ĉ22 = fma(Â2, B2, Ĉ22)
            B3 = V(unsafe_load(ptrB̂ + (u - 1) * st * micro_n + 2st))
            Ĉ13 = fma(Â1, B3, Ĉ13)
            Ĉ23 = fma(Â2, B3, Ĉ23)
            B4 = V(unsafe_load(ptrB̂ + (u - 1) * st * micro_n + 3st))
            Ĉ14 = fma(Â1, B4, Ĉ14)
            Ĉ24 = fma(Â2, B4, Ĉ24)
            B5 = V(unsafe_load(ptrB̂ + (u - 1) * st * micro_n + 4st))
            Ĉ15 = fma(Â1, B5, Ĉ15)
            Ĉ25 = fma(Â2, B5, Ĉ25)
            B6 = V(unsafe_load(ptrB̂ + (u - 1) * st * micro_n + 5st))
            Ĉ16 = fma(Â1, B6, Ĉ16)
            Ĉ26 = fma(Â2, B6, Ĉ26)
        end
        ptrÂ += 8 * st * micro_m
        ptrB̂ += 8 * st * micro_n
    end

    for _ in 1:pleft
        Â1 = vload(V, ptrÂ)
        Â2 = vload(V, ptrÂ + sv)
        B1 = V(unsafe_load(ptrB̂))
        Ĉ11 = fma(Â1, B1, Ĉ11)
        Ĉ21 = fma(Â2, B1, Ĉ21)
        B2 = V(unsafe_load(ptrB̂ + st))
        Ĉ12 = fma(Â1, B2, Ĉ12)
        Ĉ22 = fma(Â2, B2, Ĉ22)
        B3 = V(unsafe_load(ptrB̂ + 2st))
        Ĉ13 = fma(Â1, B3, Ĉ13)
        Ĉ23 = fma(Â2, B3, Ĉ23)
        B4 = V(unsafe_load(ptrB̂ + 3st))
        Ĉ14 = fma(Â1, B4, Ĉ14)
        Ĉ24 = fma(Â2, B4, Ĉ24)
        B5 = V(unsafe_load(ptrB̂ + 4st))
        Ĉ15 = fma(Â1, B5, Ĉ15)
        Ĉ25 = fma(Â2, B5, Ĉ25)
        B6 = V(unsafe_load(ptrB̂ + 5st))
        Ĉ16 = fma(Â1, B6, Ĉ16)
        Ĉ26 = fma(Â2, B6, Ĉ26)
        ptrÂ += st * micro_m
        ptrB̂ += st * micro_n
    end

    if iszero(β)
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

# (micro_m × ks) * (ks × micro_n)
# Ĉ = A[i:(i+micro_m-1), ps] * B[ps, j:(j+micro_n-1)]
@noinline function tiling_microkernel!(C::StridedMatrix{Float64}, A::StridedMatrix{Float64}, B::StridedMatrix{Float64}, α, β,
                                     i, j, p₁, p₂, ::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}
    T = Float64
    N = 4
    V = Vec{N, T}
    st = sizeof(T)
    sv = sizeof(V)
    sa2 = stride(A, 2)*st
    sb2 = stride(B, 2)*st
    sc2 = stride(C, 2)*st
    # we unroll 8 times
    punroll, pleft = divrem(p₂ - p₁ + 1, 8)

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
    p = p₁
    ptrÂ = getptr(A, i, p)
    ptrB̂ = getptr(B, p, j)

    # prefetch C matrix
    prefetcht0(ptrĈ)
    prefetcht0(ptrĈ +  sc2)
    prefetcht0(ptrĈ + 2sc2)
    prefetcht0(ptrĈ + 3sc2)
    prefetcht0(ptrĈ + 4sc2)
    prefetcht0(ptrĈ + 5sc2)
    for _ in 1:punroll
        prefetcht0(ptrÂ + 8sa2)
        @nexprs 8 u -> begin
            # iteration u
            Â1 = vload(V, ptrÂ + (u - 1) * sa2)
            Â2 = vload(V, ptrÂ + (u - 1) * sa2 + sv)
            #B1..6 = V(@inbounds B[p + (u - 1), j + 0..5])
            B1 = V(unsafe_load(ptrB̂ + (u - 1) * st))
            Ĉ11 = fma(Â1, B1, Ĉ11)
            Ĉ21 = fma(Â2, B1, Ĉ21)
            B2 = V(unsafe_load(ptrB̂ + (u - 1) * st + sb2))
            Ĉ12 = fma(Â1, B2, Ĉ12)
            Ĉ22 = fma(Â2, B2, Ĉ22)
            B3 = V(unsafe_load(ptrB̂ + (u - 1) * st + 2sb2))
            Ĉ13 = fma(Â1, B3, Ĉ13)
            Ĉ23 = fma(Â2, B3, Ĉ23)
            B4 = V(unsafe_load(ptrB̂ + (u - 1) * st + 3sb2))
            Ĉ14 = fma(Â1, B4, Ĉ14)
            Ĉ24 = fma(Â2, B4, Ĉ24)
            B5 = V(unsafe_load(ptrB̂ + (u - 1) * st + 4sb2))
            Ĉ15 = fma(Â1, B5, Ĉ15)
            Ĉ25 = fma(Â2, B5, Ĉ25)
            B6 = V(unsafe_load(ptrB̂ + (u - 1) * st + 5sb2))
            Ĉ16 = fma(Â1, B6, Ĉ16)
            Ĉ26 = fma(Â2, B6, Ĉ26)
        end
        p += 8
        ptrÂ += 8sa2
        ptrB̂ += 8st # sb1
    end

    for _ in 1:pleft
        Â1 = vload(V, ptrÂ)
        Â2 = vload(V, ptrÂ + sv)
        B1 = V(unsafe_load(ptrB̂))
        Ĉ11 = fma(Â1, B1, Ĉ11)
        Ĉ21 = fma(Â2, B1, Ĉ21)
        B2 = V(unsafe_load(ptrB̂ + sb2))
        Ĉ12 = fma(Â1, B2, Ĉ12)
        Ĉ22 = fma(Â2, B2, Ĉ22)
        B3 = V(unsafe_load(ptrB̂ + 2sb2))
        Ĉ13 = fma(Â1, B3, Ĉ13)
        Ĉ23 = fma(Â2, B3, Ĉ23)
        B4 = V(unsafe_load(ptrB̂ + 3sb2))
        Ĉ14 = fma(Â1, B4, Ĉ14)
        Ĉ24 = fma(Â2, B4, Ĉ24)
        B5 = V(unsafe_load(ptrB̂ + 4sb2))
        Ĉ15 = fma(Â1, B5, Ĉ15)
        Ĉ25 = fma(Â2, B5, Ĉ25)
        B6 = V(unsafe_load(ptrB̂ + 5sb2))
        Ĉ16 = fma(Â1, B6, Ĉ16)
        Ĉ26 = fma(Â2, B6, Ĉ26)
        p += 1
        ptrÂ += sa2
        ptrB̂ += st # sb1
    end

    if iszero(β)
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

# (mleft × nleft) = (mleft × ks) * (ks × nleft)
# Ĉ = A[i:(i+mleft-1), ps] * B[ps, j:(j+nleft-1)]
function cleanup_tiling_microkernel!(C, A, B, α, β, i, j, p₁, p₂, mleft, nleft)
    iszeroβ = iszero(β)
    @inbounds for ĵ in j:(j+nleft-1), î in i:(i+mleft-1)
        ABîĵ = zero(eltype(C))
        @simd ivdep for p̂ in p₁:p₂
            ABîĵ = muladd(A[î, p̂], B[p̂, ĵ], ABîĵ)
        end
        C[î, ĵ] = iszeroβ ? α * ABîĵ : muladd(α, ABîĵ, β * C[î, ĵ])
    end
    return nothing
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

@inline getptr(A::StridedMatrix{T}, i, j) where T = pointer(A) + ((j - 1) * stride(A, 2) + i - 1) * sizeof(T)

@inline function vecload(V::Type{Vec{N,T}}, A::StridedMatrix, ptr::Ptr{T}, i, j)::V where {N,T}
    ii = (j - 1) * stride(A, 2) + (i - 1) * N
    return vload(V, ptr + ii*sizeof(T))
end
@inline function vecstore(v::Vec{N,T}, A::StridedMatrix, ptr::Ptr{T}, i, j) where {N,T}
    ii = (j - 1) * stride(A, 2) + (i - 1) * N
    vstore(v, ptr + ii*sizeof(T))
    return nothing
end

prefetcht0(ptr::Ptr) = Base.llvmcall(raw"""
                                     %ptr = inttoptr i64 %0 to i8*
                                     call void asm "prefetcht0 $0", "*m"(i8* nonnull %ptr)
                                     ret void
                                     """, Cvoid, Tuple{Ptr{Cvoid}}, convert(Ptr{Cvoid}, ptr))
