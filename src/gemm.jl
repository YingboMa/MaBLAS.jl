using SIMDPirates
using SIMDPirates: vfmadd231
using LinearAlgebra: Transpose, Adjoint
using Base.Cartesian: @nexprs
using ..LoopInfo
using LoopVectorization: @avx
using VectorizationBase
using VectorizationBase: offset, AbstractPointer

# General direction: we want to avoid pointer arithmetic as much as possible,
# also, don't strict type the container type

# I'll try to implement everything without defining a type, and all variables
# should stay local if possible. We can always clean up the code later when we
# find a good direction/structure.

###
### Prelude
###

const N_THREADS_BUFFERS = map(_->Vector{UInt8}(undef, 0), 1:Threads.nthreads())

# alias scopes from https://github.com/JuliaLang/julia/pull/31018
struct Const{T<:Array}
    a::T
end

@eval Base.getindex(A::Const, i1::Int) = Core.const_arrayref($(Expr(:boundscheck)), A.a, i1)
@eval Base.getindex(A::Const, i1::Int, i2::Int, I::Int...) = (Base.@_inline_meta; Core.const_arrayref($(Expr(:boundscheck)), A.a, i1, i2, I...))
constarray(a::Array) = Const(a)
constarray(a) = a

macro aliasscope(body)
    sym = gensym()
    esc(quote
        $(Expr(:aliasscope))
        $sym = $body
        $(Expr(:popaliasscope))
        $sym
    end)
end

###
### User-level API
###

function mul!(C, A, B, α=true, β=false; cache_params=(cache_m=72, cache_k=256, cache_n=4080), kernel_params=(Val(8), Val(6)), packing=false)
    m, k, n = checkmulsize(C, A, B)
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
    cs1 = stride(C, 1)
    if cs1 != 1
        throw(ArgumentError("packing_mul! doesn't support nonunit leading stride C matrix. Got stride(C, 1) = $cs1."))
    end
    m, k, n = checkmulsize(C, A, B)

    # get buffer
    Abuffersize = cache_m * cache_k
    Bbuffersize = cache_k * cache_n
    ABbuffersize = micro_m * micro_n
    T = eltype(C)
    buffer = N_THREADS_BUFFERS[Threads.threadid()]
    resize!(buffer, (Abuffersize + Bbuffersize + ABbuffersize) * sizeof(T))
    ptrbuffer = convert(Ptr{T}, pointer(buffer))
    Abuffer = unsafe_wrap(Array, ptrbuffer, Abuffersize); ptrbuffer += Abuffersize * sizeof(T)
    Bbuffer = unsafe_wrap(Array, ptrbuffer, Bbuffersize); ptrbuffer += Bbuffersize * sizeof(T)
    ABbuffer = unsafe_wrap(Array, ptrbuffer, (micro_m, micro_n))

    for cachej₁ in 1:cache_n:n; cachej₂ = min(cachej₁ + cache_n - 1, n)
        for cachep₁ in 1:cache_k:k; cachep₂ = min(cachep₁ + cache_k - 1, k)
            _β = cachep₁ == 1 ? β : one(β)
            ps = cachep₂ - cachep₁ + 1
            packBbuffer!(Bbuffer, B, cachep₁, cachep₂, cachej₁, cachej₂, Val(micro_n))
            for cachei₁ in 1:cache_m:m; cachei₂ = min(cachei₁ + cache_m - 1, m)
                packAbuffer!(Abuffer, A, cachei₁, cachei₂, cachep₁, cachep₂, Val(micro_m))
                # macrokernel
                for microj₁ in cachej₁:micro_n:cachej₂; nleft = min(cachej₂ - microj₁ + 1, micro_n)
                    Boffset = (microj₁ - cachej₁) * ps
                    for microi₁ in cachei₁:micro_m:cachei₂; mleft = min(cachei₂ - microi₁ + 1, micro_m)
                        Aoffset = (microi₁ - cachei₁) * ps
                        if nleft == micro_n && mleft == micro_m
                            # (micro_m × ks) * (ks × micro_n)
                            # Ĉ = A[i:(i+micro_m-1), ps] * B[ps, j:(j+micro_n-1)]
                            Coffset = (microj₁ - 1) * stride(C, 2) + (microi₁ - 1) * stride(C, 1)
                            packing_microkernel!(C, Abuffer, Bbuffer, α, _β, Coffset, Aoffset, Boffset, ps, kernel_params)
                        else
                            # microkernel writes to the `AB` buffer
                            packing_microkernel!(ABbuffer, Abuffer, Bbuffer, α, false, 0, Aoffset, Boffset, ps, kernel_params)
                            # copy the `AB` buffer to `C` with `β` scaling
                            # Ĉ = AB[1:mleft, 1:nleft]
                            cleanup_packing!(C, ABbuffer, _β, (microi₁ - 1, microj₁ - 1), mleft, nleft)
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

function packAbuffer!(Abuffer, A, cachei₁, cachei₂, cachep₁, cachep₂, ::Val{micro_m}) where micro_m
    # terminology:
    # a `micro_m × 1` vector is a panel
    # a `micro_m × cache_k` matrix is a block
    l = 0 # write position in packing buffer
    @inbounds @aliasscope for i₁ in cachei₁:micro_m:cachei₂ # iterate over blocks
        i₂ = i₁ + micro_m - 1
        if i₂ <= cachei₂ # full panel
            # copy a panel to the buffer contiguously
            @simd for p in cachep₁:cachep₂
                @unroll for i in i₁:i₂ # has constant loop count
                    Abuffer[l += 1] = constarray(A)[i, p]
                end
            end
        else # a panel is not full
            for p in cachep₁:cachep₂ # iterate through panel columns
                for i in i₁:cachei₂  # iterate through live panel rows
                    Abuffer[l += 1] = constarray(A)[i, p]
                end
                for i in (cachei₂ + 1):i₂ # pad the rest of the panel with zero
                    Abuffer[l += 1] = zero(eltype(Abuffer))
                end
            end
        end
    end
    return nothing
end

function packBbuffer!(Bbuffer, B, cachep₁, cachep₂, cachej₁, cachej₂, ::Val{micro_n}) where micro_n
    # terminology:
    # a `1 × micro_n` vector is a panel
    # a `cache_k × micro_n` matrix is a block
    l = 0 # write position in packing buffer
    @inbounds @aliasscope for j₁ in cachej₁:micro_n:cachej₂ # iterate over blocks
        j₂ = j₁ + micro_n - 1
        if j₂ <= cachej₂ # full panel
            # copy a panel to the buffer contiguously
            @simd for p in cachep₁:cachep₂
                @unroll for j in j₁:j₂ # has constant loop count
                    Bbuffer[l += 1] = constarray(B)[p, j]
                end
            end
        else # a panel is not full
            for p in cachep₁:cachep₂ # iterate through panel columns
                for j in j₁:cachej₂  # iterate through live panel rows
                    Bbuffer[l += 1] = constarray(B)[p, j]
                end
                for j in (cachej₂ + 1):j₂ # pad the rest of the panel with zero
                    Bbuffer[l += 1] = zero(eltype(Bbuffer))
                end
            end
        end
    end
    return nothing
end

###
### Micro kernel
###

@generated function packing_microkernel!(C::StridedMatrix{T}, Abuffer::StridedVector{T}, Bbuffer::StridedVector{T}, α, β,
                                        Coffset, Aoffset, Boffset, ps, ::Tuple{Val{micro_m},Val{micro_n}}) where {T,micro_m,micro_n}
    N = VectorizationBase.pick_vector_width(T)
    V = SVec{N,T}
    MM = _MM{N}
    unroll = 4
    m2 = micro_m ÷ N

    quote
        punroll, pleft = divrem(ps, $unroll)

        ptrĈ = stridedpointer(C, Coffset + 1) # pointer with offset
        ptrÂ = stridedpointer(Abuffer, Aoffset + 1)
        ptrB̂ = stridedpointer(Bbuffer, Boffset + 1)

        # prefetch C matrix
        # TODO
        @nexprs $micro_n n̂ -> begin
            prefetcht0(ptrĈ + offset(ptrĈ, (0, n̂ - 1)) + 7*2)
        end

        # rank-1 updates
        @nexprs $micro_n n̂ -> @nexprs $m2 m̂ -> AB_m̂_n̂ = zero($V)
        for _ in 1:punroll
            @nexprs $unroll u -> begin
                #TODO
                u == 1 && prefetcht0(pointer(ptrÂ) + 64 * $micro_m + 2 * $micro_n)
                u == 3 && prefetcht0(pointer(ptrÂ) + 76 * $micro_m + 2 * $micro_n)
                ## iteration u
                @nexprs $m2 m̂ -> A_m̂ = vload($V, ptrÂ + (u - 1) * $micro_m + (m̂ - 1) * $N)
                #B1..6 = V(@inbounds B[p + (u - 1), j + 0..5])
                @nexprs $micro_n n̂ -> begin
                    B_n̂ = $V(ptrB̂[n̂ + (u - 1) * $micro_n])
                    @nexprs $m2 m̂ -> begin
                        AB_m̂_n̂ = vfmadd231(A_m̂, B_n̂, AB_m̂_n̂)
                    end
                end
            end
            ptrÂ += $(unroll * micro_m)
            ptrB̂ += $(unroll * micro_n)
        end

        for _ in 1:pleft
            #TODO
            prefetcht0(pointer(ptrÂ) + 64 * $micro_m + 2 * $micro_n)
            @nexprs $m2 m̂ -> A_m̂ = vload($V, ptrÂ + (m̂ - 1) * $N)
            #B1..6 = V(@inbounds B[p + (u - 1), j + 0..5])
            @nexprs $micro_n n̂ -> begin
                B_n̂ = $V(ptrB̂[n̂])
                @nexprs $m2 m̂ -> begin
                    AB_m̂_n̂ = vfmadd231(A_m̂, B_n̂, AB_m̂_n̂)
                end
            end
            ptrÂ += $micro_m
            ptrB̂ += $micro_n
        end

        _α = $V(α)
        if iszero(β)
            @nexprs $micro_n n̂ -> @nexprs $m2 m̂ -> begin
                C_m̂_n̂ = _α * AB_m̂_n̂
                vstore!(ptrĈ, C_m̂_n̂, ((m̂ - 1) * $N + 1, n̂))
            end
        else
            _β = $V(β)
            @nexprs $micro_n n̂ -> @nexprs $m2 m̂ -> begin
                C_m̂_n̂ = _β * vload(ptrĈ, ($MM((m̂ - 1) * $N + 1), n̂))
                C_m̂_n̂ = fma(_α, AB_m̂_n̂, C_m̂_n̂)
                vstore!(ptrĈ, C_m̂_n̂, ((m̂ - 1) * $N + 1, n̂))
            end
        end

        return nothing
    end
end

# (micro_m × ks) * (ks × micro_n)
# Ĉ = A[i:(i+micro_m-1), ps] * B[ps, j:(j+micro_n-1)]
@noinline function tiling_microkernel!(C::StridedMatrix{Float64}, A::StridedMatrix{Float64}, B::StridedMatrix{Float64}, α, β,
                                       i, j, p₁, p₂, ::Tuple{Val{8},Val{6}})
    T = Float64
    N = 4
    V = SVec{N, T}
    st = sizeof(T)
    sv = sizeof(V)
    sa2 = stride(A, 2)*st
    sb2 = stride(B, 2)*st
    sc2 = stride(C, 2)*st
    micro_m, micro_n = 8, 6
    unroll = 4
    punroll, pleft = divrem(p₂ - p₁ + 1, unroll)

    p = p₁
    ptrĈ = getptr(C, i, j) # pointer with offset
    ptrĈ3 = ptrĈ + 3sc2
    ptrÂ = getptr(A, i, p)
    ptrB̂ = getptr(B, p, j)
    # preload A
    Â1 = vload(V, ptrÂ)
    Â2 = vload(V, ptrÂ + sv)

    # prefetch C matrix
    prefetcht0(ptrĈ + 7*8)
    prefetcht0(ptrĈ +  sc2 + 7*8)
    prefetcht0(ptrĈ + 2sc2 + 7*8)
    prefetcht0(ptrĈ3)
    prefetcht0(ptrĈ3 +  sc2 + 7*8)
    prefetcht0(ptrĈ3 + 2sc2 + 7*8)

    # rank-1 updates
    Ĉ11 = Ĉ21 = Ĉ12 = Ĉ22 = Ĉ13 = Ĉ23 = Ĉ14 = Ĉ24 = Ĉ15 = Ĉ25 = Ĉ16 = Ĉ26 = zero(V)
    for _ in 1:punroll
        # TODO
        #prefetcht0(ptrÂ + 8sa2)
        @nexprs 4 u -> begin
            # iteration u
            #B1..6 = V(@inbounds B[p + (u - 1), j + 0..5])
            B1 = V(unsafe_load(ptrB̂ + (u - 1) * st))
            Ĉ11 = vfmadd231(Â1, B1, Ĉ11)
            Ĉ21 = vfmadd231(Â2, B1, Ĉ21)
            B2 = V(unsafe_load(ptrB̂ + (u - 1) * st + sb2))
            Ĉ12 = vfmadd231(Â1, B2, Ĉ12)
            Ĉ22 = vfmadd231(Â2, B2, Ĉ22)
            B3 = V(unsafe_load(ptrB̂ + (u - 1) * st + 2sb2))
            Ĉ13 = vfmadd231(Â1, B3, Ĉ13)
            Ĉ23 = vfmadd231(Â2, B3, Ĉ23)
            B4 = V(unsafe_load(ptrB̂ + (u - 1) * st + 3sb2))
            Ĉ14 = vfmadd231(Â1, B4, Ĉ14)
            Ĉ24 = vfmadd231(Â2, B4, Ĉ24)
            B5 = V(unsafe_load(ptrB̂ + (u - 1) * st + 4sb2))
            Ĉ15 = vfmadd231(Â1, B5, Ĉ15)
            Ĉ25 = vfmadd231(Â2, B5, Ĉ25)
            B6 = V(unsafe_load(ptrB̂ + (u - 1) * st + 5sb2))
            Ĉ16 = vfmadd231(Â1, B6, Ĉ16)
            Ĉ26 = vfmadd231(Â2, B6, Ĉ26)
            Â1 = vload(V, ptrÂ + u * sa2)
            Â2 = vload(V, ptrÂ + u * sa2 + sv)
        end
        p += unroll
        ptrÂ += unroll * sa2
        ptrB̂ += unroll * st # sb1
    end

    for _ in 1:pleft
        B1 = V(unsafe_load(ptrB̂))
        Ĉ11 = vfmadd231(Â1, B1, Ĉ11)
        Ĉ21 = vfmadd231(Â2, B1, Ĉ21)
        B2 = V(unsafe_load(ptrB̂ + sb2))
        Ĉ12 = vfmadd231(Â1, B2, Ĉ12)
        Ĉ22 = vfmadd231(Â2, B2, Ĉ22)
        B3 = V(unsafe_load(ptrB̂ + 2sb2))
        Ĉ13 = vfmadd231(Â1, B3, Ĉ13)
        Ĉ23 = vfmadd231(Â2, B3, Ĉ23)
        B4 = V(unsafe_load(ptrB̂ + 3sb2))
        Ĉ14 = vfmadd231(Â1, B4, Ĉ14)
        Ĉ24 = vfmadd231(Â2, B4, Ĉ24)
        B5 = V(unsafe_load(ptrB̂ + 4sb2))
        Ĉ15 = vfmadd231(Â1, B5, Ĉ15)
        Ĉ25 = vfmadd231(Â2, B5, Ĉ25)
        B6 = V(unsafe_load(ptrB̂ + 5sb2))
        Ĉ16 = vfmadd231(Â1, B6, Ĉ16)
        Ĉ26 = vfmadd231(Â2, B6, Ĉ26)
        p += 1
        ptrÂ += sa2
        ptrB̂ += st # sb1
        Â1 = vload(V, ptrÂ)
        Â2 = vload(V, ptrÂ + sv)
    end

    _α = V(α)
    if iszero(β)
        vecstore!(_α * Ĉ11, C, ptrĈ, 1, 1)
        vecstore!(_α * Ĉ21, C, ptrĈ, 2, 1)
        vecstore!(_α * Ĉ12, C, ptrĈ, 1, 2)
        vecstore!(_α * Ĉ22, C, ptrĈ, 2, 2)
        vecstore!(_α * Ĉ13, C, ptrĈ, 1, 3)
        vecstore!(_α * Ĉ23, C, ptrĈ, 2, 3)
        vecstore!(_α * Ĉ14, C, ptrĈ, 1, 4)
        vecstore!(_α * Ĉ24, C, ptrĈ, 2, 4)
        vecstore!(_α * Ĉ15, C, ptrĈ, 1, 5)
        vecstore!(_α * Ĉ25, C, ptrĈ, 2, 5)
        vecstore!(_α * Ĉ16, C, ptrĈ, 1, 6)
        vecstore!(_α * Ĉ26, C, ptrĈ, 2, 6)
    else
        _β = V(β)
        vecstore!(fma(_α, Ĉ11, _β * vecload(V, C, ptrĈ, 1, 1)), C, ptrĈ, 1, 1)
        vecstore!(fma(_α, Ĉ21, _β * vecload(V, C, ptrĈ, 2, 1)), C, ptrĈ, 2, 1)
        vecstore!(fma(_α, Ĉ12, _β * vecload(V, C, ptrĈ, 1, 2)), C, ptrĈ, 1, 2)
        vecstore!(fma(_α, Ĉ22, _β * vecload(V, C, ptrĈ, 2, 2)), C, ptrĈ, 2, 2)
        vecstore!(fma(_α, Ĉ13, _β * vecload(V, C, ptrĈ, 1, 3)), C, ptrĈ, 1, 3)
        vecstore!(fma(_α, Ĉ23, _β * vecload(V, C, ptrĈ, 2, 3)), C, ptrĈ, 2, 3)
        vecstore!(fma(_α, Ĉ14, _β * vecload(V, C, ptrĈ, 1, 4)), C, ptrĈ, 1, 4)
        vecstore!(fma(_α, Ĉ24, _β * vecload(V, C, ptrĈ, 2, 4)), C, ptrĈ, 2, 4)
        vecstore!(fma(_α, Ĉ15, _β * vecload(V, C, ptrĈ, 1, 5)), C, ptrĈ, 1, 5)
        vecstore!(fma(_α, Ĉ25, _β * vecload(V, C, ptrĈ, 2, 5)), C, ptrĈ, 2, 5)
        vecstore!(fma(_α, Ĉ16, _β * vecload(V, C, ptrĈ, 1, 6)), C, ptrĈ, 1, 6)
        vecstore!(fma(_α, Ĉ26, _β * vecload(V, C, ptrĈ, 2, 6)), C, ptrĈ, 2, 6)
    end

    return nothing
end
#TODO: (A'B), AB', and A'B'

###
### Clean up loops
###

# (mleft × nleft) = (mleft × ks) * (ks × nleft)
# Ĉ = A[i:(i+mleft-1), ps] * B[ps, j:(j+nleft-1)]
function cleanup_tiling_microkernel!(C, A, B, α, β, i, j, p₁, p₂, mleft, nleft)
    if iszero(β)
        @avx for ĵ in j:(j+nleft-1), î in i:(i+mleft-1)
            ABîĵ = zero(eltype(C))
            for p̂ in p₁:p₂
                ABîĵ = muladd(A[î, p̂], B[p̂, ĵ], ABîĵ)
            end
            C[î, ĵ] = α * ABîĵ
        end
    else
        @avx for ĵ in j:(j+nleft-1), î in i:(i+mleft-1)
            ABîĵ = zero(eltype(C))
            for p̂ in p₁:p₂
                ABîĵ = muladd(A[î, p̂], B[p̂, ĵ], ABîĵ)
            end
            Cîĵ = C[î, ĵ]
            C[î, ĵ] = muladd(α, ABîĵ, β * Cîĵ)
        end
    end
    return nothing
end

# copy the `AB` buffer to `C` with `β` scaling
# Ĉ = AB[1:mleft, 1:nleft]
function cleanup_packing!(C, ABbuffer, β, (ii, jj), mleft, nleft)
    if iszero(β)
        @avx for j in 1:nleft
            for i in 1:mleft
                C[ii + i, jj + j] = ABbuffer[i, j]
            end
        end
    else
        @avx for j in 1:nleft
            for i in 1:mleft
                Cij = C[ii + i, jj + j]
                C[ii + i, jj + j] = muladd(β, Cij, ABbuffer[i, j])
            end
        end
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

@inline function vecstore!(v::SVec{N,T}, A::StridedMatrix, ptr::Ptr{T}, i, j) where {N,T}
    ii = (j - 1) * stride(A, 2) + (i - 1) * N
    vstore!(ptr + ii*sizeof(T), v)
    return nothing
end

@inline function vecload(V::Type{SVec{N,T}}, A::StridedMatrix, ptr::Ptr{T}, i, j)::V where {N,T}
    ii = (j - 1) * stride(A, 2) + (i - 1) * N
    return vload(V, ptr + ii*sizeof(T))
end

prefetcht0(ptr::AbstractPointer) = prefetcht0(pointer(ptr))
prefetcht0(ptr::Ptr) = __prefetch(ptr, Val(:read), Val(3), Val(:data))
# From https://github.com/vchuravy/GPUifyLoops.jl/pull/5
@generated function __prefetch(ptr::T, ::Val{RW}, ::Val{Locality}, ::Val{Cache}) where {T, RW, Locality, Cache}
    decls = """
    declare void @llvm.prefetch(i8*, i32, i32, i32)
    """

    if RW == :read
        f_rw = 0
    elseif RW == :write
        f_rw = 1
    end

    f_locality = Locality

    if Cache == :data
        f_cache = 1
    elseif Cache == :instruction
        f_cache = 0
    end

    ir = """
        %ptr = inttoptr i64 %0 to i8*
        call void @llvm.prefetch(i8* %ptr, i32 $f_rw, i32 $f_locality, i32 $f_cache)
        ret void
    """

    quote
        Base.@_inline_meta
        Base.llvmcall(($decls, $ir), Nothing, Tuple{T}, ptr)
    end
end
