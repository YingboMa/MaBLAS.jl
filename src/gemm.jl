using SIMD
using LinearAlgebra: Transpose, Adjoint
using Base.Cartesian: @nexprs
using ..LoopInfo
using LoopVectorization: @avx
using VectorizationBase: VectorizationBase
using TimerOutputs: TimerOutputs, @timeit_debug, TimerOutput

# General direction: we want to avoid pointer arithmetic as much as possible,
# also, don't strict type the container type

# I'll try to implement everything without defining a type, and all variables
# should stay local if possible. We can always clean up the code later when we
# find a good direction/structure.

###
### Prelude
###

const N_THREADS_BUFFERS = map(_->Vector{UInt8}(undef, 0), 1:Threads.nthreads())
const PAGE_SIZE = ccall(:jl_getpagesize, Int, ())

const BLAS_TIMER = TimerOutput()
reset_timer!() = TimerOutputs.reset_timer!(BLAS_TIMER)
get_timer() = BLAS_TIMER
enable_timer() = TimerOutputs.enable_debug_timings(@__MODULE__)
disable_timer() = TimerOutputs.disable_debug_timings(@__MODULE__)

const FA{T,N} = SIMD.FastContiguousArray{T,N}

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

function mul!(C, A, B, α=true, β=false; cache_params=(cache_m=72, cache_k=256, cache_n=4080), kernel_params=(Val(8), Val(6)), packing=(false, false))
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
    # packing strategy
    _mul!(C, A, B, α, β, packing, cache_params, kernel_params)
    return C
end

partition_k(k, cache_k) = cld(k, cld(k, cache_k))

###
### Lower-level `_mul!`
###

function _mul!(C, A, B, α, β, (packa, packb)::NTuple{2,Bool}, (cache_m, cache_k, cache_n), kernel_params::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}
    cs1 = _stride(C, 1)
    as1 = _stride(A, 1)
    bs1 = _stride(B, 1)
    cs1 == 1 || throw(ArgumentError("Packing kernel doesn't support nonunit leading stride C matrix. Got stride(C, 1) = $cs1."))
    if !packa
        as1 == 1 || throw(ArgumentError("Kernel with packed A doesn't support nonunit leading stride C and A matrices. Got stride(C, 1) = $cs1, stride(A, 1) = $as1, and stride(B, 1) = $bs1."))
    end
    m, k, n = checkmulsize(C, A, B)

    cache_k = partition_k(k, cache_k) # More evenly divide K
    
    if packa || packb
        # get buffer
        Abuffersize = cache_m * cache_k
        Bbuffersize = cache_k * cache_n
        ABbuffersize = micro_m * micro_n
        T = eltype(C)
        buffer = N_THREADS_BUFFERS[Threads.threadid()]
        resize!(buffer, (Abuffersize + Bbuffersize + ABbuffersize) * sizeof(T) + 3PAGE_SIZE)
        ptrbuffer = align(convert(Ptr{T}, pointer(buffer)), PAGE_SIZE)
        Abuffer = packa ? unsafe_wrap(Array, ptrbuffer, Abuffersize) : A
        ptrbuffer = align(ptrbuffer + Abuffersize * sizeof(T), PAGE_SIZE)
        Bbuffer = packb ? unsafe_wrap(Array, ptrbuffer, Bbuffersize) : B
        ptrbuffer = align(ptrbuffer + Bbuffersize * sizeof(T), PAGE_SIZE)
        ABbuffer = unsafe_wrap(Array, ptrbuffer, (micro_m, micro_n))
    else
        Abuffer = A
        Bbuffer = B
    end

    for cachej₁ in 1:cache_n:n; cachej₂ = min(cachej₁ + cache_n - 1, n)
        for cachep₁ in 1:cache_k:k; cachep₂ = min(cachep₁ + cache_k - 1, k) # TODO: add adaptive packing?
            _β = cachep₁ == 1 ? β : one(β)
            ps = cachep₂ - cachep₁ + 1
            packb && @timeit_debug BLAS_TIMER "Pack B" packBbuffer!(Bbuffer, B, cachep₁, cachep₂, cachej₁, cachej₂, Val(micro_n))
            for cachei₁ in 1:cache_m:m; cachei₂ = min(cachei₁ + cache_m - 1, m)
                packa && @timeit_debug BLAS_TIMER "Pack A" packAbuffer!(Abuffer, A, cachei₁, cachei₂, cachep₁, cachep₂, Val(micro_m))
                # macrokernel
                for microj₁ in cachej₁:micro_n:cachej₂; nleft = min(cachej₂ - microj₁ + 1, micro_n)
                    for microi₁ in cachei₁:micro_m:cachei₂; mleft = min(cachei₂ - microi₁ + 1, micro_m)
                        Coffset = (microi₁ - 1) * _stride(C, 1) + (microj₁ - 1) * _stride(C, 2)
                        Aoffset = packa ? (microi₁ - cachei₁) * ps :
                            (microi₁ - 1) * _stride(A, 1) + (cachep₁ - 1) * _stride(A, 2)
                        Boffset = packb ? (microj₁ - cachej₁) * ps :
                            (cachep₁ - 1) * _stride(B, 1) + (microj₁ - 1) * _stride(B, 2)

                        if nleft == micro_n && mleft == micro_m
                            # (micro_m × ks) * (ks × micro_n)
                            # Ĉ = A[i:(i+micro_m-1), ps] * B[ps, j:(j+micro_n-1)]
                            microkernel!(C, Abuffer, Bbuffer, α, _β,  Coffset, Aoffset, Boffset, ps, kernel_params)
                        else
                            if packa && packb
                                # microkernel writes to the `AB` buffer
                                microkernel!(ABbuffer, Abuffer, Bbuffer, α, false, 0, Aoffset, Boffset, ps, kernel_params)
                                # copy the `AB` buffer to `C` with `β` scaling
                                # Ĉ = AB[1:mleft, 1:nleft]
                                cleanup_packing!(C, ABbuffer, _β, (microi₁ - 1, microj₁ - 1), mleft, nleft)
                            else
                                cleanup_tiling_microkernel!(C, A, B, α, _β, microi₁, microj₁, cachep₁, cachep₂, mleft, nleft)
                            end
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

@generated function microkernel!(C::AbstractMatrix{T}, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{T}, α, β,
                                 Coffset, Aoffset, Boffset, ps, ::Tuple{Val{micro_m},Val{micro_n}}) where {T,micro_m,micro_n}
    N = VectorizationBase.pick_vector_width(T)
    V = Vec{N,T}
    unroll = 4
    m2 = micro_m ÷ N
    matA = ndims(A) == 2
    matB = ndims(B) == 2

    kernel_code = quote
        st = sizeof(T)
        sc1 = _stride(C, 1) * st
        sc2 = _stride(C, 2) * st
        if $matA
            sa1 = _stride(A, 1) * st
            sa2 = _stride(A, 2) * st
        end
        if $matB
            sb1 = _stride(B, 1) * st
            sb2 = _stride(B, 2) * st
        end

        punroll, pleft = divrem(ps, $unroll)

        ptrĈ = _pointer(C) + Coffset * st
        ptrÂ = _pointer(A) + Aoffset * st
        ptrB̂ = _pointer(B) + Boffset * st

        # prefetch C matrix
        # TODO
        @nexprs $micro_n n̂ -> begin
            prefetcht0(ptrĈ + (n̂ - 1) * sc2 + 7 * 8)
        end

        # rank-1 updates
        @nexprs $micro_n n̂ -> @nexprs $m2 m̂ -> AB_m̂_n̂ = zero($V)
        for _ in 1:punroll
            @nexprs $unroll u -> begin
                #TODO
                if !$matA
                    u == 1 && prefetcht0(ptrÂ + 64 * $micro_m + 2 * $micro_n)
                    u == 3 && prefetcht0(ptrÂ + 76 * $micro_m + 2 * $micro_n)
                end
                ## unroll variable: u
                @nexprs $m2 m̂ -> begin
                    if $matA
                        # assumption: A has unit leading stride
                        A_m̂ = vload($V, ptrÂ + (m̂ - 1) * $N * st + (u - 1) * sa2)
                    else
                        A_m̂ = vload($V, ptrÂ + (u - 1) * $micro_m * st + (m̂ - 1) * $N * st)
                    end
                end
                @nexprs $micro_n n̂ -> begin
                    if $matB
                        B_n̂ = $V(unsafe_load(ptrB̂ + (n̂ - 1) * sb2 + (u - 1) * sb1))
                    else
                        B_n̂ = $V(unsafe_load(ptrB̂ + (n̂ - 1) * st + (u - 1) * $micro_n * st))
                    end
                    @nexprs $m2 m̂ -> begin
                        AB_m̂_n̂ = fma(A_m̂, B_n̂, AB_m̂_n̂)
                    end
                end
            end
            ptrÂ += $matA ? $unroll * sa2 : $unroll * $micro_m * st
            ptrB̂ += $matB ? $unroll * sb1 : $unroll * $micro_n * st
        end

        for _ in 1:pleft
            #TODO
            prefetcht0(ptrÂ + 64 * $micro_m + 2 * $micro_n)
            @nexprs $m2 m̂ -> begin
                # assumption: A has unit leading stride
                A_m̂ = vload($V, ptrÂ + (m̂ - 1) * $N * st)
            end
            @nexprs $micro_n n̂ -> begin
                if $matB
                    B_n̂ = $V(unsafe_load(ptrB̂ + (n̂ - 1) * sb2))
                else
                    B_n̂ = $V(unsafe_load(ptrB̂ + (n̂ - 1) * st))
                end
                @nexprs $m2 m̂ -> begin
                    AB_m̂_n̂ = fma(A_m̂, B_n̂, AB_m̂_n̂)
                end
            end
            ptrÂ += $matA ? sa2 : $micro_m * st
            ptrB̂ += $matB ? sb1 : $micro_n * st
        end

        _α = $V(α)
        if iszero(β)
            @nexprs $micro_n n̂ -> @nexprs $m2 m̂ -> begin
                C_m̂_n̂ = _α * AB_m̂_n̂
                vstore(C_m̂_n̂, ptrĈ + (m̂ - 1) * $N * sc1 + (n̂ - 1) * sc2)
            end
        else
            _β = $V(β)
            @nexprs $micro_n n̂ -> @nexprs $m2 m̂ -> begin
                addr = ptrĈ + (m̂ - 1) * $N * sc1 + (n̂ - 1) * sc2
                C_m̂_n̂ = _β * vload($V, addr)
                C_m̂_n̂ = fma(_α, AB_m̂_n̂, C_m̂_n̂)
                vstore(C_m̂_n̂, addr)
            end
        end

        return nothing
    end

    expr = quote
        @inbounds begin
            $(Expr(:meta,:noinline))
            $kernel_code
        end
    end
    return expr
end
#TODO: (A'B), AB', and A'B'

###
### Clean up loops
###

# (mleft × nleft) = (mleft × ks) * (ks × nleft)
# Ĉ = A[i:(i+mleft-1), ps] * B[ps, j:(j+nleft-1)]
function cleanup_tiling_microkernel!(C, A::FA, B::FA, α, β, i, j, p₁, p₂, mleft, nleft)
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

# fallback: @avx miss compiles non-unit stride matrix
function cleanup_tiling_microkernel!(C, A, B, α, β, i, j, p₁, p₂, mleft, nleft)
    if iszero(β)
        @inbounds @aliasscope for ĵ in j:(j+nleft-1), î in i:(i+mleft-1)
            ABîĵ = zero(eltype(C))
            for p̂ in p₁:p₂
                ABîĵ = muladd(constarray(A)[î, p̂], constarray(B)[p̂, ĵ], ABîĵ)
            end
            C[î, ĵ] = α * ABîĵ
        end
    else
        @inbounds @aliasscope for ĵ in j:(j+nleft-1), î in i:(i+mleft-1)
            ABîĵ = zero(eltype(C))
            for p̂ in p₁:p₂
                ABîĵ = muladd(constarray(A)[î, p̂], constarray(B)[p̂, ĵ], ABîĵ)
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

@inline _stride(A, n) = stride(A, n)
@inline _pointer(A) = pointer(A)
#@inline _stride(A::Union{Adjoint,Transpose}, n) = reverse(strides(parent(A)))[n]
#Not very safe, but okay
@inline _stride(A::Union{Adjoint,Transpose}, n) = n == 1 ? stride(parent(A), 2) : stride(parent(A), 1)
@inline _pointer(A::Union{Adjoint,Transpose}) = pointer(parent(A))

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

@inline align(x::Ptr{T}, n) where {T} = reinterpret(Ptr{T}, align(reinterpret(UInt, x), n))
@inline align(x, n) = (nm1 = n - 1; (x + nm1) & -n)
