using SIMDPirates
using LinearAlgebra: Transpose, Adjoint
using Base.Cartesian: @nexprs, @nif
using ..LoopInfo
using LoopVectorization: @avx
using SIMDPirates: vfmadd231
using VectorizationBase: VectorizationBase
using TimerOutputs: TimerOutputs, @timeit_debug, TimerOutput
using UnPack: @unpack

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

function mul!(C, A, B, α=true, β=false; cache_params=(cache_m=72, cache_k=256, cache_n=4080), kernel_params=(Val(8), Val(6)), packing=nothing)
    m, k, n = checkmulsize(C, A, B)
    α = convert(eltype(C), α)
    β = convert(eltype(C), β)
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
    # TODO: packing strategy
    packing === nothing && (packing = (false, false))
    packa, packb = packing
    if packa isa Bool && packb isa Bool
        if packa # manual union split
            if packb
                _mul!(C, A, B, α, β, (Val(true),Val(true)), cache_params, kernel_params)
            else
                _mul!(C, A, B, α, β, (Val(true),Val(false)), cache_params, kernel_params)
            end
        else
            if packb
                _mul!(C, A, B, α, β, (Val(false),Val(true)), cache_params, kernel_params)
            else
                _mul!(C, A, B, α, β, (Val(false),Val(false)), cache_params, kernel_params)
            end
        end
    else
        _mul!(C, A, B, α, β, packing, cache_params, kernel_params)
    end
    return C
end

partition_k(k, cache_k) = cld(k, cld(k, cache_k))

###
### Lower-level `_mul!`
###

# use lowercase here because these are not actually types
struct GemmContext{packa,packb,micro_m,micro_n,C1,C2,A1,A2,B1,B2}
    cache_m::Int
    cache_k::Int
    cache_n::Int
    cs1::C1
    cs2::C2
    as1::A1
    as2::A2
    bs1::B1
    bs2::B2
end
function GemmContext{packa,packb,micro_m,micro_n}(cache_m, cache_k, cache_n, cs1, cs2, as1, as2, bs1, bs2) where {packa,packb,micro_m,micro_n}
    return GemmContext{packa,packb,micro_m,micro_n,
                       typeof(cs1),typeof(cs2),
                       typeof(as1),typeof(as2),
                       typeof(bs1),typeof(bs2)}(
                                                cache_m, cache_k, cache_n,
                                                cs1, cs2,
                                                as1, as2,
                                                bs1, bs2
                                               )
end

function _mul!(C, A, B, α, β, packing::Tuple{Val{packa},Val{packb}}, (cache_m, cache_k, cache_n), kernel_params::Tuple{Val{micro_m},Val{micro_n}}) where {packa,packb,micro_m,micro_n}
    cs1, cs2 = _strides(C)
    as1, as2 = _strides(A)
    bs1, bs2 = _strides(B)
    cs1 == 1 || throw(ArgumentError("Packing kernel doesn't support nonunit leading stride C matrix. Got stride(C, 1) = $cs1."))
    if !packa
        as1 == 1 || throw(ArgumentError("Kernel with packed A doesn't support nonunit leading stride C and A matrices. Got stride(C, 1) = $cs1, stride(A, 1) = $as1, and stride(B, 1) = $bs1."))
    end
    ctx = GemmContext{packa,packb,micro_m,micro_n}(cache_m, cache_k, cache_n,
                                                   cs1, cs2, as1, as2, bs1, bs2)
    m, k, n = checkmulsize(C, A, B)
    A′, B′ = canonicalize(A), canonicalize(B)
    if !packa && !packb
        macrokernel!(C, A′, B′, α, β, (1, m), (1, k), (1, n), ctx)
        return C
    end

    cache_k = partition_k(k, cache_k) # More evenly divide K

    # get buffer
    Abuffersize = cache_m * cache_k
    Bbuffersize = cache_k * cache_n
    T = eltype(C)
    buffer = N_THREADS_BUFFERS[Threads.threadid()]
    resize!(buffer, (Abuffersize + Bbuffersize) * sizeof(T) + 2PAGE_SIZE)
    ptrbuffer = align(convert(Ptr{T}, pointer(buffer)), PAGE_SIZE)
    Abuffer = packa ? unsafe_wrap(Array, ptrbuffer, Abuffersize) : A′
    ptrbuffer = align(ptrbuffer + Abuffersize * sizeof(T), PAGE_SIZE)
    Bbuffer = packb ? unsafe_wrap(Array, ptrbuffer, Bbuffersize) : B′
    ptrbuffer = align(ptrbuffer + Bbuffersize * sizeof(T), PAGE_SIZE)

    for cachej₁ in 1:cache_n:n; cachej₂ = min(cachej₁ + cache_n - 1, n)
        for cachep₁ in 1:cache_k:k; cachep₂ = min(cachep₁ + cache_k - 1, k) # TODO: add adaptive packing?
            β′ = cachep₁ == 1 ? β : one(β)
            packb && @timeit_debug BLAS_TIMER "Pack B" packBbuffer!(Bbuffer, B, cachep₁, cachep₂, cachej₁, cachej₂, Val(micro_n))
            for cachei₁ in 1:cache_m:m; cachei₂ = min(cachei₁ + cache_m - 1, m)
                packa && @timeit_debug BLAS_TIMER "Pack A" packAbuffer!(Abuffer, A, cachei₁, cachei₂, cachep₁, cachep₂, Val(micro_m))
                # macrokernel
                macrokernel!(C, Abuffer, Bbuffer, α, β′, (cachei₁, cachei₂), (cachep₁, cachep₂), (cachej₁, cachej₂), ctx)
            end # cachei
        end # cachep
    end # cachej
    return C
end

###
### Macro kernel
###

@generated function macrokernel!(C, A, B, α, β, cacheis, cacheps, cachejs, ctx::GemmContext{packa,packb,micro_m,micro_n})::Nothing where {packa,packb,micro_m,micro_n}
    T = eltype(C)
    N = VectorizationBase.pick_vector_width(T)
    micro_m % N == 0 || error("`micro_m` must be an integer multiple of vector register width for efficient microkernel generation, got width = $N, micro_m = $micro_m.")
    mregister = max(1, micro_m ÷ N)

    mloopexpr = quote
        ii = cachei₁
        m′ = $micro_m
        while (m = cachei₂ - ii + 1) >= $micro_m
            Coffset = (ii - 1) * cs1 + (jj - 1) * cs2
            Aoffset′ = Aoffset + (packa ? (ii - cachei₁) * ps : (ii - 1) * as1)
            Boffset′ = Boffset + (packb ? (jjfull - cachej₁) * ps + micro_n_offset : (jj - 1) * bs2)
            microkernel!(C, A, B, α, β, Coffset, Aoffset′, Boffset′, ps, (Val(m′), Val(n′)), Val(4), nothing, ctx) # no mask
            ii += m′
        end
        m = cachei₂ - ii + 1
        mask = VectorizationBase.mask($T, m)
        Coffset = (ii - 1) * cs1 + (jj - 1) * cs2
        Aoffset′ = Aoffset + (packa ? (ii - cachei₁) * ps : (ii - 1) * as1)
        Boffset′ = Boffset + (packb ? (jjfull - cachej₁) * ps + micro_n_offset : (jj - 1) * bs2)
        @nif $(mregister+1) midx -> begin
            (m > (m′ = $micro_m - (midx - 1) * $N) - $N)
        end midx -> begin
            microkernel!(C, A, B, α, β, Coffset, Aoffset′, Boffset′, ps, (Val(m′), Val(n′)), Val(1), mask, ctx) # mask
        end midx -> nothing
        jj += n′
    end

    nloopexpr = quote
        cachei₁, cachei₂ = cacheis
        cachep₁, cachep₂ = cacheps
        cachej₁, cachej₂ = cachejs
        ps = cachep₂ - cachep₁ + 1
        @unpack cs1, cs2, as1, as2, bs1, bs2 = ctx
        jj = jjfull = cachej₁
        micro_n_offset = 0
        Aoffset = packa ? 0 : (cachep₁ - 1) * as2
        Boffset = packb ? 0 : (cachep₁ - 1) * bs1

        @nexprs $micro_n nidx -> begin
            n′ = $micro_n - nidx + 1
            if nidx == 1 # hot loop
                while (n = cachej₂ - jj + 1) >= n′
                    $mloopexpr
                    jjfull = jj
                end # full n
                jjfull = jj # set `jjfull` after completing a full `micro_n`
            else # cleanup loop
                if (n = cachej₂ - jj + 1) >= n′
                    $mloopexpr
                    micro_n_offset += n′
                end
            end
        end
        return nothing
    end
    # return @show macroexpand(Base, nloopexpr)
    return nloopexpr
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

"""
    microkernel!(C, Abuffer, Bbuffer, α, β, Coffset, Aoffset, Boffset, m, ps,
        kernel_params::Tuple{Val{micro_m},Val{micro_n}}) where {micro_m,micro_n}

A kernel that computes
```julia
Ĉ[1:m, 1:n] = α * Â[1:m, 1:ps] * B̂[1:ps, 1:n] + β * Ĉ[1:m, 1:n]
```
where ``{⋅̂}`` denotes the matrix with offset.
"""
@generated function microkernel!(C, A, B, α, β, Coffset, Aoffset, Boffset, ps, ::Tuple{Val{micro_m},Val{micro_n}},::Val{unroll}, mask,
                                 ctx::GemmContext{packa,packb,fullmicro_m,fullmicro_n})::Nothing where {micro_m,micro_n,unroll,packa,packb,fullmicro_m,fullmicro_n}
    T = eltype(C)
    TA = eltype(A)
    TB = eltype(B)
    T === TA === TB || throw(ArgumentError("eltype(C) === eltype(A) === eltype(B) must hold in the microkernel! Got eltype(C) = $T, eltype(A) = $TA, eltype(B) = $TB."))
    N = VectorizationBase.pick_vector_width(T)
    mregister = max(1, micro_m ÷ N)
    V = SVec{N,T}
    dounroll = unroll > 1
    matA = !packa
    matB = !packb
    usemask = mask != Nothing

    nounroll_loopexpr = quote
        prefetcht0(ptrÂ + 64 * $fullmicro_m + 2 * $fullmicro_n)
        @nexprs $mregister m̂ -> begin
            # assumption: A has unit leading stride
            A_m̂ = vload($V, ptrÂ + (m̂ - 1) * $N * st, mask_m̂)
        end
        @nexprs $micro_n n̂ -> begin
            if $matB
                B_n̂ = $V(vload(ptrB̂ + (n̂ - 1) * bs2))
            else
                B_n̂ = $V(vload(ptrB̂ + (n̂ - 1) * st))
            end
            @nexprs $mregister m̂ -> begin
                AB_m̂_n̂ = vfmadd231(A_m̂, B_n̂, AB_m̂_n̂)
            end
        end
        ptrÂ += $matA ? as2 : $fullmicro_m * st
        ptrB̂ += $matB ? bs1 : $fullmicro_n * st
    end

    unroll_loopexpr = quote
        @nexprs $unroll u -> begin
            #TODO
            if !$matA
                u == 1 && prefetcht0(ptrÂ + 64 * $micro_m + 2 * $fullmicro_n)
                u == 3 && prefetcht0(ptrÂ + 76 * $micro_m + 2 * $fullmicro_n)
            end
            ## unroll variable: u
            @nexprs $mregister m̂ -> begin
                # assumption: A has unit leading stride, as1 == 1
                ptrA′ = ptrÂ + (u - 1) * ($matA ? as2 : $fullmicro_m * st)
                A_m̂ = vload($V, ptrA′ + (m̂ - 1) * $N * st, mask_m̂)
            end
            @nexprs $micro_n n̂ -> begin
                ptrB′ = ptrB̂ + (n̂ - 1) * ($matB ? bs2 : st) + (u - 1) * ($matB ? bs1 : $fullmicro_n * st)
                B_n̂ = $V(unsafe_load(ptrB′))
                @nexprs $mregister m̂ -> begin
                    AB_m̂_n̂ = vfmadd231(A_m̂, B_n̂, AB_m̂_n̂)
                end
            end
        end
        ptrÂ += $matA ? $unroll * as2 : $unroll * $fullmicro_m * st
        ptrB̂ += $matB ? $unroll * bs1 : $unroll * $fullmicro_n * st
    end

    kernel_code = quote
        # tell the compiler that the iteration is nonempty
        ps < 1 && return nothing

        st = sizeof($T)
        cs1 = 1 * st # ctx.cs1 * st
        cs2 = ctx.cs2 * st
        if $matA # A is not packed
            as1 = 1 * st # ctx.as1 * st
            as2 = ctx.as2 * st
        end
        if $matB # B is not packed
            bs1 = ctx.bs1 * st
            bs2 = ctx.bs2 * st
        end
        @nexprs $mregister m̂ -> begin
            mask_m̂ = ($usemask && m̂ == $mregister) ? mask : # nontrival mask
                                                     VectorizationBase.max_mask($T) # trival mask that should be optimized away
        end

        ptrĈ = pointer(C) + Coffset * st
        ptrÂ = pointer(A) + Aoffset * st
        ptrB̂ = pointer(B) + Boffset * st

        # prefetch C matrix
        @nexprs $micro_n n̂ -> begin
            prefetcht0(ptrĈ + (n̂ - 1) * cs2 + 7 * 8)
        end

        # intializing AB registers
        @nexprs $micro_n n̂ -> @nexprs $mregister m̂ -> AB_m̂_n̂ = zero($V)

        # rank-1 updates
        if $dounroll
            punroll, pleft = divrem(ps, $unroll)
            # tell the compiler that the iteration is nonempty
            (punroll < 1 && pleft < 1) && return nothing
            for _ in 1:punroll
                $unroll_loopexpr
            end

            for _ in 1:pleft
                $nounroll_loopexpr
            end
        else
            for _ in 1:ps
                $nounroll_loopexpr
            end
        end

        _α = $V(α)
        @nexprs $micro_n n̂ -> @nexprs $mregister m̂ -> begin
            C_m̂_n̂ = _α * AB_m̂_n̂
        end

        if iszero(β)
            @nexprs $micro_n n̂ -> @nexprs $mregister m̂ -> begin
                addr = ptrĈ + (m̂ - 1) * $N * cs1 + (n̂ - 1) * cs2
                vstore!(addr, C_m̂_n̂, mask_m̂)
            end
        else
            _β = $V(β)
            @nexprs $micro_n n̂ -> @nexprs $mregister m̂ -> begin
                addr = ptrĈ + (m̂ - 1) * $N * cs1 + (n̂ - 1) * cs2
                C_m̂_n̂ = vfmadd231(_β, vload($V, addr, mask_m̂), C_m̂_n̂)
                vstore!(addr, C_m̂_n̂, mask_m̂)
            end
        end
    end

    expr = quote
        begin
            $(Expr(:meta, usemask ? :noinline : :inline))
            $kernel_code
            return nothing
        end
    end
    return expr
end

###
### Fast stride handling
###

struct One
end
@inline Base.:*(::One, x::Int) = x
@inline Base.:*(x::Int, ::One) = x
@inline Base.:+(::One, x::Int) = x + 1
@inline Base.:+(x::Int, ::One) = x + 1
@inline Base.:(==)(x::Int, ::One) = isone(x)
@inline Base.:(==)(::One, x::Int) = isone(x)
@inline _strides(A) = strides(A)
@inline _strides(A::DenseArray) = (One(), stride(A, 2))
@inline _strides(A::SubArray{<:Any,2,<:DenseArray,<:Tuple{Union{Base.Slice, AbstractUnitRange}, Vararg{Any}}}) = (One(), stride(A, 2))
@inline _strides(A::Union{Adjoint,Transpose}) = reverse(_strides(parent(A)))
@inline function _strides(A::PermutedDimsArray{T,N,perm}) where {T,N,perm}
    s = _strides(parent(A))
    return ntuple(d->s[perm[d]], Val(N))
end
@inline canonicalize(A) = A
@inline canonicalize(A::Union{Adjoint,Transpose,PermutedDimsArray}) = canonicalize(parent(A))

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
