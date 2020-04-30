module MaBLAS

# general design philosophy:
#   1: keep the number of defined types in check
#   2: don't split code into another file/module if the API and tests are not standalone

include("loopinfo.jl")
include("gemm.jl")

end # module
