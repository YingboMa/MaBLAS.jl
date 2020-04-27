using SIMD

# General direction: we want to avoid pointer arithmetic as much as possible,
# also, don't strict type the container type

# I'll try to implement everything without defining a type, and all variables
# should stays local if possible. We can always clean up the code later when we
# find a good direction/structure.

###
### BLAS parameters
###


###
### User-level API
###

function mul!(C, A, B, α=true, β=false)

end

###
### Lower-level `_mul!`
###

function _mul_with_packing!(C, A, B, α=true, β=false)

end


function _mul_without_packing!(C, A, B, α=true, β=false)

end

###
### Packing
###

function packAbuffer!

end

function packBbuffer!

end

###
### Macro kernel
###

function macrokernel!
end

###
### Micro kernel
###

function microkernel!
end


###
### Clean up loops
###
