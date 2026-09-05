# Focused driver for the profiling pass: runs only the test files named in
# ARGS, so a fix can be checked without the two-minute full suite.
#
#     julia --project=test -t 4 test/focused.jl select.jl shap.jl
using LinearTrees, Test, LinearAlgebra, StaticArrays

@testset "focused" begin
    for f in ARGS
        include(joinpath(@__DIR__, f))
    end
end
