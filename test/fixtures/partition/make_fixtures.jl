# Record reference trees from the growth code before the node-wise row
# partition, so the partitioned growth can be checked for bit-identical output.
# Run from the package root:
#   julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add(["StableRNGs","JSON3"]); include("test/fixtures/partition/make_fixtures.jl")'
using LinearTrees, StableRNGs, JSON3

include(joinpath(@__DIR__, "cases.jl"))

for (name, X, y, loss, kw) in partition_cases()
    t = fit_tree(X, y, loss; nthreads = 1, kw...)
    open(joinpath(@__DIR__, "$name.json"), "w") do io
        JSON3.write(io, to_dict(t))
    end
    @info name nodes = length(t.nodes)
end
