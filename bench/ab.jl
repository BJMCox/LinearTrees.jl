# Before/after medians for the six P1 cases, without the profilers.
#
#     julia --project=bench -t 1    bench/ab.jl
#     julia --project=bench -t auto bench/ab.jl
#
# It also writes a canonical bit-level dump of every fitted tree and of every
# prediction and SHAP matrix into `bench/profiles/dumps/`, so a fix meant to be
# behaviour-preserving can be `diff`ed against the same script run on the
# previous commit. Set `LT_AB_DUMPDIR` to send the dumps elsewhere and
# `LT_AB_DUMP_ONLY=1` to skip the benchmarks.

using BenchmarkTools, Printf, StaticArrays
include(joinpath(@__DIR__, "cases.jl"))

"""
Write a canonical text dump of one case's result. Two runs are compared by
`diff`, not by a hash: `hash` over `Any`-typed struct fields is not a contract
Base makes across processes, and a fix that is meant to be bit-identical has to
be checked against something that is.

Floats are printed by their bit pattern, so `-0.0`, `NaN` and the last mantissa
bit all show up.
"""
bits(x::Real) = string(reinterpret(UInt64, Float64(x)); base = 16, pad = 16)
bits(v::SVector) = join((bits(x) for x in v), ",")

function dump_case(io, t::LinearTree)
    println(io, "nfeatures=", t.nfeatures, " truncate=", t.truncate,
        " lo=", bits(t.lo), " hi=", bits(t.hi), " base=", bits(t.base))
    for (k, n) in enumerate(t.nodes)
        println(io, k, " ", n.feature, " ", bits(n.threshold), " ", n.left, " ", n.right,
            " ", bits(n.lcoef), " ", bits(n.lintercept), " ", bits(n.rcoef), " ", bits(n.rintercept),
            " ", bits(n.xmin), " ", bits(n.xmax), " ", bits(n.cover), " ", bits(n.xmean),
            " ", bits(n.gain), " ", n.catstart, " ", n.catwords, " ", Int(n.model))
    end
    for (k, w) in enumerate(t.catmasks)
        println(io, "mask ", k, " ", w)
    end
    return nothing
end

function dump_case(io, a::AbstractArray{<:Real})
    println(io, "size=", size(a))
    for x in a
        println(io, bits(x))
    end
    return nothing
end

const DUMPDIR = get(ENV, "LT_AB_DUMPDIR", joinpath(@__DIR__, "profiles", "dumps"))
mkpath(DUMPDIR)

const NT = Threads.nthreads()
println("Julia ", VERSION, "  threads=", NT)

const DUMP_ONLY = haskey(ENV, "LT_AB_DUMP_ONLY")

function row(name, f)
    v = f()
    path = joinpath(DUMPDIR, "$name.txt")
    open(io -> dump_case(io, v), path, "w")
    if DUMP_ONLY
        @printf("%-14s dump=%s\n", name, path)
        return nothing
    end
    b = @benchmark $f() samples = 5 seconds = 60 evals = 1
    m = median(b)
    @printf("%-14s threads=%2d  median=%9.3f ms  allocs=%9d  bytes=%12d\n",
        name, NT, m.time / 1e6, m.allocs, m.memory)
    return nothing
end

X1, y1 = case1_data(); row("case1-mse", () -> fit_tree(X1, y1; max_depth = 12))
X2, y2 = case2_data(); row("case2-softmax", () -> fit_tree(X2, y2, Softmax(3); categorical = [10], max_depth = 12))
X3, y3 = case3_data(); row("case3-mad", () -> fit_tree(X3, y3, MAD(); max_depth = 12, niter = 5))
df4, y4 = case4_data(); row("case4-table", () -> fit(LinearTreeRegressorFit, df4, y4; max_depth = 12).tree)
tree, Xp = case56_data()
out = Vector{Float64}(undef, size(Xp, 1))
row("case5-predict", () -> LinearTrees.predict(tree, Xp))
row("case5-predict!", () -> LinearTrees.predict!(out, tree, Xp))
Xs = Xp[1:10_000, :]
row("case6-shap", () -> shap(tree, Xs).values)
