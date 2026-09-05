using BenchmarkTools, LinearTrees, DecisionTree, StableRNGs

println("Julia: ", VERSION)
println("CPU: ", Sys.cpu_info()[1].model)
println("CPU_THREADS: ", Sys.CPU_THREADS)
println("Total memory: ", round(Sys.total_memory() / 2^30; digits = 1), " GiB")
println("Threads.nthreads(): ", Threads.nthreads())
println()

function bench_set(name, X, y; max_depth = 8)
    println("== ", name, " (", size(X, 1), " x ", size(X, 2), ") ==")
    println("fit_tree:")
    display(@benchmark fit_tree($X, $y; max_depth = $max_depth))
    println()
    println("DecisionTree.build_tree:")
    display(@benchmark DecisionTree.build_tree($y, $X, 0, $max_depth, 5, 10))
    println()
    t = fit_tree(X, y; max_depth = max_depth)
    println("LinearTrees.predict:")
    display(@benchmark LinearTrees.predict($t, $X))
    println()
    return nothing
end

rng = StableRNG(1)
n, p = 100_000, 20
X = rand(rng, n, p)
ylin = X * randn(rng, p) .+ 0.1 .* randn(rng, n)
ypw = [X[i, 1] > 0.5 ? 2X[i, 2] : -X[i, 3] for i in 1:n] .+ 0.1 .* randn(rng, n)

bench_set("linear", X, ylin)
bench_set("piecewise", X, ypw)

# Deep, wide tree: many step changes keep BIC splitting, so the node count is
# in the hundreds and growth cost per node matters. Reports the node count so
# the timing can be read per node.
ystep = sum(floor.(5 .* X[:, j]) for j in 1:6) .+ X[:, 7] .* X[:, 8] .+ 0.1 .* randn(rng, n)
t = fit_tree(X, ystep; max_depth = 12)
println("== step (", n, " x ", p, "), max_depth = 12, ", length(t.nodes), " nodes ==")
println("fit_tree:")
display(@benchmark fit_tree($X, $ystep; max_depth = 12))
println()

# Forced growth: MinDeviance((LIN, PCON, BLIN, PLIN)) with a loose min_leaf/min_fit
# keeps BIC from stopping early, so growth runs to the depth cap. Same target
# shape as the step case above, on its own RNG/size so the "forced growth" row
# in RESULTS.md is reproducible from this script.
rngf = StableRNG(7)
nf, pf = 200_000, 20
Xf = rand(rngf, nf, pf)
yforced = sum(floor.(5 .* Xf[:, j]) for j in 1:6) .+ Xf[:, 7] .* Xf[:, 8] .+ 0.1 .* randn(rngf, nf)
forced_rule = MinDeviance((LIN, PCON, BLIN, PLIN))
tf = fit_tree(Xf, yforced; max_depth = 12, rule = forced_rule, min_leaf = 50, min_fit = 100)
println("== forced growth (", nf, " x ", pf, "), max_depth = 12, ", length(tf.nodes), " nodes ==")
println("fit_tree:")
display(@benchmark fit_tree($Xf, $yforced; max_depth = 12, rule = $forced_rule, min_leaf = 50, min_fit = 100))
println()
