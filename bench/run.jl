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

# California Housing is fetched through MLDatasets, an optional dependency this
# package never requires: skip cleanly rather than force it onto every user.
if Base.find_package("MLDatasets") !== nothing
    @eval using MLDatasets
    housing = MLDatasets.CaliforniaHousing()
    Xh = Matrix{Float64}(Matrix(housing.features)')
    yh = Vector{Float64}(housing.targets[1, :])
    bench_set("california housing", Xh, yh)
else
    println("MLDatasets not installed -- California Housing benchmark skipped.")
end
