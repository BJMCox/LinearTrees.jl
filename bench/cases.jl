# Shared case data for the profiling pass. `bench/profile.jl` includes this.
#
# Every case builds its data from a `StableRNG` seed, so two runs on the same
# machine profile the same tree. Sizes follow the P1 brief.

using LinearTrees, StableRNGs, StaticArrays, Statistics
using CategoricalArrays, DataFrames

"Case 1 data: MSE, n = 200_000, p = 20, step target, max_depth = 12."
function case1_data()
    rng = StableRNG(1)
    n, p = 200_000, 20
    X = rand(rng, n, p)
    y = sum(floor.(5 .* X[:, j]) for j in 1:6) .+ X[:, 7] .* X[:, 8] .+ 0.1 .* randn(rng, n)
    return X, y
end

"Case 2 data: Softmax(3), n = 100_000, p = 10, column 10 categorical with 8 levels."
function case2_data()
    rng = StableRNG(2)
    n, p = 100_000, 10
    X = rand(rng, n, p)
    codes = rand(rng, 1:8, n)
    X[:, p] = Float64.(codes)
    # a step target keeps BIC splitting, so the tree reaches the hundreds of
    # nodes that make per-node growth cost visible
    u = sum(floor.(4 .* X[:, j]) for j in 1:5) .+ 0.4 .* codes .+ randn(rng, n)
    q1, q2 = quantile(u, 1 / 3), quantile(u, 2 / 3)
    y = Float64.([u[i] > q2 ? 1 : (u[i] > q1 ? 2 : 3) for i in 1:n])
    return X, y
end

"Case 3 data: MAD, n = 100_000, p = 10."
function case3_data()
    rng = StableRNG(3)
    n, p = 100_000, 10
    X = rand(rng, n, p)
    y = sum(floor.(4 .* X[:, j]) for j in 1:4) .+ X[:, 5] .* X[:, 6] .+ 0.1 .* randn(rng, n)
    return X, y
end

"Case 4 data: DataFrame with two CategoricalArray columns and eight numeric, n = 100_000."
function case4_data()
    rng = StableRNG(4)
    n = 100_000
    cols = Dict{Symbol,Any}()
    num = Matrix{Float64}(undef, n, 8)
    for j in 1:8
        num[:, j] = rand(rng, n)
        cols[Symbol("x", j)] = num[:, j]
    end
    ca = categorical(rand(rng, ["a", "b", "c", "d", "e"], n))
    cb = categorical(rand(rng, ["p", "q", "r"], n))
    cols[:ca] = ca
    cols[:cb] = cb
    df = DataFrame(cols)
    df = df[!, [Symbol("x", j) for j in 1:8] ∪ [:ca, :cb]]
    y = sum(floor.(4 .* num[:, j]) for j in 1:4) .+ Float64.(levelcode.(ca)) .+ 0.5 .* Float64.(levelcode.(cb)) .+
        0.1 .* randn(rng, n)
    return df, y
end

"Case 5/6 data: a depth-12 tree with one categorical column plus a big prediction matrix."
function case56_data()
    rng = StableRNG(5)
    n, p = 200_000, 10
    X = rand(rng, n, p)
    codes = rand(rng, 1:8, n)
    X[:, p] = Float64.(codes)
    y = sum(floor.(4 .* X[:, j]) for j in 1:4) .+ 0.3 .* codes .+ 0.1 .* randn(rng, n)
    tree = fit_tree(X, y; categorical = [p], max_depth = 12)
    Xpred = rand(rng, 1_000_000, p)
    Xpred[:, p] = Float64.(rand(rng, 1:8, 1_000_000))
    return tree, Xpred
end
