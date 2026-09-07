using Random: randperm
using StableRNGs, Statistics

@testset "one root-only round is a regularised Newton step" begin
    rng = StableRNG(110)
    n = 300
    X = rand(rng, n, 2)
    y = Float64.(rand(rng, n) .< 0.3)
    w = Float64.(rand(rng, 1:3, n))
    λ = 2.0
    b = fit_boost(X, y, Logistic(); nrounds = 1, eta = 1.0, max_depth = 0, lambda_intercept = λ, weights = w)
    f0 = initscore(Logistic(), y, w)
    g = zeros(n); h = zeros(n)
    gradhess!(g, h, Logistic(), y, fill(f0, n))
    step = -sum(w .* g) / (sum(w .* h) + λ)
    @test score(b, X) ≈ fill(f0 + step, n) atol = 1e-12
    @test nrounds(b) == 1 && length(b.trees[1].nodes) == 1
end

@testset "training deviance is non-increasing on each distinct freeze path" begin
    # Four losses, one per path through frozen_target!: smooth scalar (MSE),
    # non-smooth IRLS (Quantile), log-link count (Poisson), and vector-valued
    # SVector (Softmax). Huber, MAD, Logistic, Gamma, Tweedie and NegBin reach
    # gradhess! by the same route as MSE and Poisson and add no path.
    rng = StableRNG(111)
    n = 600
    X = rand(rng, n, 4)
    μ = exp.(0.5 .* X[:, 1] .- X[:, 2] .* X[:, 3])
    w = Float64.(rand(rng, 1:2, n))
    cases = [
        (MSE(), 3 .* X[:, 1] .+ sin.(6 .* X[:, 2]) .+ 0.1 .* randn(rng, n)),
        (Quantile(0.7), 3 .* X[:, 1] .+ randn(rng, n)),
        (Poisson(), Float64.(floor.(μ .* 3 .+ rand(rng, n)))),
        (Softmax(3), [X[i, 1] > 0.6 ? 1 : X[i, 2] > 0.5 ? 2 : 3 for i in 1:n]),
    ]
    for (loss, y) in cases
        b = fit_boost(X, y, loss; nrounds = 30, eta = 0.1, max_depth = 3, weights = w)
        @test length(b.history) == 30
        @test all(diff(b.history) .<= 1e-9 .* abs.(b.history[1:(end - 1)]))
        @test b.history[end] < b.history[1]
        yhat = predict(b, X)
        @test all(isfinite, yhat)
        loss isa Softmax && @test all(≈(1.0), sum(yhat; dims = 2))
    end
end

@testset "Softmax(2) boosting equals Logistic boosting" begin
    rng = StableRNG(112)
    n = 400
    X = rand(rng, n, 3)
    p = 1 ./ (1 .+ exp.(-(5 .* X[:, 1] .- 2.5)))
    y01 = Float64.(rand(rng, n) .< p)
    bl = fit_boost(X, y01, Logistic(); nrounds = 20, eta = 0.2, max_depth = 3)
    bs = fit_boost(X, Int.(2 .- y01), Softmax(2); nrounds = 20, eta = 0.2, max_depth = 3)
    @test predict(bs, X)[:, 1] ≈ predict(bl, X) atol = 1e-10
end

@testset "subsampling: reproducible, thread-independent, and different from the full fit" begin
    rng = StableRNG(113)
    n = 20_000
    X = rand(rng, n, 5)
    y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, n)
    kw = (nrounds = 8, eta = 0.3, max_depth = 4, subsample = 0.5, colsample = 0.6)
    fitrng = StableRNG(1)
    b1 = fit_boost(X, y; rng = fitrng, nthreads = 1, kw...)
    bn = fit_boost(X, y; rng = StableRNG(1), kw...)
    @test all(a.nodes == c.nodes for (a, c) in zip(b1.trees, bn.trees))
    @test b1.history == bn.history
    bfull = fit_boost(X, y; nrounds = 8, eta = 0.3, max_depth = 4)
    @test b1.trees[1].nodes != bfull.trees[1].nodes
    b2 = fit_boost(X, y; rng = StableRNG(2), nthreads = 1, kw...)
    @test b1.trees[1].nodes != b2.trees[1].nodes
    oracle = StableRNG(1)
    for t in b1.trees
        randperm(oracle, n)
        sampled = Set(randperm(oracle, 5)[1:3])
        used = Set(nd.feature for nd in t.nodes if !LinearTrees.isleaf(nd))
        @test used ⊆ sampled
    end
    @test rand(fitrng) == rand(oracle)
    @test_throws ArgumentError fit_boost(X, y; subsample = 0.0)
    @test_throws ArgumentError fit_boost(X, y; colsample = 1.5)
end

@testset "early stopping keeps exactly the trees up to the best validation round" begin
    # The claim is that stopping early changes nothing but the stopping: the
    # kept ensemble must equal the prefix of the never-stopped run at its own
    # argmin. Asserting `nrounds(b) == argmin(b.history)` on the truncated
    # history alone is a tautology after the resize. The `break` itself saves
    # rounds without changing the result, so nothing here distinguishes it;
    # what these assertions catch is a patience that cuts before the argmin,
    # or an argmin taken over the wrong vector.
    rng = StableRNG(114)
    n = 400
    X = rand(rng, n, 3)
    y = 2 .* X[:, 1] .+ randn(rng, n)
    Xv = rand(rng, 400, 3); yv = 2 .* Xv[:, 1] .+ randn(rng, 400)
    kw = (nrounds = 200, eta = 1.0, max_depth = 4, min_leaf = 2, min_fit = 4,
        lambda_slope = 0.0, lambda_intercept = 0.0, Xval = Xv, yval = yv)
    long = fit_boost(X, y; patience = 200, kw...)
    b = fit_boost(X, y; patience = 5, kw...)
    best = argmin(long.history)
    @test nrounds(b) == best
    @test best < 200
    @test b.history == long.history[1:best]
    @test all(t.nodes == u.nodes for (t, u) in zip(b.trees, long.trees[1:best]))
    @test b.validated
    @test !fit_boost(X, y; nrounds = 5, eta = 1.0, max_depth = 4).validated
    @test_throws ArgumentError fit_boost(X, y; Xval = Xv)
    @test_throws DimensionMismatch fit_boost(X, y; Xval = Xv, yval = yv[1:10])
end

@testset "predict clamps the ensemble score once and applies the link" begin
    rng = StableRNG(115)
    n = 300
    X = rand(rng, n, 2)
    # Separable targets make the logit exceed Logistic's +-10 bound.
    y = Float64.(X[:, 1] .> 0.5)
    b = fit_boost(X, y, Logistic(); nrounds = 40, eta = 1.0, max_depth = 2,
        lambda_slope = 0.0, lambda_intercept = 0.0, min_sum_hessian = 0.0)
    s = score(b, X; clip = false)
    @test any(abs.(s) .> 10)
    @test predict(b, X) ≈ 1 ./ (1 .+ exp.(-clamp.(s, b.lo, b.hi)))
    @test (b.lo, b.hi) == scorebound(Logistic(), y)
    out = similar(y)
    @test predict!(out, b, X) === out
    @test_throws DimensionMismatch predict!(zeros(3), b, X)
    bt = fit_boost(X, y, Logistic(); nrounds = 2, truncate = false)
    @test !bt.truncate && bt.lo == -Inf
    yk = [X[i, 1] > 0.5 ? 1 : X[i, 2] > 0.5 ? 2 : 3 for i in 1:n]
    bs = fit_boost(X, yk, Softmax(3); nrounds = 3)
    @test size(score(bs, X)) == (n, 2) && size(predict(bs, X)) == (n, 3)
end

@testset "fit_boost's own argument guards" begin
    # Only the guards fit_boost owns. Weight, NaN and target-domain checks are
    # fit_tree's and are covered there; fit_boost's copies surface through the
    # fits above if they diverge.
    X = rand(20, 2); y = rand(20)
    @test_throws DimensionMismatch fit_boost(X, y[1:10])
    @test_throws ArgumentError fit_boost(X, y; nrounds = 0)
    @test_throws ArgumentError fit_boost(X, y; nrounds = 1.5)
    @test_throws ArgumentError fit_boost(X, y; eta = 0.0)
    @test_throws ArgumentError fit_boost(X, y; eta = Inf)
    for name in (:lambda_slope, :lambda_intercept, :gamma), value in (-1.0, Inf)
        @test_throws ArgumentError fit_boost(X, y; NamedTuple{(name,)}((value,))...)
    end
    @test_throws ArgumentError fit_boost(X, y; patience = 0)
    @test_throws ArgumentError fit_boost(X, y; patience = 1.5)
    X16 = Float16.(X); y16 = Float16.(y)
    @test_throws ArgumentError fit_boost(X16, y16; eta = 1e-20)
    @test_throws ArgumentError fit_boost(X16, y16; eta = 1e100)
    @test_throws ArgumentError fit_boost(X, y; wval = ones(length(y)))
    Xv = copy(X); Xv[1, 1] = NaN
    @test_throws ArgumentError fit_boost(X, y; Xval = Xv, yval = y)
    y01 = Float64.(y .> 0.5)
    @test_throws ArgumentError fit_boost(X, y01, Logistic(); Xval = X, yval = fill(2, length(y)))
    yv = copy(y01); yv[1] = 1 + 1e-12
    @test_throws ArgumentError fit_boost(Float32.(X), Float32.(y01), Logistic(); Xval = X, yval = yv)
    @test_throws ArgumentError fit_boost(X, y; Xval = X, yval = y, wval = zeros(length(y)))
    Xv = copy(X); Xv[end, 1] = NaN
    yv = copy(y); yv[end] = floatmax(Float64)
    wv = ones(length(y)); wv[end] = 0
    b = fit_boost(X, y; nrounds = 1, Xval = Xv, yval = yv, wval = wv)
    @test isfinite(only(b.history))
    @test only(b.history) ≈ deviance(MSE(), yv[1:(end - 1)],
        score(b, Xv[1:(end - 1), :]; clip = false), wv[1:(end - 1)])
end

@testset "sampled categorical boosting equals fresh-tree recurrence" begin
    rng = StableRNG(145)
    X = rand(rng, 400, 3)
    X[:, 1] .= rand(rng, 1:6, 400)
    X[end, 1] = 7   # sampling can remove the highest category in later rounds
    y = [X[i, 1] <= 2 ? 1 : X[i, 2] > 0.5 ? 2 : 3 for i in 1:400]
    loss = Softmax(3)
    w = Float64.(rand(rng, 1:3, 400))
    fitrng = StableRNG(146)
    b = fit_boost(X, y, loss; weights = w, categorical = [1], nrounds = 5,
        subsample = 0.6, colsample = 0.6, eta = 0.2, max_depth = 3, rng = fitrng)
    oracle_rng = StableRNG(146)
    F = fill(initscore(loss, y, w), length(y))
    g = similar(F); h = similar(F)
    for tree in b.trees
        gradhess!(g, h, loss, y, F)
        target = [(-g[i] ./ h[i], h[i]) for i in eachindex(y)]
        wr = zeros(length(y))
        rows = randperm(oracle_rng, length(y))[1:240]
        wr[rows] .= w[rows]
        features = sort!(randperm(oracle_rng, 3)[1:2])
        fresh = fit_tree(X, target, Frozen{eltype(F)}(); weights = wr, categorical = [1],
            features, rule = GainRule(), max_depth = 3, nthreads = 1)
        @test tree.nodes == fresh.nodes
        @test tree.catmasks == fresh.catmasks
        F .+= 0.2 .* score(fresh, X; clip = false, nthreads = 1)
    end
    @test score(b, X; clip = false) == reduce(hcat, F)'
    @test rand(fitrng) == rand(oracle_rng)
end
