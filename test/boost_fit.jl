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
    b1 = fit_boost(X, y; rng = StableRNG(1), nthreads = 1, kw...)
    bn = fit_boost(X, y; rng = StableRNG(1), kw...)
    @test all(a.nodes == c.nodes for (a, c) in zip(b1.trees, bn.trees))
    @test b1.history == bn.history
    bfull = fit_boost(X, y; nrounds = 8, eta = 0.3, max_depth = 4)
    @test b1.trees[1].nodes != bfull.trees[1].nodes
    b2 = fit_boost(X, y; rng = StableRNG(2), nthreads = 1, kw...)
    @test b1.trees[1].nodes != b2.trees[1].nodes
    used = [Set(nd.feature for nd in t.nodes if !LinearTrees.isleaf(nd)) for t in b1.trees]
    @test all(length(u) <= 3 for u in used)
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
    @test_throws ArgumentError fit_boost(X, y; patience = 0)
    @test_throws ArgumentError fit_boost(X, y; patience = 1.5)
    Xv = copy(X); Xv[1, 1] = NaN
    @test_throws ArgumentError fit_boost(X, y; Xval = Xv, yval = y)
    y01 = Float64.(y .> 0.5)
    @test_throws ArgumentError fit_boost(X, y01, Logistic(); Xval = X, yval = fill(2, length(y)))
    @test_throws ArgumentError fit_boost(X, y; Xval = X, yval = y, wval = zeros(length(y)))
end
