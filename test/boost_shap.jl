using StableRNGs

@testset "ensemble SHAP satisfies efficiency on the unclipped score" begin
    # Three losses across the scalar and SVector output shapes, with a
    # categorical column. Logistic walks the scalar path again.
    rng = StableRNG(120)
    n = 300
    lvl = Float64.(rand(rng, 1:4, n))
    X = hcat(lvl, rand(rng, n, 3))
    ymse = 2 .* X[:, 2] .+ (lvl .<= 2) .+ 0.1 .* randn(rng, n)
    ylog = Float64.((X[:, 2] .> 0.6) .& (lvl .<= 2))
    yk = [X[i, 2] > 0.6 ? 1 : lvl[i] <= 2 ? 2 : 3 for i in 1:n]
    for (loss, y) in ((MSE(), ymse), (Logistic(), ylog), (Softmax(3), yk))
        b = fit_boost(X, y, loss; nrounds = 6, eta = 0.3, max_depth = 3, categorical = [1])
        r = shap(b, X)
        s = score(b, X; clip = false)
        if loss isa Softmax
            @test size(r.values) == (n, 4, 2)
            tot = dropdims(sum(r.values; dims = 2); dims = 2)
            @test all(isapprox(tot[i, k] + r.base[k], s[i, k]; atol = 1e-10) for i in 1:n, k in 1:2)
        else
            @test size(r.values) == (n, 4)
            @test vec(sum(r.values; dims = 2)) .+ r.base ≈ s atol = 1e-10
        end
        @test r.base ≈ b.f0 .+ b.eta .* sum(LinearTrees.expected_score(t) for t in b.trees) atol = 1e-12
    end
end

@testset "ensemble SHAP marks the rows the ensemble clamp changed" begin
    # Separable Logistic so the ensemble score runs past ±10 and `clipped` is
    # not uniformly false, which makes the assertion falsifiable.
    rng = StableRNG(123)
    n = 300
    X = rand(rng, n, 2)
    y = Float64.(X[:, 1] .> 0.5)
    b = fit_boost(X, y, Logistic(); nrounds = 40, eta = 1.0, max_depth = 2,
        lambda_slope = 0.0, lambda_intercept = 0.0, min_sum_hessian = 0.0)
    r = shap(b, X)
    @test any(r.clipped)
    @test r.clipped == (score(b, X; clip = true) .!= score(b, X; clip = false))
end

@testset "ensemble SHAP is the eta-weighted sum of tree SHAP" begin
    rng = StableRNG(121)
    X = rand(rng, 200, 3); y = X[:, 1] .* X[:, 2] .+ 0.1 .* randn(rng, 200)
    b = fit_boost(X, y; nrounds = 4, eta = 0.5, max_depth = 2)
    manual = b.eta .* sum(shap(t, X).values for t in b.trees)
    @test shap(b, X).values ≈ manual atol = 1e-12
end

@testset "ensemble SHAP is bit-identical serial and threaded" begin
    rng = StableRNG(124)
    n = 20_000
    X = rand(rng, n, 3); y = X[:, 1] .* X[:, 2] .+ 0.1 .* randn(rng, n)
    b = fit_boost(X, y; nrounds = 4, eta = 0.3, max_depth = 3)
    r1 = shap(b, X; nthreads = 1)
    rn = shap(b, X)
    @test r1.values == rn.values
    @test r1.base == rn.base
    @test r1.clipped == rn.clipped
end

@testset "ensemble importance and coeftable" begin
    rng = StableRNG(122)
    X = rand(rng, 400, 3); y = 3 .* X[:, 1] .+ 0.1 .* randn(rng, 400)
    b = fit_boost(X, y; nrounds = 5, eta = 0.3, max_depth = 2)
    imp = feature_importance(b)
    @test length(imp) == 3 && sum(imp) ≈ 1 && argmax(imp) == 1
    gains = zeros(3)
    for t in b.trees, nd in t.nodes
        LinearTrees.isleaf(nd) || (gains[nd.feature] += max(nd.gain, 0))
    end
    @test imp ≈ gains ./ sum(gains)
    @test feature_importance(fit_boost(X, y; nrounds = 2, max_depth = 0)) == zeros(3)
    x = X[7, :]
    intercept, slopes = coeftable(b, x)
    @test intercept + slopes' * x ≈ score(b, X[7:7, :]; clip = false)[1] atol = 1e-12
    yk = [X[i, 1] > 0.6 ? 1 : X[i, 2] > 0.5 ? 2 : 3 for i in axes(X, 1)]
    bs = fit_boost(X, yk, Softmax(3); nrounds = 3, eta = 0.3, max_depth = 2)
    intercept, slopes = coeftable(bs, x)
    @test collect(intercept + sum(slopes .* x)) ≈ vec(score(bs, X[7:7, :]; clip = false)) atol = 1e-12
end

@testset "ensemble score does not allocate per row" begin
    function boost_score_alloc()
        X = rand(200, 2); y = X[:, 1] .+ 0.1 .* rand(200)
        b = fit_boost(X, y; nrounds = 3, max_depth = 2)
        LinearTrees.score_row(b, X, 1, true)
        return @allocated LinearTrees.score_row(b, X, 1, true)
    end
    @test boost_score_alloc() == 0
end
