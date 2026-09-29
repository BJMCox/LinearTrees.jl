using StableRNGs

@testset "importance credits the used feature" begin
    rng = StableRNG(19)
    X = rand(rng, 200, 3); y = 4 .* X[:, 2] .+ 0.01 .* randn(rng, 200)
    imp = feature_importance(fit_tree(X, y))
    @test length(imp) == 3 && sum(imp) ≈ 1
    @test imp[2] > 0.95
    @test feature_importance(fit_tree(X, y; min_fit = 10_000)) == zeros(3)   # con root
end

@testset "invalid categorical codes take the unseen-right path" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    nodes = [N(feature = 1, left = 2, right = 3, catstart = 1, catwords = 1,
               cover = 2.0, model = PCON),
             N(lintercept = 2.0, cover = 1.0), N(lintercept = 3.0, cover = 1.0)]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[1], MSE(), -Inf, Inf, 2.5, 1, false)
    reference = [9.0;;]
    for code in (1.4, 1e30, NaN, Inf)
        X = [code;;]
        @test score(tree, X) == score(tree, reference) == [3.0]
        @test coeftable(tree, [code]) == coeftable(tree, [9.0])
        @test shap(tree, X).values == shap(tree, reference).values
    end
    @test score(tree, [1.0;;]) == [2.0]
end

@testset "coeftable matches the unclipped score" begin
    rng = StableRNG(20)
    X = rand(rng, 300, 2); y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2]
    t = fit_tree(X, y)
    for i in 1:3
        x = X[i, :]
        b, a = coeftable(t, x)
        @test b + dot(a, x) ≈ score(t, X[i:i, :]; clip = false)[1] atol = 1e-10
    end
    # out of range: clamped feature has zero slope, value folded into intercept
    xfar = [10.0, 0.5]
    b, a = coeftable(t, xfar)
    @test a[1] == 0
    @test b + dot(a, xfar) ≈ score(t, reshape(xfar, 1, 2); clip = false)[1] atol = 1e-10
end

@testset "coeftable mirrors categorical routing" begin
    rng = StableRNG(21)
    n = 300; lvl = Float64.(rand(rng, 1:5, n)); x2 = randn(rng, n)
    y = [l in (1.0, 3.0) ? 2.0 : -1.0 for l in lvl] .+ x2
    t = fit_tree([lvl x2], y; categorical = [1])
    for xq in ([2.0, 0.3], [9.0, -0.4], [NaN, 0.1])
        b, a = coeftable(t, xq)
        @test a[1] == 0
        @test b + a[2] * xq[2] ≈ score(t, reshape(xq, 1, 2); clip = false)[1] atol = 1e-10
    end
end
