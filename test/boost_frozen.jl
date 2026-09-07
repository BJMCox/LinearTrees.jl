using StableRNGs, StaticArrays

@testset "Frozen loss methods" begin
    l = LinearTrees.Frozen{Float64}()
    @test issmooth(l)
    @test linkinv(l, 0.3) == 0.3
    @test initscore(l, [(1.0, 1.0)], [1.0]) == 0.0
    @test scorebound(l, [(1.0, 1.0)]) == (-Inf, Inf)
    @test LinearTrees.coeftype(l, Float64) == Float64
    @test LinearTrees.target_eltype(l, [(1.0, 2.0)]) == Float64
    @test LinearTrees.target_eltype(MSE(), [1.0]) == Float64
    y = [(0.5, 2.0), (-1.0, 0.0)]
    g = zeros(2); h = zeros(2)
    gradhess!(g, h, l, y, [1.0, 1.0])
    @test g == [(1.0 - 0.5) * 2.0, (1.0 + 1.0) * 0.0]
    @test h == [2.0, 0.0]
    @test LinearTrees.pointloss(l, (0.5, 2.0), 1.0) == 2.0 * 0.25 / 2
    @test deviance(l, y, [1.0, 1.0], [1.0, 1.0]) ≈ 2 * (0.25 + 0.0)
    # three distinct rejection branches of one guard, plus its accept case
    @test_throws ArgumentError validate_target(l, [(NaN, 1.0)])
    @test_throws ArgumentError validate_target(l, [(1.0, -1.0)])
    @test_throws ArgumentError validate_target(l, [(1.0, Inf)])
    @test validate_target(l, [(1.0, 0.0)]) === nothing

    lv = LinearTrees.Frozen{SVector{2,Float64}}()
    @test LinearTrees.target_eltype(lv, [(SVector(1.0, 2.0), SVector(1.0, 1.0))]) == Float64
    @test scorebound(lv, []) == (SVector(-Inf, -Inf), SVector(Inf, Inf))
    gv = [SVector(0.0, 0.0)]; hv = [SVector(0.0, 0.0)]
    gradhess!(gv, hv, lv, [(SVector(1.0, 2.0), SVector(2.0, 0.0))], [SVector(0.0, 0.0)])
    @test gv[1] == SVector(-2.0, 0.0) && hv[1] == SVector(2.0, 0.0)
    @test LinearTrees.pointloss(lv, (SVector(1.0, 2.0), SVector(2.0, 4.0)), SVector(0.0, 0.0)) == (2.0 * 1.0 + 4.0 * 4.0) / 2
end

@testset "a zero-Hessian Frozen row has zero objective mass" begin
    X = reshape([0.0, 1.0], 2, 1)
    loss = LinearTrees.Frozen{Float64}()
    rule = GainRule(lambda_slope = 0.0, lambda_intercept = 0.0)
    reference = fit_tree(X[1:1, :], [(2.0, 1.0)], loss; rule, max_depth = 0,
        min_fit = 1, min_leaf = 1, truncate = false)
    fitted = fit_tree(X, [(2.0, 1.0), (1e12, 0.0)], loss; rule, max_depth = 0,
        min_fit = 1, min_leaf = 1, truncate = false)
    @test predict(fitted, X) == predict(reference, X)
end

@testset "an all-zero Frozen Hessian coordinate fits zero" begin
    X = reshape([0.0, 1.0], 2, 1)
    scalar = fit_tree(X, [(2.0, 0.0), (-3.0, 0.0)], LinearTrees.Frozen{Float64}();
        max_depth = 0, truncate = false)
    @test score(scalar, X) == [0.0, 0.0]

    V = SVector{2,Float64}
    vector = fit_tree(X,
        [(V(2.0, 10.0), V(1.0, 0.0)), (V(4.0, -10.0), V(1.0, 0.0))],
        LinearTrees.Frozen{V}(); max_depth = 0, truncate = false)
    @test score(vector, X) == fill(V(3.0, 0.0), 2)
end

@testset "a zero-mass vector coordinate preserves an identifiable PLIN fit" begin
    X = reshape(collect(1.0:20.0), 20, 1)
    rule = GainRule(lambda_slope = 0.0, lambda_intercept = 0.0, gamma = 0.0)
    scalar = fit_tree(X, [(2x + 1, 1.0) for x in X[:, 1]],
        LinearTrees.Frozen{Float64}(); rule, max_depth = 1, min_fit = 1,
        min_leaf = 5, truncate = false)
    V = SVector{2,Float64}
    vector = fit_tree(X, [(V(2x + 1, 999.0), V(1.0, 0.0)) for x in X[:, 1]],
        LinearTrees.Frozen{V}(); rule, max_depth = 1, min_fit = 1,
        min_leaf = 5, truncate = false)
    vector_score = score(vector, X)
    @test first.(vector_score) == score(scalar, X)
    @test last.(vector_score) == zeros(20)
end

@testset "a Frozen tree with unit h0 is the MSE GainRule tree on the residual" begin
    rng = StableRNG(103)
    n = 500
    X = rand(rng, n, 4)
    y = sin.(5 .* X[:, 1]) .+ 2 .* X[:, 2] .+ 0.1 .* randn(rng, n)
    F = 0.3 .* X[:, 3]                          # a pretend ensemble score
    r = y .- F
    rule = GainRule(lambda_slope = 0.0, lambda_intercept = 0.0, gamma = 0.0)
    tm = fit_tree(X, r, MSE(); rule, max_depth = 4, truncate = false)
    tf = fit_tree(X, collect(zip(r, ones(n))), LinearTrees.Frozen{Float64}(); rule, max_depth = 4, truncate = false)
    @test length(tm.nodes) == length(tf.nodes)
    for (a, b) in zip(tm.nodes, tf.nodes)
        @test a.model == b.model && a.feature == b.feature && a.left == b.left && a.right == b.right
        a.model == CON || isnan(a.threshold) || @test a.threshold == b.threshold
    end
    # MSE starts at f0 = mean(r), Frozen at 0, so intercepts differ by rounding only
    @test predict(tf, X) ≈ predict(tm, X) atol = 1e-10
    @test score(tf, X) == predict(tf, X)      # identity link
end

@testset "Frozen tree keeps the feature clamp and drops the score clamp" begin
    rng = StableRNG(104)
    X = rand(rng, 200, 2); r = 3 .* X[:, 1]
    t = fit_tree(X, collect(zip(r, ones(200))), LinearTrees.Frozen{Float64}(); rule = GainRule(gamma = 0.0), max_depth = 2)
    @test t.truncate && t.lo == -Inf && t.hi == Inf
    far = [5.0 0.5]
    @test predict(t, far)[1] == predict(t, [maximum(X[:, 1]) 0.5])[1]   # x clamped to the training range
end
