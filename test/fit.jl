using StableRNGs
import DecisionTree

@testset "exact linear data gives one lin root" begin
    X = reshape(collect(range(0, 1, length = 50)), 50, 1)
    y = 3 .* X[:, 1] .+ 2
    t = fit_tree(X, y)
    @test t.nodes[1].model == LIN
    @test t.nodes[1].lcoef ≈ 3 && t.nodes[1].lintercept ≈ 2 - 3.5   # residual after the root score mean(y) = 3.5
    @test predict(t, X) ≈ y atol = 1e-8
end

@testset "CART equivalence under MinDeviance(pcon)" begin
    rng = StableRNG(5)
    X = rand(rng, 200, 3); y = Float64.(X[:, 1] .> 0.5) .+ 0.1 .* randn(rng, 200)
    t = fit_tree(X, y; rule = MinDeviance((PCON,)), max_depth = 3, min_leaf = 5, min_fit = 10, truncate = false)
    cart = DecisionTree.build_tree(y, X, 0, 3, 5, 10)   # n_subfeatures=0, max_depth=3, min_samples_leaf=5, min_samples_split=10
    @test predict(t, X) ≈ DecisionTree.apply_tree(cart, X) atol = 1e-8
end

@testset "truncation bounds" begin
    X = reshape(collect(range(0, 1, length = 50)), 50, 1)
    y = 3 .* X[:, 1] .+ 2
    t = fit_tree(X, y)
    lo, hi = scorebound(MSE(), y)
    @test (t.lo, t.hi) == (lo, hi)
    far = predict(t, [100.0;;])
    @test lo <= far[1] <= hi
    @test far[1] ≈ predict(t, [1.0;;])[1]           # feature clamp to xmax
end

@testset "response shift changes only the root score" begin
    rng = StableRNG(6)
    X = rand(rng, 100, 2); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .+ 0.05 .* randn(rng, 100)
    t1 = fit_tree(X, y); t2 = fit_tree(X, y .+ 10)
    @test predict(t2, X) ≈ predict(t1, X) .+ 10 atol = 1e-8
    @test [n.model for n in t2.nodes] == [n.model for n in t1.nodes]
end

@testset "affine feature invariance" begin
    rng = StableRNG(7)
    X = rand(rng, 100, 2); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .+ 0.05 .* randn(rng, 100)
    X2 = copy(X); X2[:, 1] .= 4 .* X[:, 1] .- 1
    @test predict(fit_tree(X2, y), X2) ≈ predict(fit_tree(X, y), X) atol = 1e-8
end

@testset "integer weights equal duplication" begin
    rng = StableRNG(8)
    X = rand(rng, 40, 2); y = X[:, 1] .+ 0.3 .* randn(rng, 40)
    w = Float64.(rand(rng, 1:3, 40))
    rows = reduce(vcat, [fill(i, Int(w[i])) for i in 1:40])
    tw = fit_tree(X, y; weights = w, min_fit = 12, min_leaf = 6)
    td = fit_tree(X[rows, :], y[rows]; min_fit = 12, min_leaf = 6)
    @test predict(tw, X) ≈ predict(td, X) atol = 1e-8
    @test length(tw.nodes) == length(td.nodes)
end

@testset "stopping rules" begin
    X = reshape(collect(1.0:100.0), 100, 1); y = sin.(X[:, 1])
    t = fit_tree(X, y; max_depth = 2)
    depth(t, k = 1, d = 0) = LinearTrees.isleaf(t.nodes[k]) ? d :
        t.nodes[k].model == LIN ? depth(t, t.nodes[k].left, d) :
        max(depth(t, t.nodes[k].left, d + 1), depth(t, t.nodes[k].right, d + 1))
    @test depth(t) <= 2
    @test length(fit_tree(X, y; min_fit = 1000).nodes) == 1
end

@testset "threaded split search equals serial" begin
    rng = StableRNG(35)
    X = rand(rng, 40_000, 6); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 40_000)
    t1 = fit_tree(X, y; nthreads = 1, max_depth = 4)
    tn = fit_tree(X, y; max_depth = 4)                 # nthreads defaults to Threads.nthreads()
    @test t1.nodes == tn.nodes
    @test predict(tn, X) == predict(tn, X; nthreads = 1)
    @test score(tn, X; clip = false) == score(tn, X; clip = false, nthreads = 1)
end
