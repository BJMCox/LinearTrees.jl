using StableRNGs
import DecisionTree

@testset "exact linear data gives one lin root" begin
    X = reshape(collect(range(0, 1, length = 50)), 50, 1)
    y = 3 .* X[:, 1] .+ 2
    t = fit_tree(X, y)
    @test t.nodes[1].model == LIN
    @test t.nodes[1].lcoef ≈ 3 && t.nodes[1].lintercept ≈ 2   # root intercept includes the folded start score mean(y) = 3.5
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

@testset "integer weights equal duplication, every loss (spec line 779-780)" begin
    # Before the fix, median_abs took an unweighted median of the stored (not
    # duplicated) rows, so the IRLS epsilon floor moved when weights were
    # non-uniform: MAD and Quantile disagreed with duplication by ~1.8e-3 and
    # ~1.9e-3 (measured interactively at FIX_BASE) despite the other losses
    # already agreeing to 1e-11 or tighter.
    # one loss per weighting path: MSE for the smooth scan, MAD and Quantile
    # for the IRLS refit, where the defect was.
    rng = StableRNG(8)
    n = 60
    X = rand(rng, n, 2); y = X[:, 1] .+ 0.3 .* randn(rng, n)
    w = Float64.(rand(rng, 1:3, n))
    rows = reduce(vcat, [fill(i, Int(w[i])) for i in 1:n])
    for (loss, yy) in ((MSE(), y), (MAD(), y), (Quantile(0.7), y))
        tw = fit_tree(X, yy, loss; weights = w)
        td = fit_tree(X[rows, :], yy[rows], loss)
        @test maximum(abs.(predict(tw, X) .- predict(td, X))) < 1e-8
        @test length(tw.nodes) == length(td.nodes)   # the same tree, not just the same predictions
    end
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

@testset "fit_tree rejects non-finite X (spec line 228)" begin
    # one guard (`isfinite`), so one case: a NaN threshold compares false both
    # ways and would send the row down whichever branch the scan wrote last
    rng = StableRNG(9)
    X = rand(rng, 300, 3); y = X[:, 1] .+ 0.1 .* randn(rng, 300)
    X[7, 1] = NaN
    @test_throws ArgumentError fit_tree(X, y)
end

@testset "truncation_factor below 1 is rejected" begin
    # fails if fit_tree drops the truncation_factor >= 1 guard. Below 1 the
    # padding term goes negative, so the clamp band closes inside the observed
    # range of y and every extreme score is pulled toward the middle with no
    # error anywhere; at 0 the band is a single point.
    rng = StableRNG(10)
    X = rand(rng, 60, 2); y = X[:, 1] .+ 0.1 .* randn(rng, 60)
    @test_throws ArgumentError fit_tree(X, y; truncation_factor = 0.5)
    lo, hi = scorebound(MSE(), y; truncation_factor = 0.5)
    @test lo > minimum(y) && hi < maximum(y)             # the band the guard prevents
end

@testset "Float32 and Float64 fits agree (spec property test)" begin
    # A noise-free linear fixture hits the dmin floor exactly, where Float32
    # rounding of a near-zero RSS can flip the winning kind; this fixture adds
    # real noise so the winning kind (LIN) beats the runner-up (BLIN) by a
    # margin measured directly from scan_feature's BIC scores at the root, not
    # assumed: 10.395 in Float64, 10.398 in Float32 (checked interactively;
    # both comfortably above the spec's 1e-3 floor). The intercept keeps every
    # predicted value away from zero, since a value near zero would blow up
    # the relative-difference check with no fault of Float32 fitting.
    rng = StableRNG(301)
    n = 300
    x1 = rand(rng, n)
    y64 = 10.0 .+ 4.0 .* x1 .+ 0.8 .* randn(rng, n)
    X64 = reshape(x1, n, 1)
    X32 = Float32.(X64); y32 = Float32.(y64)
    t64 = fit_tree(X64, y64)
    t32 = fit_tree(X32, y32)
    @test [n.model for n in t64.nodes] == [n.model for n in t32.nodes]
    p64 = predict(t64, X64); p32 = predict(t32, X32)
    @test maximum(abs.(Float64.(p32) .- p64) ./ abs.(p64)) < 1e-4
end

@testset "Float32 Logistic fits with truncate = true" begin
    rng = StableRNG(302)
    X = Float32.(rand(rng, 100, 2)); y = Float32.(rand(rng, Bool, 100))
    t = fit_tree(X, y, Logistic(); truncate = true)
    @test all(p -> 0 <= p <= 1, predict(t, X))
end

@testset "niter keyword controls IRLS refit passes" begin
    # fails if fit_tree's `niter` kwarg is not threaded to irls_refit's iteration count
    rng = StableRNG(41)
    n = 200
    X = rand(rng, n, 2); y = X[:, 1] .+ 0.2 .* randn(rng, n)
    y[1:5] .+= 20   # outliers: node medians need several IRLS passes to settle
    t5 = fit_tree(X, y, MAD())
    t5b = fit_tree(X, y, MAD(); niter = 5)
    @test t5.nodes == t5b.nodes   # default stays 5, bit for bit
    t1 = fit_tree(X, y, MAD(); niter = 1)
    @test t1.nodes != t5.nodes
end

@testset "threaded predict and score equal serial" begin
    # the threaded *fit* is covered over every thread count in test/threads.jl;
    # this is the row-block split inside `predict`/`score`, above
    # `PARALLEL_MIN_ROWS`.
    rng = StableRNG(35)
    X = rand(rng, 20_000, 6); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 20_000)
    t = fit_tree(X, y; max_depth = 4)                  # nthreads defaults to Threads.nthreads()
    @test predict(t, X) == predict(t, X; nthreads = 1)
    @test score(t, X; clip = false) == score(t, X; clip = false, nthreads = 1)
end

@testset "features keyword restricts the split search" begin
    rng = StableRNG(105)
    n = 400
    X = rand(rng, n, 3)
    y = 4 .* (X[:, 1] .> 0.5) .+ X[:, 2] .+ 0.05 .* randn(rng, n)
    tall = fit_tree(X, y; max_depth = 3)
    @test any(nd -> !LinearTrees.isleaf(nd) && nd.feature == 1, tall.nodes)
    t23 = fit_tree(X, y; max_depth = 3, features = [2, 3])
    @test all(nd -> LinearTrees.isleaf(nd) || nd.feature in (2, 3), t23.nodes)
    # two distinct branches of one guard
    @test_throws ArgumentError fit_tree(X, y; features = [0, 2])
    @test_throws ArgumentError fit_tree(X, y; features = Int[])
end

@testset "presort keyword reproduces the sorted fit, with dropped rows" begin
    rng = StableRNG(106)
    n = 20_000
    X = rand(rng, n, 4); X[:, 4] .= round.(X[:, 4]; digits = 1)     # ties
    y = sin.(3 .* X[:, 1]) .+ X[:, 4] .+ 0.1 .* randn(rng, n)
    w = Float64.(rand(rng, 0:2, n))                                 # drops about a third of the rows
    idx = LinearTrees.presort!(Matrix{Int32}(undef, n, 4), Matrix{Float64}(X), 1)
    saved = copy(idx)
    t1 = fit_tree(X, y; weights = w, max_depth = 6)
    t2 = fit_tree(X, y; weights = w, max_depth = 6, presort = idx)
    @test t1.nodes == t2.nodes
    @test idx == saved                                      # the caller's matrix is not mutated
    @test_throws DimensionMismatch fit_tree(X, y; presort = idx[1:10, :])
    # serial equals threaded on both new paths at once
    tf = fit_tree(X, y; weights = w, max_depth = 6, features = [4, 1], presort = idx)
    @test fit_tree(X, y; weights = w, max_depth = 6, features = [1, 4], presort = idx, nthreads = 1).nodes == tf.nodes
    @test all(nd -> LinearTrees.isleaf(nd) || nd.feature in (1, 4), tf.nodes)
end

@testset "filter_presort! stably renumbers retained rows" begin
    # `fit_tree` does not expose its filtered index columns, and a tree cannot
    # force every feature to win under ties. Check this exact mapping seam
    # directly, with kept and removed rows interleaved in the full presort.
    X = Float64[2 3; 1 2; 2 1; 1 2; 1 2; 3 1; 3 1; 2 3]
    keep = [1, 3, 5, 7]
    presort = LinearTrees.presort!(Matrix{Int32}(undef, 8, 2), X, 1)
    saved = copy(presort)
    idx = LinearTrees.filter_presort!(Matrix{Int32}(undef, length(keep), 2), presort, keep, 1)
    for j in axes(X, 2)
        @test idx[:, j] == Int32.(sortperm(view(X[keep, :], :, j); alg = MergeSort))
    end
    @test presort == saved
end
