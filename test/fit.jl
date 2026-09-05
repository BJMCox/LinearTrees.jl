using StableRNGs, Statistics
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
    rng = StableRNG(8)
    n = 60
    X = rand(rng, n, 2); y = X[:, 1] .+ 0.3 .* randn(rng, n)
    w = Float64.(rand(rng, 1:3, n))
    rows = reduce(vcat, [fill(i, Int(w[i])) for i in 1:n])
    for (loss, yy) in ((MSE(), y), (Logistic(), Float64.(y .> median(y))),
                       (Poisson(), Float64.(round.(Int, abs.(y) .* 3))),
                       (MAD(), y), (Quantile(0.7), y))
        tw = fit_tree(X, yy, loss; weights = w)
        td = fit_tree(X[rows, :], yy[rows], loss)
        @test maximum(abs.(predict(tw, X) .- predict(td, X))) < 1e-8
        @test length(tw.nodes) == length(td.nodes)   # the same tree, not just the same predictions
    end
end

@testset "median_abs is a weighted median (I5)" begin
    # Unit weights must reproduce Statistics.median for both parities; general
    # weights must match Statistics.median on the row-duplicated data, which
    # is the actual invariant fit_tree needs (spec line 779-780).
    rng = StableRNG(78)
    for m in 1:12
        r = randn(rng, m)
        @test LinearTrees.median_abs(r, ones(m)) ≈ median(abs.(r))
    end
    for _ in 1:50
        m = rand(rng, 3:15)
        r = randn(rng, m); w = Float64.(rand(rng, 1:4, m))
        dup = reduce(vcat, [fill(r[i], Int(w[i])) for i in 1:m])
        @test LinearTrees.median_abs(r, w) ≈ median(abs.(dup))
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
    rng = StableRNG(9)
    X = rand(rng, 300, 3); y = X[:, 1] .+ 0.1 .* randn(rng, 300)
    Xnan = copy(X); Xnan[7, 1] = NaN
    Xinf = copy(X); Xinf[11, 2] = Inf
    @test_throws ArgumentError fit_tree(Xnan, y)
    @test_throws ArgumentError fit_tree(Xinf, y)
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
    @test LinearTrees.clampscore(maximum(y), lo, hi) == hi
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

@testset "Float32 Logistic fits with truncate = true (C1 regression)" begin
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

@testset "irls_refit does not allocate a fresh residual or sort buffer (B2)" begin
    # fails if irls_refit's residual buffer or median_abs's sort buffers revert
    # to a fresh per-call allocation. `masks` is passed explicitly (as the
    # categorical call site in grow_subtree does) so this isolates exactly the
    # buffers B2 targets: irls_refit's own `masks::Vector{UInt64}=UInt64[]`
    # default, used at the other three call sites, allocates an empty-vector
    # header (32 bytes, measured) regardless of this fix -- unrelated to the
    # residual/sort buffers here, and out of this item's scope.
    function irls_refit_alloc()
        n = 4000
        X = rand(n, 1); y = X[:, 1] .+ 0.1 .* randn(n)
        y[1:20] .+= 5   # outliers, so IRLS solves a non-degenerate residual
        loss = MAD()
        w = ones(n)
        f0 = Float64(LinearTrees.initscore(loss, y, w))
        f = fill(f0, n)
        idx = Matrix{Int32}(undef, n, 1)
        LinearTrees.presort!(idx, X, 1)
        st = LinearTrees.FitState{Float64,Float64,typeof(loss),BIC}(; X, y, w, f,
            g = zeros(n), h = zeros(n), z = zeros(n), idx, isleft = zeros(Bool, n),
            scratch = [LinearTrees.Scratch{Float64,Float64}(n)],
            nodes = LinearTrees.Node{Float64,Float64}[], catmasks = UInt64[],
            iscat = zeros(Bool, 1), nlevels = zeros(Int, 1), loss, rule = BIC(), lo = -Inf, hi = Inf,
            max_depth = 12, min_fit = 10.0, min_leaf = 5.0, min_sum_hessian = 1.0, max_lin_chain = 10,
            truncate = false, nthreads = 1, niter = 5, unith = false)   # MAD, so h is never one
        rows = collect(Int32(1):Int32(n))
        LinearTrees.refresh!(st, rows, 1)
        b = LinearTrees.fit_con(LinearTrees.node_sums(st, rows))[1]
        node = LinearTrees.Node{Float64,Float64}(; lintercept = b, cover = Float64(n))
        masks = UInt64[]
        LinearTrees.irls_refit(st, node, rows, 1, 5, masks)
        return @allocated LinearTrees.irls_refit(st, node, rows, 1, 5, masks)
    end
    @test irls_refit_alloc() == 0
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
