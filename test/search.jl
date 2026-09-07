using StableRNGs

@testset "split search" begin
    rng = StableRNG(401)
    X = rand(rng, 300, 3)
    y = sin.(5 .* X[:, 1]) .+ X[:, 2] .* X[:, 3]
    kw = (max_depth = 3, min_fit = 10, min_leaf = 5)

    @testset "exact compatibility and fallback" begin
        implicit = fit_tree(X, y; kw...)
        explicit = fit_tree(X, y; split_search = ExactSearch(), kw...)
        @test implicit.nodes == explicit.nodes
        @test implicit.catmasks == explicit.catmasks
        implicit_boost = fit_boost(X, y; nrounds = 2, max_depth = 2)
        explicit_boost = fit_boost(X, y; nrounds = 2, max_depth = 2,
            split_search = ExactSearch())
        @test all(a.nodes == b.nodes for (a, b) in zip(implicit_boost.trees, explicit_boost.trees))

        smallX = X[1:40, :]
        smally = y[1:40]
        fallback = fit_tree(smallX, smally; split_search = BinnedSearch(nbins = 64), kw...)
        exact = fit_tree(smallX, smally; split_search = ExactSearch(), kw...)
        @test fallback.nodes == exact.nodes
        @test predict(fallback, smallX) == predict(exact, smallX)
    end

    @testset "winner-bin refinement recovers a step" begin
        sx = Float64.(1:96)
        sX = reshape(sx, :, 1)
        sy = Float64.(sx .> 37)
        stepkw = (rule = MinDeviance((PCON,)), max_depth = 1, min_leaf = 5,
            truncate = false)
        exact = fit_tree(sX, sy; split_search = ExactSearch(), stepkw...)
        coarse = fit_tree(sX, sy; split_search = BinnedSearch(nbins = 4, refine = false), stepkw...)
        refined = fit_tree(sX, sy; split_search = BinnedSearch(nbins = 4, refine = true), stepkw...)
        @test predict(refined, sX) ≈ predict(exact, sX) atol = 1e-12
        @test predict(refined, sX) ≈ sy atol = 1e-12
        @test sum(abs2, predict(coarse, sX) .- sy) > 0
    end

    @testset "Float32 binned fit is accurate" begin
        x32 = collect(range(0.0f0, 1.0f0, length = 300))
        X32 = reshape(x32, :, 1)
        y32 = @. 1.5f0 + 2.0f0 * x32 + ifelse(x32 > 0.45f0, 1.0f0, 0.0f0)
        tree32 = fit_tree(X32, y32; split_search = BinnedSearch(nbins = 12), max_depth = 2)
        @test maximum(abs, predict(tree32, X32) .- y32) < 2.0f-4
    end

    @testset "weights, ties, and categorical columns" begin
        tx = repeat(Float64.(1:30), inner = 3)
        tw = Float64.(repeat(1:3, 30))
        ty = Float64.(tx .> 14)
        tied = fit_tree(reshape(tx, :, 1), ty; weights = tw,
            split_search = BinnedSearch(nbins = 6), rule = MinDeviance((PCON,)), max_depth = 1)
        @test maximum(abs, predict(tied, reshape(tx, :, 1)) .- ty) < 1e-12

        level = repeat(Float64.(1:6), inner = 30)
        cx = rand(rng, length(level))
        cy = Float64.(level .>= 4)
        cat = fit_tree([cx level], cy; categorical = [2], weights = repeat([1.0, 2.0, 3.0], 60),
            split_search = BinnedSearch(nbins = 3), rule = MinDeviance((PCON,)), max_depth = 1)
        @test predict(cat, [cx level]) == cy
        @test LinearTrees.iscategorical(cat.nodes[1])
    end

    @testset "thread and RNG invariants" begin
        bigX = rand(rng, 20_000, 4)
        bigy = Float64.(bigX[:, 1] .> 0.43) .+ 0.2 .* bigX[:, 2]
        search = BinnedSearch(nbins = 16)
        serial = fit_tree(bigX, bigy; split_search = search, max_depth = 2, nthreads = 1)
        threaded = fit_tree(bigX, bigy; split_search = search, max_depth = 2)
        @test serial.nodes == threaded.nodes

        boostkw = (split_search = search, nrounds = 2, max_depth = 2,
            subsample = 0.9, colsample = 0.75)
        rng1 = StableRNG(402)
        rngn = StableRNG(402)
        b1 = fit_boost(bigX, bigy; rng = rng1, nthreads = 1, boostkw...)
        bn = fit_boost(bigX, bigy; rng = rngn, boostkw...)
        @test all(a.nodes == b.nodes for (a, b) in zip(b1.trees, bn.trees))
        @test rand(rng1) == rand(rngn)
    end

    @testset "scalar-target support boundary" begin
        classes = [X[i, 1] > 0.5 ? 1 : X[i, 2] > 0.5 ? 2 : 3 for i in axes(X, 1)]
        @test_throws ArgumentError fit_tree(X, classes, Softmax(3); split_search = BinnedSearch())
        V = typeof(LinearTrees.coeftype(Softmax(3), Float64)(0, 0))
        target = [(V(y[i], -y[i]), V(1, 1)) for i in eachindex(y)]
        @test_throws ArgumentError fit_tree(X, target, Frozen{V}(); split_search = BinnedSearch())
    end
end
