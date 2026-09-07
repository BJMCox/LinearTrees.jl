using Random: randperm
using StableRNGs
using Statistics

@testset "hybrid split search" begin
    @testset "ties and exact fallback" begin
        Xzero = reshape([-0.0, 0.0, 1.0, 2.0], :, 1)
        yzero = [2.0, 2.0, -1.0, -1.0]
        zero_tree = fit_tree(Xzero, yzero; split_search = HybridSearch(nbins = 2),
            rule = MinDeviance((PCON,)), max_depth = 1, min_fit = 1, min_leaf = 1)
        @test predict(zero_tree, Xzero) == yzero

        x = repeat(Float64.(1:16), inner = 3)
        X = reshape(x, :, 1)
        y = sin.(x)
        kw = (weights = repeat([0.2, 0.5, 1.0], 16),
            rule = MinDeviance((PCON,)), max_depth = 2, min_leaf = 1,
            truncate = false, nthreads = 1)
        hybrid = fit_tree(X, y; split_search = HybridSearch(nbins = 32), kw...)
        exact = fit_tree(X, y; split_search = ExactSearch(), kw...)
        @test predict(hybrid, X) ≈ predict(exact, X) atol = 1e-12
    end

    @testset "Float32 and nonlinear scalar losses" begin
        x = collect(range(0.0f0, 1.0f0; length = 300))
        X = reshape(x, :, 1)
        y = @. 1.5f0 + 2.0f0 * x + ifelse(x > 0.45f0, 1.0f0, 0.0f0)
        tree = fit_tree(X, y; split_search = HybridSearch(nbins = 12), max_depth = 2)
        @test maximum(abs, predict(tree, X) .- y) < 2.0f-4

        xq = collect(range(0.0, 1.0; length = 400))
        Xq = reshape(xq, :, 1)
        yq = sin.(5 .* xq)
        yq[1:20:end] .+= 1
        quantile = fit_tree(Xq, yq, Quantile(0.5);
            split_search = HybridSearch(nbins = 32), max_depth = 3, nthreads = 1)
        @test mean(abs, predict(quantile, Xq) .- yq) < mean(abs, yq .- median(yq))
    end

    @testset "PLIN refinement considers the whole winning bin" begin
        x = Float64.(1:128)
        X = reshape(x, :, 1)
        y = ifelse.(x .<= 53, 0.4 .* x, 200 .- 2 .* x)
        kw = (rule = MinDeviance((PLIN,)), max_depth = 1, min_leaf = 1,
            nthreads = 1, truncate = false)
        hybrid = fit_tree(X, y; split_search = HybridSearch(nbins = 2), kw...)
        exact = fit_tree(X, y; split_search = ExactSearch(), kw...)
        @test predict(hybrid, X) ≈ predict(exact, X) atol = 1e-10
    end

    @testset "weights, ties, and zero filtering" begin
        x = repeat(Float64.(1:100), inner = 3)
        X = reshape(x, :, 1)
        y = repeat(Float64.(x[1:3:end] .> 50), inner = 3)
        w = repeat([0.0, 2.0, 3.0], 100)
        kw = (split_search = HybridSearch(nbins = 4),
            rule = MinDeviance((PCON,)), max_depth = 1, nthreads = 1)
        weighted = fit_tree(X, y; weights = w, kw...)
        keep = findall(>(0), w)
        filtered = fit_tree(X[keep, :], y[keep]; weights = w[keep], kw...)
        @test predict(weighted, X) ≈ predict(filtered, X) atol = 1e-12
        @test predict(weighted, X) ≈ y atol = 1e-12

        edgeX = reshape(Float64[0, 0, 0, 1, 1, 1], :, 1)
        edgey = Float64[0, 0, 0, 1, 1, 1]
        edge = fit_tree(edgeX, edgey; weights = [1e16, 1, 1, 1, 1, 1],
            split_search = HybridSearch(nbins = 4), rule = MinDeviance((PCON,)),
            max_depth = 1, min_fit = 1, min_leaf = 1)
        @test predict(edge, edgeX) ≈ edgey
    end

    @testset "single-global-bin descendants use exact fallback" begin
        x = collect(range(0.0, 1.0; length = 400))
        X = reshape(x, :, 1)
        y = Float64.(x .> 0.41) .+ 2 .* Float64.(x .> 0.44)
        tree = fit_tree(X, y; split_search = HybridSearch(nbins = 4),
            rule = MinDeviance((PCON,)), max_depth = 3, nthreads = 1)
        @test maximum(abs, predict(tree, X) .- y) < 1e-10
    end

    @testset "threaded search preserves predictions" begin
        rng = StableRNG(501)
        X = rand(rng, 20_001, 4)
        y = Float64.(X[:, 1] .> 0.43) .+ 0.2 .* X[:, 2]
        kw = (split_search = HybridSearch(nbins = 16), max_depth = 2)
        serial = fit_tree(X, y; nthreads = 1, kw...)
        threaded = fit_tree(X, y; kw...)
        @test predict(serial, X) ≈ predict(threaded, X) atol = 1e-12
    end

    @testset "sampled boosting equals a replayed Frozen recurrence" begin
        rng = StableRNG(502)
        n, p = 240, 4
        X = rand(rng, n, p)
        y = sin.(4 .* X[:, 1]) .+ X[:, 2] .* X[:, 3]
        w = Float64.(rand(rng, 1:3, n))
        eta = 0.2
        search = HybridSearch(nbins = n)
        fitrng = StableRNG(503)
        boost = fit_boost(X, y; weights = w, nrounds = 4, eta,
            max_depth = 3, subsample = 0.6, colsample = 0.5,
            split_search = search, rng = fitrng, nthreads = 1)

        oracle_rng = StableRNG(503)
        F = fill(initscore(MSE(), y, w), n)
        g = similar(F)
        h = similar(F)
        for tree in boost.trees
            gradhess!(g, h, MSE(), y, F)
            target = collect(zip(-g ./ h, h))
            wr = zeros(n)
            rows = randperm(oracle_rng, n)[1:144]
            wr[rows] .= w[rows]
            features = sort!(randperm(oracle_rng, p)[1:2])
            replay = fit_tree(X, target, Frozen{Float64}(); weights = wr,
                features, rule = GainRule(), max_depth = 3,
                split_search = search, nthreads = 1)
            @test score(tree, X; clip = false, nthreads = 1) ≈
                score(replay, X; clip = false, nthreads = 1) atol = 1e-12
            F .+= eta .* score(replay, X; clip = false, nthreads = 1)
        end
        @test score(boost, X; clip = false) ≈ F atol = 1e-12
        @test rand(fitrng) == rand(oracle_rng)

        repeatkw = (nrounds = 3, max_depth = 2, subsample = 0.7,
            colsample = 0.75, split_search = HybridSearch(nbins = 16))
        rng1 = StableRNG(504)
        rngn = StableRNG(504)
        a = fit_boost(X, y; rng = rng1, nthreads = 1, repeatkw...)
        b = fit_boost(X, y; rng = rngn, repeatkw...)
        @test predict(a, X) == predict(b, X)
        @test rand(rng1) == rand(rngn)
        @test mean(abs2, predict(a, X) .- y) < mean(abs2, y .- mean(y))
    end
end
