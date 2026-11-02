using StableRNGs, StaticArrays, Random, LinearAlgebra

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

    @testset "small suffix mass survives a large prefix" begin
        # Frequency mass and fitting moments both need independent suffix sums.
        for T in (Float32, Float64)
            sx = reshape(T[0, 0, 0, 1], :, 1)
            sy = T[0, 0, 0, 100]
            sw = T[T === Float32 ? 2^24 : 1e16, 1, 1, 5]
            for search in (ExactSearch(), BinnedSearch(nbins=2), HybridSearch(nbins=2))
                tree = fit_tree(sx, sy; weights=sw, min_fit=1, min_leaf=5,
                    max_depth=1, split_search=search)
                @test predict(tree, sx) ≈ sy atol=1e-5
            end
        end
        cx = reshape(Float32[1, 2], :, 1)
        cy = Float32[0, 100]
        cat = fit_tree(cx, cy; weights=Float32[2^24, 5], categorical=[1],
            min_fit=1, min_leaf=5, max_depth=1)
        @test predict(cat, cx) ≈ cy atol=1e-5

        # Frozen curvature can differ greatly even when frequency weights match.
        fx = reshape([0.0, 1.0], :, 1)
        frozen = fit_tree(fx, [(0.0, 1e16), (100.0, 1.0)], Frozen{Float64}();
            rule=GainRule(lambda_slope=0, lambda_intercept=0),
            min_fit=1, min_leaf=1, max_depth=1)
        @test score(frozen, fx) ≈ [0.0, 100.0] atol=1e-10
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

        classes = [row[1] > 0.43 ? 1 : row[2] > 0.57 ? 2 : 3 for row in eachrow(bigX)]
        serial =
            fit_tree(bigX, classes, Softmax(3); split_search = search, max_depth = 2, nthreads = 1)
        threaded = fit_tree(bigX, classes, Softmax(3); split_search = search, max_depth = 2)
        @test predict(serial, bigX) ≈ predict(threaded, bigX)
        rng1, rngn = Xoshiro(1104), Xoshiro(1104)
        b1 = fit_boost(bigX, classes, Softmax(3); rng = rng1, nthreads = 1, boostkw...)
        bn = fit_boost(bigX, classes, Softmax(3); rng = rngn, boostkw...)
        @test predict(b1, bigX) ≈ predict(bn, bigX)
        @test rand(rng1) == rand(rngn)
    end

    @testset "vector bins match independent weighted regression" begin
        x = collect(range(-2.0, 2.0; length = 64))
        z = hcat(@.(0.7x + 2(x > 0.3) + 0.2sin(5x)), @.(-1.1x - 1.5(x > -0.4) + 0.1cos(3x)))
        h = hcat(@.(0.4 + exp(x / 2)), @.(2.1 - 0.5sin(x)))
        w = Float64.(repeat([1, 3, 2, 4], 16))
        target = [(SVector{2}(z[i, :]), SVector{2}(h[i, :])) for i in eachindex(x)]
        coarse = collect(8:8:56)

        function raw_fit(kind, boundary)
            left = x .<= x[boundary]
            A = if kind == PCON
                hcat(left, .!left)
            elseif kind == BLIN
                hcat(x, ones(length(x)), max.(x .- x[boundary], 0))
            else
                hcat(x .* left, left, x .* .!left, .!left)
            end
            prediction = zeros(size(z))
            objective = 0.0
            for k in axes(z, 2)
                mass = w .* h[:, k]
                rootmass = sqrt.(mass)
                beta = (A .* rootmass) \ (z[:, k] .* rootmass)
                prediction[:, k] = A * beta
                residual = z[:, k] - prediction[:, k]
                objective += sum(mass .* residual .^ 2)
                for rows in (left, .!left)
                    prediction[rows, k] .+= sum(mass[rows] .* residual[rows]) / sum(mass[rows])
                end
            end
            return (; boundary, objective, prediction)
        end

        for kind in (PCON, BLIN, PLIN)
            coarse_fits = [raw_fit(kind, boundary) for boundary in coarse]
            winner = coarse_fits[argmin(getproperty.(coarse_fits, :objective))]
            for refine in (false, true)
                boundaries =
                    refine ?
                    union(coarse, max(1, winner.boundary-7):min(63, winner.boundary+8)) : coarse
                filter!(
                    boundary ->
                        sum(w[1:boundary]) >= 5 &&
                        sum(w[(boundary+1):end]) >= 5 &&
                        (kind != PLIN || 5 <= boundary <= 59),
                    boundaries,
                )
                fits = [raw_fit(kind, boundary) for boundary in boundaries]
                expected = fits[argmin(getproperty.(fits, :objective))].prediction
                model = fit_tree(
                    reshape(x, :, 1),
                    target,
                    Frozen{SVector{2,Float64}}();
                    weights = w,
                    split_search = BinnedSearch(; nbins = 8, refine),
                    rule = MinDeviance((kind,)),
                    max_depth = 1,
                    min_leaf = 5,
                    truncate = false,
                )
                observed = reduce(hcat, score(model, reshape(x, :, 1)))'
                @test observed ≈ expected rtol=1e-10 atol=1e-10
            end
        end
    end

    @testset "binned binary softmax matches logistic" begin
        bx = reshape(collect(range(-2.0f0, 2.0f0; length = 120)), :, 1)
        by = Float32[mod(i, 7) < 2 + 3(bx[i] > 0.25) for i in eachindex(bx)]
        for search in (BinnedSearch(nbins = 8, refine = false), BinnedSearch(nbins = 8))
            logistic = fit_tree(bx, by, Logistic(); split_search = search, max_depth = 3)
            softmax =
                fit_tree(bx, Int.(2 .- by), Softmax(2); split_search = search, max_depth = 3)
            @test predict(softmax, bx)[:, 1] ≈ predict(logistic, bx) rtol=2e-5 atol=2e-6
        end
    end

    @testset "multiclass trees and boosting" begin
        rng = Xoshiro(1102)
        mx = rand(rng, 600, 3)
        label(row) = row[1] > 0.43 ? 1 : row[2] > 0.57 ? 2 : 3
        my = label.(eachrow(mx))
        testx = rand(rng, 200, 3)
        testy = label.(eachrow(testx))
        for search in (BinnedSearch(nbins = 12, refine = false), BinnedSearch(nbins = 12))
            tree = fit_tree(mx, my, Softmax(3); split_search = search, max_depth = 3)
            boost = fit_boost(
                mx,
                my,
                Softmax(3);
                split_search = search,
                nrounds = 5,
                max_depth = 2,
                rng = Xoshiro(1103),
            )
            for model in (tree, boost)
                p = predict(model, testx)
                @test maximum(abs, sum(p; dims = 2) .- 1) < 1e-12
                @test count(==(true), argmax.(eachrow(p)) .== testy) > 180
            end
        end
        small = mx[1:30, :]
        exact = fit_tree(small, my[1:30], Softmax(3); split_search = ExactSearch())
        fallback =
            fit_tree(small, my[1:30], Softmax(3); split_search = BinnedSearch(nbins = 32))
        @test predict(exact, testx) ≈ predict(fallback, testx)

        levels = reshape(repeat(Float64.(1:6), inner = 12), :, 1)
        classes = repeat([1, 2, 3, 1, 2, 3], inner = 12)
        weights = repeat([1.0, 3.0, 2.0], 24)
        exactcat = fit_tree(levels, classes, Softmax(3); weights, categorical = [1])
        binnedcat = fit_tree(
            levels,
            classes,
            Softmax(3);
            weights,
            categorical = [1],
            split_search = BinnedSearch(nbins = 2),
        )
        @test predict(binnedcat, levels) ≈ predict(exactcat, levels)
    end

    @testset "vector bins retain small suffixes and zero curvature" begin
        V = SVector{3,Float64}
        fx = reshape([0.0, 0.0, 0.0, 1.0], :, 1)
        target = [(V(0, 5, 999), V(1e16, 1, 0)) for _ = 1:3]
        push!(target, (V(100, -20, -999), V(1, 1e16, 0)))
        model = fit_tree(
            fx,
            target,
            Frozen{V}();
            split_search = BinnedSearch(nbins = 2),
            weights = [4.0, 1.0, 2.0, 5.0],
            rule = GainRule(lambda_slope = 0, lambda_intercept = 0),
            min_fit = 1,
            min_leaf = 1,
            max_depth = 1,
        )
        @test score(model, fx) ≈ [V(0, 5, 0), V(0, 5, 0), V(0, 5, 0), V(100, -20, 0)] atol=1e-10
    end
end
