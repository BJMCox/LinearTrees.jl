using LinearAlgebra: qr, Diagonal
using StaticArrays: SVector

@testset "large-offset linear model trees" begin
    x = 1e9 .+ collect(1.0:50.0)
    u = x .- 1e9
    X = reshape(x, :, 1)
    y = 2 .* u .+ 3
    tree = fit_tree(X, y; rule = MinDeviance((LIN,)), max_depth = 1,
        max_lin_chain = 1, truncate = false, nthreads = 1)
    @test maximum(abs, predict(tree, X) .- y) < 1e-5
end

@testset "offset piecewise lines across search methods" begin
    u = collect(1.0:128.0)
    x = 1e9 .+ u
    X = reshape(x, :, 1)
    hinge = 4 .+ 0.5 .* u .+ 0.8 .* max.(u .- 53, 0)
    separate = ifelse.(u .<= 73, 1 .+ 0.2 .* u, 100 .- 0.7 .* u)
    kw = (max_depth = 1, min_leaf = 5, truncate = false, nthreads = 1)

    for search in (ExactSearch(), BinnedSearch(nbins = 8))
        tree = fit_tree(X, hinge; rule = MinDeviance((BLIN,)), split_search = search, kw...)
        @test maximum(abs, predict(tree, X) .- hinge) < 1e-4
    end
    tree = fit_tree(X, separate; rule = MinDeviance((PLIN,)),
        split_search = HybridSearch(nbins = 8), kw...)
    @test maximum(abs, predict(tree, X) .- separate) < 1e-4
end

@testset "a narrow far-side child still fits its line" begin
    left = 1e9 .+ collect(1.0:50.0)
    right = 1e9 + 1e6 .+ collect(1.0:10.0)
    x = vcat(left, right)
    y = vcat(1 .+ 0.25 .* (left .- 1e9),
        70 .- 0.7 .* (right .- (1e9 + 1e6)))
    X = reshape(x, :, 1)
    tree = fit_tree(X, y; rule = MinDeviance((PLIN,)), split_search = ExactSearch(),
        max_depth = 1, min_leaf = 5, truncate = false, nthreads = 1)
    @test maximum(abs, predict(tree, X) .- y) < 1e-4
end

@testset "IRLS refits agree under a feature translation" begin
    u = collect(1.0:50.0)
    x = 1e9 .+ u
    y = 5 .+ 0.4 .* u .+ 0.02 .* sin.(u)
    y[3] += 5
    y[41] -= 7
    kw = (rule = MinDeviance((LIN,)), max_depth = 1, max_lin_chain = 1,
        niter = 5, truncate = false, nthreads = 1)
    for loss in (MAD(), Quantile(0.7))
        near = fit_tree(reshape(u, :, 1), y, loss; kw...)
        far = fit_tree(reshape(x, :, 1), y, loss; kw...)
        @test maximum(abs, predict(far, reshape(x, :, 1)) .-
            predict(near, reshape(u, :, 1))) < 2e-3
    end
end

@testset "GainRule preserves the raw intercept ridge" begin
    x = 1e4 .+ collect(1.0:40.0)
    u = x .- 1e4
    z = ifelse.(u .<= 20, 7 .+ 0.8 .* u, 40 .- 0.6 .* u)
    X = reshape(x, :, 1)
    λw, λb = 0.7, 1e-5
    target = collect(zip(z, ones(length(z))))
    tree = fit_tree(X, target, LinearTrees.Frozen{Float64}();
        rule = GainRule(lambda_slope = λw, lambda_intercept = λb),
        max_depth = 1, min_leaf = 5, truncate = false, nthreads = 1)
    root = tree.nodes[1]
    @test root.model == PLIN

    # Augmented QR solves the stated penalty in raw x coordinates. The chosen
    # split and fitted branch scores should minimize that same objective.
    function ridge_oracle(ids)
        D = hcat(x[ids], ones(length(ids)))
        A = vcat(D, Diagonal(sqrt.([λw, λb])))
        β = qr(A) \ vcat(z[ids], 0.0, 0.0)
        predictions = D * β
        objective = sum(abs2, z[ids] .- predictions) + λw * β[1]^2 + λb * β[2]^2
        return predictions, objective
    end

    cut = findlast(<=(root.threshold), x)
    @test cut !== nothing && 5 <= cut <= length(x) - 5
    if cut !== nothing && 5 <= cut <= length(x) - 5
        lpred, lobj = ridge_oracle(1:cut)
        rpred, robj = ridge_oracle((cut + 1):length(x))
        fitted_left = root.lcoef .* x[1:cut] .+ root.lintercept
        fitted_right = root.rcoef .* x[(cut + 1):end] .+ root.rintercept
        fitted = vcat(fitted_left, fitted_right)
        fitted_obj = sum(abs2, z .- fitted) +
            λw * (root.lcoef^2 + root.rcoef^2) +
            λb * (root.lintercept^2 + root.rintercept^2)
        @test maximum(abs, fitted_left .- lpred) < 2e-4
        @test maximum(abs, fitted_right .- rpred) < 2e-4
        @test isapprox(fitted_obj, lobj + robj; rtol = 1e-6)
        best = minimum(ridge_oracle(1:k)[2] + ridge_oracle((k + 1):length(x))[2]
            for k in 5:(length(x) - 5))
        @test isapprox(fitted_obj, best; rtol = 1e-6)
    end
end

@testset "a massless Frozen score coordinate stays zero after centered fitting" begin
    u = collect(1.0:50.0)
    x = 1e9 .+ u
    X = reshape(x, :, 1)
    V = SVector{2,Float64}
    target = [(V(2 * value + 3, 9), V(1, 0)) for value in u]
    tree = fit_tree(X, target, LinearTrees.Frozen{V}(); rule = MinDeviance((LIN,)),
        max_depth = 1, max_lin_chain = 1, truncate = false, nthreads = 1)
    fitted = score(tree, X)
    @test maximum(abs, getindex.(fitted, 1) .- (2 .* u .+ 3)) < 1e-5
    @test all(iszero, getindex.(fitted, 2))
end
