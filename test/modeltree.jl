using Statistics: mean
using LinearAlgebra: qr, I
using StableRNGs: StableRNG

@testset "greedy model tree finds a change in slope" begin
    x = collect(range(-2.0, 2.0; length=81))
    X = hcat(x, sin.(2 .* x))
    response(x, z) = x <= 0 ? 1 + 2x + 0.2z : 1 - 3x + 0.2z
    y = [response(X[i, 1], X[i, 2]) for i in axes(X, 1)]
    xnew = collect(range(-1.95, 1.95; length=79))
    Xnew = hcat(xnew, sin.(2 .* xnew))
    ynew = [response(Xnew[i, 1], Xnew[i, 2]) for i in axes(Xnew, 1)]
    options = (; lambda=1e-8, max_depth=1, min_leaf=8, split_penalty=0.01)

    global_ridge = fit_model_tree(X, y; lambda=1e-8, max_depth=0)
    exact = fit_model_tree(X, y; options..., split_search=ExactSearch())
    binned = fit_model_tree(X, y; options..., split_search=BinnedSearch(nbins=8))
    coarse = fit_model_tree(X, y; options..., split_search=BinnedSearch(nbins=8, refine=false))
    global_rmse = sqrt(mean(abs2, predict(global_ridge, Xnew) .- ynew))
    exact_rmse = sqrt(mean(abs2, predict(exact, Xnew) .- ynew))
    binned_rmse = sqrt(mean(abs2, predict(binned, Xnew) .- ynew))
    coarse_rmse = sqrt(mean(abs2, predict(coarse, Xnew) .- ynew))
    @test exact_rmse < 0.25 * global_rmse
    @test binned_rmse < 0.35 * global_rmse
    @test coarse_rmse < 0.75 * global_rmse
    @test length(exact.leaves) >= 2

    _, left_slopes = coeftable(exact, [-1.0, sin(-2.0)])
    _, right_slopes = coeftable(exact, [1.0, sin(2.0)])
    @test left_slopes[1] > 1
    @test right_slopes[1] < -2

    repeated = fit_model_tree(X, y; options..., split_search=ExactSearch())
    @test predict(repeated, Xnew) == predict(exact, Xnew)
    repeated_route = [(n.feature, n.threshold) for n in repeated.routing.nodes]
    @test repeated_route == [(n.feature, n.threshold) for n in exact.routing.nodes]
end

@testset "exact model-tree split matches exhaustive QR child fits" begin
    x = collect(range(-1.5, 1.5; length=19))
    # The two-valued column becomes constant in some children. Together with the
    # global constant it checks solves whose active regressor set changes.
    X = hcat(x, cos.(2 .* x), ifelse.(x .> 0, 0.7, -0.2), fill(0.1, length(x)))
    y = [x[i] <= 0 ? 0.5 + 2x[i] - 0.2X[i, 2] : 0.5 - x[i] - 0.2X[i, 2]
        for i in eachindex(x)]
    w = Float64[isodd(i) ? 1 : 2 for i in eachindex(x)]
    λ = 0.2
    min_leaf = 5.0

    # Independent augmented-QR oracle: recenter and rescale each child, then
    # scan every distinct threshold in every split column.
    function child_cost(indices)
        W = sum(w[indices])
        mass = reshape(w[indices], :, 1)
        μx = vec(sum(X[indices, :] .* mass, dims=1)) / W
        μy = sum(w[indices] .* y[indices]) / W
        scales = vec(sqrt.(sum((X[indices, :] .- permutedims(μx)).^2 .* mass, dims=1) / W))
        scales[scales .== 0] .= 1
        A = sqrt.(mass) .* ((X[indices, :] .- permutedims(μx)) ./ permutedims(scales))
        b = sqrt.(w[indices]) .* (y[indices] .- μy)
        q = size(X, 2)
        coefficient = qr([A; sqrt(λ) .* Matrix{Float64}(I, q, q)]) \ [b; zeros(q)]
        return sum(abs2, A * coefficient - b) + λ * sum(abs2, coefficient)
    end

    oracle = (feature=0, threshold=0.0, objective=Inf)
    for j in axes(X, 2), threshold in sort(unique(X[:, j]))[1:end-1]
        left = findall(i -> X[i, j] <= threshold, axes(X, 1))
        right = findall(i -> X[i, j] > threshold, axes(X, 1))
        sum(w[left]) >= min_leaf && sum(w[right]) >= min_leaf || continue
        objective = child_cost(left) + child_cost(right)
        if objective < oracle.objective
            oracle = (; feature=j, threshold, objective)
        end
    end

    model = fit_model_tree(X, y; weights=w, lambda=λ, min_leaf,
        max_depth=1, split_penalty=0.0, truncate=false)
    root = model.routing.nodes[1]
    @test root.feature == oracle.feature
    @test root.threshold ≈ oracle.threshold atol=1e-12
    left_cost = child_cost(findall(i -> X[i, root.feature] <= root.threshold, axes(X, 1)))
    right_cost = child_cost(findall(i -> X[i, root.feature] > root.threshold, axes(X, 1)))
    @test left_cost + right_cost ≈ oracle.objective atol=1e-9
    @test root.gain ≈ child_cost(collect(eachindex(y))) - oracle.objective atol=1e-9
end

@testset "model-tree split retains variation lost by parent centering" begin
    # Subtracting the parent mean rounds the first two predictors to one value.
    X = reshape([-1.0, 1.0, 2.0^55, 2.0^55], :, 1)
    y = [-1.0, 1.0, 0.0, 0.0]
    model = fit_model_tree(X, y; lambda=0.2, min_leaf=2,
        max_depth=1, split_penalty=0.0, truncate=false)
    @test length(model.leaves) == 2
    @test model.routing.nodes[1].threshold == 1.0
    @test predict(model, X) ≈ [-10 / 11, 10 / 11, 0.0, 0.0] atol=1e-12
end

@testset "model-tree split resolves a small jump beside a large slope" begin
    # The moment objective subtracts quantities around 1e18. A false zero at
    # an earlier threshold must not hide the exact piecewise-linear fit.
    x = collect(range(-1.0, 1.0; length=40))
    X = reshape(x, :, 1)
    y = 1e9 .* x .+ 10 .* (x .> 0)
    model = fit_model_tree(X, y; lambda=1e-20, max_depth=1,
        min_leaf=5, split_penalty=0.1, truncate=false)
    @test model.routing.nodes[1].threshold == x[20]
    @test maximum(abs, predict(model, X) .- y) < 2e-6
end

@testset "Float16 constant columns preserve the split ranking" begin
    x = collect(range(-1.5, 1.5; length=25))
    X = Float16.(hcat(x, sin.(x), ifelse.(x .> 0, 0.7, -0.2), fill(0.1, 25)))
    y = Float16.([v <= 0 ? 1 + 2v : 1 - 3v for v in x] +
        0.1randn(StableRNG(6), length(x)))
    w = Float16[isodd(i) ? 0.7 : 1.3 for i in eachindex(x)]
    options = (; lambda=Float16(0.2), max_depth=1, min_leaf=5,
        split_penalty=0, truncate=false)
    model = fit_model_tree(X, y; weights=w, options...)
    reference = fit_model_tree(Float64.(X), Float64.(y); weights=Float64.(w), options...)
    @test (model.routing.nodes[1].feature, model.routing.nodes[1].threshold) == (1, Float16(0.125))
    @test predict(model, X) ≈ predict(reference, Float64.(X)) atol=0.02
end

@testset "near-affine model-tree scores retain child-constant regressors" begin
    x = collect(range(-1.0, 1.0; length=40))
    X = hcat(x, Float64.(x .> 0), fill(0.1, length(x)))
    y = 1e9 .* x .+ 10 .* (x .> 0)
    model = fit_model_tree(X, y; lambda=1e-20, min_leaf=5,
        max_depth=1, split_penalty=0.0, truncate=false)
    @test model.routing.nodes[1].threshold == x[20]
    @test maximum(abs, predict(model, X) .- y) < 2e-6
end

@testset "near-affine model-tree scores retain overflow fallback" begin
    X = reshape([-1e308, -0.99e308, 0.99e308, 1e308], :, 1)
    w = [1.0, 1.0, 10.0, 10.0]
    y = 1e-308 .* X[:, 1] .+ 1e-6 .* (X[:, 1] .> 0)
    # The parent-centered predictor overflows for the negative observations.
    # Weighted raw QR fits remain finite, including within each child.
    model = fit_model_tree(X, y; weights=w, lambda=1e-20, min_leaf=1,
        max_depth=1, split_penalty=0.0, truncate=false)
    @test model.routing.nodes[1].threshold == X[2, 1]
    @test predict(model, X) ≈ y atol=1e-14
end

@testset "model-tree exact split retains a tiny suffix weight" begin
    X = reshape([0.0, 1.0], :, 1)
    y = [0.0, 1.0]
    model = fit_model_tree(X, y; weights=[1e16, 1.0], max_features=0,
        min_leaf=1.0, max_depth=1, split_penalty=0.0)
    @test length(model.leaves) == 2
    @test model.routing.nodes[1].feature == 1
    @test predict(model, X) ≈ y atol=1e-12
end

@testset "model-tree constant regressors retain overflow fallback" begin
    X = hcat(fill(1e100, 4), [0.0, 0.0, 1.0, 1.0])
    y = [1e20, 1e20, 1e20 + 2.0^20, 1e20 + 2.0^20]
    model = fit_model_tree(X, y; weights=fill(1e290, 4), max_features=1,
        lambda=0.2, min_leaf=1e290, max_depth=1, split_penalty=0.0, truncate=false)
    @test length(model.leaves) == 2
    @test model.routing.nodes[1].feature == 2
    @test predict(model, X) == y
end

@testset "model-tree split is invariant to extreme predictor units" begin
    x = collect(range(-2.0, 2.0; length=41))
    X = reshape(x, :, 1)
    y = [v <= 0 ? 1 + 2v : 1 - 3v for v in x]
    options = (; lambda=1e-7, max_depth=1, min_leaf=5,
        split_penalty=0.01, truncate=false)
    ordinary = fit_model_tree(X, y; options...)
    huge = fit_model_tree(1e200 .* X, y; options...)
    tiny = fit_model_tree(1e-200 .* X, y; options...)
    @test length(huge.leaves) == length(ordinary.leaves) == 2
    @test all(isfinite, predict(huge, 1e200 .* X))
    @test predict(huge, 1e200 .* X) ≈ predict(ordinary, X) atol=1e-7
    @test predict(tiny, 1e-200 .* X) ≈ predict(ordinary, X) atol=1e-7
end

@testset "model tree weights and feature translation" begin
    x = collect(range(-2.0, 2.0; length=81))
    X = hcat(x, sin.(2 .* x))
    y = [x[i] < 0 ? 1 + 2x[i] + 0.2X[i, 2] : 1 - 3x[i] + 0.2X[i, 2] for i in eachindex(x)]
    w = Float64[isodd(i) ? 1 : 2 for i in eachindex(x)]
    options = (; weights=w, lambda=1e-7, max_depth=1, split_penalty=0.01)
    model = fit_model_tree(X, y; options...)

    Xbad = [X; NaN 7.0]
    with_zero = fit_model_tree(Xbad, [y; 1e6]; weights=[w; 0.0],
        lambda=1e-7, max_depth=1, split_penalty=0.01)
    @test predict(with_zero, X) ≈ predict(model, X) atol=1e-10

    offset = 1e8
    shifted = copy(X)
    shifted[:, 1] .+= offset
    shifted_model = fit_model_tree(shifted, y; options...)
    @test predict(shifted_model, shifted) ≈ predict(model, X) atol=1e-5
end

@testset "model tree linear leaf and schema" begin
    x = collect(range(-1.0, 1.0; length=41))
    X = hcat(x, sin.(3 .* x))
    y = 1 .+ 2 .* X[:, 1] .- 0.5 .* X[:, 2]
    model = fit_model_tree(X, y; max_depth=0, lambda=1e-8, truncate=false)
    intercept, slopes = coeftable(model, [0.2, sin(0.6)])
    @test intercept ≈ 1 atol=1e-6
    @test slopes ≈ [2.0, -0.5] atol=1e-6
    @test predict(model, X) ≈ y atol=1e-6
    @test_throws DimensionMismatch predict(model, X[:, 1:1])
    @test_throws DimensionMismatch coeftable(model, [0.2])
end
