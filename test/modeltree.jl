using Statistics: mean
using LinearAlgebra: qr, I

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
    # The binary column becomes constant in some children. Together with the
    # global constant it checks solves whose active regressor set changes.
    X = hcat(x, cos.(2 .* x), Float64.(x .> 0), ones(length(x)))
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

@testset "model-tree split is invariant to extreme predictor units" begin
    x = collect(range(-2.0, 2.0; length=41))
    X = reshape(x, :, 1)
    y = [v <= 0 ? 1 + 2v : 1 - 3v for v in x]
    options = (; lambda=1e-7, max_depth=1, min_leaf=5,
        split_penalty=0.01, truncate=false)
    ordinary = fit_model_tree(X, y; options...)
    huge = fit_model_tree(1e200 .* X, y; options...)
    @test length(huge.leaves) == length(ordinary.leaves) == 2
    @test all(isfinite, predict(huge, 1e200 .* X))
    @test predict(huge, 1e200 .* X) ≈ predict(ordinary, X) atol=1e-7
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
