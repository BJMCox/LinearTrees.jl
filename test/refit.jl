using LinearAlgebra

@testset "affine refits preserve exact constant responses" begin
    X = reshape(Float32.(1:2000), :, 1)
    y = ones(Float32, 2000)
    tree = fit_tree(X, y; max_depth = 0, truncate = false)
    model = refit_leaves(tree, X, y; features = [1], truncate = false)
    @test predict(model, X) == y
    @test coeftable(model, Float32[1]) == (1f0, Float32[0])
end

@testset "weighted affine refit matches a raw-coordinate ridge oracle" begin
    T = Float64
    tree = LinearTree{T,T,MSE}([Node{T,T}(lintercept = 50.0)], UInt64[],
        MSE(), -Inf, Inf, 50.0, 3, false)
    X = [0.0 1.0 4.0;
         1.0 0.0 3.0;
         2.0 1.0 2.0;
         3.0 4.0 1.0;
         NaN  999.0 0.0;
         4.0 2.0 5.0]
    y = [1.0, 2.8, 4.2, 5.0, 1e6, 7.1]
    w = [1.0, 2.0, 1.5, 3.0, 0.0, 2.0]
    λ = 1.25
    original_nodes = copy(tree.nodes)
    model = refit_leaves(tree, X, y; weights = w, features = [1, 2], lambda = λ)
    @test tree.nodes == original_nodes

    keep = findall(>(0), w)
    D = hcat(ones(length(keep)), X[keep, 1:2])
    wk = w[keep]
    center = vec((wk' * X[keep, 1:2]) ./ sum(wk))
    scale = [sqrt(sum(wk .* (X[keep, j] .- center[j]).^2) / sum(wk)) for j in 1:2]
    penalty = Diagonal([0.0; λ .* scale.^2])
    oracle = (D' * Diagonal(wk) * D + penalty) \ (D' * Diagonal(wk) * y[keep])
    intercept, slopes = coeftable(model, [1.5, 1.5, 3.0])
    @test [intercept; slopes[1:2]] ≈ oracle atol = 1e-10
    @test slopes[3] == 0
    @test score(model, X[keep, :]; clip = false) ≈ D * oracle atol = 1e-10
end

@testset "ridge refit preserves extreme weighted predictor scales" begin
    T = Float64
    tree = LinearTree{T,T,MSE}([Node{T,T}()], UInt64[],
        MSE(), -Inf, Inf, 0.0, 1, false)
    y = [-2.0, 2.0]
    w = fill(1e155, 2)
    λ = 2e155
    for magnitude in (1e155, 1e-170)
        X = reshape([-magnitude, magnitude], :, 1)
        model = refit_leaves(tree, X, y; weights = w, features = [1], lambda = λ)
        intercept, slopes = coeftable(model, [0.0])
        # With standardized x = [-1, 1], the ridge slope is
        # (4w)/(2w + λ) = 1; the raw slope is therefore 1 / magnitude.
        @test isapprox(slopes[1], inv(magnitude); rtol = 1e-12, atol = 0)
        @test abs(intercept) <= 1e-12
        @test score(model, X; clip = false) ≈ [-1.0, 1.0] atol = 1e-12
    end

    X = reshape([-1e308, 1e308], :, 1)
    y = [-1.0, 1.0]
    w = [1.0, 1e-10]
    model = refit_leaves(tree, X, y; weights = w, features = [1], lambda = 1.0)
    oracle = setprecision(BigFloat, 256) do
        xb = BigFloat.(X[:, 1]); yb = BigFloat.(y); wb = BigFloat.(w)
        total = sum(wb)
        meanx = sum(wb .* xb) / total
        meany = sum(wb .* yb) / total
        scale = sqrt(sum(wb .* (xb .- meanx).^2) / total)
        z = (xb .- meanx) ./ scale
        slope = sum(wb .* z .* (yb .- meany)) /
            (sum(wb .* z.^2) + BigFloat(1))
        Float64.(meany .+ slope .* z)
    end
    actual = score(model, X; clip = false)
    @test all(isfinite, actual)
    @test isapprox(actual, oracle; atol = 1e-8, rtol = 1e-8)
end

@testset "split routing refits complete terminal responses" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    root = N(feature = 1, threshold = 0.0, left = 2, right = 3,
        lintercept = 100.0, rintercept = -100.0, xmin = -4.0, xmax = 4.0,
        model = PCON)
    tree = LinearTree{T,T,MSE}([root, N(lintercept = 20.0), N(lintercept = 30.0)],
        UInt64[], MSE(), -Inf, Inf, 0.0, 1, false)
    X = reshape([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0], :, 1)
    y = [1.0 + 2.0 * x for x in X[:, 1]]
    y[5:end] .= 7.0 .- X[5:end, 1]
    model = refit_leaves(tree, X, y; lambda = 1e-8)
    @test predict(model, X) ≈ y atol = 1e-7
    @test maximum(abs, predict(tree, X) .- y) > 10
    baseline = predict(model, X)
    tree.nodes[1] = N(feature = 1, threshold = 10.0, left = 2, right = 3, model = PCON)
    @test predict(model, X) == baseline

    missing_children = LinearTree{T,T,MSE}([N(feature = 1, threshold = 0.0,
        left = 0, right = 0, model = PCON)], UInt64[], MSE(), -Inf, Inf, 0.0, 1, false)
    missing_refit = refit_leaves(missing_children, X, y; lambda = 1e-8)
    @test predict(missing_refit, X) ≈ y atol = 1e-7
end

@testset "refit clipping, local coefficients, and schema" begin
    T = Float64
    tree = LinearTree{T,T,MSE}([Node{T,T}()], UInt64[], MSE(), -2.0, 2.0, 0.0, 1, true)
    X = reshape([0.0, 1.0, 2.0, 3.0, 100.0], :, 1)
    y = 2.0 .* X[:, 1]
    w = [1.0, 1.0, 1.0, 1.0, 0.0]
    model = refit_leaves(tree, X, y; weights = w, features = [1], lambda = 1e-8)
    far = [5.0]
    b, a = coeftable(model, far)
    raw = only(score(model, reshape(far, 1, :); clip = false))
    @test a == [0.0] && b ≈ raw
    @test only(predict(model, reshape(far, 1, :))) == 2.0
    inside = [1.5]
    b, a = coeftable(model, inside)
    @test b + dot(a, inside) ≈ only(score(model, reshape(inside, 1, :); clip = false))
    free = refit_leaves(tree, X, y; weights = w, features = [1], lambda = 1e-8, truncate = false)
    @test only(predict(free, reshape(far, 1, :))) > raw

    bad = zeros(4, 2)
    out = fill(17.0, 4)
    @test_throws DimensionMismatch predict!(out, model, bad)
    @test out == fill(17.0, 4)
    @test_throws DimensionMismatch score(model, bad)
    @test_throws DimensionMismatch coeftable(model, [1.0, 2.0])
    Xalias = reshape([0.0, 1.0, 2.0, 3.0, 5.0], :, 1)
    expected = predict(model, copy(Xalias))
    reversed_out = view(Xalias, size(Xalias, 1):-1:1, 1)
    @test predict!(reversed_out, model, Xalias) === reversed_out
    @test reversed_out ≈ expected
    logistic = LinearTree{T,T,Logistic}([Node{T,T}()], UInt64[], Logistic(),
        -Inf, Inf, 0.0, 1, false)
    @test_throws ArgumentError refit_leaves(logistic, X, y)
end
