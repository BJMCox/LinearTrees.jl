using StableRNGs, StaticArrays, Statistics

@testset "Softmax(2) equals Logistic" begin
    rng = StableRNG(16)
    X = randn(rng, 300, 2)
    p = 1 ./ (1 .+ exp.(-(2 .* X[:, 1] .- X[:, 2])))
    yb = Float64.(rand(rng, 300) .< p)
    yk = Int.(2 .- yb)                     # class 1 = positive, class 2 = reference
    tl = fit_tree(X, yb, Logistic())
    ts = fit_tree(X, yk, Softmax(2))
    @test predict(ts, X)[:, 1] ≈ predict(tl, X) atol = 1e-8
    @test [n.model for n in ts.nodes] == [n.model for n in tl.nodes]
end

@testset "softmax probabilities sum to one and recover classes" begin
    rng = StableRNG(17)
    n = 600; X = randn(rng, n, 2)
    y = [X[i, 1] > 0.5 ? 1 : X[i, 2] > 0 ? 2 : 3 for i in 1:n]
    t = fit_tree(X, y, Softmax(3))
    P = predict(t, X)
    @test size(P) == (n, 3)
    @test all(isapprox.(sum(P; dims = 2), 1; atol = 1e-12))
    @test mean(argmax.(eachrow(P)) .== y) > 0.9
    S = score(t, X)
    @test size(S) == (n, 2)
end

@testset "softmax with a categorical feature" begin
    rng = StableRNG(18)
    n = 400; lvl = rand(rng, 1:5, n)
    y = [l in (1, 2) ? 1 : l == 3 ? 2 : 3 for l in lvl]
    t = fit_tree(Float64.(reshape(lvl, n, 1)), y, Softmax(3); categorical = [1])
    @test mean(argmax.(eachrow(predict(t, Float64.(reshape(lvl, n, 1))))) .== y) > 0.95
end

@testset "softmax domain" begin
    @test_throws ArgumentError validate_target(Softmax(3), [1, 2])     # class 3 absent
    @test_throws ArgumentError validate_target(Softmax(3), [1, 2, 4])
end

@testset "threaded softmax split search equals serial" begin
    rng = StableRNG(19)
    n = 20_000; X = randn(rng, n, 3)
    y = [X[i, 1] > 0.5 ? 1 : X[i, 2] > 0 ? 2 : 3 for i in 1:n]
    t1 = fit_tree(X, y, Softmax(3); nthreads = 1)
    tn = fit_tree(X, y, Softmax(3))
    @test t1.nodes == tn.nodes
end
