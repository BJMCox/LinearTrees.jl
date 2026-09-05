using StatsAPI, CategoricalArrays, DataFrames, StableRNGs, Statistics

@testset "regressor on a matrix" begin
    rng = StableRNG(27)
    X = rand(rng, 100, 2); y = X[:, 1] .+ 0.1 .* randn(rng, 100)
    m = fit(LinearTreeRegressorFit, X, y)
    @test predict(m, X) == predict(m.tree, X)
    @test residuals(m) ≈ y .- predict(m, X)
    @test nobs(m) == 100
    @test deviance(m) ≈ deviance(MSE(), y, score(m.tree, X), ones(100))
    @test dof(m) == sum(LinearTrees.ncoef(n) for n in m.tree.nodes)
    b, a = coeftable(m, X[1, :])
    @test b + a' * X[1, :] ≈ score(m.tree, X[1:1, :]; clip = false)[1]
end

@testset "tables with categorical columns and stable level maps" begin
    rng = StableRNG(28)
    n = 200
    colour = categorical(rand(rng, ["red", "green", "blue"], n))
    x = randn(rng, n)
    y = (colour .== "red") .* 3 .+ x .+ 0.05 .* randn(rng, n)
    df = DataFrame(colour = colour, x = x)
    m = fit(LinearTreeRegressorFit, df, y)
    @test m.encoder.categorical == [1]
    @test Set(m.encoder.levels[1]) == Set(["red", "green", "blue"])
    # same data, levels in another order, must encode identically
    df2 = DataFrame(colour = categorical(String.(colour); levels = ["blue", "red", "green"]), x = x)
    @test predict(m, df2) == predict(m, df)
    # unseen level policy
    df3 = DataFrame(colour = categorical(["purple"]), x = [0.0])
    m2 = fit(LinearTreeRegressorFit, df, y; unseen = :right)
    @test isfinite(predict(m2, df3)[1])
end

@testset "fit rejects an unrecognised unseen policy" begin
    # fails if a typo'd `unseen` silently selects :error instead of raising
    @test_throws ArgumentError fit(LinearTreeRegressorFit, rand(5, 2), rand(5); unseen = :nope)
end

@testset "classifier" begin
    rng = StableRNG(29)
    X = randn(rng, 300, 2); y = [X[i, 1] > 0 ? "a" : X[i, 2] > 0 ? "b" : "c" for i in 1:300]
    m = fit(LinearTreeClassifierFit, X, y)
    P = predict(m, X)
    @test size(P) == (300, 3) && m.classes == ["a", "b", "c"]
    @test mean(m.classes[argmax.(eachrow(P))] .== y) > 0.9
    # Softmax branch: score(m.tree, X) is a plain n x (K-1) Matrix, repacked into SVector rows
    S = score(m.tree, X)
    @test deviance(m) ≈ deviance(m.tree.loss, m.y, [SVector{2}(S[i, :]) for i in axes(S, 1)], m.w)
    yb = X[:, 1] .> 0
    mb = fit(LinearTreeClassifierFit, X, yb)
    @test size(predict(mb, X)) == (300, 2)
    # Logistic branch: score(mb.tree, X) is already the Vector deviance needs
    @test deviance(mb) ≈ deviance(mb.tree.loss, Float64.(mb.y .== 1), score(mb.tree, X), mb.w)
end

@testset "threaded table fit equals serial" begin
    # above the row-count gate the table is encoded in parallel blocks, so a
    # block that read the wrong rows or the wrong level map would move the tree
    rng = StableRNG(30)
    n = 20_000
    colour = categorical(rand(rng, ["red", "green", "blue"], n))
    x = randn(rng, n)
    df = DataFrame(colour = colour, x = x)
    y = x .+ 0.1 .* randn(rng, n)
    m1 = fit(LinearTreeRegressorFit, df, y; nthreads = 1)
    mn = fit(LinearTreeRegressorFit, df, y)
    @test m1.tree.nodes == mn.tree.nodes
end
