using MLJBase, MLJTestInterface, CategoricalArrays, StableRNGs, Statistics

@testset "MLJ generic interface tests" begin
    fails, _ = MLJTestInterface.test([LinearTreeRegressor], MLJTestInterface.make_regression()...;
        mod = @__MODULE__, verbosity = 0, throw = true)
    @test isempty(fails)
    fails, _ = MLJTestInterface.test([LinearTreeClassifier], MLJTestInterface.make_multiclass()...;
        mod = @__MODULE__, verbosity = 0, throw = true)
    @test isempty(fails)
    fails, _ = MLJTestInterface.test([LinearTreeClassifier], MLJTestInterface.make_binary()...;
        mod = @__MODULE__, verbosity = 0, throw = true)
    @test isempty(fails)
end

@testset "MLJ machine with quantile loss and importances" begin
    rng = StableRNG(30)
    X = (a = rand(rng, 200), b = rand(rng, 200))
    y = 3 .* X.a .+ 0.1 .* randn(rng, 200)
    mach = machine(LinearTreeRegressor(loss = Quantile(0.9)), X, y) |> fit!
    yhat = MLJBase.predict(mach, X)   # `predict` is ambiguous: LinearTrees and MLJBase both export it
    @test length(yhat) == 200 && all(isfinite, yhat)
    imps = feature_importances(mach)
    @test first(imps).first == :a
    @test fitted_params(mach).tree isa LinearTree
end

@testset "MLJ classifier returns UnivariateFinite over the training levels" begin
    rng = StableRNG(31)
    X = (a = randn(rng, 300), b = randn(rng, 300))
    y = categorical([X.a[i] > 0 ? "p" : X.b[i] > 0 ? "q" : "r" for i in 1:300])
    mach = machine(LinearTreeClassifier(), X, y) |> fit!
    yhat = MLJBase.predict(mach, X)
    @test levels(mode.(yhat)) == levels(y)
    @test mean(mode.(yhat) .== y) > 0.9
end
