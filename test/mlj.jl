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

@testset "MLJ boosting models pass the generic interface tests" begin
    fails, _ = MLJTestInterface.test([LinearBoostRegressor], MLJTestInterface.make_regression()...;
        mod = @__MODULE__, verbosity = 0, throw = true)
    @test isempty(fails)
    fails, _ = MLJTestInterface.test([LinearBoostClassifier], MLJTestInterface.make_multiclass()...;
        mod = @__MODULE__, verbosity = 0, throw = true)
    @test isempty(fails)
    fails, _ = MLJTestInterface.test([LinearBoostClassifier], MLJTestInterface.make_binary()...;
        mod = @__MODULE__, verbosity = 0, throw = true)
    @test isempty(fails)
end

@testset "MLJ boosting machine: report, importances, seeded rng" begin
    rng = StableRNG(134)
    X = (a = rand(rng, 300), b = rand(rng, 300))
    y = 3 .* X.a .+ 0.1 .* randn(rng, 300)
    model = LinearBoostRegressor(nrounds = 10, eta = 0.3, max_depth = 2, subsample = 0.7, rng = 7)
    mach = machine(model, X, y) |> fit!
    yhat = MLJBase.predict(mach, X)
    @test length(yhat) == 300 && all(isfinite, yhat)
    rep = report(mach)
    @test rep.nrounds == 10 && length(rep.history) == 10
    fp = fitted_params(mach)
    imps = feature_importances(mach)
    @test first.(imps) == [:a, :b]
    @test last.(imps) == rep.feature_importances
    @test rep.nrounds == nrounds(fp.boost)
    @test rep.history == fp.boost.history
    @test MLJBase.iteration_parameter(LinearBoostRegressor) == :nrounds
    @test MLJBase.training_losses(mach) == rep.history
    mach2 = machine(model, X, y) |> fit!
    @test MLJBase.predict(mach2, X) == yhat
    bad = @test_logs (:warn, r"subsample") LinearBoostRegressor(subsample = 1.5)
    @test bad.subsample == 1.0
    badloss = @test_logs (:warn, r"loss") LinearBoostRegressor(loss = Softmax(3))
    @test badloss.loss == MSE()
end

@testset "MLJ boosting forwards weights and rng" begin
    X = (a = [0.0, 0.0, 0.0],)
    yr = [0.0, 0.0, 12.0]
    wr = [1.0, 1.0, 2.0]
    mr = machine(LinearBoostRegressor(nrounds = 1, max_depth = 0), X, yr, wr) |> fit!
    @test MLJBase.predict(mr, X) ≈ fill(6.0, 3)

    yc = categorical(["no", "yes", "yes"])
    wc = [2.0, 1.0, 1.0]
    mc = machine(LinearBoostClassifier(nrounds = 1, max_depth = 0), X, yc, wc) |> fit!
    @test pdf.(MLJBase.predict(mc, X), yc[1]) ≈ fill(0.5, 3)

    stream = StableRNG(8)
    seeded = LinearBoostRegressor(nrounds = 2, max_depth = 0, subsample = 0.7, rng = stream)
    machine(seeded, X, yr) |> fit!
    oracle = StableRNG(8)
    StatsAPI.fit(LinearBoostRegressorFit, X, yr; nrounds = 2, max_depth = 0,
        subsample = 0.7, rng = oracle)
    @test rand(stream) == rand(oracle)
end
