using LossFunctions, StableRNGs
import Distributions

@testset "nonsmooth adapter scales preserve the ridge objective" begin
    x = collect(-10.0:10)
    X = reshape(x, :, 1)
    y = x .+ 0.2 .* sin.(x)
    opts = (; rule = GainRule(lambda_intercept = 2, lambda_slope = 2),
        max_depth = 1, min_fit = 1, min_leaf = 5)
    adapted = fit_tree(X, y, Loss(L1DistLoss(); scale = 10); opts...)
    weighted = fit_tree(X, y, MAD(); weights = fill(10.0, length(y)), opts...)
    @test predict(adapted, X) ≈ predict(weighted, X) atol = 1e-12

    # Large residuals put the native curvature floor before the loss scale.
    ylarge = 1e10 .* y
    quantile = fit_tree(X, ylarge, Loss(QuantileLoss(0.3); scale = 10); opts...)
    reference = fit_tree(X, ylarge, Quantile(0.3); weights = fill(10.0, length(y)), opts...)
    @test predict(quantile, X) ≈ predict(reference, X) rtol = 1e-12
end

@testset "L2DistLoss adapter equals MSE" begin
    rng = StableRNG(24)
    X = rand(rng, 200, 2); y = X[:, 1] .+ 0.2 .* randn(rng, 200)
    t1 = fit_tree(X, y, MSE()); t2 = fit_tree(X, y, Loss(L2DistLoss()))
    @test predict(t1, X) == predict(t2, X)
    @test [n.model for n in t1.nodes] == [n.model for n in t2.nodes]
end

@testset "HuberLoss adapter preserves robust fitting and loss scale" begin
    x = collect(range(-1.0, 1.0; length = 41))
    X, y = reshape(x, :, 1), 2 .* x
    y[1:4] .+= 20
    native = fit_tree(X, y, Huber(1.0))
    adapted = fit_tree(X, y, Loss(HuberLoss(1.0)))
    scaled = fit_tree(X, y, Loss(HuberLoss(1.0); scale = 2.0))
    @test predict(adapted, X) ≈ predict(native, X)
    # The fixed true-Hessian floor slightly changes the constant Newton step.
    @test maximum(abs, predict(scaled, X) - predict(native, X)) < 1e-6
    native_boost = fit_boost(X, y, Huber(1.0); nrounds = 3, eta = 0.5, max_depth = 1)
    adapted_boost = fit_boost(X, y, Loss(HuberLoss(1.0)); nrounds = 3, eta = 0.5, max_depth = 1)
    @test predict(adapted_boost, X) ≈ predict(native_boost, X)
    @test adapted_boost.history ≈ native_boost.history
end

@testset "HuberLoss keeps representable scaled Float32 directions" begin
    loss = Loss(HuberLoss(1e-50); scale = 1e50)
    X = reshape(Float32[-3, -2, -1, 1, 2, 3], :, 1)
    y = Float32[-6, -4, -2, 2, 4, 6]
    tree = fit_tree(X, y, loss; max_depth = 1, min_fit = 2,
        min_sum_hessian = 0, max_lin_chain = 1, rule = MinDeviance((LIN,)))
    @test predict(tree, X) ≈ y
    Xb, yb = zeros(Float32, 3, 1), Float32[-3, -2, 5]
    boost = fit_boost(Xb, yb, loss; nrounds = 1, eta = 0.5, max_depth = 0)
    @test all(score(boost, Xb; clip = false) .< 0)
    @test only(boost.history) < deviance(loss, yb, zeros(Float32, 3), ones(Float32, 3))
    # Nonpositive adapter scales retain the existing generic loss path.
    inverted = Loss(HuberLoss(1.0); scale = -1.0)
    inverse_boost = fit_boost(Xb, yb, inverted; nrounds = 1, eta = 0.5, max_depth = 0)
    @test only(inverse_boost.history) < deviance(inverted, yb, zeros(Float32, 3), ones(Float32, 3))
end

@testset "LogitMarginLoss adapter preserves logistic damping" begin
    X = reshape([0.0, 0.0, 1.0, 1.0], :, 1)
    y = [0.0, 1.0, 0.0, 1.0]
    w = [89.0, 1.0, 2.0, 8.0]
    native = fit_tree(X, y, Logistic(); weights = w, max_depth = 1)
    adapted = fit_tree(X, y, Loss(LogitMarginLoss(), LinearTrees.LogitLink()); weights = w, max_depth = 1)
    @test predict(adapted, X) ≈ predict(native, X)
end

@testset "LogitMarginLoss adapter equals Logistic on {0,1} targets" begin
    rng = StableRNG(25)
    X = randn(rng, 200, 2); y = Float64.(X[:, 1] .+ 0.3 .* randn(rng, 200) .> 0)
    t1 = fit_tree(X, y, Logistic()); t2 = fit_tree(X, y, Loss(LogitMarginLoss(), LinearTrees.LogitLink()))
    @test predict(t1, X) ≈ predict(t2, X) atol = 1e-10
end

@testset "scale changes min_sum_hessian behaviour" begin
    rng = StableRNG(26)
    X = rand(rng, 60, 1); y = X[:, 1] .+ 0.1 .* randn(rng, 60)
    # scale 1 doubles h relative to MSE, so a threshold that stops MSE may not stop the adapter
    a = fit_tree(X, y, Loss(L2DistLoss(); scale = 1.0); min_sum_hessian = 90.0)
    b = fit_tree(X, y, MSE(); min_sum_hessian = 90.0)
    @test length(a.nodes) >= length(b.nodes)
end

@testset "QuantileLoss adapter matches native Quantile at the zero-residual tie" begin
    rng = StableRNG(30)
    X = randn(rng, 500, 3); y = X[:, 1] .+ 0.5 .* X[:, 2] .+ 0.3 .* randn(rng, 500)
    t1 = fit_tree(X, y, Quantile(0.3)); t2 = fit_tree(X, y, Loss(QuantileLoss(0.3)))
    @test predict(t1, X) == predict(t2, X)
end

@testset "PoissonLoss adapter equals native Poisson through LogLink" begin
    rng = StableRNG(31)
    X = randn(rng, 300, 2)
    μ = exp.(X[:, 1])
    y = Float64.(rand.(rng, Distributions.Poisson.(μ)))
    t1 = fit_tree(X, y, Poisson()); t2 = fit_tree(X, y, Loss(PoissonLoss(), LinearTrees.LogLink()))
    f = 0.3 .* randn(rng, 20); yy = Float64.(rand(rng, 0:6, 20))
    g1 = similar(f); h1 = similar(f); g2 = similar(f); h2 = similar(f)
    gradhess!(g1, h1, Poisson(), yy, f)
    gradhess!(g2, h2, Loss(PoissonLoss(), LinearTrees.LogLink()), yy, f)
    @test g1 ≈ g2 atol = 1e-12
    @test h1 ≈ h2 atol = 1e-12
    @test predict(t1, X) == predict(t2, X)
end
