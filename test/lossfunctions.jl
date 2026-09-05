using LossFunctions, StableRNGs
import Distributions

@testset "L2DistLoss adapter equals MSE" begin
    rng = StableRNG(24)
    X = rand(rng, 200, 2); y = X[:, 1] .+ 0.2 .* randn(rng, 200)
    t1 = fit_tree(X, y, MSE()); t2 = fit_tree(X, y, Loss(L2DistLoss()))
    @test predict(t1, X) == predict(t2, X)
    @test [n.model for n in t1.nodes] == [n.model for n in t2.nodes]
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
