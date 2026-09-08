using StableRNGs, InteractiveUtils
import Distributions
using Statistics

@testset "rare-event logistic steps preserve the group signal" begin
    X = reshape(vcat(zeros(90), ones(10)), :, 1)
    y = vcat(1.0, zeros(89), ones(8), zeros(2))
    w = ones(100)
    baseline = deviance(Logistic(), y, fill(initscore(Logistic(), y, w), 100), w)
    for truncate in (true, false)
        tree = fit_tree(X, y, Logistic(); max_depth = 1, nthreads = 1, truncate)
        p = predict(tree, X)
        @test deviance(Logistic(), y, score(tree, X), w) < baseline
        @test p[1] < mean(y) < p[end]
    end
    # Frequency weights represent the same sample, including when feature
    # routing uses a category mask and arithmetic uses Float32.
    compressed = reshape(Float32[1, 1, 2, 2], :, 1)
    weighted = fit_tree(compressed, Float32[0, 1, 0, 1], Logistic();
        weights = Float32[89, 1, 2, 8], categorical = [1], max_depth = 1, nthreads = 1)
    replicated = fit_tree(X, y, Logistic(); max_depth = 1, nthreads = 1)
    @test predict(weighted, compressed)[[1, 4]] ≈ predict(replicated, X)[[1, 100]] rtol = 2e-5
end

@testset "one lin node fit equals one IRLS step" begin
    rng = StableRNG(9)
    x = randn(rng, 200); X = reshape(x, 200, 1)
    loss = Logistic()
    y = Float64.(rand(rng, 200) .< 1 ./ (1 .+ exp.(-x)))
    # one lin node at depth 0 with a fixed start score f0 = initscore
    t = fit_tree(X, y, loss; max_depth = 1, max_lin_chain = 1, rule = MinDeviance((LIN,)), truncate = false)
    f0 = initscore(loss, y, ones(200))
    # IRLS step from f0 by hand: weighted LS of z on [x 1] with weights h
    g = similar(x); h = similar(x)
    gradhess!(g, h, loss, y, fill(f0, 200))
    z = -g ./ h
    D = [x ones(200)]
    β = (D' * (h .* D)) \ (D' * (h .* z))
    n = t.nodes[1]
    @test n.lcoef ≈ β[1] atol = 1e-10
    # fit_tree folds the clamped start score into the root intercept
    @test n.lintercept - f0 ≈ β[2] atol = 1e-10
end

@testset "logistic tree improves deviance and stays in (0,1)" begin
    rng = StableRNG(10)
    X = randn(rng, 300, 2); p = 1 ./ (1 .+ exp.(-(2 .* X[:, 1] .- X[:, 2])))
    y = Float64.(rand(rng, 300) .< p)
    t = fit_tree(X, y, Logistic())
    pr = predict(t, X)
    @test all(0 .< pr .< 1)
    @test deviance(Logistic(), y, score(t, X), ones(300)) < deviance(Logistic(), y, fill(initscore(Logistic(), y, ones(300)), 300), ones(300))
end

@testset "quantile oracle on a con root (n = 21, exact single-step case)" begin
    # A CON leaf's IRLS fixed point is exact when the target's rank k satisfies
    # k - 1 == τ*(n - 1); n = 21 satisfies this for all three τ below, giving
    # a tight, near-exact oracle for the quantile fixed point and pinball optimality.
    n = 21
    y = collect(1.0:n); X = zeros(n, 1)
    for τ in (0.25, 0.9)
        t = fit_tree(X, y, Quantile(τ); min_fit = 100)
        q = initscore(Quantile(τ), y, ones(n))
        @test predict(t, X)[1] ≈ q atol = 1e-9
        pin(v) = sum(r >= 0 ? τ * r : (τ - 1) * r for r in y .- v)
        ys = sort(y); k = findfirst(==(q), ys)
        k > 1 && @test pin(q) <= pin(ys[k - 1]) + 1e-12
        k < n && @test pin(q) <= pin(ys[k + 1]) + 1e-12
    end
end

@testset "MAD tree fits a robust line" begin
    rng = StableRNG(11)
    x = collect(range(0, 1, length = 200)); y = 2 .* x .+ 1
    y[1:10] .+= 50                                   # gross outliers
    t = fit_tree(reshape(x, 200, 1), y, MAD())
    clean = 11:200
    @test maximum(abs, predict(t, reshape(x, 200, 1))[clean] .- y[clean]) < 0.5
end

@testset "count losses return positive means" begin
    rng = StableRNG(12)
    X = rand(rng, 200, 1); μ = exp.(1 .+ X[:, 1])
    y = Float64.(rand.(rng, Distributions.Poisson.(μ)))
    pr = predict(fit_tree(X, y, Poisson()), X)
    @test all(>(0), pr)
    @test cor(pr, μ) > 0.8
    yg = rand.(rng, Distributions.Gamma.(2.0, μ ./ 2))
    @test all(>(0), predict(fit_tree(X, yg, Gamma()), X))

    # A con root predicts `linkinv(loss, initscore(loss, y, w))`, which for a
    # log-link loss is the mean of y, and the truncation band has to be wide
    # enough to leave it alone. One fit per remaining count loss: both are
    # exported, so their init score and clamp band are public behaviour even
    # while they share a method with `Poisson`.
    for loss in (Tweedie(1.5), NegBin(3.0))
        t = fit_tree(X, y, loss; min_fit = 10_000)   # min_fit above n: a con root
        @test length(t.nodes) == 1
        @test predict(t, X)[1] ≈ mean(y)
    end
end

# Every Loss subtype is a public contract, so each gets one fit through
# `fit_tree` and one prediction on its response scale. The last assertion
# fails when a `Loss` subtype is added without a row here, so pruning or
# extension cannot leave a loss unexercised again.
@testset "every Loss subtype fits and predicts on its response scale" begin
    rng = StableRNG(13)
    n = 300
    X = rand(rng, n, 3)
    yreal = X[:, 1] .- 2 .* X[:, 2] .+ 0.05 .* randn(rng, n)
    yfrozen = collect(zip(yreal, ones(n)))
    ypos = exp.(yreal)
    ycount = Float64.(rand(rng, 0:5, n))
    ybin = Float64.(X[:, 1] .> 0.5)
    yclass = [X[i, 1] > 0.5 ? 1 : X[i, 2] > 0.5 ? 2 : 3 for i in 1:n]
    unbounded = (-Inf, Inf)
    cases = [
        (MSE(), yreal, unbounded), (Huber(0.5), yreal, unbounded),
        (Quantile(0.3), yreal, unbounded), (MAD(), yreal, unbounded),
        (Logistic(), ybin, (0.0, 1.0)),
        (Poisson(), ycount, (0.0, Inf)), (NegBin(2.0), ycount, (0.0, Inf)),
        (Gamma(), ypos, (0.0, Inf)), (Tweedie(1.5), ypos, (0.0, Inf)),
        (LinearTrees.Frozen{Float64}(), yfrozen, unbounded),
    ]
    for (loss, y, (lo, hi)) in cases
        pr = predict(fit_tree(X, y, loss), X)
        @test all(isfinite, pr)
        @test all(p -> lo <= p <= hi, pr)
    end
    P = predict(fit_tree(X, yclass, Softmax(3)), X)
    @test all(isfinite, P)
    @test all(isapprox.(sum(P; dims = 2), 1.0; atol = 1e-12))
    # `AdaptedLoss` wraps LossFunctions losses and is fitted in lossfunctions.jl
    fitted = Set(nameof(typeof(c[1])) for c in cases) ∪ Set([:Softmax, :AdaptedLoss])
    @test Set(nameof.(InteractiveUtils.subtypes(LinearTrees.Loss))) == fitted
end
