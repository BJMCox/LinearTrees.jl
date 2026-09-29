@testset "log-link updates retain finite true loss across response scales" begin
    X = reshape(vcat(zeros(10), ones(10)), :, 1)
    y = vcat(fill(1e-6, 10), ones(10))
    for loss in (Gamma(), Tweedie(1.99))
        tree = fit_tree(X, y, loss; max_depth=1, min_fit=1, min_leaf=1)
        scores = score(tree, X)
        initial = fill(initscore(loss, y, ones(20)), 20)
        @test all(isfinite, predict(tree, X))
        @test isfinite(deviance(loss, y, scores, ones(20)))
        @test deviance(loss, y, scores, ones(20)) <= deviance(loss, y, initial, ones(20))
    end
    loss = NegBin(2.0)
    @test isfinite(LinearTrees.pointloss(loss, 0.0, -744.0))
    @test isfinite(LinearTrees.pointloss(loss, 1.0, -744.0))
end

@testset "weak nonzero posterior directions remain in the model" begin
    n = 100
    x = collect(range(-1.0, 1.0; length=n))
    y = sinpi.(x)
    X = hcat(x, x .+ 1e-14 .* y)
    λ = 1e-30
    before = precision(BigFloat)
    model = fit_continuous_tree(X, y; max_splits=0, coefficient_precision=λ)
    @test precision(BigFloat) == before
    Z = LinearTrees._continuous_input(model, X)
    B = BigFloat.(hcat(ones(n), Z))
    response = BigFloat.((y .- model.ycenter[1]) ./ model.yscale[1])
    ridge = BigFloat(λ)
    A = B' * B + ridge * I
    coef = A \ (B' * response)
    rate = 1 + (sum(abs2, response - B * coef) + ridge * sum(abs2, coef)) / 2
    evidence = size(B, 2) / 2 * log(ridge) - logdet(A) / 2 - (2 + n / 2) * log(rate)
    @test model.fit.post.rate[1] ≈ Float64(rate) rtol=1e-10
    @test model.fit.post.score ≈ Float64(evidence) rtol=1e-10
    @test model.fit.post.rate[1] < 1.1
    @test sqrt(sum(abs2, predict(model, X) .- y) / n) < 0.03
end
