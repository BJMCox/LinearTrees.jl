@testset "Huber boosting safeguards an overshooting root update" begin
    X = reshape(collect(1.0:100.0), :, 1)
    y = vcat(zeros(90), fill(100.0, 10))
    loss = Huber(1.0)
    boost = fit_boost(X, y, loss; nrounds = 5, eta = 0.1, max_depth = 0)
    @test length(boost.history) == 5
    @test all(isfinite, boost.history)
    @test all(diff(boost.history) .<= 1e-9)
    raw = score(boost, X; clip = false)
    @test boost.history[end] ≈ deviance(loss, y, raw, ones(length(y)))
    validated = fit_boost(X, y, loss; nrounds = 1, eta = 0.1, max_depth = 0,
        Xval = X[1:20, :], yval = y[1:20])
    @test only(validated.history) ≈ deviance(loss, y[1:20],
        score(validated, X[1:20, :]; clip = false), ones(20))
end

@testset "log-link boosting retains finite raw-score history" begin
    X = reshape(vcat(fill(-1.0, 90), fill(1.0, 10)), :, 1)
    cases = ((Gamma(), vcat(fill(0.01, 90), fill(100.0, 10))),
             (Tweedie(1.5), vcat(zeros(90), fill(100.0, 10))))
    for (loss, y) in cases
        boost = fit_boost(X, y, loss; nrounds = 5, eta = 1.0,
            max_depth = 1, min_fit = 10, min_leaf = 5)
        @test length(boost.history) == 5
        @test all(isfinite, boost.history)
        @test all(diff(boost.history) .<= 1e-8 .* max.(1.0, abs.(boost.history[1:end-1])))
        raw = score(boost, X; clip = false)
        @test boost.history[end] ≈ deviance(loss, y, raw, ones(length(y)))
    end
end
