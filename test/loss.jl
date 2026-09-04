using StableRNGs

pointloss(::MSE, y, f) = (y - f)^2 / 2
pointloss(l::Huber, y, f) = (r = y - f; abs(r) <= l.δ ? r^2 / 2 : l.δ * (abs(r) - l.δ / 2))
pointloss(::Logistic, y, f) = log1p(exp(f)) - y * f
pointloss(::Poisson, y, f) = exp(f) - y * f
pointloss(::Gamma, y, f) = y * exp(-f) + f
pointloss(l::Tweedie, y, f) = (μ = exp(f); ρ = l.ρ; -y * μ^(1 - ρ) / (1 - ρ) + μ^(2 - ρ) / (2 - ρ))
pointloss(l::NegBin, y, f) = (μ = exp(f); θ = l.θ; -y * log(μ / (μ + θ)) + θ * log(1 + μ / θ))

@testset "smooth loss derivatives" begin
    rng = StableRNG(1)
    for (loss, y) in [(MSE(), randn(rng, 5)), (Huber(1.0), 3 .* randn(rng, 5)),
                      (Logistic(), Float64.(rand(rng, Bool, 5))), (Poisson(), Float64.(rand(rng, 0:6, 5))),
                      (Gamma(), rand(rng, 5) .+ 0.5), (Tweedie(1.5), rand(rng, 5) .* 3),
                      (NegBin(2.0), Float64.(rand(rng, 0:6, 5)))]
        f = 0.3 .* randn(rng, 5)
        g = similar(f); h = similar(f)
        gradhess!(g, h, loss, y, f)
        ε = 1e-4  # central second-difference round-off is eps()/ε²: ~2e-8 here, ~2e-4 at 1e-6
        for i in eachindex(f)
            up = f[i] + ε; dn = f[i] - ε
            gfd = (pointloss(loss, y[i], up) - pointloss(loss, y[i], dn)) / (2ε)
            hfd = (pointloss(loss, y[i], up) - 2pointloss(loss, y[i], f[i]) + pointloss(loss, y[i], dn)) / ε^2
            @test g[i] ≈ gfd atol = 1e-6
            @test h[i] ≈ max(hfd, LinearTrees.HMIN) atol = 1e-4
        end
        @test issmooth(loss)
    end
end

@testset "quantile and MAD gradients and IRLS weights" begin
    y = [1.0, 2.0, 3.0]; f = [2.0, 2.0, 2.0]
    g = similar(y); h = similar(y)
    gradhess!(g, h, Quantile(0.25), y, f)
    @test g == [1 - 0.25, 0.0, -0.25]                 # 1(y<f) - τ, zero at equality
    gradhess!(g, h, MAD(), y, f)
    @test g == [1.0, 0.0, -1.0]
    irls_weights!(h, MAD(), y, f)
    @test h[1] ≈ 1.0 && h[3] ≈ 1.0 && isfinite(h[2]) && h[2] > 0   # exact residual uses the positive floor
    @test !issmooth(MAD()) && !issmooth(Quantile(0.5))

    irls_weights!(h, Quantile(0.25), y, f)
    @test h[1] ≈ 0.75 && h[3] ≈ 0.25          # l1weight(r) / |r| with |r| = 1
    @test h[2] > 100                           # r = 0: l1weight gives τ, not gh's exact-zero gradient
end

@testset "irls_weights! uses l1weight, not gh's boundary gradient" begin
    # regression for the defect in irls_weights!: gh reports an exact-zero
    # gradient at r == 0, which used to floor that row's weight to HMIN and
    # bias the Newton step away from a boundary quantile. l1weight gives the
    # row its correct one-sided weight (τ here) instead.
    y = [1.0, 2.0, 2.5, 4.0, 10.0]; f = fill(10.0, 5)
    h = similar(y)
    irls_weights!(h, Quantile(0.9), y, f)
    @test h[5] > 100
end

@testset "loss parameter validation" begin
    @test_throws ArgumentError Huber(-1.0)
    @test_throws ArgumentError Huber(0.0)
    @test_throws ArgumentError Quantile(1.5)
    @test_throws ArgumentError Quantile(0.0)
    @test_throws ArgumentError Tweedie(0.5)
    @test_throws ArgumentError Tweedie(2.0)
    @test_throws ArgumentError NegBin(-2.0)
    @test Huber(1).δ === 1.0 && Quantile(1//4).τ === 0.25
end

@testset "init scores, domains, bounds" begin
    @test initscore(MSE(), [1.0, 3.0], [1.0, 1.0]) == 2.0
    @test initscore(MSE(), [1.0, 3.0], [3.0, 1.0]) == 1.5
    @test initscore(Logistic(), [1.0, 1.0], ones(2)) == log((1 - 1e-6) / 1e-6)
    @test initscore(Poisson(), [0.0, 0.0], ones(2)) == log(1e-6)
    @test initscore(Quantile(0.5), [1.0, 2.0, 10.0], ones(3)) == 2.0
    @test_throws ArgumentError validate_target(Logistic(), [0.0, 2.0])
    @test_throws ArgumentError validate_target(Poisson(), [1.5])
    @test_throws ArgumentError validate_target(Gamma(), [0.0])
    @test_throws ArgumentError validate_target(MSE(), [NaN])
    @test scorebound(MSE(), [0.0, 2.0]) == (-2.0, 4.0)          # B = 1, factor 3
    @test scorebound(MSE(), [0.0, 2.0]; truncation_factor = 1) == (0.0, 2.0)
    @test scorebound(Logistic(), [0.0, 1.0]) == (-10.0, 10.0)
    @test scorebound(Poisson(), [0.0, 5.0]) == (-(log(5) + 3), log(5) + 3)
    @test scorebound(Poisson(), [0.0, 0.0]) == (-3.0, 3.0)
    @test deviance(MSE(), [1.0, 2.0], [0.0, 0.0], ones(2)) == 5.0    # Σ 2ℓ = Σ r²
    @test linkinv(Logistic(), 0.0) == 0.5 && linkinv(Poisson(), 0.0) == 1.0
end
