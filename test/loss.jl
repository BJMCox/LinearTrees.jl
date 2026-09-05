using StableRNGs, StaticArrays
using Statistics: median
import Distributions   # not `using`: Distributions exports its own `Logistic`, colliding with LinearTrees's

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

@testset "gh is type-stable at Float32 (deferred item 4)" begin
    # Huber's `copysign(l.δ, r)` and Quantile's `1 - l.τ`/`-l.τ` used to promote a
    # Float32 `f` to Float64 on one branch, giving a non-concrete Union return
    # type; Tweedie and NegBin widened fully via their Float64-typed fields.
    losses = (MSE(), Huber(1.0), Quantile(0.3), MAD(), Logistic(),
              Poisson(), NegBin(2.0), Gamma(), Tweedie(1.5))
    for loss in losses
        @test only(Base.return_types(LinearTrees.gh, Tuple{typeof(loss),Float32,Float32})) == Tuple{Float32,Float32}
        @test @inferred(LinearTrees.gh(loss, 1.0f0, 0.3f0)) isa Tuple{Float32,Float32}
    end
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
    # Minor 1: log1p(exp(f)) alone overflows to Inf past f = 709, where the true
    # deviance is ≈ 0 since y == 1 makes the loss f - y*f = f - f in the f > 0 limit
    @test isfinite(deviance(Logistic(), [1.0], [800.0], [1.0]))
    @test deviance(Logistic(), [1.0], [800.0], [1.0]) ≈ 0 atol = 1e-6
    @test linkinv(Logistic(), 0.0) == 0.5 && linkinv(Poisson(), 0.0) == 1.0
end

@testset "deviance differences match Distributions.jl logpdf (C1)" begin
    # `deviance(loss, y, f1, w) - deviance(loss, y, f2, w)` must equal
    # `-2 Σ w (logpdf(D(f1), y) - logpdf(D(f2), y))` for the matching
    # Distributions.jl model `D`. Catches a wrong factor of 2, a dropped
    # weight, or a wrong sign in `pointloss`.
    function devdiff(loss, D, y, f1, f2, w)
        lhs = deviance(loss, y, f1, w) - deviance(loss, y, f2, w)
        rhs = -2 * sum(w[i] * (Distributions.logpdf(D(f1[i]), y[i]) - Distributions.logpdf(D(f2[i]), y[i])) for i in eachindex(y))
        return lhs, rhs
    end

    w = [1.0, 2.5, 0.3, 4.0]
    f1 = [0.1, -0.3, 0.5, 1.2]
    f2 = [0.4, 0.2, -0.1, 0.9]

    @testset "Poisson" begin
        y = [0.0, 1.0, 3.0, 5.0]
        lhs, rhs = devdiff(Poisson(), f -> Distributions.Poisson(exp(f)), y, f1, f2, w)
        @test lhs ≈ rhs atol = 1e-10
    end
    @testset "Gamma" begin
        y = [0.5, 2.0, 1.3, 3.7]
        lhs, rhs = devdiff(Gamma(), f -> Distributions.Gamma(1.0, exp(f)), y, f1, f2, w)
        @test lhs ≈ rhs atol = 1e-10
    end
    @testset "NegBin" begin
        θ = 2.0
        y = [0.0, 1.0, 4.0, 2.0]
        lhs, rhs = devdiff(NegBin(θ), f -> Distributions.NegativeBinomial(θ, θ / (exp(f) + θ)), y, f1, f2, w)
        @test lhs ≈ rhs atol = 1e-10
    end
    @testset "Logistic" begin
        y = [0.0, 1.0, 1.0, 0.0]
        lhs, rhs = devdiff(Logistic(), f -> Distributions.Bernoulli(1 / (1 + exp(-f))), y, f1, f2, w)
        @test lhs ≈ rhs atol = 1e-10
    end
    @testset "Softmax(3)" begin
        loss = Softmax(3)
        y = [1, 2, 3, 1]
        sf1 = [SVector(0.2, -0.1), SVector(-0.3, 0.4), SVector(0.1, 0.1), SVector(0.5, -0.5)]
        sf2 = [SVector(-0.1, 0.3), SVector(0.2, -0.2), SVector(-0.4, 0.6), SVector(0.0, 0.0)]
        D(f) = Distributions.Categorical(collect(LinearTrees.probs(loss, f)))
        lhs, rhs = devdiff(loss, D, y, sf1, sf2, w)
        @test lhs ≈ rhs atol = 1e-10
    end
    @testset "MSE" begin
        y = [0.5, -1.2, 3.3, 0.0]
        lhs, rhs = devdiff(MSE(), f -> Distributions.Normal(f, 1.0), y, f1, f2, w)
        @test lhs ≈ rhs atol = 1e-10
    end
end

@testset "initscore and scorebound for Huber, MAD, Gamma, Tweedie, NegBin (C2)" begin
    y = [1.0, 5.0, 100.0]; w = [1.0, 2.0, 0.5]
    @test initscore(Huber(1.0), y, w) ≈ sum(w .* y) / sum(w)   # Huber initscore diverging from the weighted mean

    # integer weights equal row duplication: weighted median against Statistics.median.
    # Holds away from a cumulative-weight tie...
    yq = [3.0, 1.0, 4.0, 1.0, 5.0]
    iw_odd = [2, 1, 2, 1, 1]   # total 7: no cumulative tie at the half-point
    dup_odd = reduce(vcat, [fill(yq[i], iw_odd[i]) for i in eachindex(yq)])
    @test initscore(MAD(), yq, Float64.(iw_odd)) ≈ median(dup_odd)

    # ...but not at one: DEFECT, reported not fixed (no src/ changes in this
    # brief). `wquantile` (used by `initscore(MAD,...)`/`initscore(Quantile,...)`)
    # returns the first value whose cumulative weight reaches the target, with no
    # tie-average at an exact boundary; `median_abs` (the IRLS ε floor's weighted
    # median) does average there. The two disagree under integer-weight
    # duplication exactly at a half-point tie: `wquantile` gives 3.0, the true
    # (duplicated-row) median is 3.5.
    iw_tie = [2, 1, 3, 1, 1]   # total 8: cumulative weight lands exactly at half (4)
    dup_tie = reduce(vcat, [fill(yq[i], iw_tie[i]) for i in eachindex(yq)])
    @test_broken initscore(MAD(), yq, Float64.(iw_tie)) ≈ median(dup_tie)

    yg = [1.0, 2.0, 3.0]; wg = [1.0, 1.0, 2.0]
    @test initscore(Gamma(), yg, wg) ≈ log(sum(wg .* yg) / sum(wg))   # Gamma initscore missing the log link

    y0 = zeros(3); wu = ones(3)
    @test initscore(Tweedie(1.5), y0, wu) == log(1e-6)   # Tweedie floor case y .= 0
    @test initscore(NegBin(2.0), y0, wu) == log(1e-6)    # NegBin floor case y .= 0
    yt = [1.0, 2.0, 3.0]
    @test initscore(Tweedie(1.5), yt, wu) ≈ log(sum(wu .* yt) / sum(wu))
    @test initscore(NegBin(2.0), yt, wu) ≈ log(sum(wu .* yt) / sum(wu))

    # identity losses: padded range formula
    @test scorebound(Huber(1.0), [0.0, 2.0]) == (-2.0, 4.0)   # B = 1, factor 3
    @test scorebound(MAD(), [0.0, 2.0]; truncation_factor = 1) == (0.0, 2.0)

    # log-link losses: S = log(max(maximum(y), 1)) + 3
    @test scorebound(Gamma(), [0.1, 5.0]) == (-(log(5) + 3), log(5) + 3)
    @test scorebound(Gamma(), [0.1, 0.5]) == (-3.0, 3.0)   # maximum(y) < 1 gives S = 3
    @test scorebound(Tweedie(1.5), [0.0, 5.0]) == (-(log(5) + 3), log(5) + 3)
    @test scorebound(Tweedie(1.5), [0.0, 0.5]) == (-3.0, 3.0)
    @test scorebound(NegBin(2.0), [0.0, 5.0]) == (-(log(5) + 3), log(5) + 3)
    @test scorebound(NegBin(2.0), [0.0, 0.5]) == (-3.0, 3.0)
end
