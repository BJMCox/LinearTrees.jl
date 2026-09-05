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
    LinearTrees.irls_weights!(h, MAD(), y, f; ε = 1e-3)   # 1e-3 · median|r|, median|r| = 1
    @test h[1] ≈ 1.0 && h[3] ≈ 1.0 && isfinite(h[2]) && h[2] > 0   # exact residual uses the positive floor
    @test !issmooth(MAD()) && !issmooth(Quantile(0.5))

    LinearTrees.irls_weights!(h, Quantile(0.25), y, f; ε = 1e-3)
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
    LinearTrees.irls_weights!(h, Quantile(0.9), y, f; ε = 7.5e-3)   # median|r| = 7.5
    @test h[5] > 100
end

@testset "gh is type-stable at Float32 (deferred item 4)" begin
    # Two widening paths: Huber's `copysign(l.δ, r)` and Quantile's `1 - l.τ` promoted
    # one branch to Float64 (a Union return); Tweedie's `μ^(2 - ρ)` with a Float64
    # exponent and NegBin's Float64 `θ` promoted every branch (a plain Float64 return).
    for loss in (Huber(1.0), Quantile(0.3), Tweedie(1.5), NegBin(2.0))
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

    # ...and at one: the cumulative weight lands exactly on half, so `wquantile`
    # must average the two adjacent order statistics like `median` does on the
    # duplicated sample (a first-value-reaching-the-target rule returns 3.0, not 3.5)
    iw_tie = [2, 1, 3, 1, 1]   # total 8: cumulative weight lands exactly at half (4)
    dup_tie = reduce(vcat, [fill(yq[i], iw_tie[i]) for i in eachindex(yq)])
    @test initscore(MAD(), yq, Float64.(iw_tie)) ≈ median(dup_tie)

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

@testset "wquantile_select! matches a sort-then-walk oracle (Q2)" begin
    # regression for `median_abs!`'s full sort, profiled at 41% of a MAD fit
    # (bench/PROFILE.md case 3): fails if the O(m) quickselect in
    # `wquantile_select!` ever disagrees with a plain sort on the weighted
    # quantile it returns, including at the exact-boundary tie, with zero
    # weights, and with heavy duplicates.
    #
    # A zero-weight row contributes no copies to the duplicated sample
    # `wquantile` means to match, so this oracle drops zero-weight rows before
    # sorting, exactly as `wquantile_select!` does (see its docstring): once
    # dropped, every remaining run of equal values has positive total weight,
    # so `sortperm`'s unspecified tie order among that run's rows can no
    # longer change which row the walk returns on. The unpatched sort-based
    # code this replaced (`wquantile_sorted`, walking the *unfiltered* `o`)
    # does not have that guarantee: an exact weight boundary immediately
    # followed, in ascending order, by a zero-weight row -- the two rows need
    # not share a value -- lets whatever tie order `sortperm`/`sort!` happens
    # to produce decide the exact-boundary answer. That is a preexisting
    # inconsistency with its own documented invariant ("tie order among equal
    # values never changes the median"), not a target this replacement
    # reproduces.
    function oracle_wquantile(y, w, τ)
        total = sum(w)
        total > 0 || throw(ArgumentError("weights must have a positive sum"))
        idx = [i for i in eachindex(y, w) if w[i] > 0]
        o = idx[sortperm(y[idx])]
        target = τ * total
        cum = zero(total)
        for (k, i) in enumerate(o)
            cum += w[i]
            cum > target && return y[i]
            cum == target && return k < length(o) ? (y[i] + y[o[k + 1]]) / 2 : y[i]
        end
        return y[o[end]]
    end
    oracle_median_abs(r, w) = oracle_wquantile(abs.(r), w, 0.5)

    # n mostly small (1:64): the exact-boundary and zero-weight branches are
    # driven by ties, which small n hits far more often per draw than large n
    # does, so this covers the same branches as a much bigger loop over
    # n in 1:3000 in a fraction of the time; a handful of large-n draws stay
    # in the mix so the O(m) behavior at bigger sizes still gets exercised
    rng = StableRNG(202609)
    ncases = 0
    for trial in 1:3_000
        n = trial <= 2_950 ? rand(rng, 1:64) : rand(rng, 65:3000)
        y = rand(rng, Bool) ? Float64.(rand(rng, -5:5, n)) : round.(randn(rng, n); digits = 1)
        w = rand(rng, Bool) ? Float64.(rand(rng, 0:3, n)) : rand(rng, n) .* 2
        sum(w) > 0 || continue
        ncases += 1
        τ = rand(rng, (0.1, 0.25, 0.3, 0.5, 0.7, 0.75, 0.9))
        buf = similar(y); perm = Vector{Int32}(undef, n)
        got_q = LinearTrees.wquantile_select!(copy(y), w, collect(eachindex(y)), τ)
        got_m = LinearTrees.median_abs!(buf, perm, y, w)
        @test got_q == oracle_wquantile(y, w, τ)
        @test got_m == oracle_median_abs(y, w)
    end
    @test ncases > 2_200   # sanity: the loop actually ran on most of the 3,000 draws

    # zero-weight rows, direct: `wquantile_select!` must ignore them, and the
    # skipped row need not share a value with its neighbors (rev-Q2 finding --
    # `main` returns 1.5 here, averaging the exact boundary at row 1 into row
    # 2's value even though row 2's weight is zero)
    @test LinearTrees.wquantile_select!([1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [1, 2, 3], 0.5) == 2.0
    # duplicate values, direct: a run of equal values counts as one order statistic
    @test LinearTrees.wquantile_select!([1.0, 2.0, 3.0], [0.0, 5.0, 0.0], [1, 2, 3], 0.5) == 2.0
    @test LinearTrees.wquantile_select!(fill(3.0, 10), Float64.([0, 1, 0, 2, 0, 3, 0, 4, 0, 5]), collect(1:10), 0.5) == 3.0
    @test_throws ArgumentError LinearTrees.wquantile_select!([1.0, 2.0], [0.0, 0.0], [1, 2], 0.5)
end

@testset "median_abs! benchmark sizes agree with median_abs (Q2)" begin
    # m = 10^3 and 10^5, the sizes bench/RESULTS.md reports median_abs! at
    rng = StableRNG(3)
    for m in (1_000, 100_000)
        r = randn(rng, m); w = Float64.(rand(rng, 0:3, m))
        buf = similar(r); perm = Vector{Int32}(undef, m)
        @test LinearTrees.median_abs!(buf, perm, r, w) == LinearTrees.median_abs(r, w)
    end
end

@testset "MAD and Quantile(0.3) fits are unchanged by the median_abs! rewrite (Q2)" begin
    # fails if `median_abs!`'s quickselect ever returns a different value than
    # the sort it replaced on any node of these trees: `nodes` and `predict`
    # are recorded from a fit against the pre-rewrite (sort-based)
    # `median_abs!`/`wquantile_sorted`, compared here bit for bit
    #
    # guarded: test/partition.jl also includes this file (and runs after
    # loss.jl in runtests.jl), so an unconditional include here would make
    # its own unconditional include overwrite the method a second time
    @isdefined(partition_cases) || include(joinpath(@__DIR__, "fixtures", "partition", "cases.jl"))
    # the hash is `reduce(xor, reinterpret(UInt64, predict(t, X)))`: exactly
    # associative and commutative, so it does not depend (unlike a floating
    # sum) on thread count or reduction order, only on predict's bit pattern
    golden = Dict(
        ("mse_bic", MAD) => (72, UInt64(18394097934875351018)),
        ("mse_bic", Quantile) => (61, UInt64(18404503152402939754)),
        ("mse_forced", MAD) => (331, UInt64(9153398030783347154)),
        ("mse_forced", Quantile) => (391, UInt64(9130169035490933018)),
        ("softmax_cat", MAD) => (121, UInt64(2251799644130564)),
        ("softmax_cat", Quantile) => (121, UInt64(9221120236987907005)),
        ("quantile_weighted", MAD) => (175, UInt64(9221989060130567109)),
        ("quantile_weighted", Quantile) => (193, UInt64(9226809526271653953)),
    )
    for (name, X, y, loss0, kw) in partition_cases()
        for loss in (MAD(), Quantile(0.3))
            t = fit_tree(X, y, loss; kw...)
            p = predict(t, X)
            nnodes, hash = golden[(name, typeof(loss))]
            @test length(t.nodes) == nnodes
            @test reduce(xor, reinterpret(UInt64, p)) == hash
        end
    end
end
