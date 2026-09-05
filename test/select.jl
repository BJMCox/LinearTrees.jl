@testset "BIC rule" begin
    r = BIC()
    n = 100.0
    dmin = 1e-12
    # equal deviance: lower dof wins
    @test LinearTrees.selection_score(r, CON, 50.0, n, dmin) < LinearTrees.selection_score(r, LIN, 50.0, n, dmin)
    # exact formula
    @test LinearTrees.selection_score(r, PLIN, 10.0, n, dmin) ≈ n * log(10.0 / n) + 7 * log(n)
    # guard: zero deviance uses dmin
    @test LinearTrees.selection_score(r, CON, 0.0, n, dmin) == n * log(dmin / n) + log(n)
    @test LinearTrees.selection_score(r, CON, NaN, n, dmin) == Inf
    custom = BIC(dof = (1.0, 2.0, 3.0, 3.0, 4.0))
    @test LinearTrees.selection_score(custom, PLIN, 10.0, n, dmin) ≈ n * log(10.0 / n) + 4 * log(n)
end

@testset "hoisted log(n) gives the same score to the last bit" begin
    # `scan_feature` evaluates `score_logn` once per feature and hands it to
    # `selection_score` for every split point, so the hoisted call must agree
    # with the unhoisted one exactly (`===`), not approximately: one ULP flips
    # `sc < best.score` and picks a different split. Fails if `score_logn`
    # stops being `log(n)`, or if the penalty term is folded differently.
    dmin = 1e-12
    for r in (BIC(), BIC(dof = (1.0, 2.0, 3.0, 3.0, 4.0)), MinDeviance((PCON, PLIN)))
        for k in (CON, LIN, PCON, BLIN, PLIN), n in (10.0, 100.0, 3.7, 1e6), nc in (1, 2, 4)
            for s in (0.0, 1e-14, 3.5, 50.0, Inf, NaN)
                @test LinearTrees.selection_score(r, k, s, n, dmin, nc) ===
                    LinearTrees.selection_score(r, k, s, n, dmin, nc, LinearTrees.score_logn(r, n))
            end
        end
    end
    # a degenerate node weight must not reach `log`: the score is Inf either way
    @test LinearTrees.score_logn(BIC(), 0.0) == 0.0
    @test LinearTrees.score_logn(BIC(), -1.0) == 0.0
    @test LinearTrees.score_logn(BIC(), NaN) == 0.0
    @test LinearTrees.selection_score(BIC(), CON, 1.0, 0.0, dmin) == Inf
    @test_throws ArgumentError LinearTrees.score_logn(GainRule(0.1), 10.0)
end

@testset "MinDeviance rule" begin
    r = MinDeviance((PCON,))
    @test LinearTrees.allowed(r, PCON) && !LinearTrees.allowed(r, CON) && !LinearTrees.allowed(r, LIN)
    @test LinearTrees.selection_score(r, PCON, 3.0, 10.0, 1e-12) == 3.0
    @test LinearTrees.selection_score(r, CON, 0.0, 10.0, 1e-12) == Inf
end

@testset "GainRule raises ArgumentError, not a bare error (Minor 17)" begin
    r = GainRule(0.1)
    @test_throws ArgumentError LinearTrees.allowed(r, CON)
    @test_throws ArgumentError LinearTrees.selection_score(r, CON, 1.0, 10.0, 1e-12)
end

@testset "devkey is the deviance in the form its rule compares" begin
    # `scan_feature` carries the lowest `devkey` per kind through the split
    # sweep and hands that key to `selection_score` in place of the raw
    # deviance, so a rule's key must (a) give the identical score, to the bit,
    # and (b) order candidates the way the score does, non-decreasing. Fails
    # if a rule's key stops matching its own floor -- if `MinDeviance` gained
    # `BIC`'s `dmin` floor, or `BIC` lost it. One `(kind, n, ncoord)` setting:
    # the property depends on none of the three (`max` is idempotent whatever
    # the penalty term is), and `selection_score`'s own dependence on them is
    # covered by the two testsets above.
    dmin = 1e-3
    devs = (-Inf, 0.0, 1e-9, 1e-3, 0.5, 3.5, 50.0, Inf, NaN)
    for r in (BIC(), BIC(dof = (1.0, 2.0, 3.0, 3.0, 4.0)), MinDeviance((PCON, BLIN, PLIN)),
            LinearTrees.PconOnly(BIC()))
        for s in devs
            @test LinearTrees.selection_score(r, PLIN, LinearTrees.devkey(r, s, dmin), 108.0, dmin) ===
                LinearTrees.selection_score(r, PLIN, s, 108.0, dmin)
        end
        keys = filter(isfinite, [LinearTrees.devkey(r, s, dmin) for s in devs])
        for a in keys, b in keys
            sa = LinearTrees.selection_score(r, PLIN, a, 108.0, dmin)
            sb = LinearTrees.selection_score(r, PLIN, b, 108.0, dmin)
            a < b && isfinite(sa) && @test sa <= sb
        end
    end
    @test_throws ArgumentError LinearTrees.devkey(GainRule(0.1), 1.0, 1e-12)

    # `BIC`'s score is NOT injective in the key: `n log(dev / n)` maps a run of
    # adjacent `Float64` keys to one score, ten of them wide at n = 200_000.
    # This is why the contract is non-decreasing, and it is what makes the
    # within-kind winner the lowest-deviance split point rather than the
    # earliest: both reach the same score, so scoring inside the sweep kept the
    # first and carrying the key keeps the smallest. Fails if `selection_score`
    # ever becomes injective here, which would make the two rules agree.
    # A run of adjacent keys is measured rather than one hardcoded pair: a
    # given pair can straddle the boundary between two runs.
    for (n, minrun) in ((108.0, 2), (200_000.0, 5))
        dmin = eps() * n
        ks = Float64[0.01 * n]
        for _ in 1:39
            push!(ks, prevfloat(ks[end]))
        end
        scs = [LinearTrees.selection_score(BIC(), PLIN, x, n, dmin) for x in ks]
        @test issorted(scs; rev = true)                      # non-decreasing in the key
        @test length(ks) / length(unique(scs)) >= minrun     # this many keys share one score
    end

    # a `-Inf` surrogate must key to `Inf` and so never win its kind, which is
    # what scoring it inside the sweep did (`selection_score` returns `Inf` for
    # any non-finite surrogate). Without the `isfinite` guard the generic
    # `devkey` floors it to `dmin`, the lowest key reachable, and the candidate
    # wins outright. No `fit_*` return has been seen to reach `-Inf`, so this
    # guards a latent case.
    @test LinearTrees.devkey(BIC(), -Inf, 1e-3) === Inf
    @test LinearTrees.devkey(BIC(), NaN, 1e-3) === Inf
    @test LinearTrees.devkey(BIC(), Inf, 1e-3) === Inf
    @test LinearTrees.devkey(BIC(), 1.0f-9, 1.0f-3) === 1.0f-3   # stays in the caller's float type
    @test !isfinite(LinearTrees.devkey(MinDeviance((PCON,)), -Inf, 1e-3))   # dropped by the sweep's own isfinite
end
