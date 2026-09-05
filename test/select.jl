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
