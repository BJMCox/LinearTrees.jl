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
