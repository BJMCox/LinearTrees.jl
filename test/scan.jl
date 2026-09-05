using StableRNGs

@testset "scan picks lin on exact linear data" begin
    x = collect(range(-1, 1, length = 40)); z = 2 .* x .+ 1
    c = LinearTrees.scan_feature(x, z, ones(40), ones(40), BIC(), 5, 1e-12)
    @test c.kind == LIN
    @test c.lcoef ≈ 2 && c.lintercept ≈ 1
    @test c.surrogate ≈ 0 atol = 1e-10
end

@testset "scan picks pcon at the step" begin
    x = collect(1.0:40.0); z = [i <= 20 ? 0.0 : 5.0 for i in 1:40]
    c = LinearTrees.scan_feature(x, z, ones(40), ones(40), BIC(), 5, 1e-12)
    @test c.kind == PCON
    @test c.threshold == 20.0
    @test c.lintercept ≈ 0 && c.rintercept ≈ 5
end

@testset "scan picks blin on a hinge" begin
    # kink at 0 lands exactly on a grid point: the split point is the largest left value
    # (PILOT parity), so blin's knot only lands on the true kink when 0 is itself a candidate
    x = collect(-2.0:0.05:2.0); z = max.(x, 0)
    c = LinearTrees.scan_feature(x, z, ones(81), ones(81), BIC(), 5, 1e-12)
    @test c.kind == BLIN
    @test c.threshold ≈ 0 atol = 1e-8
    @test c.lcoef ≈ 0 atol = 1e-8
    @test c.rcoef ≈ 1 atol = 1e-8
end

@testset "scan picks con on noise and honours min_leaf" begin
    rng = StableRNG(3)
    x = sort(randn(rng, 30)); z = randn(rng, 30)
    c = LinearTrees.scan_feature(x, z, ones(30), ones(30), BIC(), 5, 1e-12)
    @test c.kind == CON
    # min_leaf = 15 leaves exactly one legal split position: at x = 15
    z2 = [i <= 15 ? 0.0 : 5.0 for i in 1:30]
    c2 = LinearTrees.scan_feature(collect(1.0:30.0), z2, ones(30), ones(30), MinDeviance((PCON,)), 15, 1e-12)
    @test c2.threshold == 15.0
end

@testset "scan with MinDeviance never returns con" begin
    x = collect(1.0:10.0); z = randn(StableRNG(4), 10)
    c = LinearTrees.scan_feature(x, z, ones(10), ones(10), MinDeviance((PCON,)), 1, 1e-12)
    @test c.kind == PCON
end

@testset "integer weights equal row duplication (C3a)" begin
    # `hs` already carries the frequency weight (fit.jl multiplies `st.h[i] *=
    # st.w[i]` before the scan), so an integer-weighted call with hs == ws ==
    # w must agree with unit-weight rows duplicated w[i] times: a dropped or
    # misapplied `ws`/`hs` in `scan_feature` or the moment sums would break this.
    w = Float64[1, 2, 1, 3, 2, 1, 1, 4, 1, 2, 2, 1, 3, 1, 1, 2, 4, 1, 1, 2]

    @testset "pcon design" begin
        x = collect(1.0:20.0)
        z = [i <= 10 ? 1.0 : 4.0 for i in 1:20]
        cw = LinearTrees.scan_feature(x, z, w, w, BIC(), 5, 1e-12)

        xd = reduce(vcat, [fill(x[i], Int(w[i])) for i in eachindex(x)])
        zd = reduce(vcat, [fill(z[i], Int(w[i])) for i in eachindex(x)])
        cd = LinearTrees.scan_feature(xd, zd, ones(length(xd)), ones(length(xd)), BIC(), 5, 1e-12)

        @test cw.kind == cd.kind == PCON
        @test cw.threshold == cd.threshold
        @test cw.lintercept ≈ cd.lintercept atol = 1e-12
        @test cw.rintercept ≈ cd.rintercept atol = 1e-12
        @test cw.score ≈ cd.score atol = 1e-12
    end

    @testset "plin design" begin
        x = collect(1.0:20.0)
        z = [xi <= 10 ? 2xi + 1 : -3xi + 100 for xi in x]   # a jump at the boundary: plin, not blin
        cw = LinearTrees.scan_feature(x, z, w, w, BIC(), 5, 1e-12)

        xd = reduce(vcat, [fill(x[i], Int(w[i])) for i in eachindex(x)])
        zd = reduce(vcat, [fill(z[i], Int(w[i])) for i in eachindex(x)])
        cd = LinearTrees.scan_feature(xd, zd, ones(length(xd)), ones(length(xd)), BIC(), 5, 1e-12)

        @test cw.kind == cd.kind == PLIN
        @test cw.threshold == cd.threshold
        @test cw.lcoef ≈ cd.lcoef atol = 1e-12
        @test cw.lintercept ≈ cd.lintercept atol = 1e-12
        @test cw.rcoef ≈ cd.rcoef atol = 1e-12
        @test cw.rintercept ≈ cd.rintercept atol = 1e-12
        @test cw.score ≈ cd.score atol = 1e-12
    end
end

@testset "duplicate xs values are never split between equal values (C3b)" begin
    # a wrong tie-skip (comparing the wrong index, or dropping the `xs[i] <
    # xs[i + 1]` guard) would let a threshold fall strictly inside a block of
    # equal xs, splitting rows that share a feature value between children.
    x = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]
    z = [xi <= 3.0 ? 0.0 : 5.0 for xi in x]
    n = length(x)
    c = LinearTrees.scan_feature(x, z, ones(n), ones(n), BIC(), 1, 1e-12)
    @test c.kind == PCON
    @test c.threshold in unique(x)           # threshold is always a distinct value, never between two equal ones
    @test c.lintercept ≈ 0.0 atol = 1e-12    # every x == 3.0 row landed left; none split off to the right
    @test c.rintercept ≈ 5.0 atol = 1e-12
end

@testset "nocandidate when the rule allows nothing or min_leaf is too big (C3c)" begin
    x = collect(1.0:10.0); z = randn(StableRNG(5), 10)

    # allowed(rule, ::ModelKind) is false for every kind, including con
    c1 = LinearTrees.scan_feature(x, z, ones(10), ones(10), MinDeviance(()), 1, 1e-12)
    @test c1.score == Inf && c1.kind == CON

    # con isn't offered by this rule, and min_leaf (100) exceeds Σw (10), so no
    # split position qualifies either
    c2 = LinearTrees.scan_feature(x, z, ones(10), ones(10), MinDeviance((PCON,)), 100, 1e-12)
    @test c2.score == Inf && c2.kind == CON
end
