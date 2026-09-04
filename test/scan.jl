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
    @test c.threshold == 20.0 && c.nleft == 20
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
    # min_leaf = 15 leaves exactly one legal split position
    z2 = [i <= 15 ? 0.0 : 5.0 for i in 1:30]
    c2 = LinearTrees.scan_feature(collect(1.0:30.0), z2, ones(30), ones(30), MinDeviance((PCON,)), 15, 1e-12)
    @test c2.nleft == 15
end

@testset "scan with MinDeviance never returns con" begin
    x = collect(1.0:10.0); z = randn(StableRNG(4), 10)
    c = LinearTrees.scan_feature(x, z, ones(10), ones(10), MinDeviance((PCON,)), 1, 1e-12)
    @test c.kind == PCON
end
