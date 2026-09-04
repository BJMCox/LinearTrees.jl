@testset "node storage" begin
    n = Node{Float64,Float64}(; feature = 2, threshold = 1.5, left = 2, right = 3,
        lcoef = 0.3, lintercept = 0.2, rcoef = 0.8, rintercept = -0.3,
        xmin = 0.0, xmax = 4.0, cover = 10.0, xmean = 2.0, model = PLIN)
    @test n.feature == 2 && n.left == 2 && n.right == 3
    @test n.model == PLIN
    @test isbits(n)
    leaf = Node{Float64,Float64}(; lintercept = 1.0, cover = 4.0)
    @test leaf.feature == 0 && leaf.left == 0 && leaf.right == 0
    @test leaf.model == CON
    @test leaf.xmin == -Inf && leaf.xmax == Inf
    @test isbits(MomentSums{Float64}(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    t = LinearTree{Float64,Float64,MSE}([leaf], UInt64[], MSE(), -1.0, 1.0, 0.0, 3, true)
    @test length(t.nodes) == 1 && t.nfeatures == 3
end
