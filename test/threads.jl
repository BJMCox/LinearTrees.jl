using StableRNGs

@testset "subtree-parallel fit equals serial fit" begin
    rng = StableRNG(34)
    X = rand(rng, 6_000, 5); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 6_000)
    t1 = fit_tree(X, y; nthreads = 1)
    tn = fit_tree(X, y)
    @test t1.nodes == tn.nodes
    # categorical and softmax paths
    lvl = Float64.(rand(rng, 1:6, 6_000))
    Xc = hcat(lvl, X)
    yc = [X[i, 1] > 0.5 ? 1 : lvl[i] <= 3 ? 2 : 3 for i in 1:6_000]
    tc1 = fit_tree(Xc, yc, Softmax(3); categorical = [1], nthreads = 1)
    tcn = fit_tree(Xc, yc, Softmax(3); categorical = [1])
    @test tc1.nodes == tcn.nodes
    @test tc1.catmasks == tcn.catmasks
end

@testset "odd scratch splits and nested row threading" begin
    rng = StableRNG(35)
    X = rand(rng, 20_000, 4); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 20_000)
    lvl = Float64.(rand(rng, 1:7, 20_000))
    Xc = hcat(lvl, X)
    t1 = fit_tree(Xc, y; categorical = [1], nthreads = 1, max_depth = 5)
    for k in (2, 3, 5)
        k > Threads.nthreads() && continue
        tk = fit_tree(Xc, y; categorical = [1], nthreads = k, max_depth = 5)
        @test tk.nodes == t1.nodes
        @test tk.catmasks == t1.catmasks
    end
end
