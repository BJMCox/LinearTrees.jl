using StableRNGs

@testset "categorical pcon split routes by level set" begin
    rng = StableRNG(13)
    n = 300
    lvl = rand(rng, 1:6, n)
    means = [0.0, 5.0, 0.0, 5.0, 5.0, 0.0]        # levels 2, 4, 5 are high
    y = means[lvl] .+ 0.1 .* randn(rng, n)
    X = Float64.(reshape(lvl, n, 1))
    t = fit_tree(X, y; categorical = [1], max_depth = 1)
    root = t.nodes[1]
    @test root.model == PCON && LinearTrees.iscategorical(root)
    lefts = [LinearTrees.category_is_left(t, root, c) for c in 1:6]
    @test lefts == (means .== 0.0) || lefts == (means .== 5.0)
    pr = predict(t, X)
    @test maximum(abs, pr .- means[lvl]) < 0.2
    # unseen level 9 routes right
    @test predict(t, [9.0;;])[1] == predict(t, [findfirst(!, lefts) * 1.0;;])[1]
    # hand-built mask decode: score() must route by the literal catmask bits,
    # independent of fit_tree's search (was predict.jl's "categorical routing" test)
    hnodes = [Node{Float64,Float64}(feature = 1, left = 2, right = 3, lintercept = 1.0, rintercept = 2.0,
                  catstart = 1, catwords = 1, model = PCON),
              Node{Float64,Float64}(), Node{Float64,Float64}()]
    hmasks = [UInt64(0b101)]                         # levels 1 and 3 go left
    htree = LinearTree{Float64,Float64,MSE}(hnodes, hmasks, MSE(), -10.0, 10.0, 0.0, 1, true)
    @test score(htree, [1.0; 2.0; 3.0; 7.0;;]) == [1.0, 2.0, 1.0, 2.0]   # 7 is unseen, routes right
end

@testset "categorical columns never get linear pieces" begin
    rng = StableRNG(14)
    n = 200; lvl = rand(rng, 1:4, n); x2 = randn(rng, n)
    y = Float64.(lvl) .+ 2 .* x2
    t = fit_tree([Float64.(lvl) x2], y; categorical = [1])
    for nd in t.nodes
        nd.feature == 1 && @test nd.model == PCON
    end
end

@testset "mask pool holds more than 64 levels" begin
    rng = StableRNG(15)
    n = 2000; L = 130
    lvl = rand(rng, 1:L, n); high = Set(1:2:L)
    y = [l in high ? 1.0 : 0.0 for l in lvl] .+ 0.01 .* randn(rng, n)
    t = fit_tree(Float64.(reshape(lvl, n, 1)), y; categorical = [1], max_depth = 1)
    root = t.nodes[1]
    @test root.catwords == 3
    @test all(LinearTrees.category_is_left(t, root, l) == (l in high) for l in 1:L) ||
          all(LinearTrees.category_is_left(t, root, l) == !(l in high) for l in 1:L)
end

@testset "threaded categorical split search equals serial" begin
    rng = StableRNG(16)
    n = 20_000
    lvl = rand(rng, 1:12, n)
    levelmean = [Float64(l % 3) * 4.0 for l in 1:12]
    x2 = rand(rng, n); x3 = rand(rng, n)
    y = levelmean[lvl] .+ x2 .+ 0.3 .* x3 .+ 0.1 .* randn(rng, n)
    X = [Float64.(lvl) x2 x3]
    t1 = fit_tree(X, y; categorical = [1], nthreads = 1, max_depth = 4)
    tn = fit_tree(X, y; categorical = [1], max_depth = 4)
    @test t1.nodes == tn.nodes
    @test t1.catmasks == tn.catmasks
end

@testset "categorical guards" begin
    rng = StableRNG(17)
    n = 4000; L = 20
    lvl = rand(rng, 1:L, n); means = [l <= 10 ? Float64(l) : 40.0 + l for l in 1:L]
    y = means[lvl] .+ 0.1 .* randn(rng, n)
    X = Float64.(reshape(lvl, n, 1))
    # a rule that allows linear kinds still gives pcon on a categorical column
    t = fit_tree(X, y; categorical = [1], rule = MinDeviance((PCON, PLIN)), max_depth = 4)
    @test all(nd.model == PCON for nd in t.nodes if !LinearTrees.isleaf(nd))
    @test maximum(abs, predict(t, X) .- means[lvl]) < 5
    # non-finite code routes right like an unseen level
    tb = fit_tree(X, y; categorical = [1], max_depth = 1)
    right = findfirst(c -> !LinearTrees.category_is_left(tb, tb.nodes[1], c), 1:L)
    @test predict(tb, [NaN;;])[1] == predict(tb, [Float64(right);;])[1]
    # non-integer codes would index the wrong level silently, so they are rejected
    @test_throws ArgumentError fit_tree(X .+ 0.5, y; categorical = [1])
end
