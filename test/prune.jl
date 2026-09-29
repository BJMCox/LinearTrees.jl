@testset "validation pruning replaces a noisy split with a full affine leaf" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    root = N(feature = 1, threshold = 0.0, left = 2, right = 3,
        xmin = -4.0, xmax = 4.0, model = PCON)
    tree = LinearTree{T,T,MSE}([root, N(), N()], UInt64[], MSE(), -Inf, Inf, 0.0, 2, false)
    x2 = repeat([-1.0, 0.0, 1.0, 2.0], 2)
    Xtrain = hcat([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0], x2)
    ytrain = [i <= 4 ? 2.0 + 2.0 * x2[i] : 8.0 + 2.0 * x2[i] for i in 1:8]
    Xval = copy(Xtrain)
    yval = 5.0 .+ 2.0 .* x2
    original = refit_leaves(tree, Xtrain, ytrain; features = [2], lambda = 1e-8)
    before = predict(original, Xval)
    pruned = prune_refit(original, Xtrain, ytrain, Xval, yval;
        features = [2], lambda = 1e-8)
    after = predict(pruned, Xval)
    @test sum(abs2, after .- yval) < sum(abs2, before .- yval)
    @test length(pruned.routing.nodes) == 1
    @test predict(original, Xval) == before
    guarded = prune_refit(original, Xtrain, ytrain, Xval, yval;
        features = [2], lambda = 1e-8,
        tolerance = sum(abs2, before .- yval) + 1.0)
    @test predict(guarded, Xval) == before
    Xzero = vcat(Xtrain, [NaN NaN])
    yzero = vcat(ytrain, 1e6)
    train_weights = vcat(ones(8), 0.0)
    zero_pruned = prune_refit(original, Xzero, yzero, Xval, yval;
        train_weights, features = [2], lambda = 1e-8)
    @test predict(zero_pruned, Xval) == after
    original.routing.nodes[1] = N(feature = 1, threshold = 10.0, left = 2, right = 3, model = PCON)
    @test predict(pruned, Xval) == after

    # A zero-weight validation row cannot make a worse candidate acceptable.
    original = refit_leaves(tree, Xtrain, ytrain; features = [2], lambda = 1e-8)
    Xguard = vcat(Xval, [0.5 0.5])
    yguard = vcat(predict(original, Xval), 5.0)
    wguard = vcat(ones(8), 0.0)
    kept = prune_refit(original, Xtrain, ytrain, Xguard, yguard;
        val_weights = wguard, features = [2], lambda = 1e-8)
    @test predict(kept, Xval) == predict(original, Xval)
    @test length(kept.routing.nodes) == 3
    none = prune_refit(original, Xtrain, ytrain, Xval, yval;
        val_weights = zeros(8), features = [2])
    @test predict(none, Xval) == predict(original, Xval)
end

@testset "LIN and categorical routes remain valid when validation misses a region" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    nodes = [N(feature = 1, left = 2, right = 4, catstart = 1, catwords = 1,
               model = PCON),
             N(feature = 2, left = 3, right = 3, threshold = NaN,
               xmin = -1.0, xmax = 1.0, model = LIN),
             N(), N()]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[1], MSE(), -Inf, Inf, 0.0, 2, false)
    Xtrain = [1.0 -1.0; 1.0 0.0; 1.0 1.0;
              2.0 -1.0; 2.0 0.0; 2.0 1.0]
    ytrain = [0.0, 2.0, 4.0, 10.0, 10.0, 10.0]
    original = refit_leaves(tree, Xtrain, ytrain; features = [2], lambda = 1e-8)
    Xval = [2.0 -1.0; 2.0 0.0; 2.0 1.0]
    yval = predict(original, Xval)
    pruned = prune_refit(original, Xtrain, ytrain, Xval, yval;
        features = [2], lambda = 1e-8)
    @test predict(pruned, Xtrain) == predict(original, Xtrain)
    @test length(pruned.routing.nodes) == 4
end

@testset "pruning compares served clipped predictions" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    tree = LinearTree{T,T,MSE}([N(feature = 1, threshold = 0.0, left = 2, right = 3,
        xmin = -2.0, xmax = 2.0, model = PCON), N(), N()],
        UInt64[], MSE(), 0.0, 0.0, 0.0, 1, true)
    X = reshape([-2.0, -1.0, 1.0, 2.0], :, 1)
    y = [-10.0, -10.0, 10.0, 10.0]
    original = refit_leaves(tree, X, y; features = [1], lambda = 1.0)
    pruned = prune_refit(original, X, y, X, zeros(4);
        features = [1], lambda = 1.0)
    @test length(pruned.routing.nodes) == 3
    @test predict(pruned, X) == zeros(4)
end
