using StableRNGs, StaticArrays, Statistics

@testset "Softmax(2) equals Logistic" begin
    rng = StableRNG(16)
    X = randn(rng, 300, 2)
    p = 1 ./ (1 .+ exp.(-(2 .* X[:, 1] .- X[:, 2])))
    yb = Float64.(rand(rng, 300) .< p)
    yk = Int.(2 .- yb)                     # class 1 = positive, class 2 = reference
    tl = fit_tree(X, yb, Logistic())
    ts = fit_tree(X, yk, Softmax(2))
    @test predict(ts, X)[:, 1] ≈ predict(tl, X) atol = 1e-8
    @test [n.model for n in ts.nodes] == [n.model for n in tl.nodes]
end

@testset "binary softmax preserves logistic damping" begin
    X = reshape([0.0, 0.0, 1.0, 1.0], :, 1)
    y = [0.0, 1.0, 0.0, 1.0]
    w = [89.0, 1.0, 2.0, 8.0]
    logistic = fit_tree(X, y, Logistic(); weights = w, max_depth = 1)
    softmax = fit_tree(X, Int.(2 .- y), Softmax(2); weights = w, max_depth = 1)
    @test predict(softmax, X)[:, 1] ≈ predict(logistic, X)
end

@testset "softmax probabilities sum to one and recover classes" begin
    rng = StableRNG(17)
    n = 600; X = randn(rng, n, 2)
    y = [X[i, 1] > 0.5 ? 1 : X[i, 2] > 0 ? 2 : 3 for i in 1:n]
    t = fit_tree(X, y, Softmax(3))
    P = predict(t, X)
    @test size(P) == (n, 3)
    @test all(isapprox.(sum(P; dims = 2), 1; atol = 1e-12))
    @test mean(argmax.(eachrow(P)) .== y) > 0.9
    S = score(t, X)
    @test size(S) == (n, 2)
end

@testset "softmax with a categorical feature" begin
    rng = StableRNG(18)
    n = 400; lvl = rand(rng, 1:5, n)
    y = [l in (1, 2) ? 1 : l == 3 ? 2 : 3 for l in lvl]
    t = fit_tree(Float64.(reshape(lvl, n, 1)), y, Softmax(3); categorical = [1])
    @test mean(argmax.(eachrow(predict(t, Float64.(reshape(lvl, n, 1))))) .== y) > 0.95
end

@testset "softmax domain" begin
    # missing class: silently fitting with an absent class would misalign the class axis
    @test_throws ArgumentError validate_target(Softmax(3), [1, 2])     # class 3 absent
    # out-of-range class: an unchecked label 4 would index past the K=3 score/prob columns
    @test_throws ArgumentError validate_target(Softmax(3), [1, 2, 4])
end

@testset "vector closed forms equal scalar per coordinate" begin
    rng = StableRNG(19)
    V = SVector{2,Float64}
    sl = zero(LinearTrees.MomentSums{V}); sr = zero(LinearTrees.MomentSums{V})
    a1 = zero(LinearTrees.MomentSums{Float64}); a2 = zero(LinearTrees.MomentSums{Float64})
    b1 = zero(LinearTrees.MomentSums{Float64}); b2 = zero(LinearTrees.MomentSums{Float64})
    x = sort(randn(rng, 30)); t = x[15]
    for i in 1:30
        z = V(randn(rng), randn(rng)); h = V(rand(rng) + 0.5, rand(rng) + 0.5)
        if i <= 15
            sl = LinearTrees.addrow(sl, x[i], z, h); a1 = LinearTrees.addrow(a1, x[i], z[1], h[1]); a2 = LinearTrees.addrow(a2, x[i], z[2], h[2])
        else
            sr = LinearTrees.addrow(sr, x[i], z, h); b1 = LinearTrees.addrow(b1, x[i], z[1], h[1]); b2 = LinearTrees.addrow(b2, x[i], z[2], h[2])
        end
    end
    s = sl + sr
    cv = LinearTrees.fit_con(s); c1 = LinearTrees.fit_con(a1 + b1); c2 = LinearTrees.fit_con(a2 + b2)
    @test cv[1] ≈ V(c1[1], c2[1]) && cv[2] ≈ V(c1[2], c2[2])
    lv = LinearTrees.fit_lin(s); l1 = LinearTrees.fit_lin(a1 + b1); l2 = LinearTrees.fit_lin(a2 + b2)
    @test all(lv[k] ≈ V(l1[k], l2[k]) for k in 1:3)
    bv = LinearTrees.fit_blin(sl, sr, t); bb1 = LinearTrees.fit_blin(a1, b1, t); bb2 = LinearTrees.fit_blin(a2, b2, t)
    @test all(bv[k] ≈ V(bb1[k], bb2[k]) for k in 1:5)
    # probabilities stay finite at extreme logits
    p = LinearTrees.probs(Softmax(3), V(50.0, -50.0))
    @test sum(p) ≈ 1 && all(isfinite, p)
    # BIC penalty scales with the number of coordinates
    @test LinearTrees.selection_score(BIC(), PLIN, 10.0, 100.0, 1e-12, 2) ≈ 100 * log(10 / 100) + 14 * log(100)
end

@testset "Softmax fits without the score clamp" begin
    rng = StableRNG(11)
    X = rand(rng, 300, 3)
    y = [X[i, 1] > 0.5 ? 1 : X[i, 2] > 0.5 ? 2 : 3 for i in 1:300]
    t = fit_tree(X, y, Softmax(3); truncate = false)
    P = predict(t, X)
    @test all(isapprox.(sum(P; dims = 2), 1.0; atol = 1e-12))
    # an unclipped fit's bounds are infinite in the score type, so the clamp is a no-op
    @test score(t, X; clip = true) == score(t, X; clip = false)
end
