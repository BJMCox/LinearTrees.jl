using StableRNGs, StaticArrays, Combinatorics

"Brute-force oracle game. Mirrors `shap_recurse!`'s routing exactly, LIN nodes included."
function game_value(tree, x, S, k = 1)
    n = tree.nodes[k]
    if LinearTrees.isleaf(n)
        return n.lintercept
    end
    j = n.feature
    lw = tree.nodes[n.left].cover / n.cover; rw = 1 - lw
    piece(goleft, xv) = goleft ? n.lcoef * xv + n.lintercept : n.rcoef * xv + n.rintercept
    if LinearTrees.iscategorical(n)
        if j in S
            xv = x[j]
            gl = isfinite(xv) && LinearTrees.category_is_left(tree, n, round(Int, xv))
            return (gl ? n.lintercept : n.rintercept) + game_value(tree, x, S, gl ? n.left : n.right)
        else
            return lw * (n.lintercept + game_value(tree, x, S, n.left)) +
                   rw * (n.rintercept + game_value(tree, x, S, n.right))
        end
    end
    if j in S
        # score_row/coeftable only clamp the feature when tree.truncate is set;
        # the brief's oracle clamps unconditionally, which disagrees with the
        # real game whenever a test uses truncate = false
        xv = tree.truncate ? clamp(x[j], n.xmin, n.xmax) : x[j]
        # LIN node: threshold is NaN, both children equal, always take the left/lcoef piece
        gl = n.left == n.right ? true : xv <= n.threshold
        return piece(gl, xv) + game_value(tree, x, S, gl ? n.left : n.right)
    else
        return lw * (piece(true, n.xmean) + game_value(tree, x, S, n.left)) +
               rw * (piece(false, n.xmean) + game_value(tree, x, S, n.right))
    end
end

"`combinations(v)` includes the empty subset, so this already covers S = ∅."
function brute_shap(tree, x)
    p = tree.nfeatures
    φ = zeros(p)
    for i in 1:p, S in combinations(setdiff(1:p, i))
        w = factorial(length(S)) * factorial(p - length(S) - 1) / factorial(p)
        φ[i] += w * (game_value(tree, x, Set([S; i])) - game_value(tree, x, Set(S)))
    end
    return φ, game_value(tree, x, Set{Int}())
end

@testset "shap matches brute force on a small tree" begin
    rng = StableRNG(21)
    X = rand(rng, 400, 4); y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ 0.05 .* randn(rng, 400)
    t = fit_tree(X, y; max_depth = 3, truncate = false)
    res = shap(t, X[1:10, :])
    for i in 1:10
        φ, base = brute_shap(t, X[i, :])
        @test res.values[i, :] ≈ φ atol = 1e-10
        @test res.base ≈ base atol = 1e-10
    end
end

@testset "shap matches brute force with a categorical feature" begin
    rng = StableRNG(24)
    n = 300; lvl = Float64.(rand(rng, 1:5, n)); x2 = rand(rng, n)
    y = [l in (1.0, 3.0) ? 2.0 : -1.0 for l in lvl] .+ 3 .* x2
    t = fit_tree([lvl x2], y; categorical = [1], max_depth = 2, truncate = false)
    res = shap(t, [lvl x2][1:10, :])
    for i in 1:10
        x = [lvl x2][i, :]
        φ, base = brute_shap(t, x)
        @test res.values[i, :] ≈ φ atol = 1e-10
        @test res.base ≈ base atol = 1e-10
    end
end

@testset "shap additivity and clipped flag" begin
    rng = StableRNG(22)
    X = rand(rng, 300, 3); y = 5 .* X[:, 1] .+ 0.1 .* randn(rng, 300)
    t = fit_tree(X, y)
    Xq = vcat(X[1:5, :], [50.0 0.5 0.5])           # last row forces a score clip if the bound allows
    res = shap(t, Xq)
    unclipped = score(t, Xq; clip = false)
    @test vec(sum(res.values; dims = 2)) .+ res.base ≈ unclipped atol = 1e-10
    @test res.clipped == (score(t, Xq) .!= unclipped)
end

@testset "shap for softmax has a class axis" begin
    rng = StableRNG(23)
    X = randn(rng, 300, 2); y = [X[i, 1] > 0 ? 1 : X[i, 2] > 0 ? 2 : 3 for i in 1:300]
    t = fit_tree(X, y, Softmax(3); max_depth = 3)
    res = shap(t, X[1:5, :])
    @test size(res.values) == (5, 2, 2)
    S = score(t, X[1:5, :]; clip = false)
    for k in 1:2
        @test vec(sum(res.values[:, :, k]; dims = 2)) .+ res.base[k] ≈ S[:, k] atol = 1e-10
    end
end

@testset "shap base matches training mean when branches share a slope" begin
    rng = StableRNG(25)
    X = rand(rng, 300, 3); y = 2 .* X[:, 1] .+ X[:, 2] .+ 0.01 .* randn(rng, 300)
    t = fit_tree(X, y; rule = MinDeviance((PCON,)), truncate = false)
    res = shap(t, X)
    @test res.base ≈ t.base atol = 1e-10
end

@testset "tree.base is the SHAP empty-coalition value on a general (asymmetric) tree (I6)" begin
    # Before the fix, tree.base was the cover-weighted mean training score, not
    # the empty-coalition value shap uses; the two differed by 28% on a 62-node
    # tree with mixed lcoef/rcoef (measured interactively: 1.5016 vs 1.9255).
    # The PCON-only tree above can't show this, since every lcoef == rcoef == 0
    # there; this one has LIN/PLIN/BLIN nodes where the two genuinely differ.
    rng = StableRNG(50)
    X = rand(rng, 400, 4)
    y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ X[:, 4] .^ 2 .+ 0.05 .* randn(rng, 400)
    t = fit_tree(X, y; max_depth = 5)
    @test t.base == LinearTrees.expected_score(t)
    res = shap(t, X[1:5, :])
    @test res.base == t.base
end

"Walk `tree.nodes` from the root; true if a LIN node's feature also splits an ancestor."
function has_lin_under_same_feature(tree)
    found = false
    function walk(k, ancestors)
        n = tree.nodes[k]
        LinearTrees.isleaf(n) && return
        j = n.feature
        islin = !LinearTrees.iscategorical(n) && n.left == n.right
        islin && j in ancestors && (found = true)
        newanc = islin ? ancestors : (ancestors ∪ (j,))
        walk(n.left, newanc)
        n.right != n.left && walk(n.right, newanc)
    end
    walk(1, Set{Int}())
    return found
end

@testset "shap matches brute force with a LIN node under an ancestor split on the same feature" begin
    rng = StableRNG(2)
    X = rand(rng, 300, 4)
    y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ X[:, 4] .^ 2 .+ 0.05 .* randn(rng, 300)
    t = fit_tree(X, y; max_depth = 4, truncate = false)
    @test has_lin_under_same_feature(t)   # node 19 (LIN, feature 4) sits under node 14 (SPLIT, feature 4)

    res = shap(t, X[1:10, :])
    for i in 1:10
        φ, base = brute_shap(t, X[i, :])
        @test res.values[i, :] ≈ φ atol = 1e-10
        @test res.base ≈ base atol = 1e-10
    end
    @test vec(sum(res.values; dims = 2)) .+ res.base ≈ score(t, X[1:10, :]; clip = false) atol = 1e-10
end

@testset "shap threaded matches serial on a large set" begin
    rng = StableRNG(26)
    n = 20_000; X = rand(rng, n, 3); y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .+ 0.05 .* randn(rng, n)
    t = fit_tree(X, y; max_depth = 3)
    r1 = shap(t, X; nthreads = 1)
    rn = shap(t, X)
    @test r1.values == rn.values
    @test r1.clipped == rn.clipped
end

@testset "shap on a deep tree is efficient and thread-invariant" begin
    # max_depth = 10 drives the path pool past whatever depth the shallow
    # testsets reach, so a pool that failed to grow (or that let a child
    # overwrite a live parent buffer) breaks efficiency here, not above.
    rng = StableRNG(101)
    n = 2000
    lvl = Float64.(rand(rng, 1:6, n))
    X = hcat(lvl, rand(rng, n, 4))
    y = sin.(3 .* X[:, 2]) .+ 2 .* X[:, 3] .* (X[:, 4] .> 0.5) .+
        [l in (1.0, 4.0) ? 1.5 : -0.5 for l in lvl] .+ 0.05 .* randn(rng, n)
    t = fit_tree(X, y; categorical = [1], max_depth = 10)
    @test length(t.nodes) > 40

    Xq = X[1:200, :]
    r1 = shap(t, Xq; nthreads = 1)
    rn = shap(t, Xq)
    @test r1.values == rn.values                 # bit-for-bit, not approximately
    @test r1.clipped == rn.clipped
    @test vec(sum(r1.values; dims = 2)) .+ r1.base ≈ score(t, Xq; clip = false) atol = 1e-10
end

@testset "shap base comes from the nodes, not the base field" begin
    # a hand-built tree with a stale `base` (the constructor default pattern) must still satisfy efficiency
    N(; kw...) = LinearTrees.Node{Float64,Float64}(; kw...)
    nodes = [N(feature = 1, threshold = 0.5, left = 2, right = 3, lcoef = 0.3, lintercept = 0.2, rcoef = 0.8,
               rintercept = -0.3, cover = 10.0, xmean = 0.5, xmin = 0.0, xmax = 1.0, model = PLIN),
             N(lintercept = 0.1, cover = 4.0), N(lintercept = -0.2, cover = 6.0)]
    t = LinearTree{Float64,Float64,MSE}(nodes, UInt64[], MSE(), -100.0, 100.0, 0.0, 1, false)
    X = reshape(collect(0.05:0.1:0.95), :, 1)
    res = shap(t, X)
    @test res.base != t.base
    @test vec(sum(res.values; dims = 2)) .+ res.base ≈ score(t, X; clip = false) atol = 1e-12
end
