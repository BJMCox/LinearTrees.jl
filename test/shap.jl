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
    res = shap(t, X[1:5, :])
    for i in 1:5
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
    res = shap(t, [lvl x2][1:5, :])
    for i in 1:5
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

    res = shap(t, X[1:5, :])
    for i in 1:5
        φ, base = brute_shap(t, X[i, :])
        @test res.values[i, :] ≈ φ atol = 1e-10
        @test res.base ≈ base atol = 1e-10
    end
    @test vec(sum(res.values; dims = 2)) .+ res.base ≈ score(t, X[1:5, :]; clip = false) atol = 1e-10
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

@testset "shap threads below PARALLEL_MIN_ROWS and stays bit-identical" begin
    # Fails while `shap!` uses the default `row_blocks` gate: a 2_000-row call
    # then runs in a single block whatever `nthreads` says, so a depth-12
    # tree gets no parallelism at exactly the sizes SHAP is used at.
    rng = StableRNG(37)
    n = 2_000
    X = rand(rng, n, 5)
    y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ 0.05 .* randn(rng, n)
    t = fit_tree(X, y; max_depth = 10)
    nt = min(4, Threads.nthreads())
    r1 = shap(t, X; nthreads = 1)
    rn = shap(t, X; nthreads = nt)
    @test r1.values == rn.values                 # bit-for-bit
    @test r1.clipped == rn.clipped
    nblocks = Threads.Atomic{Int}(0)
    LinearTrees.row_blocks(_ -> Threads.atomic_add!(nblocks, 1), n, nt; minrows = LinearTrees.shap_min_rows(t))
    @test nblocks[] == nt
end

"""
Three tree shapes the SHAP recursion is exercised on beyond the shallow trees
above: a deep tree with a categorical column, a vector-valued `Softmax` tree
and a chain of LIN nodes. Each returns 4000 query rows, which is above
`shap_min_rows` for all three, so the threaded call really splits into blocks.
"""
function shap_reference_shapes()
    out = Tuple{String,Any,Matrix{Float64}}[]

    rng = StableRNG(101)
    n = 4000
    lvl = Float64.(rand(rng, 1:6, n))
    X = hcat(lvl, rand(rng, n, 4))
    y = sin.(3 .* X[:, 2]) .+ 2 .* X[:, 3] .* (X[:, 4] .> 0.5) .+
        [l in (1.0, 4.0) ? 1.5 : -0.5 for l in lvl] .+ 0.05 .* randn(rng, n)
    push!(out, ("depth10-cat", fit_tree(X, y; categorical = [1], max_depth = 10), X))

    rng = StableRNG(23)
    X = randn(rng, n, 3)
    y = [X[i, 1] > 0 ? 1 : X[i, 2] > 0 ? 2 : 3 for i in 1:n]
    push!(out, ("softmax3", fit_tree(X, y, Softmax(3); max_depth = 4), X))

    rng = StableRNG(77)
    X = rand(rng, n, 3)
    y = 2 .* X[:, 1] .+ 3 .* X[:, 2] .- X[:, 3] .+ 0.05 .* randn(rng, n)
    t = fit_tree(X, y; rule = MinDeviance((LIN,)), max_lin_chain = 8, max_depth = 8, truncate = false)
    push!(out, ("linchain", t, X))
    return out
end

"""
Rebuild, from the path `P` at a split node on feature `j`, the hot and cold
paths `visit!` hands to the two child recursions. `prev` is the position of
`j`'s stale element in `P`, or `nothing` when no ancestor split on `j`.
"""
function shap_child_paths(P, j, prev, hotcov)
    Q = copy(P)
    izero = 1.0; ione = 1.0
    if prev !== nothing
        izero = P[prev].zerofrac; ione = P[prev].onefrac
        LinearTrees.unwind!(Q, prev)
    end
    hot = LinearTrees.extend!(copy(Q), hotcov * izero, ione, j)
    cold = LinearTrees.extend!(copy(Q), (1 - hotcov) * izero, 0.0, j)
    return hot, cold
end

@testset "a constant attributed at a split node telescopes to its two children" begin
    # `visit!` does not attribute a split node's two branch constants at the
    # node any more: it adds them to `acc` and attributes the running sum once
    # per leaf. That is sound only because of this identity. Reinstating the
    # `izero` factor wrongly -- `hotcov` in place of `hotcov * izero` in
    # `visit!`'s hot path, say -- or changing `extend!`'s weight recurrence
    # breaks it, and then a constant credited at a leaf no longer equals the
    # same constant credited at the ancestor it came from.
    c = 1.7
    p = 3
    root = LinearTrees.extend!(LinearTrees.PathElem[], 1.0, 1.0, 0)

    # Two levels, child feature fresh: the root splits on feature 1 (cover 0.6
    # to the taken side), and its hot child splits on feature 2.
    P = LinearTrees.extend!(copy(root), 0.6, 1.0, 1)
    hot, cold = shap_child_paths(P, 2, nothing, 0.25)
    φnode = zeros(1, p); φkids = zeros(1, p); φhot = zeros(1, p)
    LinearTrees.attribute_constant!(φnode, P, c, 1)
    LinearTrees.attribute_constant!(φkids, hot, c, 1)
    LinearTrees.attribute_constant!(φkids, cold, c, 1)
    LinearTrees.attribute_constant!(φhot, hot, c, 1)
    @test φkids ≈ φnode atol = 1e-12
    @test φhot[1, 2] != 0        # the split feature's two child credits really cancel
    @test φkids[1, 2] ≈ φnode[1, 2] atol = 1e-12

    # Same shape with an ancestor split on the child's own feature, so `visit!`
    # unwinds a stale element and both child covers carry its `izero`. Here the
    # split feature's parent credit is nonzero, so the two children have to
    # reproduce it rather than cancel.
    Ps = LinearTrees.extend!(LinearTrees.extend!(copy(P), 0.3, 1.0, 2), 0.45, 1.0, 3)
    prev = findfirst(e -> e.feature == 2, Ps)
    hots, colds = shap_child_paths(Ps, 2, prev, 0.25)
    ψnode = zeros(1, p); ψkids = zeros(1, p)
    LinearTrees.attribute_constant!(ψnode, Ps, c, 1)
    LinearTrees.attribute_constant!(ψkids, hots, c, 1)
    LinearTrees.attribute_constant!(ψkids, colds, c, 1)
    @test ψnode[1, 2] != 0
    @test ψkids ≈ ψnode atol = 1e-12
    @test ψkids[1, 2] ≈ ψnode[1, 2] atol = 1e-12
end

@testset "shap on three reference shapes is thread-invariant and efficient" begin
    # These shapes reach cases the shallow testsets above do not: depth 10 with
    # a categorical split, a `Softmax` class axis on a depth-4 tree, and a
    # chain of eight LIN nodes. Both assertions fail on any production change
    # that makes the block split observable -- a `PathPool` buffer shared
    # across tasks, or a child overwriting a parent's live hot/cold path --
    # and the efficiency check fails on any change that drops or double-counts
    # a node's contribution, such as attributing a split node's own-feature
    # term against the wrong path.
    #
    # There is deliberately no bit-level golden here. The exact mantissa bits
    # of these values depend on the target architecture and on
    # `--check-bounds`, so a hard constant fails in CI for reasons unrelated
    # to the code. `bench/ab.jl`'s dumps are the tool for bit regressions:
    # they compare two commits on one machine under one flag.
    for (name, t, X) in shap_reference_shapes()
        @test size(X, 1) >= LinearTrees.shap_min_rows(t)   # so the threaded call splits
        r1 = shap(t, X; nthreads = 1)
        rn = shap(t, X; nthreads = min(4, Threads.nthreads()))
        @test r1.values == rn.values                       # bit-for-bit, not approximately
        @test r1.clipped == rn.clipped
        S = score(t, X; clip = false)
        if ndims(r1.values) == 3
            for k in axes(r1.values, 3)
                @test vec(sum(r1.values[:, :, k]; dims = 2)) .+ r1.base[k] ≈ S[:, k] atol = 1e-10
            end
        else
            @test vec(sum(r1.values; dims = 2)) .+ r1.base ≈ S atol = 1e-10
        end
    end
end
