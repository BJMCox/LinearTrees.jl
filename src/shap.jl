# TreeSHAP value computation: path-dependent Lundberg recursion generalised
# to per-branch linear increments, plus the game's own expected value.

"""
    ShapResult{A,B,C}

`values` is `n × p` for scalar-score trees, `n × p × (K-1)` for `Softmax`.
`base` is the game's empty-coalition value, `expected_score(tree)`; `fit_tree` stores the same number in `tree.base`.
`clipped[i]` is true when the score bound changed row `i`'s prediction.
"""
struct ShapResult{A,B,C}
    values::A
    base::B
    clipped::C
end

"""
One path element. The four fields stay in one struct rather than in parallel
arrays on [`PathPool`](@ref): the split was measured on case 6 and lost, by 25%
with all four fields split apart and by 20% with `weight` alone split out. The
paths are at most one element per tree level and stay in L1, so there is no
locality to win, while every extra array costs another bounds check per read
(this package forbids `@inbounds`) and another `resize!`/`copyto!` pair in
`copyinto!`, which runs three times per split node per row.
"""
struct PathElem
    feature::Int
    zerofrac::Float64
    onefrac::Float64
    weight::Float64
end

"""
    PathPool()

Per-row scratch for the SHAP recursion, indexed by recursion depth. A split
node at depth `d` builds a hot and a cold path that both stay live across the
two child recursions, so each depth owns its own pair; the own-path is dead
once `attribute_constant!` returns, so one scratch per depth is enough. The
root's path sits outside the depth arrays because `visit!` at depth 1 already
claims `hot[1]`/`cold[1]`. Grows lazily: `visit!` calls `ensure_depth!` on
entry, and the first row of a `row_blocks` block already recurses into both
children at every split, so it drives the pool to the tree's full depth
before any later row needs it.

One pool per `row_blocks` block: it is mutated for every row, so it must never
be shared across tasks.
"""
struct PathPool
    hot::Vector{Vector{PathElem}}
    own::Vector{Vector{PathElem}}
    cold::Vector{Vector{PathElem}}
    root::Vector{PathElem}
end

PathPool() = PathPool(Vector{PathElem}[], Vector{PathElem}[], Vector{PathElem}[], PathElem[])

function ensure_depth!(pool::PathPool, depth::Integer)
    while length(pool.hot) < depth
        push!(pool.hot, PathElem[])
        push!(pool.own, PathElem[])
        push!(pool.cold, PathElem[])
    end
    return pool
end

"Refill `dst` with `src`, keeping `dst`'s allocation. Replaces `copy(src)`."
function copyinto!(dst::Vector{PathElem}, src::Vector{PathElem})
    resize!(dst, length(src))
    copyto!(dst, src)
    return dst
end

function extend!(path::Vector{PathElem}, zerofrac, onefrac, feature)
    push!(path, PathElem(feature, zerofrac, onefrac, isempty(path) ? 1.0 : 0.0))
    L = length(path)
    for i in (L - 1):-1:1
        path[i + 1] = PathElem(path[i + 1].feature, path[i + 1].zerofrac, path[i + 1].onefrac,
            path[i + 1].weight + onefrac * path[i].weight * i / L)
        path[i] = PathElem(path[i].feature, path[i].zerofrac, path[i].onefrac,
            zerofrac * path[i].weight * (L - i) / L)
    end
    return path
end

function unwind!(path::Vector{PathElem}, i)
    L = length(path)
    n = path[L].weight
    onefrac = path[i].onefrac; zerofrac = path[i].zerofrac
    # weight recompute runs the whole path, not just down to i: removing
    # position i changes every surviving weight, so the range must be full.
    # `onefrac` does not change inside the loop, so the guard is hoisted out
    # of it -- the two arms are the same arithmetic in the same order.
    if onefrac != 0
        for j in (L - 1):-1:1
            t = path[j].weight
            path[j] = PathElem(path[j].feature, path[j].zerofrac, path[j].onefrac, n * L / (j * onefrac))
            n = t - path[j].weight * zerofrac * (L - j) / L
        end
    else
        for j in (L - 1):-1:1
            path[j] = PathElem(path[j].feature, path[j].zerofrac, path[j].onefrac, path[j].weight * L / (zerofrac * (L - j)))
        end
    end
    for j in i:(L - 1)
        path[j] = PathElem(path[j + 1].feature, path[j + 1].zerofrac, path[j + 1].onefrac, path[j].weight)
    end
    pop!(path)
    return path
end

function unwound_sum(path::Vector{PathElem}, i)
    L = length(path)
    n = path[L].weight
    onefrac = path[i].onefrac; zerofrac = path[i].zerofrac
    total = 0.0
    # same full-path range and hoisted guard as unwind!: see note there
    if onefrac != 0
        for j in (L - 1):-1:1
            t = n * L / (j * onefrac)
            total += t
            n = path[j].weight - t * zerofrac * (L - j) / L
        end
    else
        for j in (L - 1):-1:1
            total += path[j].weight * L / (zerofrac * (L - j))
        end
    end
    return total
end

"Attribute a constant `v` reached along `path` to the path's features, scalar `V`."
function attribute_constant!(φ::AbstractMatrix, path::Vector{PathElem}, v::Real, row)
    for i in 2:length(path)
        e = path[i]
        # a degenerate element (onefrac == zerofrac, e.g. an own-term inheriting
        # a zeroed-out onefrac from an ancestor) carries exactly zero credit;
        # skip it rather than risk 0/0 in unwound_sum turning into NaN * 0
        e.onefrac == e.zerofrac && continue
        w = unwound_sum(path, i)
        φ[row, e.feature] += w * (e.onefrac - e.zerofrac) * v
    end
    return nothing
end

"Vector-`V` variant: spreads the class axis into the trailing dimension of `φ`."
function attribute_constant!(φ::AbstractArray{T,3}, path::Vector{PathElem}, v::SVector, row) where {T}
    for i in 2:length(path)
        e = path[i]
        e.onefrac == e.zerofrac && continue
        w = unwound_sum(path, i)
        c = w * (e.onefrac - e.zerofrac)
        @views φ[row, e.feature, :] .+= c .* v
    end
    return nothing
end

"Extend the pool's root path by one element, then dispatch. Called once per row, for the root only -- a split node's own hot/cold recursion already holds an extended path and calls `visit!` on it directly."
function shap_recurse!(φ, tree::LinearTree{T,V}, x, row, k, pool::PathPool,
        zerofrac, onefrac, feature) where {T,V}
    path = extend!(empty!(pool.root), zerofrac, onefrac, feature)
    visit!(φ, tree, x, row, k, path, pool, 1, zero(V))
    return nothing
end

"""
Dispatch on node `k` against an already-extended `path`. Split off from
`shap_recurse!` because a LIN node's single child is reached without adding a
path dimension for it -- see the LIN branch below.

`path` points into `pool`, at a depth strictly below `depth`, and every path
this node builds comes from `pool` at `depth`. Children run at `depth + 1`, so
they cannot touch the hot and cold paths this node keeps live across both
recursions.

`acc` carries the branch constants collected on the way down from the root;
they are attributed only at the leaves. A constant credited against a node's
path equals the same constant credited against both of its child paths,
because the total path weight `unwound_sum` computes is linear in the appended
element's `(zerofrac, onefrac)` and the two children's fractions sum
componentwise to the parent's (`hotcov * izero + coldcov * izero == izero`,
`ione + 0 == ione`). That trades three `attribute_constant!` calls per split
node for one per leaf, worth 1.3x on case 6 -- see `bench/PROFILE.md`'s
"3. SHAP". It reassociates, so values move in the last few bits. The
own-feature linear term is the exception: it is attributed at the node,
against a path no child has.
"""
function visit!(φ, tree::LinearTree{T,V}, x, row, k, path::Vector{PathElem},
        pool::PathPool, depth::Int, acc::V) where {T,V}
    n = tree.nodes[k]
    if isleaf(n)
        attribute_constant!(φ, path, acc + n.lintercept, row)
        return nothing
    end
    ensure_depth!(pool, depth)   # lazy growth: extends the pool the first time recursion reaches this depth
    j = n.feature
    xraw = T(x[j])
    if !iscategorical(n) && n.left == n.right
        # LIN node: no real split, single child, cover unchanged. Present vs.
        # absent give the same constant (lcoef * xmean + lintercept) either
        # way, so it rides down in acc against the path as-is -- the child is
        # reached without extending path for j, since a (zerofrac=1,
        # onefrac=1) element is not a no-op mid-path and would rescale every
        # other element's weight for no reason.
        # Only the own-feature term lcoef * (x - xmean) is conditional on
        # presence. If an ancestor already split on j, that ancestor fixed
        # j's coalition state (its onefrac); the own term reuses that state
        # (unwind the stale element, extend with zerofrac 0 and the
        # ancestor's onefrac) on its own copy of path, leaving path itself and
        # the child recursion untouched.
        xc = tree.truncate ? min(max(xraw, n.xmin), n.xmax) : xraw
        val = n.lcoef * n.xmean + n.lintercept
        own = n.lcoef * (xc - n.xmean)
        prev = findfirst(e -> e.feature == j, path)
        ownpath = copyinto!(pool.own[depth], path)
        if prev === nothing
            extend!(ownpath, 0.0, 1.0, j)
        else
            extend!(unwind!(ownpath, prev), 0.0, path[prev].onefrac, j)
        end
        attribute_constant!(φ, ownpath, own, row)
        visit!(φ, tree, x, row, n.left, path, pool, depth + 1, acc + val)
        return nothing
    end
    covl = tree.nodes[n.left].cover / n.cover; covr = 1 - covl
    if iscategorical(n)
        goleft = isfinite(xraw) && category_is_left(tree, n, round(Int, xraw))
        lval = n.lintercept; rval = n.rintercept
        ownl = zero(V); ownr = zero(V)
    else
        xc = tree.truncate ? min(max(xraw, n.xmin), n.xmax) : xraw
        goleft = xc <= n.threshold
        lval = n.lcoef * n.xmean + n.lintercept; rval = n.rcoef * n.xmean + n.rintercept
        ownl = n.lcoef * (xc - n.xmean); ownr = n.rcoef * (xc - n.xmean)
    end
    prev = findfirst(e -> e.feature == j, path)
    izero = 1.0; ione = 1.0
    if prev !== nothing
        izero = path[prev].zerofrac; ione = path[prev].onefrac
        unwind!(path, prev)
    end
    hot, cold = goleft ? (n.left, n.right) : (n.right, n.left)
    hotcov, coldcov = goleft ? (covl, covr) : (covr, covl)
    hotval, coldval = goleft ? (lval, rval) : (rval, lval)
    hotown = goleft ? ownl : ownr
    # own-feature linear term: only present when j is in the coalition, so
    # zerofrac = 0 -- the term vanishes when j is absent, not weighted by
    # cover. That also makes ownpath no child's path, so it cannot ride down
    # in acc; pushing it down would cost each leaf one path per ancestor split
    # feature instead.
    ownpath = extend!(copyinto!(pool.own[depth], path), 0.0, ione, j)
    attribute_constant!(φ, ownpath, hotown, row)
    # hotpath and coldpath are already the paths the hot/cold recursion needs,
    # so they are passed straight into visit! rather than rebuilt there --
    # shap_recurse! would otherwise refill a buffer and extend! a second time
    # for the same result. The two branch constants ride down in acc.
    hotpath = extend!(copyinto!(pool.hot[depth], path), hotcov * izero, ione, j)
    coldpath = extend!(copyinto!(pool.cold[depth], path), coldcov * izero, 0.0, j)
    visit!(φ, tree, x, row, hot, hotpath, pool, depth + 1, acc + hotval)
    visit!(φ, tree, x, row, cold, coldpath, pool, depth + 1, acc + coldval)
    return nothing
end

"Recursion for `expected_score`, indexed by node `k`."
function _expected_score(tree::LinearTree{T,V}, k::Integer) where {T,V}
    n = tree.nodes[k]
    isleaf(n) && return n.lintercept
    if !iscategorical(n) && n.left == n.right
        # LIN node: single child, covr == 0 -- skip the right term entirely
        # rather than recurse into n.right (== n.left) and multiply by 0
        lval = n.lcoef * n.xmean + n.lintercept
        return lval + _expected_score(tree, n.left)
    end
    if iscategorical(n)
        lval = n.lintercept; rval = n.rintercept
    else
        lval = n.lcoef * n.xmean + n.lintercept
        rval = n.rcoef * n.xmean + n.rintercept
    end
    covl = tree.nodes[n.left].cover / n.cover; covr = 1 - covl
    return covl * (lval + _expected_score(tree, n.left)) + covr * (rval + _expected_score(tree, n.right))
end

"""
    expected_score(tree) -> V

Cover-weighted `game_value(tree, ·, ∅)` over the whole tree: at each split
the absent feature's piece is evaluated at the node's `xmean` on both
children, weighted by `cover(child) / cover(parent)`. Independent of `x`.
`fit_tree` stores it in `tree.base`. `shap`/`shap!` recompute it from the
nodes (`O(nodes)`) so a hand-built tree whose `base` field is stale still
satisfies the efficiency identity.
"""
expected_score(tree::LinearTree) = _expected_score(tree, 1)

"""
Row gate for `shap`'s `row_blocks`. `PARALLEL_MIN_ROWS` is calibrated on one
`score_row` walk per row; one SHAP row instead visits every node of the tree,
so the same amount of work is reached after `PARALLEL_MIN_ROWS / nodes` rows.
On the shared gate a depth-12 tree got no parallelism at all below 2^14 rows,
which is above every size SHAP is normally called at.
"""
shap_min_rows(tree::LinearTree) = max(2, cld(PARALLEL_MIN_ROWS, length(tree.nodes)))

"""
    shap!(values, clipped, tree, X; nthreads=Threads.nthreads())

In-place [`shap`](@ref): write into `values` (`n × p`, or `n × p × (K-1)` for
`Softmax`) and `clipped` (`Vector{Bool}`, length `n`), then return a
[`ShapResult`](@ref) wrapping them.
"""
function shap!(values, clipped::Vector{Bool}, tree::LinearTree{T,V}, X::AbstractMatrix;
        nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1)
    fill!(values, 0)
    row_blocks(n, nthreads; minrows = shap_min_rows(tree)) do rs
        pool = PathPool()   # one per block: never shared between tasks; grows lazily, see PathPool
        for i in rs
            shap_recurse!(values, tree, view(X, i, :), i, 1, pool, 1.0, 1.0, 0)
            clipped[i] = score_row(tree, X, i, true) != score_row(tree, X, i, false)
        end
    end
    return ShapResult(values, expected_score(tree), clipped)   # derived from the nodes, not trusted from tree.base
end

"""
    shap(tree, X; nthreads=Threads.nthreads()) -> ShapResult

Path-dependent TreeSHAP values for every row of `X` against every feature of
`tree`, on the unclipped score scale. Each row's SHAP values sum to
`score_row(tree, X, i, false) - result.base`, the game's efficiency identity.
"""
function shap(tree::LinearTree{T,V}, X::AbstractMatrix; nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1); p = tree.nfeatures
    values = V <: SVector ? zeros(T, n, p, length(V)) : zeros(T, n, p)
    return shap!(values, Vector{Bool}(undef, n), tree, X; nthreads)
end
