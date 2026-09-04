# TreeSHAP value computation: path-dependent Lundberg recursion generalised
# to per-branch linear increments, plus the game's own expected value.

"""
    ShapResult{A,B,C}

`values` is `n × p` for scalar-score trees, `n × p × (K-1)` for `Softmax`.
`base` is the game's empty-coalition value (see `expected_score`), not
`tree.base`. `clipped[i]` is true when the score bound changed row `i`'s
prediction.
"""
struct ShapResult{A,B,C}
    values::A
    base::B
    clipped::C
end

struct PathElem
    feature::Int
    zerofrac::Float64
    onefrac::Float64
    weight::Float64
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
    # position i changes every surviving weight, so the range must be full
    for j in (L - 1):-1:1
        if onefrac != 0
            t = path[j].weight
            path[j] = PathElem(path[j].feature, path[j].zerofrac, path[j].onefrac, n * L / (j * onefrac))
            n = t - path[j].weight * zerofrac * (L - j) / L
        else
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
    # same full-path range as unwind!: see note there
    for j in (L - 1):-1:1
        if onefrac != 0
            t = n * L / (j * onefrac)
            total += t
            n = path[j].weight - t * zerofrac * (L - j) / L
        else
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

function shap_recurse!(φ, tree::LinearTree{T,V}, x, row, k, path::Vector{PathElem},
        zerofrac, onefrac, feature) where {T,V}
    path = extend!(copy(path), zerofrac, onefrac, feature)
    visit!(φ, tree, x, row, k, path)
    return nothing
end

"""
Dispatch on node `k` against an already-extended `path`. Split off from
`shap_recurse!` because a LIN node's single child is reached without adding a
path dimension for it -- see the LIN branch below.
"""
function visit!(φ, tree::LinearTree{T,V}, x, row, k, path::Vector{PathElem}) where {T,V}
    n = tree.nodes[k]
    if isleaf(n)
        attribute_constant!(φ, path, n.lintercept, row)
        return nothing
    end
    j = n.feature
    xraw = T(x[j])
    if !iscategorical(n) && n.left == n.right
        # LIN node: no real split, single child, cover unchanged. Present vs.
        # absent give the same constant (lcoef * xmean + lintercept) either
        # way, so it is attributed against the path as-is -- extending it for
        # j here would rescale every other element's weight for no reason,
        # since a (zerofrac=1, onefrac=1) element is not a no-op mid-path.
        # Only the own-feature term lcoef * (x - xmean) is conditional on
        # presence, and only it adds a real dimension to the path.
        xc = tree.truncate ? min(max(xraw, n.xmin), n.xmax) : xraw
        val = n.lcoef * n.xmean + n.lintercept
        own = n.lcoef * (xc - n.xmean)
        prev = findfirst(e -> e.feature == j, path)
        prev !== nothing && unwind!(path, prev)
        attribute_constant!(φ, path, val, row)
        ownpath = extend!(copy(path), 0.0, 1.0, j)
        attribute_constant!(φ, ownpath, own, row)
        visit!(φ, tree, x, row, n.left, path)
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
    # constant part of the taken branch's piece, one level down
    hotpath = extend!(copy(path), hotcov * izero, ione, j)
    attribute_constant!(φ, hotpath, hotval, row)
    # own-feature linear term: only present when j is in the coalition, so
    # zerofrac = 0 -- the term vanishes when j is absent, not weighted by cover
    ownpath = extend!(copy(path), 0.0, ione, j)
    attribute_constant!(φ, ownpath, hotown, row)
    coldpath = extend!(copy(path), coldcov * izero, 0.0, j)
    attribute_constant!(φ, coldpath, coldval, row)
    shap_recurse!(φ, tree, x, row, hot, path, hotcov * izero, ione, j)
    shap_recurse!(φ, tree, x, row, cold, path, coldcov * izero, 0.0, j)
    return nothing
end

"""
    expected_score(tree) -> V

Cover-weighted `game_value(tree, ·, ∅)` over the whole tree: at each split
the absent feature's piece is evaluated at the node's `xmean` on both
children, weighted by `cover(child) / cover(parent)`. Independent of `x`.
Equal to `tree.base` only when every non-leaf node has `lcoef == rcoef`.
"""
function expected_score(tree::LinearTree{T,V}, k::Integer = 1) where {T,V}
    n = tree.nodes[k]
    isleaf(n) && return n.lintercept
    if iscategorical(n)
        lval = n.lintercept; rval = n.rintercept
    else
        lval = n.lcoef * n.xmean + n.lintercept
        rval = n.rcoef * n.xmean + n.rintercept
    end
    covl = tree.nodes[n.left].cover / n.cover; covr = 1 - covl
    return covl * (lval + expected_score(tree, n.left)) + covr * (rval + expected_score(tree, n.right))
end

function shap!(values, clipped::Vector{Bool}, tree::LinearTree{T,V}, X::AbstractMatrix;
        nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1)
    fill!(values, 0)
    row_blocks(n, nthreads) do rs
        for i in rs
            shap_recurse!(values, tree, view(X, i, :), i, 1, PathElem[], 1.0, 1.0, 0)
            clipped[i] = score_row(tree, X, i, true) != score_row(tree, X, i, false)
        end
    end
    return ShapResult(values, expected_score(tree), clipped)
end

function shap(tree::LinearTree{T,V}, X::AbstractMatrix; nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1); p = tree.nfeatures
    values = V <: SVector ? zeros(T, n, p, length(V)) : zeros(T, n, p)
    return shap!(values, Vector{Bool}(undef, n), tree, X; nthreads)
end
