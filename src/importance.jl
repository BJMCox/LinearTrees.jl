# Split-gain feature importance.

"""
    feature_importance(tree)

Surrogate-deviance drop per feature, normalised to sum to one. Zeros when
no node has positive gain.
"""
function feature_importance(tree::LinearTree)
    imp = zeros(Float64, tree.nfeatures)
    for n in tree.nodes
        isleaf(n) && continue
        imp[n.feature] += max(n.gain, 0)
    end
    tot = sum(imp)
    return tot > 0 ? imp ./ tot : imp
end

"""
    coeftable(tree, x)

Intercept and per-feature slopes of the unclipped score at `x`. A feature
clamped at `x` contributes zero slope and its piece value goes to the
intercept. Categorical pieces are constants.
"""
function coeftable(tree::LinearTree{T,V}, x::AbstractVector) where {T,V}
    slopes = zeros(V, tree.nfeatures)
    intercept = zero(V)
    k = 1
    while true
        n = tree.nodes[k]
        if isleaf(n)
            intercept += n.lintercept
            return intercept, slopes
        end
        xraw = T(x[n.feature])
        if iscategorical(n)
            goleft = isfinite(xraw) && category_is_left(tree, n, round(Int, xraw))   # non-finite codes route right, as in score_row
            intercept += goleft ? n.lintercept : n.rintercept
        else
            xc = tree.truncate ? min(max(xraw, n.xmin), n.xmax) : xraw
            goleft = xc <= n.threshold
            a = goleft ? n.lcoef : n.rcoef
            b = goleft ? n.lintercept : n.rintercept
            if xc == xraw
                slopes[n.feature] += a; intercept += b
            else
                intercept += a * xc + b            # clamped: constant piece
            end
        end
        k = goleft ? n.left : n.right
    end
end
