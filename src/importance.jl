# Node-model gain feature importance.

"Add each nonconstant node's positive gain to `imp[feature]`, including LIN nodes."
function gain_sums!(imp::Vector{Float64}, tree::LinearTree)
    for n in tree.nodes
        isleaf(n) && continue
        imp[n.feature] += max(n.gain, 0)
    end
    return imp
end

normalise_importance(imp) = (tot = sum(imp); tot > 0 ? imp ./ tot : imp)

"""
    feature_importance(tree)

Surrogate-deviance drop per feature from nonconstant node models, including
unsplit `LIN` nodes, normalised to sum to one. Zeros when no node has positive gain.
"""
function feature_importance(tree::LinearTree)
    imp = gain_sums!(zeros(Float64, tree.nfeatures), tree)
    return normalise_importance(imp)
end

"""
    coeftable(tree, x)

Intercept and per-feature slopes of the unclipped score at `x`. A feature
clamped at `x` contributes zero slope and its piece value goes to the
intercept. Categorical pieces are constants.
"""
function coeftable(tree::LinearTree{T,V}, x::AbstractVector) where {T,V}
    check_feature_width(tree, x)
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
            goleft = category_is_left(tree, n, xraw)
            intercept += goleft ? n.lintercept : n.rintercept
        else
            xc = tree.truncate ? min(max(xraw, n.xmin), n.xmax) : xraw
            # a LIN node stores threshold = NaN, so this is always false and the
            # right branch runs -- see the same invariant at src/predict.jl:score_row
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
