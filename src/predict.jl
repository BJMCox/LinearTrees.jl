# Tree traversal and score prediction.

@inline function clampscore(s::T, lo::T, hi::T) where {T<:Real}
    return min(max(s, lo), hi)
end

"""
Unclipped or clipped path sum for row `i`. `clip=false` keeps the feature
clamp and drops the score clamp, which is what SHAP explains.
"""
@inline function score_row(tree::LinearTree{T,V}, X::AbstractMatrix, i::Integer, clip::Bool) where {T,V}
    nodes = tree.nodes
    doclip = clip & tree.truncate
    s = zero(V)
    k = 1
    @inbounds while true
        n = nodes[k]
        if isleaf(n)
            s += n.lintercept
            return doclip ? clampscore(s, tree.lo, tree.hi) : s
        end
        xraw = T(X[i, n.feature])
        if iscategorical(n)
            goleft = category_is_left(tree, n, round(Int, xraw))
            x = xraw
        else
            x = tree.truncate ? min(max(xraw, n.xmin), n.xmax) : xraw
            goleft = x <= n.threshold
        end
        if goleft
            s += n.lcoef * x + n.lintercept
            k = n.left
        else
            s += n.rcoef * x + n.rintercept
            k = n.right
        end
        doclip && (s = clampscore(s, tree.lo, tree.hi))
    end
end

"""
    score(tree, X; clip=true)

Raw path sum per row on the link scale. `clip=false` returns the additive
unclipped sum.
"""
function score(tree::LinearTree{T,V}, X::AbstractMatrix; clip::Bool = true) where {T,V}
    n = size(X, 1)
    out = Vector{V}(undef, n)
    @inbounds for i in 1:n
        out[i] = score_row(tree, X, i, clip)
    end
    return out
end

score_type(::LinearTree{T,V}) where {T,V} = V

"""
    predict(tree, X)

Prediction on the response scale, `linkinv(tree.loss, score)`.
"""
predict(tree::LinearTree, X::AbstractMatrix) = predict!(Vector{eltype(score_type(tree))}(undef, size(X, 1)), tree, X)

function predict!(out::AbstractVector, tree::LinearTree, X::AbstractMatrix)
    @inbounds for i in eachindex(out)
        out[i] = linkinv(tree.loss, score_row(tree, X, i, true))
    end
    return out
end
