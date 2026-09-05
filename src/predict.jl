# Tree traversal and score prediction.

@inline function clampscore(s::T, lo::T, hi::T) where {T<:Real}
    return min(max(s, lo), hi)
end
clampscore(s::SVector, lo::SVector, hi::SVector) = min.(max.(s, lo), hi)

"""
Unclipped or clipped path sum for row `i`. `clip=false` keeps the feature
clamp and drops the score clamp, which is what SHAP explains.
"""
@inline function score_row(tree::LinearTree{T,V}, X::AbstractMatrix, i::Integer, clip::Bool) where {T,V}
    nodes = tree.nodes
    doclip = clip & tree.truncate
    s = zero(V)
    k = 1
    while true
        n = nodes[k]
        if isleaf(n)
            s += n.lintercept
            return doclip ? clampscore(s, tree.lo, tree.hi) : s
        end
        xraw = T(X[i, n.feature])
        if iscategorical(n)
            # non-finite or non-integer codes are unseen levels and route right
            goleft = isfinite(xraw) && category_is_left(tree, n, round(Int, xraw))
            x = zero(T)      # categorical pieces are constants
        else
            x = tree.truncate ? min(max(xraw, n.xmin), n.xmax) : xraw
            # a LIN node stores threshold = NaN, so this is always false and the
            # right branch runs -- correct only because grow_subtree gives a LIN
            # node matching left/right coefficients, intercepts, and node index
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
    score(tree, X; clip=true, nthreads=Threads.nthreads())

Raw path sum per row on the link scale. `clip=false` returns the additive
unclipped sum. Rows split into `nthreads` contiguous blocks when there are at
least `PARALLEL_MIN_ROWS` of them; each row writes only its own output slot,
so the result matches the serial loop exactly.
"""
function score(tree::LinearTree{T,V}, X::AbstractMatrix; clip::Bool = true, nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1)
    out = Vector{V}(undef, n)
    row_blocks(n, nthreads) do rs
        for i in rs
            out[i] = score_row(tree, X, i, clip)
        end
    end
    return out
end

"""
    score(tree, X; clip=true, nthreads=Threads.nthreads())

`Softmax` override: `n × (K-1)` matrix of raw reference-class logits.
"""
function score(tree::LinearTree{T,V,<:Softmax}, X::AbstractMatrix; clip::Bool = true, nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1)
    Km1 = nclasses(tree.loss) - 1
    out = Matrix{T}(undef, n, Km1)
    row_blocks(n, nthreads) do rs
        for i in rs
            s = score_row(tree, X, i, clip)
            for k in 1:Km1
                out[i, k] = s[k]
            end
        end
    end
    return out
end

score_type(::LinearTree{T,V}) where {T,V} = V

"""
    predict(tree, X; nthreads=Threads.nthreads())

Prediction on the response scale, `linkinv(tree.loss, score)`.
"""
predict(tree::LinearTree, X::AbstractMatrix; nthreads = Threads.nthreads()) =
    predict!(Vector{eltype(score_type(tree))}(undef, size(X, 1)), tree, X; nthreads)

function predict!(out::AbstractVector, tree::LinearTree, X::AbstractMatrix; nthreads = Threads.nthreads())
    length(out) == size(X, 1) || throw(DimensionMismatch("out has length $(length(out)), X has $(size(X, 1)) rows"))
    row_blocks(length(out), nthreads) do rs
        for i in rs
            out[i] = linkinv(tree.loss, score_row(tree, X, i, true))
        end
    end
    return out
end

"""
    predict(tree, X; nthreads=Threads.nthreads())

`Softmax` override: `n × K` matrix of class probabilities, `linkinv` applied
row by row to the `K-1`-vector score.
"""
function predict(tree::LinearTree{T,V,<:Softmax}, X::AbstractMatrix; nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1)
    K = nclasses(tree.loss)
    return predict!(Matrix{T}(undef, n, K), tree, X; nthreads)
end

"In-place `Softmax` prediction into an `n × K` matrix."
function predict!(out::AbstractMatrix, tree::LinearTree{T,V,<:Softmax}, X::AbstractMatrix; nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1); K = nclasses(tree.loss)
    size(out) == (n, K) || throw(DimensionMismatch("out must be $n × $K, got $(size(out))"))
    row_blocks(n, nthreads) do rs
        for i in rs
            p = linkinv(tree.loss, score_row(tree, X, i, true))
            for k in 1:K
                out[i, k] = p[k]
            end
        end
    end
    return out
end
predict!(::AbstractVector, ::LinearTree{T,V,<:Softmax}, ::AbstractMatrix; kw...) where {T,V} =
    throw(ArgumentError("Softmax prediction needs an n × K matrix output"))

"""
Run `f` over `nthreads` contiguous row blocks, threaded once there are
`minrows` rows. The default, `PARALLEL_MIN_ROWS`, is calibrated on one
`score_row` walk per row, which is what `score` and `predict!` do; a caller
whose per-row work is orders of magnitude heavier passes its own, lower, gate.
"""
function row_blocks(f, n::Integer, nthreads::Integer; minrows::Integer = PARALLEL_MIN_ROWS)
    nthreads = clamp(nthreads, 1, Threads.nthreads())
    if nthreads == 1 || n < minrows
        f(1:n)
    else
        chunk = cld(n, nthreads)
        Threads.@threads for t in 1:nthreads
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, n)
            lo <= hi && f(lo:hi)
        end
    end
    return nothing
end
