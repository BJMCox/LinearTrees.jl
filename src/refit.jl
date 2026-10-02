# Fixed-partition affine ridge refits for scalar MSE trees.

using LinearAlgebra: norm, qr

"One terminal region's full-response affine model and training feature bounds."
struct RidgeLeaf{T}
    features::Vector{Int}
    slopes::Vector{T}
    intercept::T
    xmin::Vector{T}
    xmax::Vector{T}
end

"""
    RefitTree{T}

A snapshot of an MSE tree's routing geometry with one multivariate ridge
predictor per terminal region. `routing` owns copies of the original nodes
and categorical masks. Node score increments are ignored.
"""
struct RefitTree{T}
    routing::LinearTree{T,T,MSE}
    leaf_index::Vector{Int}
    leaves::Vector{RidgeLeaf{T}}
end

"The left/right slot for a node, also used for an explicit CON terminal."
refit_slot(k::Integer, left::Bool) = 2 * k - (left ? 1 : 0)

"Walk terminal slots without a boxed recursive closure."
function refit_regions_walk!(regions::Vector{Tuple{Int,Vector{Int}}}, tree::LinearTree,
        k::Int, path::Vector{Int}, max_features::Int,
        explicit_features::Union{Nothing,Vector{Int}})
    n = tree.nodes[k]
    if isleaf(n)
        push!(regions, (refit_slot(k, true), explicit_features === nothing ? path : explicit_features))
        return nothing
    end
    nextpath = path
    if !iscategorical(n) && explicit_features === nothing &&
            !(Int(n.feature) in path) && length(path) < max_features
        nextpath = [path; Int(n.feature)]
    end
    if n.model == LIN && n.left == n.right
        child = n.right
        if child == 0
            push!(regions, (refit_slot(k, false), explicit_features === nothing ? nextpath : explicit_features))
        else
            refit_regions_walk!(regions, tree, Int(child), nextpath, max_features, explicit_features)
        end
        return nothing
    end
    for (left, child) in ((true, n.left), (false, n.right))
        slot = refit_slot(k, left)
        if child == 0
            push!(regions, (slot, explicit_features === nothing ? nextpath : explicit_features))
        else
            refit_regions_walk!(regions, tree, Int(child), nextpath, max_features, explicit_features)
        end
    end
    return nothing
end

"Collect terminal slots and the bounded numeric features on each route."
function refit_regions(tree::LinearTree, max_features::Int,
        explicit_features::Union{Nothing,Vector{Int}})
    regions = Tuple{Int,Vector{Int}}[]
    refit_regions_walk!(regions, tree, 1, Int[], max_features, explicit_features)
    return regions
end

"Route a row to its terminal slot, without adding the source tree's node scores."
function refit_route(tree::LinearTree{T,T,MSE}, x) where {T}
    k = 1
    while true
        n = tree.nodes[k]
        isleaf(n) && return refit_slot(k, true)
        if n.model == LIN && n.left == n.right
            left = false
        elseif iscategorical(n)
            left = category_is_left(tree, n, T(x[n.feature]))
        else
            v = T(x[n.feature])
            v = tree.truncate ? min(max(v, n.xmin), n.xmax) : v
            left = v <= n.threshold
        end
        child = left ? n.left : n.right
        child == 0 && return refit_slot(k, left)
        k = child
    end
end

"Multiply a centered difference, scaling operands first only if subtraction overflows."
function refit_centered_product(x::T, mean::T, factor::T) where {T<:AbstractFloat}
    difference = x - mean
    return isfinite(difference) ? factor * difference : factor * x - factor * mean
end

"Divide a centered difference without forming the reciprocal of a tiny scale."
function refit_centered_ratio(x::T, mean::T, scale::T) where {T<:AbstractFloat}
    difference = x - mean
    return isfinite(difference) ? difference / scale : x / scale - mean / scale
end

"Solve one centered, scaled weighted ridge problem with an unpenalized intercept."
function fit_ridge_leaf(X::Matrix{T}, y::Vector{T}, w::Vector{T},
        rows::Vector{Int}, features::Vector{Int}, lambda::T) where {T<:AbstractFloat}
    wr = view(w, rows)
    W = sum(wr)
    isfinite(W) && W > 0 || throw(ArgumentError("terminal region must have finite positive weight"))
    q = length(features)
    ymin = T(wmean(view(y, rows), wr))
    xmin = T[minimum(X[i, j] for i in rows) for j in features]
    xmax = T[maximum(X[i, j] for i in rows) for j in features]
    q == 0 && return RidgeLeaf{T}(Int[], T[], ymin, xmin, xmax)
    means = T[wmean(view(X, rows, j), wr) for j in features]
    deviations = Vector{T}(undef, length(rows))
    scales = Vector{T}(undef, q)
    for (c, j) in enumerate(features)
        for (r, i) in enumerate(rows)
            deviations[r] = refit_centered_product(X[i, j], means[c], sqrt(w[i] / W))
        end
        scales[c] = norm(deviations)
    end
    for c in eachindex(scales)
        iszero(scales[c]) && (scales[c] = one(T))
    end
    A = zeros(T, length(rows) + q, q)
    b = zeros(T, length(rows) + q)
    for (r, i) in enumerate(rows)
        sw = sqrt(w[i])
        b[r] = refit_centered_product(y[i], ymin, sw)
        for (c, j) in enumerate(features)
            A[r, c] = sw * refit_centered_ratio(X[i, j], means[c], scales[c])
        end
    end
    for c in 1:q
        A[length(rows) + c, c] = sqrt(lambda)
    end
    standardized = qr(A) \ b
    slopes = standardized ./ scales
    intercept = ymin - sum(slopes .* means)
    return RidgeLeaf{T}(copy(features), slopes, intercept, xmin, xmax)
end

"""
    refit_leaves(tree, X, y; weights=nothing, features=:path,
                 max_features=8, lambda=1.0, truncate=tree.truncate)

Freeze an MSE tree's partitions and fit one full-response affine model per
terminal region. `:path` takes the first `max_features` distinct numeric
features on each route; an explicit vector selects the same features in every
region. Categorical split columns route rows but cannot be regressors.

The per-region objective is weighted squared error plus `lambda` times the
squared slopes on weighted-RMS-standardized predictors. The intercept is not
penalized. Every terminal region needs positive training weight. Zero-weight
rows are excluded before checking predictors or computing feature bounds.
The refit uses the source tree's final score bounds when `truncate=true` and
clamps used leaf features to their positive-weight training extrema.
"""
function refit_leaves(tree::LinearTree{T,T,MSE}, X::AbstractMatrix, y::AbstractVector;
        weights = nothing, features = :path, max_features = 8, lambda = 1.0,
        truncate::Bool = tree.truncate) where {T}
    n, p = size(X)
    p == tree.nfeatures || throw(DimensionMismatch("X has $p columns, tree expects $(tree.nfeatures)"))
    length(y) == n || throw(DimensionMismatch("X has $n rows, y has $(length(y))"))
    max_features isa Integer && max_features >= 0 || throw(ArgumentError("max_features must be a non-negative integer"))
    λ = T(lambda)
    isfinite(λ) && λ > 0 || throw(ArgumentError("lambda must be finite and positive"))
    validate_target(MSE(), y)
    w = weights === nothing ? ones(T, n) : Vector{T}(weights)
    length(w) == n || throw(DimensionMismatch("weights has length $(length(w)), X has $n rows"))
    all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("weights must be finite and non-negative"))
    keep = findall(>(0), w)
    isempty(keep) && throw(ArgumentError("total weight must be positive"))
    Xm = Matrix{T}(view(X, keep, :))
    all(isfinite, Xm) || throw(ArgumentError("X contains NaN or Inf on positive-weight rows"))
    yv = Vector{T}(view(y, keep))
    wv = w[keep]
    categorical = Set(Int(n.feature) for n in tree.nodes if iscategorical(n))
    chosen = nothing
    if features !== :path
        features isa AbstractVector || throw(ArgumentError("features must be :path or a vector of indices"))
        all(v -> v isa Integer, features) || throw(ArgumentError("features must be integer indices"))
        chosen = Int[features...]
        length(unique(chosen)) == length(chosen) || throw(ArgumentError("features must not repeat"))
        length(chosen) <= max_features || throw(ArgumentError("features exceed max_features"))
        all(j -> 1 <= j <= p && !(j in categorical), chosen) ||
            throw(ArgumentError("features must be numeric column indices in 1:$p"))
    end
    routing = LinearTree{T,T,MSE}(copy(tree.nodes), copy(tree.catmasks), MSE(),
        tree.lo, tree.hi, tree.base, tree.nfeatures, truncate)
    regions = refit_regions(routing, Int(max_features), chosen)
    slot_rows = [Int[] for _ in 1:(2 * length(routing.nodes))]
    for i in axes(Xm, 1)
        push!(slot_rows[refit_route(routing, view(Xm, i, :))], i)
    end
    leaf_index = zeros(Int, length(slot_rows))
    leaves = RidgeLeaf{T}[]
    for (slot, used_features) in regions
        rows = slot_rows[slot]
        isempty(rows) && throw(ArgumentError("terminal region $slot has no positive-weight rows"))
        push!(leaves, fit_ridge_leaf(Xm, yv, wv, rows, used_features, λ))
        leaf_index[slot] = length(leaves)
    end
    return RefitTree{T}(routing, leaf_index, leaves)
end

refit_leaves(tree::LinearTree, X::AbstractMatrix, y::AbstractVector; kwargs...) =
    throw(ArgumentError("refit_leaves supports scalar MSE trees only"))

function refit_width(model::RefitTree, X::AbstractMatrix)
    size(X, 2) == model.routing.nfeatures ||
        throw(DimensionMismatch("X has $(size(X, 2)) columns, model expects $(model.routing.nfeatures)"))
    return nothing
end

function refit_width(model::RefitTree, x::AbstractVector)
    length(x) == model.routing.nfeatures ||
        throw(DimensionMismatch("x has $(length(x)) features, model expects $(model.routing.nfeatures)"))
    return nothing
end

function refit_score_row(model::RefitTree{T}, X::AbstractMatrix, i, clip) where {T}
    tree = model.routing
    slot = refit_route(tree, view(X, i, :))
    leaf = model.leaves[model.leaf_index[slot]]
    s = leaf.intercept
    for (c, j) in enumerate(leaf.features)
        x = T(X[i, j])
        tree.truncate && (x = min(max(x, leaf.xmin[c]), leaf.xmax[c]))
        s += leaf.slopes[c] * x
    end
    return clip && tree.truncate ? clampscore(s, tree.lo, tree.hi) : s
end

"Raw full-response score, optionally clipped to the source tree's score bounds."
function score(model::RefitTree{T}, X::AbstractMatrix; clip::Bool = true,
        nthreads = Threads.nthreads()) where {T}
    refit_width(model, X)
    out = Vector{T}(undef, size(X, 1))
    row_blocks(length(out), nthreads) do rows
        for i in rows
            out[i] = refit_score_row(model, X, i, clip)
        end
    end
    return out
end

"Response prediction for an MSE refit; equal to the clipped score."
predict(model::RefitTree{T}, X::AbstractMatrix; nthreads = Threads.nthreads()) where {T} =
    predict!(Vector{T}(undef, size(X, 1)), model, X; nthreads)

function predict!(out::AbstractVector, model::RefitTree, X::AbstractMatrix;
        nthreads = Threads.nthreads())
    refit_width(model, X)
    length(out) == size(X, 1) ||
        throw(DimensionMismatch("out has length $(length(out)), X has $(size(X, 1)) rows"))
    if Base.mightalias(out, X)
        copyto!(out, score(model, X; nthreads))
        return out
    end
    row_blocks(length(out), nthreads) do rows
        for i in rows
            out[i] = refit_score_row(model, X, i, true)
        end
    end
    return out
end

"Intercept and slopes of the locally active, unclipped affine predictor."
function coeftable(model::RefitTree{T}, x::AbstractVector) where {T}
    refit_width(model, x)
    tree = model.routing
    slot = refit_route(tree, x)
    leaf = model.leaves[model.leaf_index[slot]]
    slopes = zeros(T, tree.nfeatures)
    intercept = leaf.intercept
    for (c, j) in enumerate(leaf.features)
        raw = T(x[j])
        clamped = tree.truncate ? min(max(raw, leaf.xmin[c]), leaf.xmax[c]) : raw
        if raw == clamped
            slopes[j] = leaf.slopes[c]
        else
            intercept += leaf.slopes[c] * clamped
        end
    end
    return intercept, slopes
end
