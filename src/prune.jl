# Opt-in bottom-up validation pruning of an affine-refit tree.

"Copy a leaf's owned vectors so the returned model does not alias its input."
copy_ridge_leaf(leaf::RidgeLeaf{T}) where {T} =
    RidgeLeaf{T}(copy(leaf.features), copy(leaf.slopes), leaf.intercept,
        copy(leaf.xmin), copy(leaf.xmax))

"A row's next routing node, or zero when this node ends its route."
function prune_next(tree::LinearTree{T,T,MSE}, n::Node, x) where {T}
    if n.model == LIN && n.left == n.right
        return n.right
    elseif iscategorical(n)
        return category_is_left(tree, n, T(x[n.feature])) ? n.left : n.right
    end
    v = T(x[n.feature])
    v = tree.truncate ? min(max(v, n.xmin), n.xmax) : v
    return v <= n.threshold ? n.left : n.right
end

"Record all ancestor nodes reached by each positive-weight row."
function prune_node_rows(tree::LinearTree, X)
    rows = [Int[] for _ in tree.nodes]
    for i in axes(X, 1)
        k = 1
        while true
            push!(rows[k], i)
            n = tree.nodes[k]
            isleaf(n) && break
            k = prune_next(tree, n, view(X, i, :))
            k == 0 && break
        end
    end
    return rows
end

"Walk reachable internal nodes in postorder without a recursive closure."
function prune_order_walk!(order::Vector{Int}, paths::Vector{Vector{Int}},
        tree::LinearTree, k::Int, path::Vector{Int}, max_features::Int,
        explicit_features::Union{Nothing,Vector{Int}})
    n = tree.nodes[k]
    isleaf(n) && return nothing
    nextpath = path
    if explicit_features === nothing && !iscategorical(n) &&
            !(Int(n.feature) in path) && length(path) < max_features
        nextpath = [path; Int(n.feature)]
    end
    paths[k] = explicit_features === nothing ? nextpath : explicit_features
    if n.model == LIN && n.left == n.right
        n.left == 0 || prune_order_walk!(order, paths, tree, Int(n.left), nextpath,
            max_features, explicit_features)
    else
        n.left == n.right && n.left != 0 &&
            throw(ArgumentError("shared children are supported only for LIN nodes"))
        n.left == 0 || prune_order_walk!(order, paths, tree, Int(n.left), nextpath,
            max_features, explicit_features)
        n.right == 0 || prune_order_walk!(order, paths, tree, Int(n.right), nextpath,
            max_features, explicit_features)
    end
    push!(order, k)
    return nothing
end

"Return reachable internal nodes in postorder and candidate path features."
function prune_order(tree::LinearTree, max_features::Int,
        explicit_features::Union{Nothing,Vector{Int}})
    order = Int[]
    paths = [Int[] for _ in tree.nodes]
    prune_order_walk!(order, paths, tree, 1, Int[], max_features, explicit_features)
    return order, paths
end

"Evaluate a candidate leaf under the same feature and final score clamps as `predict`."
function prune_candidate_score(leaf::RidgeLeaf{T}, tree::LinearTree{T,T,MSE}, X, i) where {T}
    s = leaf.intercept
    for (c, j) in enumerate(leaf.features)
        x = T(X[i, j])
        tree.truncate && (x = min(max(x, leaf.xmin[c]), leaf.xmax[c]))
        s += leaf.slopes[c] * x
    end
    return tree.truncate ? clampscore(s, tree.lo, tree.hi) : s
end

"Copy one terminal leaf into its compacted slot."
function compact_refit_add_leaf!(leaf_index::Vector{Int}, leaves::Vector{RidgeLeaf{T}},
        model::RefitTree{T}, oldslot::Int, newslot::Int) where {T}
    oldindex = model.leaf_index[oldslot]
    oldindex > 0 || throw(ArgumentError("routing terminal has no fitted leaf"))
    push!(leaves, copy_ridge_leaf(model.leaves[oldindex]))
    leaf_index[newslot] = length(leaves)
    return nothing
end

"Copy one reachable routing subtree without a boxed recursive closure."
function compact_refit_add_node!(nodes::Vector{Node{T,T}}, masks::Vector{UInt64},
        leaf_index::Vector{Int}, leaves::Vector{RidgeLeaf{T}},
        model::RefitTree{T}, k::Int) where {T}
    old = model.routing
    n = old.nodes[k]
    newk = length(nodes) + 1
    push!(nodes, n)
    append!(leaf_index, (0, 0))
    if isleaf(n)
        compact_refit_add_leaf!(leaf_index, leaves, model,
            refit_slot(k, true), refit_slot(newk, true))
        return newk
    end
    catstart = Int32(0)
    if iscategorical(n)
        catstart = Int32(length(masks) + 1)
        append!(masks, view(old.catmasks, n.catstart:(n.catstart + n.catwords - 1)))
    end
    if n.model == LIN && n.left == n.right
        child = n.left == 0 ? 0 : compact_refit_add_node!(nodes, masks, leaf_index,
            leaves, model, Int(n.left))
        if child == 0
            compact_refit_add_leaf!(leaf_index, leaves, model,
                refit_slot(k, false), refit_slot(newk, false))
        end
        nodes[newk] = Node{T,T}(n; left = Int32(child), right = Int32(child), catstart)
        return newk
    end
    left = n.left == 0 ? 0 : compact_refit_add_node!(nodes, masks, leaf_index,
        leaves, model, Int(n.left))
    left == 0 && compact_refit_add_leaf!(leaf_index, leaves, model,
        refit_slot(k, true), refit_slot(newk, true))
    right = n.right == 0 ? 0 : compact_refit_add_node!(nodes, masks, leaf_index,
        leaves, model, Int(n.right))
    right == 0 && compact_refit_add_leaf!(leaf_index, leaves, model,
        refit_slot(k, false), refit_slot(newk, false))
    nodes[newk] = Node{T,T}(n; left = Int32(left), right = Int32(right), catstart)
    return newk
end

"Keep only reachable routing nodes, masks, and leaves after pruning."
function compact_refit(model::RefitTree{T}) where {T}
    old = model.routing
    nodes = Node{T,T}[]
    masks = UInt64[]
    leaf_index = Int[]
    leaves = RidgeLeaf{T}[]
    compact_refit_add_node!(nodes, masks, leaf_index, leaves, model, 1)
    routing = LinearTree{T,T,MSE}(nodes, masks, MSE(), old.lo, old.hi, old.base,
        old.nfeatures, old.truncate)
    return RefitTree{T}(routing, leaf_index, leaves)
end

"Validate and keep only positive-weight rows of a pruning dataset."
function prune_data(X::AbstractMatrix, y::AbstractVector, weights, p, ::Type{T}) where {T}
    n = size(X, 1)
    size(X, 2) == p || throw(DimensionMismatch("X has $(size(X, 2)) columns, model expects $p"))
    length(y) == n || throw(DimensionMismatch("X has $n rows, y has $(length(y))"))
    validate_target(MSE(), y)
    w = weights === nothing ? ones(T, n) : Vector{T}(weights)
    length(w) == n || throw(DimensionMismatch("weights has length $(length(w)), X has $n rows"))
    all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("weights must be finite and non-negative"))
    keep = findall(>(0), w)
    Xm = Matrix{T}(view(X, keep, :))
    all(isfinite, Xm) || throw(ArgumentError("X contains NaN or Inf on positive-weight rows"))
    return Xm, Vector{T}(view(y, keep)), w[keep]
end

"""
    prune_refit(model, Xtrain, ytrain, Xval, yval;
                train_weights=nothing, val_weights=nothing, features=:path,
                max_features=8, lambda=1.0, tolerance=0.0)

Visit a [`RefitTree`](@ref)'s reachable internal nodes bottom-up. A candidate
replaces a whole subtree with a full-response affine ridge leaf fitted only
on training rows reaching that node. Accept it only when its served, weighted
validation squared loss plus the absolute `tolerance` is strictly below the
current subtree's loss. Nodes without positive-weight validation rows are
skipped. The returned model owns a compact topology snapshot. This operation
can create discontinuities across region boundaries.
"""
function prune_refit(model::RefitTree{T}, Xtrain::AbstractMatrix, ytrain::AbstractVector,
        Xval::AbstractMatrix, yval::AbstractVector; train_weights = nothing,
        val_weights = nothing, features = :path, max_features = 8,
        lambda = 1.0, tolerance = 0.0) where {T}
    max_features isa Integer && max_features >= 0 ||
        throw(ArgumentError("max_features must be a non-negative integer"))
    λ = T(lambda)
    isfinite(λ) && λ > 0 || throw(ArgumentError("lambda must be finite and positive"))
    tol = T(tolerance)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tolerance must be finite and non-negative"))
    p = model.routing.nfeatures
    Xt, yt, wt = prune_data(Xtrain, ytrain, train_weights, p, T)
    Xv, yv, wv = prune_data(Xval, yval, val_weights, p, T)
    isempty(wt) && throw(ArgumentError("total training weight must be positive"))
    categorical = Set(Int(n.feature) for n in model.routing.nodes if iscategorical(n))
    chosen = nothing
    if features !== :path
        features isa AbstractVector && all(v -> v isa Integer, features) ||
            throw(ArgumentError("features must be :path or a vector of integer indices"))
        chosen = Int[features...]
        length(unique(chosen)) == length(chosen) || throw(ArgumentError("features must not repeat"))
        length(chosen) <= max_features || throw(ArgumentError("features exceed max_features"))
        all(j -> 1 <= j <= p && !(j in categorical), chosen) ||
            throw(ArgumentError("features must be numeric column indices in 1:$p"))
    end
    routing = model.routing
    copied = LinearTree{T,T,MSE}(copy(routing.nodes), copy(routing.catmasks), MSE(),
        routing.lo, routing.hi, routing.base, p, routing.truncate)
    working = RefitTree{T}(copied, copy(model.leaf_index),
        [copy_ridge_leaf(leaf) for leaf in model.leaves])
    isempty(wv) && return compact_refit(working)
    order, paths = prune_order(copied, Int(max_features), chosen)
    train_rows = prune_node_rows(copied, Xt)
    val_rows = prune_node_rows(copied, Xv)
    for k in order
        tr = train_rows[k]; vr = val_rows[k]
        (isempty(tr) || isempty(vr)) && continue
        candidate = fit_ridge_leaf(Xt, yt, wt, tr, paths[k], λ)
        current_loss = sum(wv[i] * (yv[i] - refit_score_row(working, Xv, i, true))^2 for i in vr)
        candidate_loss = sum(wv[i] * (yv[i] - prune_candidate_score(candidate, copied, Xv, i))^2 for i in vr)
        if candidate_loss + tol < current_loss
            n = copied.nodes[k]
            copied.nodes[k] = Node{T,T}(; lintercept = zero(T), cover = n.cover)
            push!(working.leaves, candidate)
            working.leaf_index[refit_slot(k, true)] = length(working.leaves)
        end
    end
    return compact_refit(working)
end
