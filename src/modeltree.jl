# Greedy numeric partitions scored by full-response linear ridge child fits.

using LinearAlgebra: Symmetric, cholesky!, dot, issuccess, ldiv!, norm

"The weighted penalized objective minimized by `fit_ridge_leaf` on these rows."
function _modeltree_objective(X::Matrix{T}, y::Vector{T}, w::Vector{T}, rows::Vector{Int},
        leaf::RidgeLeaf{T}, lambda::T,
        deviations::Vector{T}=Vector{T}(undef, length(rows))) where {T<:AbstractFloat}
    rss = zero(T)
    for i in rows
        fitted = leaf.intercept
        for (c, j) in enumerate(leaf.features)
            fitted += leaf.slopes[c] * X[i, j]
        end
        rss += abs2(sqrt(w[i]) * (y[i] - fitted))
    end
    W = sum(w[i] for i in rows)
    penalty = zero(T)
    resize!(deviations, length(rows))
    for (c, j) in enumerate(leaf.features)
        μ = sum((w[i] / W) * X[i, j] for i in rows)
        for (r, i) in enumerate(rows)
            deviations[r] = sqrt(w[i] / W) * refit_centered_product(X[i, j], μ, leaf.slopes[c])
        end
        penalty += abs2(norm(deviations))
    end
    return rss + lambda * penalty
end

"Weighted raw moments centered at the parent; only the upper triangle of sxx is stored."
mutable struct _ModelTreeMoments{T<:AbstractFloat}
    W::T
    sx::Vector{T}
    sy::T
    sxx::Matrix{T}
    sxy::Vector{T}
    syy::T
    first_row::Int
    varies::Vector{Bool}
end

_modeltree_moments(::Type{T}, q::Int) where {T<:AbstractFloat} =
    _ModelTreeMoments{T}(zero(T), zeros(T, q), zero(T), zeros(T, q, q),
        zeros(T, q), zero(T), 0, fill(false, q))

"Accumulate one centered weighted observation in a child's moments."
function _modeltree_update!(m::_ModelTreeMoments{A}, z::Matrix{A}, r::Int,
        target::A, weight::T, X::Matrix{T}, row::Int,
        regressors::Vector{Int}) where {T<:AbstractFloat,A<:AbstractFloat}
    mass = A(weight)
    m.first_row == 0 && (m.first_row = row)
    m.W += mass
    m.sy += mass * target
    m.syy += mass * abs2(target)
    for b in axes(z, 1)
        j = regressors[b]
        if !m.varies[b] && X[row, j] != X[m.first_row, j]
            m.varies[b] = true
        end
        zb = z[b, r]
        m.sx[b] += mass * zb
        m.sxy[b] += mass * zb * target
        for a in 1:b
            m.sxx[a, b] += mass * z[a, r] * zb
        end
    end
    return m
end

"Scratch space reused by child solves; it never owns accumulated moments."
struct _ModelTreeSolve{T<:AbstractFloat,R}
    factor::Matrix{T}
    rhs::Vector{T}
    active::Vector{Int}
    reference::R
end

function _ModelTreeSolve(::Type{T}, q::Int, reference) where {T<:AbstractFloat}
    return _ModelTreeSolve(Matrix{T}(undef, q, q), Vector{T}(undef, q),
        Vector{Int}(undef, q), reference)
end

"Child penalized ridge objective from centered covariance and cross moments."
function _modeltree_moment_objective!(work::_ModelTreeSolve{T},
        m::_ModelTreeMoments{T}, lambda::T) where {T<:AbstractFloat}
    W = m.W
    W > 0 || return nothing
    cYY = m.syy - abs2(m.sy) / W
    isfinite(cYY) && cYY >= 0 || return nothing
    q = length(m.sx)
    q == 0 && return max(zero(T), cYY)
    # A child ridge solve standardizes each regressor by its own weighted RMS.
    # In raw centered coordinates that penalty is lambda*diag(Cxx)/W.
    nactive = 0
    for c in 1:q
        # Check original observations: parent centering can erase small gaps.
        m.varies[c] || continue
        variance = m.sxx[c, c] - abs2(m.sx[c]) / W
        isfinite(variance) && variance > 0 || return nothing
        nactive += 1
        work.active[nactive] = c
    end
    nactive == 0 && return max(zero(T), cYY)
    baseline = cYY
    for b in 1:nactive
        j = work.active[b]
        work.rhs[b] = m.sxy[j] - m.sx[j] * (m.sy / W)
        for a in 1:b
            i = work.active[a]
            covariance = m.sxx[i, j] - m.sx[i] * m.sx[j] / W
            work.factor[a, b] = covariance
        end
        penalty = lambda * work.factor[b, b] / W
        work.factor[b, b] += penalty
        if work.reference !== nothing
            baseline += abs2(sqrt(penalty) * work.reference[j])
            work.rhs[b] -= penalty * work.reference[j]
        end
    end
    isfinite(baseline) || return nothing
    # Leading views retain stride-one columns for the library factorization.
    A = Symmetric(view(work.factor, 1:nactive, 1:nactive), :U)
    F = cholesky!(A; check=false)
    issuccess(F) || return nothing
    smallest, largest = T(Inf), zero(T)
    for c in 1:nactive
        diagonal = abs(F.U[c, c])
        smallest = min(smallest, diagonal)
        largest = max(largest, diagonal)
    end
    smallest > sqrt(eps(T)) * largest || return nothing
    rhs = view(work.rhs, 1:nactive)
    # b'A^-1 b = ||U'\b||² for A = U'U, so scoring needs only one solve.
    ldiv!(adjoint(F.U), rhs)
    # Translate slopes by the parent fit, including its ridge penalty. This
    # avoids subtracting the full response scatter for near-affine targets.
    objective = baseline - dot(rhs, rhs)
    # A small nonnegative difference can be as inaccurate as a negative one.
    # Rescore with QR before it can win against an accurately scored split.
    tolerance = T(128) * eps(T) * abs(baseline)
    isfinite(objective) && objective > tolerance || return nothing
    return objective
end

"Approximately equal-count boundaries, keeping tied split values together."
function _modeltree_bin_edges(X, rows, order, j, nbins)
    n = length(rows)
    width = cld(n, nbins)
    edges = Int[]
    k = width
    while k < n
        while k < n && X[rows[order[k]], j] == X[rows[order[k + 1]], j]
            k += 1
        end
        k < n && push!(edges, k)
        k += width
    end
    return edges
end

"Scan independently accumulated prefix/suffix moments; QR covers singular cases."
function _modeltree_scan(X::Matrix{T}, y::Vector{T}, w::Vector{T}, rows::Vector{Int},
        regressors::Vector{Int}, z::Matrix{A}, centered_y::Vector{A},
        order::Vector{Int}, j::Int,
        eligible::BitVector, lambda::T, min_leaf::T,
        work::_ModelTreeSolve{A}) where {T<:AbstractFloat,A<:AbstractFloat}
    n = length(order)
    fallback_rows = Int[]
    deviations = T[]
    right = _modeltree_moments(A, length(regressors))
    right_objectives = fill(A(Inf), n - 1)
    # Subtracting a large prefix from total moments can round a small suffix
    # weight to zero. Build right-child moments from the right instead.
    for k in (n - 1):-1:1
        r = order[k + 1]
        _modeltree_update!(right, z, r, centered_y[r], w[rows[r]], X, rows[r], regressors)
        eligible[k] || continue
        X[rows[order[k]], j] < X[rows[r], j] || continue
        right.W >= min_leaf || continue
        objective = _modeltree_moment_objective!(work, right, A(lambda))
        if objective === nothing
            map!(r -> rows[r], resize!(fallback_rows, n - k), view(order, k+1:n))
            leaf = fit_ridge_leaf(X, y, w, fallback_rows, regressors, lambda)
            objective = _modeltree_objective(X, y, w, fallback_rows, leaf, lambda, deviations)
        end
        right_objectives[k] = objective
    end
    left = _modeltree_moments(A, length(regressors))
    best_k = 0
    best_objective = A(Inf)
    for k in 1:(n - 1)
        r = order[k]
        _modeltree_update!(left, z, r, centered_y[r], w[rows[r]], X, rows[r], regressors)
        isfinite(right_objectives[k]) && left.W >= min_leaf || continue
        left_objective = _modeltree_moment_objective!(work, left, A(lambda))
        if left_objective === nothing
            map!(r -> rows[r], resize!(fallback_rows, k), view(order, 1:k))
            leaf = fit_ridge_leaf(X, y, w, fallback_rows, regressors, lambda)
            left_objective = _modeltree_objective(X, y, w, fallback_rows, leaf, lambda, deviations)
        end
        objective = A(left_objective) + right_objectives[k]
        if isfinite(objective) && objective < best_objective
            best_k = k
            best_objective = objective
        end
    end
    return best_k, best_objective
end

"Best greedy split by penalized child fit, with final augmented-QR verification."
function _modeltree_split(X::Matrix{T}, y::Vector{T}, w::Vector{T}, rows::Vector{Int},
        parent_leaf::RidgeLeaf{T}, parent_objective::T,
        features::Vector{Int}, regressors::Vector{Int},
        lambda::T, min_leaf::T, split_penalty::T,
        search::Union{ExactSearch,BinnedSearch}) where {T<:AbstractFloat}
    # Penalized child objectives are nonnegative, so no split can improve this.
    parent_objective <= split_penalty && return nothing
    # Narrow moments cannot resolve near-affine residuals and would repeatedly
    # require full child QR fits. Widen only scan arithmetic, retaining T leaves.
    A = promote_type(T, Float64)
    W = sum(A(w[i]) for i in rows)
    means = A[sum((A(w[i]) / W) * A(X[i, j]) for i in rows) for j in regressors]
    ymean = sum((A(w[i]) / W) * A(y[i]) for i in rows)
    # Each observation's regressors are contiguous during moment accumulation.
    z = Matrix{A}(undef, length(regressors), length(rows))
    centered_y = Vector{A}(undef, length(rows))
    scatter = zero(A)
    for (r, i) in enumerate(rows)
        for (c, j) in enumerate(regressors)
            z[c, r] = A(X[i, j]) - means[c]
        end
        centered_y[r] = A(y[i]) - ymean
        scatter += A(w[i]) * abs2(centered_y[r])
    end
    # Shift only near-affine nodes: ordinary scores need no penalty correction.
    reference = nothing
    if !isempty(regressors) && A(parent_objective) < sqrt(eps(A)) * scatter
        reference = A.(parent_leaf.slopes)
        for r in eachindex(rows), c in eachindex(reference)
            centered_y[r] = muladd(-reference[c], z[c, r], centered_y[r])
        end
    end
    # Use one coordinate system for the whole node, including extreme units.
    if reference !== nothing && (!all(isfinite, reference) || !all(isfinite, centered_y))
        reference = nothing
        for (r, i) in enumerate(rows)
            centered_y[r] = A(y[i]) - ymean
        end
    end
    best = nothing
    best_objective = A(parent_objective) - A(split_penalty)
    work = _ModelTreeSolve(A, length(regressors), reference)
    for j in features
        order = sortperm(rows; by=i -> X[i, j], alg=MergeSort)
        ncuts = length(order) - 1
        ncuts < 1 && continue
        eligible = trues(ncuts)
        edges = Int[]
        if search isa BinnedSearch && length(rows) > search.nbins
            edges = _modeltree_bin_edges(X, rows, order, j, search.nbins)
            fill!(eligible, false)
            eligible[edges] .= true
        end
        cut, objective = _modeltree_scan(X, y, w, rows, regressors, z,
            centered_y, order, j, eligible, lambda, min_leaf, work)
        if search isa BinnedSearch && search.refine && !isempty(edges) && cut > 0
            where = searchsortedfirst(edges, cut)
            lo = where == 1 ? 1 : edges[where - 1] + 1
            hi = where == length(edges) ? ncuts : edges[where + 1]
            eligible[lo:hi] .= true
            cut, objective = _modeltree_scan(X, y, w, rows, regressors, z,
                centered_y, order, j, eligible, lambda, min_leaf, work)
        end
        if cut > 0 && objective < best_objective
            best_objective = objective
            best = (feature=j, threshold=X[rows[order[cut]], j], order, cut)
        end
    end
    best === nothing && return nothing
    left = Int[rows[best.order[t]] for t in 1:best.cut]
    right = Int[rows[best.order[t]] for t in (best.cut + 1):length(rows)]
    leftleaf = fit_ridge_leaf(X, y, w, left, regressors, lambda)
    rightleaf = fit_ridge_leaf(X, y, w, right, regressors, lambda)
    left_objective = _modeltree_objective(X, y, w, left, leftleaf, lambda)
    right_objective = _modeltree_objective(X, y, w, right, rightleaf, lambda)
    gain = parent_objective - left_objective - right_objective
    isfinite(gain) && gain > split_penalty || return nothing
    return (feature=best.feature, threshold=best.threshold, left, right,
        leftleaf, rightleaf, left_objective, right_objective, gain)
end

"Typed fit state for recursive node growth."
struct _ModelTreeContext{T<:AbstractFloat,S<:Union{ExactSearch,BinnedSearch}}
    X::Matrix{T}
    y::Vector{T}
    w::Vector{T}
    features::Vector{Int}
    regressors::Vector{Int}
    lambda::T
    max_depth::Int
    min_leaf::T
    split_penalty::T
    search::S
    nodes::Vector{Node{T,T}}
    leaf_index::Vector{Int}
    leaves::Vector{RidgeLeaf{T}}
end

"Grow one node from a previously fitted ridge leaf and its actual objective."
function _modeltree_grow!(ctx::_ModelTreeContext{T}, rows::Vector{Int}, depth::Int,
        leaf::RidgeLeaf{T}, objective::T)::Int where {T<:AbstractFloat}
    k = length(ctx.nodes) + 1
    push!(ctx.nodes, Node{T,T}())
    append!(ctx.leaf_index, (0, 0))
    W = sum(ctx.w[i] for i in rows)
    if depth < ctx.max_depth && W >= 2 * ctx.min_leaf
        split = _modeltree_split(ctx.X, ctx.y, ctx.w, rows, leaf, objective,
            ctx.features, ctx.regressors, ctx.lambda, ctx.min_leaf,
            ctx.split_penalty, ctx.search)
        if split !== nothing
            left = _modeltree_grow!(ctx, split.left, depth + 1,
                split.leftleaf, split.left_objective)
            right = _modeltree_grow!(ctx, split.right, depth + 1,
                split.rightleaf, split.right_objective)
            j = split.feature
            ctx.nodes[k] = Node{T,T}(; feature=j, threshold=split.threshold, left, right,
                xmin=minimum(ctx.X[i, j] for i in rows),
                xmax=maximum(ctx.X[i, j] for i in rows), cover=W,
                xmean=sum(ctx.w[i] * ctx.X[i, j] for i in rows) / W,
                gain=split.gain - ctx.split_penalty, model=PCON)
            return k
        end
    end
    push!(ctx.leaves, leaf)
    ctx.leaf_index[refit_slot(k, true)] = length(ctx.leaves)
    return k
end

"""
    fit_model_tree(X, y; features=1:size(X, 2), max_features=8, lambda=1.0,
                   max_depth=4, min_leaf=8, split_penalty=1.0,
                   split_search=ExactSearch(), weights=nothing,
                   truncate=true) -> RefitTree

Fit a scalar-MSE tree with full-response ridge linear leaves. At each node,
`ExactSearch()` evaluates every eligible distinct-value threshold by the
combined weighted penalized ridge objective of its children. `BinnedSearch()`
evaluates approximately equal-count bin edges and optionally refines near its
best edge. `features` are candidate split columns; their first `max_features`
also form the bounded leaf-regressor basis. The intercept is unpenalized.
Each accepted split improves the augmented-QR-verified objective by more than
`split_penalty`. This is greedy parametric partitioning, not a calibrated
statistical significance test.
"""
function fit_model_tree(X::AbstractMatrix, y::AbstractVector;
        features=1:size(X, 2), max_features=8, lambda=1.0, max_depth=4,
        min_leaf=8, split_penalty=1.0, split_search::SplitSearch=ExactSearch(),
        weights=nothing, truncate::Bool=true)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X has $n rows, y has $(length(y))"))
    features isa AbstractVector && all(j -> j isa Integer, features) ||
        throw(ArgumentError("features must be integer column indices"))
    chosen = Int[features...]
    isempty(chosen) && throw(ArgumentError("features must not be empty"))
    length(unique(chosen)) == length(chosen) || throw(ArgumentError("features must not repeat"))
    all(j -> 1 <= j <= p, chosen) || throw(ArgumentError("features must lie in 1:$p"))
    max_features isa Integer && max_features >= 0 ||
        throw(ArgumentError("max_features must be a non-negative integer"))
    max_depth isa Integer && max_depth >= 0 ||
        throw(ArgumentError("max_depth must be a non-negative integer"))
    split_search isa Union{ExactSearch,BinnedSearch} ||
        throw(ArgumentError("fit_model_tree supports ExactSearch and BinnedSearch only"))
    T = float(promote_type(eltype(X), eltype(y)))
    λ = T(lambda)
    isfinite(λ) && λ > 0 || throw(ArgumentError("lambda must be finite and positive"))
    minimum_weight = T(min_leaf)
    isfinite(minimum_weight) && minimum_weight > 0 ||
        throw(ArgumentError("min_leaf must be finite and positive"))
    penalty = T(split_penalty)
    isfinite(penalty) && penalty >= 0 ||
        throw(ArgumentError("split_penalty must be finite and non-negative"))
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
    rows = collect(eachindex(yv))
    regressors = chosen[1:min(length(chosen), Int(max_features))]
    ctx = _ModelTreeContext(Xm, yv, wv, chosen, regressors, λ, Int(max_depth),
        minimum_weight, penalty, split_search, Node{T,T}[], Int[], RidgeLeaf{T}[])
    root_leaf = fit_ridge_leaf(Xm, yv, wv, rows, regressors, λ)
    root_objective = _modeltree_objective(Xm, yv, wv, rows, root_leaf, λ)
    _modeltree_grow!(ctx, rows, 0, root_leaf, root_objective)
    lo, hi = truncate ? map(T, scorebound(MSE(), yv)) : (T(-Inf), T(Inf))
    base = wmean(yv, wv)
    routing = LinearTree{T,T,MSE}(ctx.nodes, UInt64[], MSE(), lo, hi, base, p, truncate)
    return RefitTree{T}(routing, ctx.leaf_index, ctx.leaves)
end
