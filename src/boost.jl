# Gradient boosting over Frozen-loss trees.

using Random: AbstractRNG, default_rng, randperm

@inline function score_row(b::LinearBoost{T,V}, X::AbstractMatrix, i::Integer, clip::Bool) where {T,V}
    s = b.f0
    for t in b.trees
        s += b.eta * score_row(t, X, i, false)
    end
    return (clip & b.truncate) ? clampscore(s, b.lo, b.hi) : s
end

"""
Fill `target[i] = (-g0/h0, h0)` at ensemble score `F`. Smooth losses use
their fitting curvature (Huber uses its residual majorizer); non-smooth losses
take the IRLS weight at `F` as the frozen Hessian, the same weight the single tree uses in
its split search, so the round is one IRLS step rather than an exact L1 fit.
"""
function frozen_target!(target::Vector{Tuple{V,V}}, g0::Vector{V}, h0::Vector{V}, loss::L,
        y::Vector{T}, F::Vector{V}, w::Vector{T}) where {T,V,L<:Loss}
    working_gradhess!(g0, h0, loss, y, F)
    if !issmooth(loss)
        S = eltype(V)
        ε = max(irls_epsilon(y .- F, w), sqrt(eps(S)) * max(maximum(abs, y), one(S)))
        irls_weights!(h0, loss, y, F; ε)
    end
    for i in eachindex(target)
        target[i] = (-g0[i] ./ h0[i], h0[i])
    end
    return target
end

"Weighted base-loss objective after a raw-score step along `increment`."
function boost_step_loss(loss::Loss, y, F, w, increment, scale)
    total = zero(eltype(w))
    if iszero(scale)
        for i in eachindex(y)
            total += w[i] * pointloss(loss, y[i], F[i])
        end
    else
        for i in eachindex(y)
            total += w[i] * pointloss(loss, y[i], F[i] + scale * increment[i])
        end
    end
    return total
end

"Halve a proposed base-loss step until its raw objective is finite and nonincreasing."
function boost_backtrack_scale(loss::Loss, y, F, w, increment, start::T) where {T}
    baseline = boost_step_loss(loss, y, F, w, increment, zero(T))
    scale = start
    while scale >= eps(T)
        candidate = boost_step_loss(loss, y, F, w, increment, scale)
        isfinite(candidate) && candidate <= baseline && return scale
        scale /= 2
    end
    return zero(T)
end

boost_step_scale(loss::Loss, y, F, w, increment, start::T) where {T} =
    boost_backtrack_scale(loss, y, F, w, increment, start)

"Minimize Huber's convex directional loss before checking the actual objective."
function boost_step_scale(loss::Union{Huber,AdaptedLoss{<:HuberLoss}}, y, F, w, increment, start::T) where {T}
    huber_transition(loss) === nothing &&
        return boost_backtrack_scale(loss, y, F, w, increment, start)
    derivative(scale) = sum(w[i] * increment[i] *
        first(gh(loss, y[i], F[i] + scale * increment[i])) for i in eachindex(y))
    scale = if derivative(zero(T)) >= 0
        zero(T)
    elseif derivative(one(T)) <= 0
        one(T)
    else
        lo, hi = zero(T), one(T)
        for _ in 1:60
            mid = lo / 2 + hi / 2
            if derivative(mid) < 0
                lo = mid
            else
                hi = mid
            end
        end
        lo / 2 + hi / 2
    end
    iszero(scale) && return scale
    return boost_backtrack_scale(loss, y, F, w, increment, scale)
end

"Scale a just-fitted tree in place before it becomes part of the ensemble."
function scale_boost_tree(tree::LinearTree{T,V,L}, scale::T) where {T,V,L}
    isone(scale) && return tree
    for k in eachindex(tree.nodes)
        n = tree.nodes[k]
        if iszero(scale)
            tree.nodes[k] = Node{T,V}(n; lcoef = zero(V),
                lintercept = zero(V), rcoef = zero(V), rintercept = zero(V))
        else
            tree.nodes[k] = Node{T,V}(n; lcoef = scale * n.lcoef,
                lintercept = scale * n.lintercept, rcoef = scale * n.rcoef,
                rintercept = scale * n.rintercept)
        end
    end
    return LinearTree{T,V,L}(tree.nodes, tree.catmasks, tree.loss, tree.lo,
        tree.hi, iszero(scale) ? zero(V) : scale * tree.base, tree.nfeatures, tree.truncate)
end

"Reuse `g0` for the proposed whole-ensemble increment after tree fitting."
function safeguard_boost_tree!(g0, loss, y, F, w, tree, X, eta, nthreads)
    row_blocks(length(g0), nthreads) do rows
        for i in rows
            g0[i] = eta * score_row(tree, X, i, false)
        end
    end
    scale = boost_step_scale(loss, y, F, w, g0, one(eta))
    if !iszero(scale)
        for i in eachindex(F)
            F[i] += scale * g0[i]
        end
    end
    return scale_boost_tree(tree, scale)
end

"""
    fit_boost(X, y, loss = MSE(); nrounds = 100, eta = 0.1, max_depth = 5,
              min_fit = 10, min_leaf = 5, min_sum_hessian = 1.0,
              lambda_slope = 1.0, lambda_intercept = 1.0, gamma = 0.0,
              subsample = 1.0, colsample = 1.0, rng = Random.default_rng(),
              weights = nothing, categorical = Int[], truncate = true,
              Xval = nothing, yval = nothing, wval = nothing, patience = 10,
              split_search = ExactSearch(), nthreads = Threads.nthreads()) -> LinearBoost

Second-order gradient boosting (Guryanov 2019) with the linear model tree as
base learner. Each round computes the gradient and Hessian of `loss` at the
ensemble score, freezes them into a [`Frozen`](@ref) target, and fits one
tree with [`GainRule`](@ref)`(lambda_slope, lambda_intercept, gamma)`, depth
`max_depth`, and the row and column samples drawn from `rng`. The tree's raw
score times `eta` is added to the ensemble. With `Xval`/`yval`, the
validation deviance is recorded per round and fitting stops after `patience`
rounds without improvement, keeping the trees up to the best round. Exact and
node-local binned searches presort `X` once. [`HybridSearch`](@ref) prepares
global bin IDs once from all positive-weight training rows, then selects the
sampled rows' IDs each round. It supports numeric features and scalar losses.
Sampled-out rows have weight
zero for that tree only. `history` holds the per-round deviance and the
ensemble's `validated` flag says whether it is the validation deviance, and so
whether early stopping was in play. It uses raw accumulated scores, before
the ensemble's prediction clamp. For base losses with `backtracks(loss)`, a
round's retained tree is scaled until the training objective on those raw
scores is finite and nonincreasing; validation receives the same scaled tree.

For `Quantile` and `MAD` the frozen Hessian is the IRLS weight at the
ensemble score, so a round is one IRLS step, not an exact L1 fit.
"""
function fit_boost(X::AbstractMatrix, y::AbstractVector, loss::Loss = MSE();
        nrounds = 100, eta = 0.1, max_depth = 5, min_fit = 10, min_leaf = 5, min_sum_hessian = 1.0,
        lambda_slope = 1.0, lambda_intercept = 1.0, gamma = 0.0,
        subsample = 1.0, colsample = 1.0, rng::AbstractRNG = default_rng(),
        weights = nothing, categorical = Int[], truncate = true,
        split_search::SplitSearch = ExactSearch(),
        Xval = nothing, yval = nothing, wval = nothing, patience = 10,
        nthreads = Threads.nthreads())
    nthreads = clamp(nthreads, 1, Threads.nthreads())
    check_tree_options(max_depth, 10, min_fit, min_leaf, min_sum_hessian, 3, 5)
    isinteger(nrounds) && nrounds >= 1 || throw(ArgumentError("nrounds must be an integer >= 1, got $nrounds"))
    isfinite(eta) && eta > 0 || throw(ArgumentError("eta must be finite and positive"))
    0 < subsample <= 1 || throw(ArgumentError("subsample must lie in (0, 1]"))
    0 < colsample <= 1 || throw(ArgumentError("colsample must lie in (0, 1]"))
    isinteger(patience) && patience >= 1 || throw(ArgumentError("patience must be an integer >= 1, got $patience"))
    (Xval === nothing) == (yval === nothing) || throw(ArgumentError("Xval and yval must be given together"))
    wval === nothing || Xval !== nothing || throw(ArgumentError("wval requires Xval and yval"))
    T = float(promote_type(eltype(X), eltype(y)))
    nrounds = Int(nrounds)
    patience = Int(patience)
    etaT = T(eta)
    isfinite(etaT) && etaT > 0 || throw(ArgumentError("eta must remain finite and positive in the working type"))
    n0, p = size(X)
    length(y) == n0 || throw(DimensionMismatch("X has $n0 rows, y has $(length(y))"))
    validate_target(loss, y)
    w = weights === nothing ? ones(T, n0) : Vector{T}(weights)
    length(w) == n0 || throw(DimensionMismatch("weights has length $(length(w)), X has $n0 rows"))
    all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("weights must be finite and non-negative"))
    keep = findall(>(0), w)
    isempty(keep) && throw(ArgumentError("total weight must be positive"))
    Xm = Matrix{T}(view(X, keep, :)); yv = Vector{T}(view(y, keep)); w = w[keep]
    all(isfinite, Xm) || throw(ArgumentError("X contains NaN or Inf"))
    n = length(keep)
    V = coeftype(loss, T)
    V <: Real || split_search isa ExactSearch ||
        throw(ArgumentError("$(typeof(split_search)) requires scalar coefficients; use ExactSearch() with $(typeof(loss))"))
    f0 = V(initscore(loss, yv, w))
    lo, hi = truncate ? map(V, scorebound(loss, yv)) : infbounds(V)
    F = fill(f0, n)
    g0 = zeros(V, n); h0 = zeros(V, n)
    target = Vector{Tuple{V,V}}(undef, n)
    allfeat = collect(1:p)
    iscat, nlevels = categorical_schema(Xm, categorical)
    split_search = prepare_search(split_search, Xm, allfeat, iscat, axes(Xm, 1), nothing, nthreads)
    idx = index_workspace(split_search, n, p)
    initialize_index!(idx, Xm, nothing, keep, allfeat, nthreads, split_search)
    presort = isempty(idx) ? nothing : idx
    rule = GainRule(; lambda_slope, lambda_intercept, gamma)
    frozen = Frozen{V}()
    trees = LinearTree{T,V,Frozen{V}}[]
    history = Float64[]
    hasval = Xval !== nothing
    Xv = Matrix{T}(undef, 0, p)
    yvv = T[]
    wv = T[]
    Fv = V[]
    if hasval
        size(Xval, 2) == p || throw(DimensionMismatch("Xval has $(size(Xval, 2)) columns, X has $p"))
        length(yval) == size(Xval, 1) || throw(DimensionMismatch("Xval has $(size(Xval, 1)) rows, yval has $(length(yval))"))
        validate_validation_target(loss, yval)
        wv = wval === nothing ? ones(T, length(yval)) : Vector{T}(wval)
        length(wv) == length(yval) || throw(DimensionMismatch("wval has length $(length(wv)), yval has $(length(yval))"))
        all(v -> isfinite(v) && v >= 0, wv) || throw(ArgumentError("wval must be finite and non-negative"))
        keepv = findall(>(0), wv)
        isempty(keepv) && throw(ArgumentError("total validation weight must be positive"))
        Xv = Matrix{T}(view(Xval, keepv, :)); yvv = Vector{T}(view(yval, keepv)); wv = wv[keepv]
        all(isfinite, Xv) || throw(ArgumentError("Xval contains NaN or Inf"))
        Fv = fill(f0, length(yvv))
    end
    wr = similar(w)
    nrow = max(1, round(Int, subsample * n))
    nfeat = max(1, ceil(Int, colsample * p))
    workspace = TreeWorkspace{T,V}(nrow, p, nthreads, split_search)
    # Round data are read-only during growth and are not retained by a tree.
    # A full-data round borrows the existing arrays; a sampled round gathers
    # into fixed-size buffers in the same ascending row order as findall.
    sampled = nrow != n
    round_X = sampled ? Matrix{T}(undef, nrow, p) : Xm
    round_y = sampled ? Vector{Tuple{V,V}}(undef, nrow) : target
    round_w = sampled ? Vector{T}(undef, nrow) : w
    round_keep = collect(1:nrow)
    for t in 1:nrounds
        frozen_target!(target, g0, h0, loss, yv, F, w)
        validate_target(frozen, target)
        if subsample < 1
            fill!(wr, zero(T))
            for i in view(randperm(rng, n), 1:nrow)
                wr[i] = w[i]
            end
        else
            copyto!(wr, w)
        end
        features = colsample < 1 ? sort!(randperm(rng, p)[1:nfeat]) : allfeat
        if sampled
            k = 0
            for i in eachindex(wr)
                iszero(wr[i]) && continue
                k += 1
                round_keep[k] = i
                round_y[k] = target[i]
                round_w[k] = wr[i]
            end
            for j in 1:p, k in 1:nrow
                round_X[k, j] = Xm[round_keep[k], j]
            end
        end
        round_search = prepare_search(split_search, round_X, features, iscat, round_keep, workspace, nthreads)
        tree = _fit_tree(round_X, round_y, round_w, frozen, rule, V, iscat, nlevels,
            round_keep, features, presort, workspace; max_depth, min_fit, min_leaf,
            min_sum_hessian, max_lin_chain = 10, truncate, truncation_factor = 3,
            nthreads, niter = 5, split_search = round_search)::LinearTree{T,V,Frozen{V}}
        if backtracks(loss)
            tree = safeguard_boost_tree!(g0, loss, yv, F, w, tree, Xm, etaT, nthreads)
        else
            add_tree_score!(F, tree, Xm, etaT, nthreads)
        end
        push!(trees, tree)
        if hasval
            add_tree_score!(Fv, tree, Xv, etaT, nthreads)
            push!(history, deviance(loss, yvv, Fv, wv))
            t - argmin(history) >= patience && break
        else
            push!(history, deviance(loss, yv, F, w))
        end
    end
    if hasval
        best = argmin(history)
        resize!(trees, best); resize!(history, best)
    end
    return LinearBoost{T,V,typeof(loss)}(trees, loss, f0, etaT, lo, hi, p, truncate, hasval, history)
end

"Add one tree's raw scores directly to the ensemble scores without a temporary vector."
function add_tree_score!(F, tree::LinearTree, X, eta, nthreads)
    row_blocks(length(F), nthreads) do rows
        for i in rows
            F[i] += eta * score_row(tree, X, i, false)
        end
    end
    return F
end

"""
    feature_importance(boost)

Positive split gains summed over every tree, normalised to sum to one.
"""
function feature_importance(b::LinearBoost)
    imp = zeros(Float64, b.nfeatures)
    for t in b.trees
        gain_sums!(imp, t)
    end
    return normalise_importance(imp)
end

"""
    coeftable(boost, x)

Intercept and per-feature slopes of the unclipped ensemble score at `x`:
`f0` plus `eta` times the sum of each tree's `coeftable`.
"""
function coeftable(b::LinearBoost{T,V}, x::AbstractVector) where {T,V}
    check_feature_width(b, x)
    slopes = zeros(V, b.nfeatures)
    intercept = b.f0
    for t in b.trees
        bi, si = coeftable(t, x)
        intercept += b.eta * bi
        slopes .+= b.eta .* si
    end
    return intercept, slopes
end

"""
Row gate for an ensemble's `row_blocks`, the tree rule of `shap_min_rows`
applied to the ensemble's total node count: one SHAP row visits every node of
every tree. An empty hand-built ensemble uses one node as the divisor.
"""
boost_shap_min_rows(b::LinearBoost) =
    max(2, cld(PARALLEL_MIN_ROWS,
        max(1, sum(length(t.nodes) for t in b.trees; init = 0))))

"""
    shap!(values, clipped, boost, X; nthreads=Threads.nthreads())

In-place [`shap`](@ref) for an ensemble: `eta` times the sum of every tree's
path-dependent SHAP values, base `f0 + eta Σ expected_score(tree)`, so each
row sums to `score(boost, x; clip = false) − base`. `clipped[i]` is true when
the ensemble clamp changed row `i`. One `PathPool` per row block,
reused across the ensemble's trees.
"""
function shap!(values, clipped::Vector{Bool}, b::LinearBoost{T,V}, X::AbstractMatrix;
        nthreads = Threads.nthreads()) where {T,V}
    check_feature_width(b, X)
    n = size(X, 1)
    expected = V <: SVector ? (n, b.nfeatures, length(V)) : (n, b.nfeatures)
    size(values) == expected || throw(DimensionMismatch("values must have size $expected, got $(size(values))"))
    length(clipped) == n || throw(DimensionMismatch("clipped must have length $n, got $(length(clipped))"))
    fill!(values, 0)
    row_blocks(n, nthreads; minrows = boost_shap_min_rows(b)) do rs
        pool = PathPool()   # one per block: never shared between tasks, see PathPool
        for i in rs
            x = view(X, i, :)
            for t in b.trees
                shap_recurse!(values, t, x, i, 1, pool, 1.0, 1.0, 0)
            end
            clipped[i] = score_row(b, X, i, true) != score_row(b, X, i, false)
        end
    end
    values .*= b.eta
    base = b.f0
    for t in b.trees
        base += b.eta * expected_score(t)
    end
    return ShapResult(values, base, clipped)
end

"""
    shap(boost, X; nthreads=Threads.nthreads()) -> ShapResult

Path-dependent TreeSHAP for a [`LinearBoost`](@ref); see [`shap!`](@ref).
"""
function shap(b::LinearBoost{T,V}, X::AbstractMatrix; nthreads = Threads.nthreads()) where {T,V}
    n = size(X, 1); p = b.nfeatures
    values = V <: SVector ? zeros(T, n, p, length(V)) : zeros(T, n, p)
    return shap!(values, Vector{Bool}(undef, n), b, X; nthreads)
end
