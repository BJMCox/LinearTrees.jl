# Tree growth: node splitting and stopping rules.

struct Scratch{T,V}
    xs::Vector{T}; zs::Vector{V}; hs::Vector{V}; ws::Vector{T}
end
Scratch{T,V}(n) where {T,V} = Scratch{T,V}(Vector{T}(undef, n), Vector{V}(undef, n), Vector{V}(undef, n), Vector{T}(undef, n))

mutable struct FitState{T,V,L<:Loss,R<:SelectionRule}
    X::Matrix{T}
    y::Vector{T}
    w::Vector{T}
    f::Vector{V}
    g::Vector{V}
    h::Vector{V}
    z::Vector{V}
    idx::Matrix{Int32}          # n × p, column j = row order sorted by feature j
    scratch::Vector{Scratch{T,V}}   # one per worker
    nodes::Vector{Node{T,V}}
    catmasks::Vector{UInt64}
    loss::L
    rule::R
    lo::V; hi::V
    max_depth::Int; min_fit::T; min_leaf::T
    min_sum_hessian::T; max_lin_chain::Int
    truncate::Bool
    nthreads::Int
end

"""
    fit_tree(X, y, loss=MSE(); kwargs...)

Fit a PILOT-style linear model tree. See the design spec section 4 for the
keyword contract.
"""
function fit_tree(X::AbstractMatrix, y::AbstractVector, loss::Loss = MSE();
        weights = nothing, categorical = Int[], rule::SelectionRule = BIC(),
        max_depth = 12, min_fit = 10, min_leaf = 5, min_sum_hessian = 1.0,
        max_lin_chain = 10, truncate = true, truncation_factor = 3,
        nthreads = Threads.nthreads())
    isempty(categorical) || throw(ArgumentError("categorical features arrive in Task 9"))
    nthreads = clamp(nthreads, 1, Threads.nthreads())
    T = float(promote_type(eltype(X), eltype(y)))
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X has $n rows, y has $(length(y))"))
    validate_target(loss, y)
    w = weights === nothing ? ones(T, n) : Vector{T}(weights)
    all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("weights must be finite and non-negative"))
    keep = findall(>(0), w)
    isempty(keep) && throw(ArgumentError("total weight must be positive"))
    Xm = Matrix{T}(X[keep, :]); yv = Vector{T}(y[keep]); w = w[keep]
    n = length(keep)
    V = T
    f0 = V(initscore(loss, yv, w))
    lo, hi = truncate ? scorebound(loss, yv; truncation_factor) : (V(-Inf), V(Inf))
    f = fill(clampscore(f0, lo, hi), n)
    idx = Matrix{Int32}(undef, n, p)
    presort!(idx, Xm, nthreads)
    st = FitState{T,V,typeof(loss),typeof(rule)}(Xm, yv, w, f, zeros(V, n), zeros(V, n), zeros(V, n), idx,
        [Scratch{T,V}(n) for _ in 1:nthreads],
        Node{T,V}[], UInt64[], loss, rule, lo, hi, max_depth, T(min_fit), T(min_leaf),
        T(min_sum_hessian), max_lin_chain, truncate, nthreads)
    rows = collect(Int32(1):Int32(n))
    refresh!(st, rows)
    grow!(st, rows, 0, 0)
    # Fold the clamped start score into the root node: training accumulates it in
    # st.f before any node is fit, but prediction starts score_row at zero, so the
    # root's own intercepts must carry it. Exact for every loss, clamped or not.
    f0c = clampscore(f0, lo, hi)
    st.nodes[1] = Node{T,V}(st.nodes[1]; lintercept = st.nodes[1].lintercept + f0c,
        rintercept = st.nodes[1].rintercept + f0c)
    base = sum(w .* st.f) / sum(w)      # SHAP base value: cover-weighted mean of the training score
    return LinearTree{T,V,typeof(loss)}(st.nodes, st.catmasks, loss, lo, hi, base, p, truncate)
end

"Stable per-feature sort orders. Features are independent, so this threads over columns."
function presort!(idx::Matrix{Int32}, X::Matrix, nthreads)
    p = size(X, 2)
    if nthreads == 1 || size(X, 1) < PARALLEL_MIN_ROWS
        for j in 1:p
            idx[:, j] .= Int32.(sortperm(view(X, :, j); alg = MergeSort))
        end
    else
        Threads.@threads for j in 1:p
            idx[:, j] .= Int32.(sortperm(view(X, :, j); alg = MergeSort))
        end
    end
    return idx
end

"""
Recompute `g`, `h`, `z` for `rows` at the current score. Floor first, then
weight. Rows are independent, so large row sets are processed in chunks on
separate threads; there is no cross-row reduction, so the result is identical
to the serial pass.
"""
function refresh!(st::FitState, rows)
    if st.nthreads == 1 || length(rows) < PARALLEL_MIN_ROWS
        refresh_chunk!(st, rows, node_epsilon(st, rows))
    else
        ε = node_epsilon(st, rows)
        chunks = collect(Iterators.partition(rows, cld(length(rows), st.nthreads)))
        Threads.@threads for ch in chunks
            refresh_chunk!(st, ch, ε)
        end
    end
    return st
end

# `ε` for IRLS is computed once over the whole `rows` set so a chunked pass
# matches the serial one exactly; smooth losses ignore it.
node_epsilon(st::FitState{T}, rows) where {T} = issmooth(st.loss) ? zero(T) :
    max(T(1e-3) * median_abs(view(st.y, rows) .- view(st.f, rows)),
        sqrt(eps(T)) * max(maximum(abs, view(st.y, rows)), one(T)))

function refresh_chunk!(st::FitState, rows, ε)
    yv = view(st.y, rows); fv = view(st.f, rows)
    gv = view(st.g, rows); hv = view(st.h, rows)
    gradhess!(gv, hv, st.loss, yv, fv)
    issmooth(st.loss) || irls_weights!(hv, st.loss, yv, fv; ε)
    for i in rows
        st.g[i] *= st.w[i]
        st.h[i] *= st.w[i]
        st.z[i] = -st.g[i] / st.h[i]
    end
    return st
end

"Gather the node rows in the order of feature `j` into one worker's scratch."
function gather!(st::FitState, sc::Scratch, inrow::BitVector, j)
    m = 0
    for i in view(st.idx, :, j)
        inrow[i] || continue
        m += 1
        sc.xs[m] = st.X[i, j]; sc.zs[m] = st.z[i]; sc.hs[m] = st.h[i]; sc.ws[m] = st.w[i]
    end
    return m
end

"Serial search over `features` using scratch set `tid`. Returns the best candidate and its feature."
function best_split_serial(st::FitState{T,V}, rows, inrow::BitVector, dmin, features, tid) where {T,V}
    sc = st.scratch[tid]
    best = nocandidate(T, V); bestj = 0
    for j in features
        m = gather!(st, sc, inrow, j)
        c = scan_feature(view(sc.xs, 1:m), view(sc.zs, 1:m), view(sc.hs, 1:m), view(sc.ws, 1:m),
            st.rule, st.min_leaf, dmin)
        if c.score < best.score
            best = c; bestj = j
        end
    end
    return best, bestj
end

"""
Search all features. Large nodes split the feature range across `nthreads`
tasks, each with its own scratch. The reduction takes the lowest score and,
on ties, the lowest feature index, so the result equals the serial search.
"""
function best_split(st::FitState{T,V}, rows, inrow::BitVector, dmin) where {T,V}
    p = size(st.X, 2)
    if size(st.X, 1) < PARALLEL_MIN_ROWS || st.nthreads == 1 || p == 1   # gather! scans the full presorted column, so the cost is O(n) per feature regardless of node size
        return best_split_serial(st, rows, inrow, dmin, 1:p, 1)
    end
    chunks = collect(Iterators.partition(1:p, cld(p, st.nthreads)))
    tasks = [Threads.@spawn best_split_serial(st, rows, inrow, dmin, ch, tid) for (tid, ch) in enumerate(chunks)]
    best = nocandidate(T, V); bestj = 0
    for t in tasks
        c, j = fetch(t)
        if c.score < best.score || (c.score == best.score && j != 0 && (bestj == 0 || j < bestj))
            best = c; bestj = j
        end
    end
    return best, bestj
end

function dmin_for(st::FitState, rows)
    s = zero(eltype(st.h))
    for i in rows
        s += st.h[i] * st.z[i]^2
    end
    n = sum(view(st.w, rows))
    return eps(eltype(st.h)) * max(s, n)
end

leafnode(st::FitState{T,V}, rows, b) where {T,V} =
    Node{T,V}(; lintercept = b, cover = sum(view(st.w, rows)))

"""
Grow the subtree for `rows`. Returns the index of the created node.
`linchain` counts consecutive `lin` fits in this node position.
"""
function grow!(st::FitState{T,V}, rows::Vector{Int32}, depth::Int, linchain::Int) where {T,V}
    nw = sum(view(st.w, rows))
    sumh = sum(view(st.h, rows))
    push!(st.nodes, leafnode(st, rows, zero(V)))   # placeholder, filled below
    me = Int32(length(st.nodes))
    if nw < st.min_fit || depth >= st.max_depth || sumh < st.min_sum_hessian || linchain >= st.max_lin_chain
        b = fit_con(node_sums(st, rows))[1]
        st.nodes[me] = leafnode(st, rows, b)
        refit_node!(st, me, rows)
        update_score!(st, rows, me)
        return me
    end
    dmin = dmin_for(st, rows)
    inrow = falses(length(st.y)); inrow[rows] .= true
    best, bestj = best_split(st, rows, inrow, dmin)
    if best.kind == CON || bestj == 0
        st.nodes[me] = leafnode(st, rows, best.kind == CON ? best.lintercept : fit_con(node_sums(st, rows))[1])
        refit_node!(st, me, rows)
        update_score!(st, rows, me)
        return me
    end
    xj = view(st.X, rows, bestj)
    xmin, xmax = extrema(xj)
    xmean = sum(st.w[i] * st.X[i, bestj] for i in rows) / nw
    if best.kind == LIN
        st.nodes[me] = Node{T,V}(; feature = bestj, threshold = T(NaN), lcoef = best.lcoef, lintercept = best.lintercept,
            rcoef = best.lcoef, rintercept = best.lintercept, xmin, xmax, cover = nw, xmean, model = LIN)
        refit_node!(st, me, rows)
        update_score!(st, rows, me); refresh!(st, rows)
        child = grow!(st, rows, depth, linchain + 1)
        st.nodes[me] = Node{T,V}(st.nodes[me]; left = child, right = child)
        return me
    end
    leftrows = Int32[]; rightrows = Int32[]
    for i in rows
        (st.X[i, bestj] <= best.threshold ? push!(leftrows, i) : push!(rightrows, i))
    end
    st.nodes[me] = Node{T,V}(; feature = bestj, threshold = best.threshold, lcoef = best.lcoef, lintercept = best.lintercept,
        rcoef = best.rcoef, rintercept = best.rintercept, xmin, xmax, cover = nw, xmean, model = best.kind)
    refit_node!(st, me, rows)
    update_score!(st, rows, me)
    refresh!(st, rows)
    left = grow!(st, leftrows, depth + 1, 0)
    right = grow!(st, rightrows, depth + 1, 0)
    st.nodes[me] = Node{T,V}(st.nodes[me]; left, right)
    return me
end

"""
IRLS refit of one node's own coefficients on its own rows. `niter` passes:
each recomputes the residual against the node's current fit, the scale-aware
`ε`, and the L1-majorizer weight, then re-solves the node's model kind.
Serial: this is one node's `MomentSums` reduction, not worth threading.
"""
function irls_refit!(st::FitState{T,V}, me::Integer, rows, niter) where {T,V}
    n = st.nodes[me]
    j = n.feature
    resid = Vector{V}(undef, length(rows))
    yscale = max(maximum(abs, view(st.y, rows)), one(T))
    for _ in 1:niter
        for (k, i) in enumerate(rows)
            pred = if n.model == CON
                n.lintercept
            else
                x = st.X[i, j]
                goleft = n.model == LIN || x <= n.threshold
                goleft ? n.lcoef * x + n.lintercept : n.rcoef * x + n.rintercept
            end
            resid[k] = st.z[i] - pred
        end
        ε = max(T(1e-3) * median_abs(resid), sqrt(eps(T)) * yscale)
        left = zero(MomentSums{V}); right = zero(MomentSums{V})
        for (k, i) in enumerate(rows)
            r = resid[k]
            hi = st.w[i] * l1weight(st.loss, r) / max(abs(r), ε)
            if n.model == CON
                left = addrow(left, zero(T), st.z[i], hi)
            else
                x = st.X[i, j]
                goleft = n.model == LIN || x <= n.threshold
                if n.model != LIN && !goleft
                    right = addrow(right, x, st.z[i], hi)
                else
                    left = addrow(left, x, st.z[i], hi)
                end
            end
        end
        if n.model == CON
            b = fit_con(left)[1]
            n = Node{T,V}(n; lintercept = b, rintercept = b)
        elseif n.model == LIN
            r = fit_lin(left); r === nothing && break
            a, b, _ = r
            n = Node{T,V}(n; lcoef = a, lintercept = b, rcoef = a, rintercept = b)
        elseif n.model == PCON
            bl = fit_con(left)[1]; br = fit_con(right)[1]
            n = Node{T,V}(n; lintercept = bl, rintercept = br)
        elseif n.model == PLIN
            rl = fit_lin(left); rr = fit_lin(right)
            (rl === nothing || rr === nothing) && break
            n = Node{T,V}(n; lcoef = rl[1], lintercept = rl[2], rcoef = rr[1], rintercept = rr[2])
        else # BLIN
            r = fit_blin(left, right, n.threshold); r === nothing && break
            n = Node{T,V}(n; lcoef = r[1], lintercept = r[2], rcoef = r[3], rintercept = r[4])
        end
    end
    st.nodes[me] = n
    return st
end

function node_sums(st::FitState{T,V}, rows) where {T,V}
    s = zero(MomentSums{V})
    for i in rows
        s = addrow(s, zero(T), st.z[i], st.h[i])
    end
    return s
end

"Add node `me`'s piece to the score of `rows`, then clamp."
function update_score!(st::FitState, rows, me::Integer)
    n = st.nodes[me]
    for i in rows
        if isleaf(n)
            inc = n.lintercept
        else
            x = st.X[i, n.feature]
            x = st.truncate ? min(max(x, n.xmin), n.xmax) : x
            goleft = n.model == LIN || x <= n.threshold
            inc = goleft ? n.lcoef * x + n.lintercept : n.rcoef * x + n.rintercept
        end
        s = st.f[i] + inc
        st.f[i] = st.truncate ? clampscore(s, st.lo, st.hi) : s
    end
    return st
end
