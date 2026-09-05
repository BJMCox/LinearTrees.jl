# Tree growth: node splitting and stopping rules.

"""
One worker's buffers, all of length `n`. A worker index (`tid`) picks the set
a task owns, and no two concurrent tasks are ever given the same one, so
nothing here needs a lock. Each buffer is reused for several unrelated
purposes over a node's life; the docstring of every user says why its own use
is free at that point.
"""
struct Scratch{T,V}
    xs::Vector{T}
    zs::Vector{V}
    hs::Vector{V}
    ws::Vector{T}
    perm::Vector{Int32}
end
Scratch{T,V}(n) where {T,V} = Scratch{T,V}(Vector{T}(undef, n), Vector{V}(undef, n), Vector{V}(undef, n),
    Vector{T}(undef, n), Vector{Int32}(undef, n))

"""
Everything one `fit_tree` call carries through growth. Built by keyword
(`Base.@kwdef`) because the positional form is twenty arguments wide and a
field reorder in it would corrupt a fit silently.
"""
Base.@kwdef mutable struct FitState{T,V,L<:Loss,R<:SelectionRule}
    X::Matrix{T}
    y::Vector{T}
    w::Vector{T}
    f::Vector{V}
    g::Vector{V}
    h::Vector{V}
    z::Vector{V}
    idx::Matrix{Int32}          # n × p, column j = row order sorted by feature j, partitioned node by node
    isleft::Vector{Bool}        # n, per-row left marker used by `partition!`; each node touches only its own rows
    scratch::Vector{Scratch{T,V}}   # one per worker
    nodes::Vector{Node{T,V}}
    catmasks::Vector{UInt64}
    iscat::Vector{Bool}         # length p, true for the columns `fit_tree` was given as categorical
    nlevels::Vector{Int}
    loss::L
    rule::R
    lo::V
    hi::V
    max_depth::Int
    min_fit::T
    min_leaf::T
    min_sum_hessian::T
    max_lin_chain::Int
    truncate::Bool
    nthreads::Int
    niter::Int
    # `h` is exactly one on every row: `MSE` with unit weights. The split scan
    # then reads `hs` as `UnitHessians` and accumulates without a multiply.
    unith::Bool
end

"""
    fit_tree(X, y, loss=MSE(); kwargs...)

Fit a PILOT-style linear model tree. See the design spec section 4 for the
keyword contract. `niter` is the number of IRLS refit passes non-smooth
losses (`MAD`, `Quantile`) take at each node; smooth losses ignore it.
"""
function fit_tree(X::AbstractMatrix, y::AbstractVector, loss::Loss = MSE();
        weights = nothing, categorical = Int[], rule::SelectionRule = BIC(),
        max_depth = 12, min_fit = 10, min_leaf = 5, min_sum_hessian = 1.0,
        max_lin_chain = 10, truncate = true, truncation_factor = 3,
        nthreads = Threads.nthreads(), niter = 5)
    # `niter` reaches an `Int` field, so a non-integer would surface as an
    # `InexactError` from deep inside the fit rather than as a rejected argument
    isinteger(niter) && niter >= 1 || throw(ArgumentError("niter must be an integer >= 1, got $niter"))
    isinteger(max_depth) && max_depth >= 0 || throw(ArgumentError("max_depth must be an integer >= 0, got $max_depth"))
    isinteger(max_lin_chain) && max_lin_chain >= 1 || throw(ArgumentError("max_lin_chain must be an integer >= 1, got $max_lin_chain"))
    min_fit >= 1 || throw(ArgumentError("min_fit must be >= 1, got $min_fit"))
    min_leaf >= 1 || throw(ArgumentError("min_leaf must be >= 1, got $min_leaf"))
    min_sum_hessian >= 0 || throw(ArgumentError("min_sum_hessian must be >= 0, got $min_sum_hessian"))
    # below 1 the padding term goes negative, so the clamp band closes inside the
    # observed range of `y` and every extreme score is pulled toward the middle
    truncation_factor >= 1 || throw(ArgumentError("truncation_factor must be >= 1, got $truncation_factor"))
    nthreads = clamp(nthreads, 1, Threads.nthreads())
    T = float(promote_type(eltype(X), eltype(y)))
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X has $n rows, y has $(length(y))"))
    validate_target(loss, y)
    w = weights === nothing ? ones(T, n) : Vector{T}(weights)
    all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("weights must be finite and non-negative"))
    keep = findall(>(0), w)
    isempty(keep) && throw(ArgumentError("total weight must be positive"))
    # No row is dropped and the caller already has the working element type, so
    # the copy would be pure cost. `st.X` is read-only for the whole fit, so the
    # tree holds a reference to the caller's matrix only until `fit_tree` returns.
    Xm = length(keep) == n && X isa Matrix{T} ? X : Matrix{T}(X[keep, :])
    yv = Vector{T}(y[keep]); w = w[keep]
    all(isfinite, Xm) || throw(ArgumentError("X contains NaN or Inf"))
    nlevels = zeros(Int, p)
    iscat = zeros(Bool, p)
    for j in categorical
        1 <= j <= p || throw(ArgumentError("categorical column $j is outside 1:$p"))
        col = view(Xm, :, j)
        all(x -> isfinite(x) && x >= 1 && x == round(x), col) ||
            throw(ArgumentError("categorical column $j must hold integer codes >= 1"))
        nlevels[j] = Int(maximum(col))
        iscat[j] = true
    end
    return _fit_tree(Xm, yv, w, loss, rule, coeftype(loss, T), iscat, nlevels;
        max_depth, min_fit, min_leaf, min_sum_hessian, max_lin_chain, truncate, truncation_factor,
        nthreads, niter)
end

"""
The body of `fit_tree` once every argument is concrete: `Xm::Matrix{T}`,
`yv`/`w::Vector{T}`, a concrete loss and rule, and the coefficient type `V`.
A function barrier, so growth specializes on those types instead of
re-dispatching on them at run time in every node. `fit_tree` stays the
validating front end.
"""
function _fit_tree(Xm::Matrix{T}, yv::Vector{T}, w::Vector{T}, loss::L, rule::R, ::Type{V},
        iscat::Vector{Bool}, nlevels::Vector{Int};
        max_depth, min_fit, min_leaf, min_sum_hessian, max_lin_chain, truncate, truncation_factor,
        nthreads, niter) where {T,V,L<:Loss,R<:SelectionRule}
    n, p = size(Xm)
    f0 = V(initscore(loss, yv, w))
    # every `scorebound` method returns bounds in its own working type (often
    # `Float64`, regardless of `V`), so convert here rather than trust each method
    lo, hi = truncate ? map(V, scorebound(loss, yv; truncation_factor)) : (V(-Inf), V(Inf))
    f = fill(clampscore(f0, lo, hi), n)
    idx = Matrix{Int32}(undef, n, p)
    presort!(idx, Xm, nthreads)
    st = FitState{T,V,L,R}(; X = Xm, y = yv, w, f, g = zeros(V, n), h = zeros(V, n), z = zeros(V, n),
        idx, isleft = zeros(Bool, n), scratch = [Scratch{T,V}(n) for _ in 1:nthreads],
        nodes = Node{T,V}[], catmasks = UInt64[], iscat, nlevels, loss, rule, lo, hi,
        max_depth = Int(max_depth), min_fit = T(min_fit), min_leaf = T(min_leaf),
        min_sum_hessian = T(min_sum_hessian), max_lin_chain = Int(max_lin_chain),
        truncate = Bool(truncate), nthreads = Int(nthreads), niter = Int(niter),
        unith = unit_hessian(loss) && all(isone, w))
    rows = collect(Int32(1):Int32(n))
    refresh!(st, rows, 1)
    grow_subtree(st, rows, 1:n, 0, 0, 1:nthreads, st.nodes, st.catmasks)
    # Fold the clamped start score into the root node: training accumulates it in
    # st.f before any node is fit, but prediction starts score_row at zero, so the
    # root's own intercepts must carry it. Exact for every loss, clamped or not.
    f0c = clampscore(f0, lo, hi)
    st.nodes[1] = Node{T,V}(st.nodes[1]; lintercept = st.nodes[1].lintercept + f0c,
        rintercept = st.nodes[1].rintercept + f0c)
    # `expected_score` only reads `nodes`/`catmasks` (see shap.jl), so build once with
    # a placeholder base to get the real one, the empty-coalition value (spec line 669-670)
    prelim = LinearTree{T,V,L}(st.nodes, st.catmasks, loss, lo, hi, zero(V), p, truncate)
    base = expected_score(prelim)
    return LinearTree{T,V,L}(st.nodes, st.catmasks, loss, lo, hi, base, p, truncate)
end

"""
Run `f(cols, t)` over `nthreads` contiguous column blocks of `1:p`, threaded
only when a column is long enough (`n` rows) to pay for the tasks. `t` is the
block's index in `1:nthreads`, for callers that keep one buffer per worker.
The row-shaped analogue is `row_blocks`.
"""
function column_blocks(f, p::Integer, n::Integer, nthreads::Integer)
    nthreads = clamp(nthreads, 1, Threads.nthreads())
    if nthreads == 1 || n < PARALLEL_MIN_ROWS || p == 1
        f(1:p, 1)
    else
        chunk = cld(p, nthreads)
        Threads.@threads for t in 1:nthreads
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, p)
            lo <= hi && f(lo:hi, t)
        end
    end
    return nothing
end

"Stable per-feature sort orders. Features are independent, so this threads over columns."
function presort!(idx::Matrix{Int32}, X::Matrix, nthreads)
    column_blocks(size(X, 2), size(X, 1), nthreads) do cols, _
        for j in cols
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
function refresh!(st::FitState, rows, tid)
    ε = node_epsilon(st, rows, tid)
    if st.nthreads == 1 || length(rows) < PARALLEL_MIN_ROWS
        refresh_chunk!(st, rows, ε)
    else
        # plain index vectors, not views of `rows`: a view-of-a-view into `st.y` makes
        # Base's alias check recurse during inference and leaves dynamic dispatch behind
        chunks = [rows[r] for r in Iterators.partition(eachindex(rows), cld(length(rows), st.nthreads))]
        Threads.@threads for ch in chunks
            refresh_chunk!(st, ch, ε)
        end
    end
    return st
end

"""
`ε` for IRLS, computed once over the whole `rows` set so a chunked pass
matches the serial one exactly; smooth losses ignore it. Runs on worker
`tid`'s scratch: the residual goes in `.ws`, `median_abs!`'s abs-value buffer
is `.xs` and its sort order is `.perm`. All three are free here for the
same reason they are free in `irls_refit` -- `best_split` has returned and
`partition!` has not yet run -- so the node's ε costs no allocation.
"""
function node_epsilon(st::FitState{T}, rows, tid) where {T}
    issmooth(st.loss) && return zero(T)
    sc = st.scratch[tid]
    resid = view(sc.ws, 1:length(rows))
    for (k, i) in enumerate(rows)
        resid[k] = st.y[i] - st.f[i]
    end
    return max(irls_epsilon!(sc.xs, sc.perm, resid, view(st.w, rows)),
        sqrt(eps(T)) * max(maximum(abs, view(st.y, rows)), one(T)))
end

function refresh_chunk!(st::FitState, rows, ε)
    yv = view(st.y, rows); fv = view(st.f, rows)
    gv = view(st.g, rows); hv = view(st.h, rows)
    gradhess!(gv, hv, st.loss, yv, fv)
    issmooth(st.loss) || irls_weights!(hv, st.loss, yv, fv; ε)
    for i in rows
        st.g[i] *= st.w[i]
        st.h[i] *= st.w[i]
        st.z[i] = -st.g[i] ./ st.h[i]   # `./` since `/` between two `V`s is undefined for `SVector`
    end
    return st
end

"""
Gather the node rows, `st.idx[span, j]` in feature-`j` order, into one
worker's scratch. `O(length(span))`. On the unit-hessian path `sc.hs` is not
written: `scan_hessians` hands the scan a `UnitHessians` instead, and the
branch is hoisted out of the row loop so neither loop tests it per row.
"""
function gather!(st::FitState, sc::Scratch, span::UnitRange{Int}, j)
    m = 0
    if st.unith
        for k in span
            i = st.idx[k, j]
            m += 1
            sc.xs[m] = st.X[i, j]; sc.zs[m] = st.z[i]; sc.ws[m] = st.w[i]
        end
    else
        for k in span
            i = st.idx[k, j]
            m += 1
            sc.xs[m] = st.X[i, j]; sc.zs[m] = st.z[i]; sc.hs[m] = st.h[i]; sc.ws[m] = st.w[i]
        end
    end
    return m
end

"""
Scan the first `m` rows of `sc` under `rule`. Two calls, not one on a
`Union` of `hs` types: on the unit-hessian path the scan gets `UnitHessians`,
which drops the multiply from `addrow` and `subrow`, and each branch stays a
call with a concrete argument type. Both give the same sums to the bit, since
the multiply dropped is by exactly one.
"""
@inline function scan_gathered(st::FitState{T,V}, sc::Scratch{T,V}, m, rule, dmin) where {T,V}
    xs = view(sc.xs, 1:m); zs = view(sc.zs, 1:m); ws = view(sc.ws, 1:m)
    st.unith && return scan_feature(xs, zs, UnitHessians{V}(m), ws, rule, st.min_leaf, dmin)
    return scan_feature(xs, zs, view(sc.hs, 1:m), ws, rule, st.min_leaf, dmin)
end

"""
Stable partition of `idx[span]` into rows with `isleft[i]` true, then the
rest, each side in its original order. `perm` is a buffer of at least
`length(span)`. Returns the left count. Positions outside `span` are untouched.
"""
function partition_column!(idx::AbstractVector{Int32}, span::UnitRange{Int}, isleft::Vector{Bool}, perm::Vector{Int32})
    nleft = 0
    for k in span
        nleft += isleft[idx[k]]
    end
    a = 0; b = nleft
    for k in span
        i = idx[k]
        if isleft[i]
            perm[a += 1] = i
        else
            perm[b += 1] = i
        end
    end
    copyto!(idx, first(span), perm, 1, length(span))
    return nleft
end

"""
Partition every column of `idx` over `span` so that `leftrows` come first, in
place and stable, so each child's rows stay sorted by every feature. Columns
are independent: with at least `PARALLEL_MIN_ROWS` rows they split across
`tids`, each task using its own `scratch[tid].perm` buffer. `isleft` is marked for
`leftrows` on entry and cleared on exit, so concurrent sibling subtrees, which
own disjoint rows, never see each other's marks. Returns the left count.
"""
function partition!(idx::Matrix{Int32}, span::UnitRange{Int}, leftrows::AbstractVector{Int32}, isleft::Vector{Bool},
        scratch, tids::UnitRange{Int})
    for i in leftrows
        isleft[i] = true
    end
    # every column returns the same left count, so take it here rather than from
    # whichever task happens to finish last
    nleft = 0
    for k in span
        nleft += isleft[idx[k, 1]]
    end
    column_blocks(size(idx, 2), length(span), length(tids)) do cols, t
        perm = scratch[first(tids) + t - 1].perm
        for j in cols
            partition_column!(view(idx, :, j), span, isleft, perm)
        end
    end
    for i in leftrows
        isleft[i] = false
    end
    return nleft
end

"Wrap a rule so only `con` and `pcon` are offered, for categorical scans."
struct PconOnly{R<:SelectionRule} <: SelectionRule
    inner::R
end
allowed(r::PconOnly, k::ModelKind) = (k == CON || k == PCON) && allowed(r.inner, k)
score_logn(r::PconOnly, n) = score_logn(r.inner, n)
devkey(r::PconOnly, surrogate, dmin) = devkey(r.inner, surrogate, dmin)
selection_score(r::PconOnly, k, s, n, dmin, ncoord::Integer = 1, logn = score_logn(r, n)) =
    allowed(r, k) ? selection_score(r.inner, k, s, n, dmin, ncoord, logn) : Inf

"""
Order the node's levels by weighted mean working response, scan a `pcon`
split over the ranks, and return the candidate plus the left level codes.
Runs on the worker's own scratch `sc`, so it is safe inside the threaded
feature search: no shared mutable state.

For a vector `V` (`Softmax`), one working response per coordinate gives one
candidate order per coordinate `k`; each is scanned in turn and the
lowest-scoring candidate wins. A scalar `V` has exactly one coordinate, so
this reduces to the original single-order scan.
"""
function scan_categorical(st::FitState{T,V}, sc::Scratch{T,V}, rows, j, dmin) where {T,V}
    L = st.nlevels[j]
    # `counts` is offset by one so it doubles as the counting sort's histogram
    sz = zeros(V, L); sw = zeros(V, L); counts = zeros(Int, L + 1)
    for i in rows
        c = Int(st.X[i, j])
        sz[c] += st.h[i] .* st.z[i]
        sw[c] += st.h[i]
        counts[c + 1] += 1
    end
    present = [c for c in 1:L if counts[c + 1] > 0]
    length(present) < 2 && return nocandidate(T, V), Int[]
    # Bucket rows by level in one O(L + |rows|) counting-sort pass, then walk
    # the buckets in rank order to fill the scratch. No O(L · |rows|) rescans.
    # The bucketing itself doesn't depend on level order, so it is built once
    # and reused for every coordinate's ordering below. `sc.perm` is free until
    # `partition!` runs, well after the scan.
    cumoffset = zeros(Int, L + 1)
    for c in 1:L
        cumoffset[c + 1] = cumoffset[c] + counts[c + 1]
    end
    cursor = copy(cumoffset)
    bucketed = sc.perm
    for i in rows
        c = Int(st.X[i, j])
        cursor[c] += 1
        bucketed[cursor[c]] = i
    end
    rule = PconOnly(st.rule)      # a categorical column never carries a linear piece, whatever the rule allows
    Km = length(zero(V))
    best = nocandidate(T, V); bestleft = Int[]
    for k in 1:Km
        order = sort(present; by = c -> sz[c][k] / sw[c][k])
        rank = zeros(Int32, L)
        for (r, lc) in enumerate(order)
            rank[lc] = r
        end
        m = 0
        for lc in order
            for b in (cumoffset[lc] + 1):cumoffset[lc + 1]
                i = bucketed[b]
                m += 1
                sc.xs[m] = T(rank[lc]); sc.zs[m] = st.z[i]; sc.hs[m] = st.h[i]; sc.ws[m] = st.w[i]
            end
        end
        cand = scan_gathered(st, sc, m, rule, dmin)
        if cand.score < best.score
            best = cand
            bestleft = cand.kind == PCON ? [lc for lc in order if rank[lc] <= cand.threshold] : Int[]
        end
    end
    return best, bestleft
end

"""
Append a packed left-level mask to the mask pool `masks` the growing subtree
writes into, and return its `(catstart, catwords)` in that pool.
"""
function push_mask!(masks::Vector{UInt64}, leftcodes, L)
    words = (L + 63) >> 6
    start = length(masks) + 1
    append!(masks, zeros(UInt64, words))
    for c in leftcodes
        masks[start + ((c - 1) >> 6)] |= UInt64(1) << ((c - 1) & 63)
    end
    return Int32(start), Int32(words)
end

"Serial search over `features` using scratch set `tid`. Returns the best candidate, its feature, and (for a categorical winner) its left level codes."
function best_split_serial(st::FitState{T,V}, rows, span::UnitRange{Int}, dmin, features, tid) where {T,V}
    sc = st.scratch[tid]
    best = nocandidate(T, V); bestj = 0; bestleft = Int[]
    for j in features
        if st.iscat[j]
            c, leftcodes = scan_categorical(st, sc, rows, j, dmin)
        else
            m = gather!(st, sc, span, j)
            c = scan_gathered(st, sc, m, st.rule, dmin)
            leftcodes = Int[]
        end
        if c.score < best.score
            best = c; bestj = j; bestleft = leftcodes
        end
    end
    return best, bestj, bestleft
end

"""
Search all features using only the subtree's own scratch sets `tids`. Large
nodes split the feature range across `length(tids)` tasks, each with its own
scratch. The reduction takes the lowest score and, on ties, the lowest
feature index, so the result equals the serial search regardless of task
completion order.
"""
function best_split(st::FitState{T,V}, rows, span::UnitRange{Int}, dmin, tids::UnitRange{Int}) where {T,V}
    p = size(st.X, 2)
    if length(rows) < PARALLEL_MIN_ROWS || length(tids) == 1 || p == 1
        return best_split_serial(st, rows, span, dmin, 1:p, first(tids))
    end
    nt = length(tids)
    chunks = collect(Iterators.partition(1:p, cld(p, nt)))
    tasks = [Threads.@spawn best_split_serial(st, rows, span, dmin, ch, tid) for (tid, ch) in zip(tids, chunks)]
    best = nocandidate(T, V); bestj = 0; bestleft = Int[]
    for t in tasks
        # `fetch` infers `Any`; without this the winning candidate stays boxed and
        # `grow_subtree`'s whole body dispatches on it once per node
        c, j, lc = fetch(t)::Tuple{Candidate{T,V},Int,Vector{Int}}
        if c.score < best.score || (c.score == best.score && j != 0 && (bestj == 0 || j < bestj))
            best = c; bestj = j; bestleft = lc
        end
    end
    return best, bestj, bestleft
end

"`nw` is the node's total weight, which `grow_subtree` already holds; recomputing it here would be a second O(|rows|) pass."
function dmin_for(st::FitState{T,V}, rows, nw) where {T,V}
    s = zero(T)
    for i in rows
        s += sum(st.h[i] .* st.z[i] .^ 2)   # sum over coordinates for vector V; a no-op for scalar V
    end
    return eps(T) * max(s, nw)
end

leafnode(st::FitState{T,V}, rows, b) where {T,V} =
    Node{T,V}(; lintercept = b, cover = sum(view(st.w, rows)))

"Depth gate for sibling-subtree parallelism: a node shallower than this may spawn its two children as separate tasks."
const SUBTREE_PARALLEL_DEPTH = 3

"""
Add `idxoff` to every non-zero child index and `maskoff` to every non-zero
`catstart` in `nodes`. Used once per spawned sibling subtree, to relocate the
local node vector and mask words it grew concurrently into the parent's.
"""
function shift_subtree(nodes::Vector{Node{T,V}}, idxoff::Int32, maskoff::Int32) where {T,V}
    return [Node{T,V}(n; left = n.left == 0 ? Int32(0) : n.left + idxoff,
                        right = n.right == 0 ? Int32(0) : n.right + idxoff,
                        catstart = n.catwords == 0 ? Int32(0) : n.catstart + maskoff) for n in nodes]
end

"""
Grow the subtree for `rows`, appending its nodes to `nodes` in preorder and
its level masks to `masks`. The subtree root lands at `length(nodes) + 1` and
child indices and `catstart` values are absolute in those two vectors, so the
caller has nothing to splice: a node is written once and moved never.

`span` is the range of `st.idx` this node owns: `st.idx[span, j]` holds
exactly `rows`, sorted by feature `j`, for every `j`. A split partitions the
span in place, so children own disjoint sub-ranges and no other node reads or
writes them. `linchain` counts consecutive `lin` fits in this node position.
`tids` is the range of `st.scratch` sets this subtree, and only this subtree,
may use; a spawned sibling gets a disjoint sub-range, so no two concurrent
subtrees ever touch the same scratch set.

Only the calling task appends to `nodes` and `masks`. A spawned sibling grows
into its own pair and is shifted and appended once when it returns, so that
subtree is the one and only case where a node is copied.
"""
function grow_subtree(st::FitState{T,V}, rows::Vector{Int32}, span::UnitRange{Int}, depth::Int, linchain::Int,
        tids::UnitRange{Int}, nodes::Vector{Node{T,V}}, masks::Vector{UInt64}) where {T,V}
    mine = length(nodes) + 1
    nw = sum(view(st.w, rows))
    sumh = sum(sum(h) for h in view(st.h, rows))   # sum over coordinates too, for vector V
    if nw < st.min_fit || depth >= st.max_depth || sumh < st.min_sum_hessian || linchain >= st.max_lin_chain
        b = fit_con(node_sums(st, rows))[1]
        me = leafnode(st, rows, b)
        me = refit_node(st, me, rows, first(tids))
        update_score!(st, rows, me)
        push!(nodes, me)
        return nothing
    end
    dmin = dmin_for(st, rows, nw)
    best, bestj, leftcodes = best_split(st, rows, span, dmin, tids)
    con_intercept, con_surrogate = fit_con(node_sums(st, rows))
    gain = T(sum(con_surrogate) - sum(best.surrogate))
    if best.kind == CON || bestj == 0
        me = leafnode(st, rows, best.kind == CON ? best.lintercept : con_intercept)
        me = refit_node(st, me, rows, first(tids))
        update_score!(st, rows, me)
        push!(nodes, me)
        return nothing
    end
    iscat = st.iscat[bestj]
    if iscat
        xmin = xmax = xmean = zero(T)
    else
        xj = view(st.X, rows, bestj)
        xmin, xmax = extrema(xj)
        xmean = sum(st.w[i] * st.X[i, bestj] for i in rows) / nw
    end
    if best.kind == LIN
        # A LIN node stores `threshold = NaN`, so `x <= n.threshold` is false for
        # every row and the routing falls to the right branch. That is harmless
        # only because both branches are the same fit: `rcoef == lcoef`,
        # `rintercept == lintercept` and, below, `right == left`. `score_row`
        # (predict.jl) and `coeftable`'s walk (importance.jl) both rely on it.
        me = Node{T,V}(; feature = bestj, threshold = T(NaN), lcoef = best.lcoef, lintercept = best.lintercept,
            rcoef = best.lcoef, rintercept = best.lintercept, xmin, xmax, cover = nw, xmean, gain, model = LIN)
        me = refit_node(st, me, rows, first(tids))
        update_score!(st, rows, me); refresh!(st, rows, first(tids))
        push!(nodes, Node{T,V}(me; left = Int32(mine + 1), right = Int32(mine + 1)))
        grow_subtree(st, rows, span, depth, linchain + 1, tids, nodes, masks)
        return nothing
    end
    catstart = Int32(0); catwords = Int32(0); threshold = best.threshold
    if iscat
        catstart, catwords = push_mask!(masks, leftcodes, st.nlevels[bestj])
        threshold = T(NaN)
    end
    me = Node{T,V}(; feature = bestj, threshold, lcoef = best.lcoef, lintercept = best.lintercept,
        rcoef = best.rcoef, rintercept = best.rintercept, xmin, xmax, cover = nw, xmean, gain, model = best.kind,
        catstart, catwords)
    me = refit_node(st, me, rows, first(tids), masks)
    update_score!(st, rows, me, masks)
    refresh!(st, rows, first(tids))
    # count first so both sides are sized exactly: growing them from empty costs
    # about 2·log2(|rows|) reallocations per split node. The push order is the
    # parent's row order, which every downstream sum depends on, so it stays.
    nleftrows = 0
    for i in rows
        nleftrows += goes_left(st, me, i, masks)
    end
    leftrows = Int32[]; rightrows = Int32[]
    sizehint!(leftrows, nleftrows); sizehint!(rightrows, length(rows) - nleftrows)
    for i in rows
        (goes_left(st, me, i, masks) ? push!(leftrows, i) : push!(rightrows, i))
    end
    nl = partition!(st.idx, span, leftrows, st.isleft, st.scratch, tids)
    lspan = first(span):(first(span) + nl - 1); rspan = (first(span) + nl):last(span)
    push!(nodes, me)   # placeholder: the child indices are only known once both children have grown
    if st.nthreads > 1 && depth < SUBTREE_PARALLEL_DEPTH && length(tids) >= 2
        mid = first(tids) + length(tids) ÷ 2 - 1
        rnodes = Node{T,V}[]; rmasks = UInt64[]
        task = Threads.@spawn grow_subtree(st, rightrows, rspan, depth + 1, 0, (mid + 1):last(tids), rnodes, rmasks)
        grow_subtree(st, leftrows, lspan, depth + 1, 0, first(tids):mid, nodes, masks)
        wait(task)
        # the sibling grew local indices from 1, so shift by what now precedes it
        rightidx = length(nodes) + 1
        append!(nodes, shift_subtree(rnodes, Int32(length(nodes)), Int32(length(masks))))
        append!(masks, rmasks)
    else
        grow_subtree(st, leftrows, lspan, depth + 1, 0, tids, nodes, masks)
        rightidx = length(nodes) + 1
        grow_subtree(st, rightrows, rspan, depth + 1, 0, tids, nodes, masks)
    end
    nodes[mine] = Node{T,V}(me; left = Int32(mine + 1), right = Int32(rightidx))
    return nothing
end

"""
True when row `i` is routed left by node `n`. Categorical nodes route by
mask; others by threshold. `masks` must be the mask pool `n.catstart`
indexes into: during growth that is the pool the subtree is appending to
(`st.catmasks` on the main chain, a local vector inside a spawned sibling);
after fitting it is `tree.catmasks`. Non-categorical nodes never touch `masks`,
so the empty default is safe wherever the node cannot be categorical.
"""
function goes_left(st::FitState, n::Node, i, masks::Vector{UInt64} = UInt64[])
    iscategorical(n) && return category_is_left(masks, n, Int(st.X[i, n.feature]))
    return n.model == LIN || st.X[i, n.feature] <= n.threshold
end

"""
IRLS refit of one node's own coefficients on its own rows. `niter` passes:
each recomputes the residual against the node's current fit, the scale-aware
`ε`, and the L1-majorizer weight, then re-solves the node's model kind.
Serial: this is one node's `MomentSums` reduction, not worth threading.
Takes and returns a `Node` value so it works on a node still local to a
growing subtree, before it has a place in `st.nodes`. `masks` is the pool
`n.catstart` indexes into (see `goes_left`); irrelevant, so omitted, for a
node that cannot be categorical. `tid` is this subtree's own worker index
(`first(tids)` at the call site): `st.scratch[tid].zs` (the residual buffer)
and `.xs` (`median_abs`'s abs-value buffer) and `.perm` (its sort
permutation) are all free at this point -- `best_split` has already returned
its winning candidate, and `partition!` has not yet run for this node -- so
reusing them here avoids a fresh allocation on every call.
"""
function irls_refit(st::FitState{T,V}, n::Node{T,V}, rows, tid, niter, masks::Vector{UInt64} = UInt64[]) where {T,V}
    j = n.feature
    m = length(rows)
    sc = st.scratch[tid]
    resid = view(sc.zs, 1:m)
    permbuf = sc.perm
    yscale = max(maximum(abs, view(st.y, rows)), one(T))
    for _ in 1:niter
        for (k, i) in enumerate(rows)
            pred = if n.model == CON
                n.lintercept
            else
                x = st.X[i, j]
                goleft = goes_left(st, n, i, masks)
                goleft ? n.lcoef * x + n.lintercept : n.rcoef * x + n.rintercept
            end
            resid[k] = st.z[i] - pred
        end
        ε = max(irls_epsilon!(sc.xs, permbuf, resid, view(st.w, rows)), sqrt(eps(T)) * yscale)
        left = zero(MomentSums{V}); right = zero(MomentSums{V})
        for (k, i) in enumerate(rows)
            r = resid[k]
            # floor the unweighted pseudo-hessian at HMIN before scaling by st.w[i],
            # the same order irls_weights! applies it in, so a zero-weight row can't
            # be the only thing keeping a non-smooth refit's Hessian away from zero
            hi = st.w[i] * max(l1weight(st.loss, r) / max(abs(r), ε), oftype(r, HMIN))
            if n.model == CON
                left = addrow(left, zero(T), st.z[i], hi)
            else
                x = st.X[i, j]
                goleft = goes_left(st, n, i, masks)
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
    return n
end

function node_sums(st::FitState{T,V}, rows) where {T,V}
    s = zero(MomentSums{V})
    for i in rows
        s = addrow(s, zero(T), st.z[i], st.h[i])
    end
    return s
end

"Add node `n`'s piece to the score of `rows`, then clamp. `masks` is the pool `n.catstart` indexes into (see `goes_left`)."
function update_score!(st::FitState, rows, n::Node, masks::Vector{UInt64} = UInt64[])
    for i in rows
        if isleaf(n)
            inc = n.lintercept
        else
            goleft = goes_left(st, n, i, masks)
            if iscategorical(n)
                inc = goleft ? n.lintercept : n.rintercept
            else
                x = st.X[i, n.feature]
                x = st.truncate ? min(max(x, n.xmin), n.xmax) : x
                inc = goleft ? n.lcoef * x + n.lintercept : n.rcoef * x + n.rintercept
            end
        end
        s = st.f[i] + inc
        st.f[i] = st.truncate ? clampscore(s, st.lo, st.hi) : s
    end
    return st
end
