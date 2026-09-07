"""
    HybridSearch(; nbins=64)

Approximate numeric split search for scalar-score losses. Assign rows to at
most `nbins` equal-count bins once per fit, keeping equal feature values in the
same bin. Tree nodes scan these global bins and refine the best coarse boundary
over its two adjacent occupied bins. Nodes with at most `nbins` rows use exact
search.

The sufficient statistics use the original feature values and responses, and
weights retain their frequency meaning. This can change the selected model and
predictions; it does not guarantee the exact optimum or an error bound.
Use `ExactSearch()` or `BinnedSearch()` for categorical features, and
`ExactSearch()` for vector-score losses. `nbins` must lie in `2:65535`.
Boosting reuses bins from all positive-weight training rows across rounds.
"""
struct HybridSearch <: SplitSearch
    nbins::Int
    function HybridSearch(; nbins::Integer = 64)
        2 <= nbins <= typemax(UInt16) ||
            throw(ArgumentError("nbins must be in 2:$(typemax(UInt16))"))
        return new(Int(nbins))
    end
end

struct PreparedHybridSearch <: SplitSearch
    nbins::Int
    ids::Matrix{UInt16}
end

struct HybridBinWorkspace{T,V}
    moments::Vector{MomentSums{V}}
    masses::Vector{T}
    counts::Vector{Int}
    suffix_counts::Vector{Int}
    values::Matrix{T}
    maxima::Vector{T}
    rowids::Vector{Int32}
    xs::Vector{T}
    zs::Vector{V}
    hs::Vector{V}
    ws::Vector{T}
end

function HybridBinWorkspace{T,V}(nbins) where {T,V}
    return HybridBinWorkspace(
        fill(zero(MomentSums{V}), nbins), zeros(T, nbins), zeros(Int, nbins),
        zeros(Int, nbins + 1), Matrix{T}(undef, MIN_UNIQUE_LIN, nbins),
        Vector{T}(undef, nbins), Int32[], T[], V[], V[], T[])
end

search_workspace(search::HybridSearch, ::Type{T}, ::Type{V}) where {T,V} =
    HybridBinWorkspace{T,V}(search.nbins)
search_workspace(search::PreparedHybridSearch, ::Type{T}, ::Type{V}) where {T,V} =
    HybridBinWorkspace{T,V}(search.nbins)

index_workspace(::PreparedHybridSearch, n, p) = Matrix{Int32}(undef, 0, 0)
sampled_index_workspace(search::PreparedHybridSearch, n, p) =
    n == size(search.ids, 1) ? nothing : Matrix{UInt16}(undef, n, p)
initialize_index!(idx, X, presort, keep, features, nthreads, ::PreparedHybridSearch) = idx

function prepare_search(search::HybridSearch, X::Matrix{T}, features::Vector{Int},
        iscat, keep, workspace, nthreads) where {T}
    any(iscat[features]) &&
        throw(ArgumentError("HybridSearch supports numeric features only; use ExactSearch() with categorical features"))
    n, p = size(X)
    ids = Matrix{UInt16}(undef, n, p)
    # Sort scratch grows with rows per worker. Four blocks retain most of the
    # measured parallel gain without allocating a full sort set for every thread.
    column_blocks(length(features), n, min(nthreads, 4)) do columns, _
        prepare_hybrid_columns!(ids, search.nbins, X, features, columns)
    end
    return PreparedHybridSearch(search.nbins, ids)
end

function prepare_hybrid_columns!(ids, nbins, X::Matrix{T}, features::Vector{Int},
        columns::UnitRange{Int}) where {T}
    n = size(X, 1)
    # Each column block owns its sort scratch and writes disjoint ID columns.
    order = Vector{Int32}(undef, n)
    U = radix_uint(T)
    buffer = U === nothing ? nothing : RadixBuffers(U, n)
    for k in columns
        j = features[k]
        x = view(X, :, j)
        if buffer === nothing
            copyto!(order, sortperm(x; alg = MergeSort))
        else
            radix_sortperm!(order, x, buffer)
        end
        width = cld(n, nbins)
        lo = 1
        bin = 1
        while lo <= n
            hi = min(n, lo + width - 1)
            while hi < n && x[order[hi + 1]] == x[order[hi]]
                hi += 1
            end
            for k in lo:hi
                ids[order[k], j] = UInt16(bin)
            end
            bin += 1
            lo = hi + 1
        end
    end
    return ids
end

function prepare_search(search::PreparedHybridSearch, X, features, iscat, keep, workspace, nthreads)
    any(iscat[features]) &&
        throw(ArgumentError("HybridSearch supports numeric features only; use ExactSearch() with categorical features"))
    size(search.ids, 2) == size(X, 2) ||
        throw(DimensionMismatch("prepared HybridSearch has $(size(search.ids, 2)) columns, X has $(size(X, 2))"))
    length(keep) == size(search.ids, 1) && return search
    # Boosting rounds are sequential and fitted trees retain no search state.
    # Copy into fit-owned scratch without changing the full training IDs.
    ids = workspace === nothing ? nothing : workspace.sampled_ids
    ids === nothing && (ids = Matrix{UInt16}(undef, length(keep), size(X, 2)))
    for j in features, i in eachindex(keep)
        ids[i, j] = search.ids[keep[i], j]
    end
    return PreparedHybridSearch(search.nbins, ids)
end

@inline function record_unique!(buffer::HybridBinWorkspace{T}, bin, x) where {T}
    count = buffer.counts[bin]
    count >= MIN_UNIQUE_LIN && return nothing
    for k in 1:count
        buffer.values[k, bin] == x && return nothing
    end
    count += 1
    buffer.counts[bin] = count
    buffer.values[count, bin] = x
    return nothing
end

function gather_hybrid!(st, rows, j, buffer, lo, hi)
    # Sort owned row-ID scratch, never the node's membership order.
    empty!(buffer.rowids)
    for i in rows
        lo <= st.split_search.ids[i, j] <= hi && push!(buffer.rowids, i)
    end
    sort!(buffer.rowids; by = i -> st.X[i, j], alg = QuickSort)
    m = length(buffer.rowids)
    for values in (buffer.xs, buffer.zs, buffer.hs, buffer.ws)
        ensure_len!(values, m)
    end
    for k in 1:m
        i = buffer.rowids[k]
        buffer.xs[k] = st.X[i, j]
        buffer.zs[k] = st.z[i]
        buffer.hs[k] = st.h[i]
        buffer.ws[k] = st.w[i]
    end
    return view(buffer.xs, 1:m), view(buffer.zs, 1:m),
        view(buffer.hs, 1:m), view(buffer.ws, 1:m)
end

function refine_hybrid(st::FitState{T,V}, rows, j, dmin, buffer, unsplit,
        candidates, total, n) where {T,V}
    nbins = st.split_search.nbins
    occupied = count(>(0), buffer.counts)
    if length(rows) <= nbins || (occupied == 1 && sum(buffer.counts) > 1)
        xs, zs, hs, ws = gather_hybrid!(st, rows, j, buffer, 1, nbins)
        return scan_feature(xs, zs, hs, ws, st.rule, st.min_leaf, dmin)
    end
    best = score_splits(unsplit, candidates, st.rule, n, dmin)
    isfinite(best.threshold) || return best
    # Every coarse threshold belongs to an occupied bin with occupied right rows.
    lo = something(findfirst(b -> buffer.counts[b] > 0 && buffer.maxima[b] == best.threshold, 1:nbins))
    hi = something(findnext(>(0), buffer.counts, lo + 1))
    xs, zs, hs, ws = gather_hybrid!(st, rows, j, buffer, lo, hi)
    left = zero(MomentSums{V})
    wleft = zero(T)
    prefix = 0
    suffix = 0
    for b in 1:(lo - 1)
        left += buffer.moments[b]
        wleft += buffer.masses[b]
        prefix += buffer.counts[b]
    end
    for b in (hi + 1):nbins
        suffix += buffer.counts[b]
    end
    # Whole-bin counts may stay capped, but local rows leave the right child
    # one distinct value at a time, so the window count must be exact.
    nu = prefix + nunique(xs) + suffix
    uleft = prefix
    right = total - left
    wright = n - wleft
    rule = st.rule
    dopcon = allowed(rule, PCON)
    doblin = allowed(rule, BLIN) && nu >= MIN_UNIQUE_LIN
    doplin = allowed(rule, PLIN)
    for k in eachindex(xs)
        left = addrow(left, xs[k], zs[k], hs[k])
        right = subrow(right, xs[k], zs[k], hs[k])
        wleft += ws[k]
        wright -= ws[k]
        (k == 1 || xs[k] != xs[k - 1]) && (uleft += 1)
        k == length(xs) && suffix == 0 && continue
        (k == length(xs) || xs[k] < xs[k + 1]) || continue
        (wleft >= st.min_leaf && wright >= st.min_leaf) || continue
        candidates = split_candidates(candidates, left, right, xs[k], uleft, nu - uleft,
            rule, dmin, dopcon, doblin, doplin, Val(true))
    end
    return score_splits(unsplit, candidates, rule, n, dmin)
end

function hybrid_scan_feature(st::FitState{T,V}, rows, j, dmin,
        buffer::HybridBinWorkspace{T,V}) where {T,V<:Real}
    nbins = st.split_search.nbins
    fill!(buffer.moments, zero(MomentSums{V}))
    fill!(buffer.masses, zero(T))
    fill!(buffer.counts, 0)
    fill!(buffer.maxima, typemin(T))
    # Membership rows and selected columns are validated by the fit. Preparation
    # assigns every selected entry a bin in 1:nbins, matching these buffers.
    @inbounds for i in rows
        bin = Int(st.split_search.ids[i, j])
        x = st.X[i, j]
        buffer.moments[bin] = addrow(buffer.moments[bin], x, st.z[i], st.h[i])
        buffer.masses[bin] += st.w[i]
        buffer.maxima[bin] = max(buffer.maxima[bin], x)
        record_unique!(buffer, bin, x)
    end
    buffer.suffix_counts[nbins + 1] = 0
    for bin in nbins:-1:1
        buffer.suffix_counts[bin] =
            min(MIN_UNIQUE_LIN, buffer.counts[bin] + buffer.suffix_counts[bin + 1])
    end
    total = reduce(+, buffer.moments)
    n = sum(buffer.masses)
    nu = min(MIN_UNIQUE_LIN, sum(buffer.counts))
    rule = st.rule
    nc = ncoord(V)
    unsplit = nocandidate(T, V)
    if allowed(rule, CON)
        intercept, rss = fit_con(total, rule)
        score = selection_score(rule, CON, sum(rss), n, dmin, nc)
        score < unsplit.score && (unsplit = Candidate{T,V}(
            CON, T(NaN), zero(V), intercept, zero(V), intercept, rss, score))
    end
    if allowed(rule, LIN) && nu >= MIN_UNIQUE_LIN
        fit = fit_lin(total, rule)
        if fit !== nothing
            coef, intercept, rss = fit
            score = selection_score(rule, LIN, sum(rss), n, dmin, nc)
            score < unsplit.score && (unsplit = Candidate{T,V}(
                LIN, T(NaN), coef, intercept, coef, intercept, rss, score))
        end
    end
    candidates = (nocandidate(T, V), nocandidate(T, V), nocandidate(T, V))
    dopcon = allowed(rule, PCON)
    doblin = allowed(rule, BLIN) && nu >= MIN_UNIQUE_LIN
    doplin = allowed(rule, PLIN)
    left = zero(MomentSums{V})
    right = total
    wleft = zero(T)
    wright = n
    uleft = 0
    for bin in 1:(nbins - 1)
        left += buffer.moments[bin]
        right -= buffer.moments[bin]
        wleft += buffer.masses[bin]
        wright -= buffer.masses[bin]
        uleft = min(MIN_UNIQUE_LIN, uleft + buffer.counts[bin])
        buffer.counts[bin] > 0 || continue
        # A positive rounded mass cannot supply a child without any rows.
        buffer.suffix_counts[bin + 1] > 0 || continue
        threshold = buffer.maxima[bin]
        (wleft >= st.min_leaf && wright >= st.min_leaf && isfinite(threshold)) || continue
        candidates = split_candidates(candidates, left, right, threshold, uleft,
            buffer.suffix_counts[bin + 1], rule, dmin, dopcon, doblin, doplin)
    end
    return refine_hybrid(st, rows, j, dmin, buffer, unsplit, candidates, total, n)
end

function best_split_serial(st::FitState{T,V,Y,L,R,S}, rows, span::UnitRange{Int},
        dmin, features, tid) where {T,V<:Real,Y,L<:Loss,R<:SelectionRule,S<:PreparedHybridSearch}
    buffer = st.scratch[tid].search
    best = nocandidate(T, V)
    bestj = 0
    for j in features
        candidate = hybrid_scan_feature(st, rows, j, dmin, buffer)
        if candidate.score < best.score
            best = candidate
            bestj = j
        end
    end
    return best, bestj, NO_LEVELS
end

function partition_node!(st::FitState{T,V,Y,L,R,S}, span, ids) where
        {T,V,Y,L<:Loss,R<:SelectionRule,S<:PreparedHybridSearch}
    nleft = count(k -> st.isleft[st.roworder[k]], span)
    partition_column!(st.roworder, span, st.isleft,
        ensure_len!(st.scratch[first(ids)].perm, length(span)), nleft)
    return nleft
end
