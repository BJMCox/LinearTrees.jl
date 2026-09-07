"Policy for choosing the numeric thresholds evaluated during tree fitting."
abstract type SplitSearch end

"""
    ExactSearch()

Evaluate every eligible numeric threshold. This is the default split search.
"""
struct ExactSearch <: SplitSearch end

"""
    BinnedSearch(; nbins=64, refine=true)

Approximate numeric split search for scalar-score losses. Build at most `nbins`
equal-count bins within each node, keeping tied values together. Evaluate bin
edges using sufficient statistics of the original feature values and responses.
Weights retain their frequency meaning. Categorical search remains exact.

With `refine=true`, scan all eligible row boundaries in the two bins adjacent
to the winning coarse split. No refinement runs when a constant or unsplit
line wins. Nodes with at most `nbins` rows use exact search.

This can change the chosen model and predictions. It does not guarantee the
exact optimum or an error bound. `nbins` must be at least two. Vector-score
losses, including `Softmax`, require `ExactSearch()`.
"""
struct BinnedSearch <: SplitSearch
    nbins::Int
    refine::Bool
    function BinnedSearch(; nbins::Integer = 64, refine::Bool = true)
        nbins >= 2 || throw(ArgumentError("nbins must be at least two"))
        return new(Int(nbins), refine)
    end
end

# Each buffer belongs to one task-owned scratch set, not a physical thread.
struct BinSums{T,V}
    moments::Vector{MomentSums{V}}
    edges::Vector{T}
    masses::Vector{T}
    uniques::Vector{Int}
end

search_workspace(::ExactSearch, ::Type{T}, ::Type{V}) where {T,V} = nothing
search_workspace(::BinnedSearch, ::Type{T}, ::Type{V}) where {T,V} =
    BinSums{T,V}(MomentSums{V}[], T[], T[], Int[])

@inline scan_feature(xs, zs, hs, ws, rule, min_leaf, dmin, ::ExactSearch, ::Nothing) =
    scan_feature(xs, zs, hs, ws, rule, min_leaf, dmin)

function bin_sums!(buf::BinSums{T,V}, xs, zs, hs, ws, nbins) where {T,V}
    moments, edges, masses, uniques = buf.moments, buf.edges, buf.masses, buf.uniques
    empty!(moments); empty!(edges); empty!(masses); empty!(uniques)
    sizehint!(moments, nbins); sizehint!(edges, nbins)
    sizehint!(masses, nbins); sizehint!(uniques, nbins)
    m = length(xs)
    width = cld(m, nbins)
    lo = 1
    while lo <= m
        hi = min(m, lo + width - 1)
        while hi < m && xs[hi] == xs[hi + 1]
            hi += 1
        end
        s = zero(MomentSums{V}); mass = zero(T); nu = 0
        for i in lo:hi
            s = addrow(s, xs[i], zs[i], hs[i])
            mass += ws[i]
            (i == lo || xs[i] != xs[i - 1]) && (nu += 1)
        end
        push!(moments, s); push!(edges, xs[hi]); push!(masses, mass); push!(uniques, nu)
        lo = hi + 1
    end
    return buf
end

# Score once per kind, after choosing thresholds by deviance. Reusing the
# unsplit winner keeps the fixed CON, LIN, PCON, BLIN, PLIN score-tie order.
function score_splits(best::Candidate{T,V}, candidates, rule, n, dmin) where {T,V}
    nc = ncoord(V); logn = score_logn(rule, n)
    for cand in candidates
        isfinite(cand.score) || continue
        sc = selection_score(rule, cand.kind, cand.score, n, dmin, nc, logn)
        sc < best.score && (best = Candidate{T,V}(cand.kind, cand.threshold, cand.lcoef,
            cand.lintercept, cand.rcoef, cand.rintercept, cand.surrogate, sc))
    end
    return best
end

function scan_feature(xs::AbstractVector{T}, zs::AbstractVector{V}, hs, ws,
        rule, min_leaf, dmin, search::BinnedSearch, buf::BinSums{T,V}) where {T,V<:Real}
    m = length(xs)
    m <= search.nbins && return scan_feature(xs, zs, hs, ws, rule, min_leaf, dmin)
    bin_sums!(buf, xs, zs, hs, ws, search.nbins)
    moments, edges, masses, uniques = buf.moments, buf.edges, buf.masses, buf.uniques
    total = reduce(+, moments)
    n = sum(masses); nu = sum(uniques); nc = ncoord(V)
    unsplit = nocandidate(T, V)
    if allowed(rule, CON)
        b, rss = fit_con(total, rule)
        sc = selection_score(rule, CON, sum(rss), n, dmin, nc)
        sc < unsplit.score && (unsplit = Candidate{T,V}(CON, T(NaN), zero(V), b, zero(V), b, rss, sc))
    end
    if allowed(rule, LIN) && nu >= MIN_UNIQUE_LIN
        r = fit_lin(total, rule)
        if r !== nothing
            a, b, rss = r
            sc = selection_score(rule, LIN, sum(rss), n, dmin, nc)
            sc < unsplit.score && (unsplit = Candidate{T,V}(LIN, T(NaN), a, b, a, b, rss, sc))
        end
    end
    candidates = (nocandidate(T, V), nocandidate(T, V), nocandidate(T, V))
    dopcon = allowed(rule, PCON)
    doblin = allowed(rule, BLIN) && nu >= MIN_UNIQUE_LIN
    doplin = allowed(rule, PLIN)
    left = zero(MomentSums{V}); right = total
    wleft = zero(T); wright = n; uleft = 0
    for i in 1:(length(moments) - 1)
        left += moments[i]; right -= moments[i]
        wleft += masses[i]; wright -= masses[i]; uleft += uniques[i]
        (wleft >= min_leaf && wright >= min_leaf) || continue
        candidates = split_candidates(candidates, left, right, edges[i], uleft, nu - uleft,
            rule, dmin, dopcon, doblin, doplin)
    end
    best = score_splits(unsplit, candidates, rule, n, dmin)
    (!search.refine || !isfinite(best.threshold)) && return best

    # Seed the prefix from complete bins, then refine both adjacent bins.
    # Keep coarse candidates in the per-kind reduction to settle score ties
    # by deviance, and deviance ties by the earliest visited threshold.
    boundary = searchsortedfirst(edges, best.threshold)
    lo = boundary == 1 ? 1 : searchsortedlast(xs, edges[boundary - 1]) + 1
    hi = min(m - 1, searchsortedlast(xs, edges[boundary + 1]))
    left = zero(MomentSums{V}); wleft = zero(T); uleft = 0
    for i in 1:(boundary - 1)
        left += moments[i]; wleft += masses[i]; uleft += uniques[i]
    end
    right = total - left; wright = n - wleft
    for i in lo:hi
        left = addrow(left, xs[i], zs[i], hs[i])
        right = subrow(right, xs[i], zs[i], hs[i])
        wleft += ws[i]; wright -= ws[i]
        (i == 1 || xs[i] != xs[i - 1]) && (uleft += 1)
        xs[i] < xs[i + 1] || continue
        (wleft >= min_leaf && wright >= min_leaf) || continue
        candidates = split_candidates(candidates, left, right, xs[i], uleft, nu - uleft,
            rule, dmin, dopcon, doblin, doplin, Val(true))
    end
    return score_splits(unsplit, candidates, rule, n, dmin)
end
