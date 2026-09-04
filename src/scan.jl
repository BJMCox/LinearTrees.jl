# Split-point scan over sorted feature columns.

"""
Best split candidate for one feature. `threshold` is `NaN` for `con` and
`lin`. `nleft` is the row count left of the split, `0` for `con` and `lin`.
"""
struct Candidate{T,V}
    kind::ModelKind
    threshold::T
    lcoef::V
    lintercept::V
    rcoef::V
    rintercept::V
    surrogate::V
    score::Float64
    nleft::Int
end

"Number of coefficient coordinates: `K-1` for softmax, 1 otherwise."
ncoord(::Type{<:Real}) = 1
ncoord(::Type{V}) where {V<:SVector} = length(V)

"`zero(V) .+ Inf` gives `Inf` for scalar `V` and an all-`Inf` `SVector` for a vector `V`."
nocandidate(::Type{T}, ::Type{V}) where {T,V} =
    Candidate{T,V}(CON, T(NaN), zero(V), zero(V), zero(V), zero(V), zero(V) .+ Inf, Inf, 0)

"Count of distinct values in a sorted vector."
function nunique(xs::AbstractVector)
    isempty(xs) && return 0
    c = 1
    for i in 2:length(xs)
        c += xs[i] != xs[i - 1]
    end
    return c
end

"""
Evaluate all five model kinds on one presorted feature and return the best
candidate under `rule`. Rows arrive sorted by `xs`. `ws` are frequency
weights. `min_leaf` is the least `Σ w` per child. `dmin` is the BIC log
floor.
"""
function scan_feature(xs::AbstractVector{T}, zs::AbstractVector{V}, hs::AbstractVector{V},
        ws::AbstractVector{T}, rule::SelectionRule, min_leaf, dmin) where {T<:Real,V}
    m = length(xs)
    n = sum(ws)
    total = zero(MomentSums{V})
    for i in 1:m
        total = addrow(total, xs[i], zs[i], hs[i])
    end
    best = nocandidate(T, V)
    nu = nunique(xs)
    nc = ncoord(V)

    # con
    if allowed(rule, CON)
        b, rss = fit_con(total)
        sc = selection_score(rule, CON, sum(rss), n, dmin, nc)
        sc < best.score && (best = Candidate{T,V}(CON, T(NaN), zero(V), b, zero(V), b, rss, sc, 0))
    end
    # lin
    if allowed(rule, LIN) && nu >= 5
        r = fit_lin(total)
        if r !== nothing
            a, b, rss = r
            sc = selection_score(rule, LIN, sum(rss), n, dmin, nc)
            sc < best.score && (best = Candidate{T,V}(LIN, T(NaN), a, b, a, b, rss, sc, 0))
        end
    end

    # splits: sweep the boundary from left to right
    left = zero(MomentSums{V}); right = total
    wleft = zero(T); wright = n
    uleft = 0
    for i in 1:(m - 1)
        left = addrow(left, xs[i], zs[i], hs[i])
        right = subrow(right, xs[i], zs[i], hs[i])
        wleft += ws[i]; wright -= ws[i]
        (i == 1 || xs[i] != xs[i - 1]) && (uleft += 1)
        xs[i] < xs[i + 1] || continue                  # only between distinct values
        (wleft >= min_leaf && wright >= min_leaf) || continue
        t = xs[i]                                      # PILOT parity: split point and blin knot are the largest left value
        uright = nu - uleft

        if allowed(rule, PCON)
            bl, rl = fit_con(left); br, rr = fit_con(right)
            rss = rl + rr
            sc = selection_score(rule, PCON, sum(rss), n, dmin, nc)
            sc < best.score && (best = Candidate{T,V}(PCON, t, zero(V), bl, zero(V), br, rss, sc, i))
        end
        if allowed(rule, BLIN) && nu >= 5
            r = fit_blin(left, right, t)
            if r !== nothing
                al, bl, ar, br, rss = r
                sc = selection_score(rule, BLIN, sum(rss), n, dmin, nc)
                sc < best.score && (best = Candidate{T,V}(BLIN, t, al, bl, ar, br, rss, sc, i))
            end
        end
        if allowed(rule, PLIN) && uleft >= 5 && uright >= 5
            rl = fit_lin(left); rr = fit_lin(right)
            if rl !== nothing && rr !== nothing
                al, bl, rssl = rl; ar, br, rssr = rr
                rss = rssl + rssr
                sc = selection_score(rule, PLIN, sum(rss), n, dmin, nc)
                sc < best.score && (best = Candidate{T,V}(PLIN, t, al, bl, ar, br, rss, sc, i))
            end
        end
    end
    return best
end
