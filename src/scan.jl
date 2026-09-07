# Split-point scan over sorted feature columns.

"""
Best split candidate for one feature. `threshold` is `NaN` for `con` and `lin`.
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
end

"Number of coefficient coordinates: `K-1` for softmax, 1 otherwise."
ncoord(::Type{<:Real}) = 1
ncoord(::Type{V}) where {V<:SVector} = length(V)

"`zero(V) .+ Inf` gives `Inf` for scalar `V` and an all-`Inf` `SVector` for a vector `V`."
nocandidate(::Type{T}, ::Type{V}) where {T,V} =
    Candidate{T,V}(CON, T(NaN), zero(V), zero(V), zero(V), zero(V), zero(V) .+ Inf, Inf)

"""
PILOT offers a linear piece only where the feature has at least this many
distinct values on the rows in hand: `lin` and `blin` need it over the node,
`plin` over each child separately.
"""
const MIN_UNIQUE_LIN = 5

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
floor. `hs` holds the row hessians as a `Vector{V}` or, when every one of
them is exactly one, as `UnitHessians{V}`, which accumulates without the
multiply and gives the same sums to the bit.

`selection_score` runs once per kind, not once per split point: the sweep
carries the lowest `devkey` reached by each of `pcon`, `blin` and `plin`, and
scores those three at the end. The winner is chosen by

1. the lowest score, then
2. the fixed kind order `(con, lin, pcon, blin, plin)` on an exact score tie
   (an ordering by parameter count only for `BIC`'s default `dof`), then
3. the lowest feature index, applied by `best_split`;

and within one kind by the lowest `devkey`, then the earliest split point on
an exact key tie. Because the score is monotone but not injective in the key
(see `devkey`), split points that share one score are separated by deviance
here where scoring inside the sweep kept the earliest of them.

The `nu >= MIN_UNIQUE_LIN`, `uleft`/`uright` and `min_leaf` restrictions still
apply per split point, exactly as they would with the score inside the loop.
"""
function scan_feature(xs::AbstractVector{T}, zs::AbstractVector{V}, hs::AbstractVector,
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
    # `n` is fixed for the whole scan, so the penalty term's `log(n)` is
    # computed once here instead of once per candidate in the sweep below
    logn = score_logn(rule, n)

    # con
    if allowed(rule, CON)
        b, rss = fit_con(total, rule)
        sc = selection_score(rule, CON, sum(rss), n, dmin, nc)
        sc < best.score && (best = Candidate{T,V}(CON, T(NaN), zero(V), b, zero(V), b, rss, sc))
    end
    # lin
    if allowed(rule, LIN) && nu >= MIN_UNIQUE_LIN
        r = fit_lin(total, rule)
        if r !== nothing
            a, b, rss = r
            sc = selection_score(rule, LIN, sum(rss), n, dmin, nc)
            sc < best.score && (best = Candidate{T,V}(LIN, T(NaN), a, b, a, b, rss, sc))
        end
    end

    # splits: sweep the boundary from left to right, carrying the lowest
    # `devkey` per kind. The `score` field of these three holds that key, not
    # a score, until the sweep ends; `nocandidate`'s `Inf` is the empty state,
    # so a non-finite key is never stored and never wins.
    pcon = nocandidate(T, V); blin = nocandidate(T, V); plin = nocandidate(T, V)
    dopcon = allowed(rule, PCON)
    doblin = allowed(rule, BLIN) && nu >= MIN_UNIQUE_LIN
    doplin = allowed(rule, PLIN)
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

        pcon, blin, plin = split_candidates((pcon, blin, plin), left, right, t,
            uleft, uright, rule, dmin, dopcon, doblin, doplin)
    end

    # con and lin are already in `best`, so scoring pcon, blin and plin in that
    # order with `<` gives the kind order the docstring states. The key stands
    # in for the raw deviance here: `devkey`'s contract is that it scores the
    # same, to the bit.
    for cand in (pcon, blin, plin)
        isfinite(cand.score) || continue
        sc = selection_score(rule, cand.kind, cand.score, n, dmin, nc, logn)
        sc < best.score && (best = Candidate{T,V}(cand.kind, cand.threshold, cand.lcoef,
            cand.lintercept, cand.rcoef, cand.rintercept, cand.surrogate, sc))
    end
    return best
end

@inline better_split(key, threshold, candidate, ::Val{false}) = key < candidate.score
@inline better_split(key, threshold, candidate, ::Val{true}) = key < candidate.score ||
    (isfinite(key) && key == candidate.score && threshold < candidate.threshold)

# Keep raw deviance keys until each kind has its best threshold.
@inline function split_candidates(candidates, left, right, t, uleft, uright, rule, dmin,
        dopcon, doblin, doplin, revisit::Val = Val(false))
    pcon, blin, plin = candidates
    T = typeof(t); V = typeof(left.sw)
    if dopcon
        bl, rl = fit_con(left, rule); br, rr = fit_con(right, rule)
        rss = rl + rr
        dk = devkey(rule, sum(rss), dmin)
        better_split(dk, t, pcon, revisit) && (pcon = Candidate{T,V}(PCON, t, zero(V), bl, zero(V), br, rss, dk))
    end
    if doblin
        r = fit_blin(left, right, t)
        if r !== nothing
            al, bl, ar, br, rss = r
            dk = devkey(rule, sum(rss), dmin)
            better_split(dk, t, blin, revisit) && (blin = Candidate{T,V}(BLIN, t, al, bl, ar, br, rss, dk))
        end
    end
    if doplin && uleft >= MIN_UNIQUE_LIN && uright >= MIN_UNIQUE_LIN
        rl = fit_lin(left, rule); rr = fit_lin(right, rule)
        if rl !== nothing && rr !== nothing
            al, bl, rssl = rl; ar, br, rssr = rr
            rss = rssl + rssr
            dk = devkey(rule, sum(rss), dmin)
            better_split(dk, t, plin, revisit) && (plin = Candidate{T,V}(PLIN, t, al, bl, ar, br, rss, dk))
        end
    end
    return pcon, blin, plin
end
