"""
Scores a candidate `(kind, surrogate deviance, n)`. Lower is better.
"""
abstract type SelectionRule end

"""
PILOT's BIC rule: `n log(dev/n) + dof(kind) log(n)` with `dof` per kind
`(con, lin, pcon, blin, plin)`. The defaults are the paper's `(1, 2, 5, 5, 7)`.
"""
struct BIC <: SelectionRule
    dof::NTuple{5,Float64}
end
BIC(; dof = (1.0, 2.0, 5.0, 5.0, 7.0)) = BIC(Float64.(dof))

"Test-only rule: pick the lowest surrogate deviance among `kinds`, never stop on `con`."
struct MinDeviance <: SelectionRule
    kinds::Tuple{Vararg{ModelKind}}
end

"Reserved for sub-project two."
struct GainRule <: SelectionRule
    threshold::Float64
end

allowed(::BIC, ::ModelKind) = true
function allowed(r::MinDeviance, k::ModelKind)
    # explicit loop: `in` over a Vararg tuple infers Union{Missing,Bool} and allocates in the scan
    for kk in r.kinds
        kk == k && return true
    end
    return false
end
allowed(::GainRule, ::ModelKind) = throw(ArgumentError("GainRule is reserved for boosting"))

"""
    score_logn(rule, n)

`log(n)` for `selection_score`'s penalty term. The split sweep holds one value
per feature instead of calling `log` once per candidate; the term itself stays
inside `selection_score`, so the arithmetic that produces the score, and hence
the score, is unchanged to the last bit.

Returns zero where `selection_score` returns `Inf` without reaching the
penalty (`n <= 0`, or a non-finite `n`), which also keeps `log` off a negative
argument.
"""
score_logn(::SelectionRule, n) = (isfinite(n) && n > 0) ? log(n) : zero(float(n))   # BIC and any rule with a log-n penalty
score_logn(::MinDeviance, n) = zero(float(n))
score_logn(::GainRule, n) = throw(ArgumentError("GainRule is reserved for boosting"))

"""
    devkey(rule, surrogate, dmin)

The surrogate deviance in the form `rule` compares it in. `scan_feature`
carries the lowest key per model kind through the split sweep and scores each
kind once, at the end, so every rule owes two properties:

1. its `selection_score` is strictly increasing in the key, so the lowest key
   is the lowest score;
2. `selection_score(rule, k, devkey(rule, s, dmin), n, dmin, ...)` is
   `selection_score(rule, k, s, n, dmin, ...)` to the last bit, so the key can
   be handed to the score in place of the raw deviance.

`BIC` scores `max(surrogate, dmin)`, so its key is that floored value and (2)
holds because `max` is idempotent. `MinDeviance` scores the raw deviance and
so keeps it: flooring it would make two deviances below `dmin` tie where the
rule separates them.
"""
devkey(::SelectionRule, surrogate, dmin) = max(surrogate, dmin)   # BIC and any rule with the same log floor
devkey(::MinDeviance, surrogate, dmin) = surrogate
devkey(::GainRule, surrogate, dmin) = throw(ArgumentError("GainRule is reserved for boosting"))

# `ncoord` is the number of coefficient coordinates (K-1 for softmax, 1 otherwise):
# every model kind fits `ncoord` times its scalar parameter count, so the BIC
# penalty scales with it while the deviance term already sums over coordinates.
@inline function selection_score(r::BIC, kind::ModelKind, surrogate, n, dmin, ncoord::Integer = 1,
        logn = score_logn(r, n))
    (isfinite(surrogate) && isfinite(n) && n > 0) || return Inf
    dev = max(surrogate, dmin)
    return n * log(dev / n) + ncoord * r.dof[Int(kind) + 1] * logn
end

@inline function selection_score(r::MinDeviance, kind::ModelKind, surrogate, n, dmin, ncoord::Integer = 1,
        logn = zero(float(n)))
    allowed(r, kind) || return Inf
    return isfinite(surrogate) ? Float64(surrogate) : Inf
end

selection_score(::GainRule, args...) = throw(ArgumentError("GainRule is reserved for boosting"))
