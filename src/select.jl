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

"""
    GainRule(; lambda_slope = 1.0, lambda_intercept = 1.0, gamma = 0.0)

Boosting's selection rule (Guryanov 2019, eq. 12 and 13). Offers `con`,
`pcon`, and `plin`. The regularised deviance of a fit is `Σ h (z − a x − b)²
+ lambda_slope a² + lambda_intercept b²`, obtained by adding the two ridges
to the Gram sums in `fit_lin` and `fit_con`. A split's score is its
children's regularised deviance plus `gamma` per coefficient coordinate, so a
split is taken exactly when its gain over the parent's constant fit exceeds
`gamma`. `BIC`'s `dof` penalty and `dmin` floor play no part.
"""
struct GainRule <: SelectionRule
    lambda_slope::Float64
    lambda_intercept::Float64
    gamma::Float64
    function GainRule(lambda_slope::Float64, lambda_intercept::Float64, gamma::Float64)
        isfinite(lambda_slope) && lambda_slope >= 0 || throw(ArgumentError("lambda_slope must be finite and non-negative"))
        isfinite(lambda_intercept) && lambda_intercept >= 0 || throw(ArgumentError("lambda_intercept must be finite and non-negative"))
        isfinite(gamma) && gamma >= 0 || throw(ArgumentError("gamma must be finite and non-negative"))
        return new(lambda_slope, lambda_intercept, gamma)
    end
end
GainRule(lambda_slope, lambda_intercept, gamma) =
    GainRule(Float64(lambda_slope), Float64(lambda_intercept), Float64(gamma))
GainRule(; lambda_slope = 1.0, lambda_intercept = 1.0, gamma = 0.0) =
    GainRule(Float64(lambda_slope), Float64(lambda_intercept), Float64(gamma))

allowed(::BIC, ::ModelKind) = true
function allowed(r::MinDeviance, k::ModelKind)
    # explicit loop: `in` over a Vararg tuple infers Union{Missing,Bool} and allocates in the scan
    for kk in r.kinds
        kk == k && return true
    end
    return false
end
allowed(::GainRule, k::ModelKind) = k == CON || k == PCON || k == PLIN

"L2 ridges `(λ_slope, λ_intercept)` a rule adds to the closed forms. Zero for every rule but `GainRule`."
ridge(::SelectionRule) = (0.0, 0.0)
ridge(r::GainRule) = (r.lambda_slope, r.lambda_intercept)

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
# no log-n penalty, so the sweep's hoisted `logn` is zero and unused
score_logn(::GainRule, n) = zero(float(n))

"""
    devkey(rule, surrogate, dmin)

The surrogate deviance in the form `rule` compares it in. `scan_feature`
carries the lowest key per model kind through the split sweep and scores each
kind once, at the end, so every rule owes two properties:

1. its `selection_score` is **monotone non-decreasing** in the key, so the
   lowest key is a lowest score;
2. `selection_score(rule, k, devkey(rule, s, dmin), n, dmin, ...)` is
   `selection_score(rule, k, s, n, dmin, ...)` to the last bit, so the key can
   be handed to the score in place of the raw deviance.

Property (1) is not strict: `n log(dev / n)` is not injective in `Float64`, so
adjacent keys can share one score (two at `n = 108`, up to ten at
`n = 200_000`). Where several split points of one kind share a score the sweep
keeps the lowest key, the lowest surrogate deviance, while scoring inside the
sweep kept the earliest split point. Both reach the same score.

`BIC` scores `max(surrogate, dmin)`, so its key is that floored value and (2)
holds because `max` is idempotent. `MinDeviance` scores the raw deviance and
so keeps it: flooring it would make two deviances below `dmin` tie where the
rule separates them.

The generic method floors the deviance and keys a non-finite one to `Inf`, so
a `-Inf` surrogate can never win its kind, as when it was scored in the sweep.
No fit has been observed to produce `-Inf`; the guard is for a latent case.
"""
# BIC and any rule with the same log floor
@inline function devkey(::SelectionRule, surrogate, dmin)
    d = max(surrogate, dmin)
    # `max` already carries `NaN` and `+Inf` through to a key that cannot win;
    # `-Inf` is the one that would otherwise floor to `dmin` and win outright
    return isfinite(surrogate) ? d : oftype(d, Inf)
end
devkey(::MinDeviance, surrogate, dmin) = surrogate
# the score is the deviance plus a constant, so the raw deviance is already
# the key; a non-finite one is keyed to Inf so it can never win its kind,
# matching the Inf its own `selection_score` returns
@inline devkey(::GainRule, surrogate, dmin) =
    isfinite(surrogate) ? surrogate : oftype(float(surrogate), Inf)

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

@inline function selection_score(r::GainRule, kind::ModelKind, surrogate, n, dmin, ncoord::Integer = 1,
        logn = zero(float(n)))
    allowed(r, kind) || return Inf
    isfinite(surrogate) || return Inf
    return kind == CON ? Float64(surrogate) : Float64(surrogate) + r.gamma * ncoord
end

# A rule with no ridge reaches the unridged closed forms rather than adding a
# literal zero: `sw .+ 0.0` is a no-op in value but not in instruction count,
# and the BIC scan calls these twice per split point.
@inline fit_con(s::MomentSums, ::SelectionRule) = fit_con(s)
@inline fit_lin(s::MomentSums, ::SelectionRule; tol = SINGULAR_TOL) = fit_lin(s; tol)
@inline fit_con(s::MomentSums, r::GainRule) = fit_con(s, r.lambda_intercept)
@inline fit_lin(s::MomentSums, r::GainRule; tol = SINGULAR_TOL) = fit_lin(s, r.lambda_slope, r.lambda_intercept; tol)
