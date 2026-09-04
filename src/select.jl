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
allowed(::GainRule, ::ModelKind) = error("GainRule is reserved for boosting")

@inline function selection_score(r::BIC, kind::ModelKind, surrogate, n, dmin)
    (isfinite(surrogate) && isfinite(n) && n > 0) || return Inf
    dev = max(surrogate, dmin)
    return n * log(dev / n) + r.dof[Int(kind) + 1] * log(n)
end

@inline function selection_score(r::MinDeviance, kind::ModelKind, surrogate, n, dmin)
    allowed(r, kind) || return Inf
    return isfinite(surrogate) ? Float64(surrogate) : Inf
end

selection_score(::GainRule, args...) = error("GainRule is reserved for boosting")
