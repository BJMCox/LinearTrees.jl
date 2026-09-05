using StaticArrays

"""
A twice-differentiable or IRLS-approximated loss. Implement `gradhess!`,
`linkinv`, `initscore`, `deviance`, `scorebound`, and `validate_target`.
"""
abstract type Loss end

"Floor applied to every unweighted row hessian before frequency weighting."
const HMIN = 1e-6

"Squared-error loss for a real-valued target. Identity link."
struct MSE <: Loss end

"Huber loss with transition `δ`: quadratic within `δ` of the target, linear beyond it. Identity link."
struct Huber <: Loss
    δ::Float64
    Huber(δ::Real) = (δ > 0 || throw(ArgumentError("δ must be positive")); new(Float64(δ)))
end

"Pinball (quantile) loss at quantile `τ`. Non-smooth: fit by IRLS. Identity link."
struct Quantile <: Loss
    τ::Float64
    Quantile(τ::Real) = (0 < τ < 1 || throw(ArgumentError("τ must lie in (0, 1)")); new(Float64(τ)))
end

"Mean absolute deviation loss. Non-smooth: fit by IRLS. Identity link."
struct MAD <: Loss end

"Logistic (binomial cross-entropy) loss for a `{0,1}` target. Logit link."
struct Logistic <: Loss end

"Poisson deviance loss for a non-negative integer count target. Log link."
struct Poisson <: Loss end

"Negative binomial deviance loss with dispersion `θ`, for an overdispersed count target. Log link."
struct NegBin <: Loss
    θ::Float64
    NegBin(θ::Real) = (θ > 0 || throw(ArgumentError("θ must be positive")); new(Float64(θ)))
end

"Gamma deviance loss for a positive real target. Log link."
struct Gamma <: Loss end

"Tweedie deviance loss with power `ρ ∈ (1, 2)`, for a non-negative target with a point mass at zero. Log link."
struct Tweedie <: Loss
    ρ::Float64
    Tweedie(ρ::Real) = (1 < ρ < 2 || throw(ArgumentError("ρ must lie in (1, 2)")); new(Float64(ρ)))
end

"""
    Softmax(K)

K-class softmax with class K as the reference at logit zero. Diagonal Hessian.
`K` is a type parameter so the coefficient type `SVector{K-1,T}` is known at
compile time and every fit specialises on it.
"""
struct Softmax{K} <: Loss
    function Softmax{K}() where {K}
        K isa Int && K >= 2 || throw(ArgumentError("K must be an integer >= 2, got $K"))
        return new{K}()
    end
end
Softmax(K::Integer) = Softmax{Int(K)}()

"Number of classes of a `Softmax` loss."
nclasses(::Softmax{K}) where {K} = K

"""
    issmooth(loss)

`true` when `loss` has a well-defined Hessian everywhere and fits by Newton
steps; `false` for the L1-type losses (`Quantile`, `MAD`), which fit by IRLS.
"""
issmooth(::Loss) = true
issmooth(::Union{Quantile,MAD}) = false

"""
    coeftype(loss, T)

Coefficient type for `loss` given feature type `T`. `T` for every scalar
loss, `SVector{K-1,T}` for `Softmax(K)`.
"""
coeftype(::Loss, ::Type{T}) where {T} = T
coeftype(::Softmax{K}, ::Type{T}) where {K,T} = SVector{K - 1,T}

# ---- links -----------------------------------------------------------------
"""
    linkinv(loss, s)

Map raw score `s` to the response scale: the identity for `MSE`, `Huber`,
`Quantile`, and `MAD`, the logistic sigmoid for `Logistic`, `exp` for the
count and rate losses, and class probabilities for `Softmax`.
"""
linkinv(::Union{MSE,Huber,Quantile,MAD}, s) = s
linkinv(::Logistic, s) = 1 / (1 + exp(-s))
linkinv(::Union{Poisson,NegBin,Gamma,Tweedie}, s) = exp(s)

"""
    probs(loss, s)

`K` class probabilities from the `K-1` reference-class logits `s`, with
class `K` fixed at logit zero. Shifts by `max(maximum(s), 0)` for overflow
safety without changing the result.
"""
function probs(l::Softmax, s::SVector{Km1,T}) where {Km1,T}
    m = max(maximum(s), zero(T))
    e = exp.(s .- m); e0 = exp(-m)
    z = sum(e) + e0
    return vcat(e ./ z, SVector{1,T}(e0 / z))
end
linkinv(l::Softmax, s::SVector) = probs(l, s)

# ---- pointwise gradient and hessian in the score --------------------------
@inline gh(::MSE, y, f) = (f - y, one(f))
@inline function gh(l::Huber, y, f)
    r = float(f - y)
    # `l.δ` is always `Float64`; `oftype(r, ...)` keeps both branches at `r`'s type
    abs(r) <= l.δ ? (r, one(r)) : (copysign(oftype(r, l.δ), r), zero(r))
end
@inline gh(l::Quantile, y, f) = (y < f ? oftype(f, 1 - l.τ) : y > f ? oftype(f, -l.τ) : zero(f), zero(f))
@inline gh(::MAD, y, f) = (sign(f - y), zero(f))
@inline function gh(::Logistic, y, f)
    p = 1 / (1 + exp(-f))
    (p - y, p * (1 - p))
end
@inline gh(::Poisson, y, f) = (μ = exp(f); (μ - y, μ))
@inline gh(::Gamma, y, f) = (μ = exp(f); (1 - y / μ, y / μ))
@inline function gh(l::Tweedie, y, f)
    # `l.ρ` is always `Float64`; `^` between a narrower base and a `Float64`
    # exponent promotes to `Float64`, so match `ρ` to `f`'s type first
    μ = exp(f); ρ = oftype(f, l.ρ)
    (μ^(2 - ρ) - y * μ^(1 - ρ), (2 - ρ) * μ^(2 - ρ) - (1 - ρ) * y * μ^(1 - ρ))
end
@inline function gh(l::NegBin, y, f)
    μ = exp(f); θ = oftype(f, l.θ)   # `l.θ` is always `Float64`
    (θ * (μ - y) / (μ + θ), θ * μ * (θ + y) / (μ + θ)^2)
end

"""
    unit_hessian(loss)

True when every row hessian this loss produces is exactly one before frequency
weighting. Only `MSE` qualifies: `gh(::MSE, y, f)` returns `one(f)` and the
`HMIN` floor leaves it alone, so with unit weights the whole `h` vector is
ones and the scan can take an accumulation path with no multiply in it. Any
loss that does not implement this keeps the general path.
"""
unit_hessian(::Loss) = false
unit_hessian(::MSE) = true

"""
    gradhess!(g, h, loss, y, f)

Unweighted `g = ∂ℓ/∂f` and `h = max(∂²ℓ/∂f², HMIN)` per row. Frequency
weights are applied by the caller after the floor.
"""
function gradhess!(g::AbstractVector, h::AbstractVector, loss::Loss, y::AbstractVector, f::AbstractVector)
    for i in eachindex(g, h, y, f)
        gi, hi = gh(loss, y[i], f[i])
        g[i] = gi
        h[i] = max(hi, oftype(hi, HMIN))
    end
    return g
end

"""
    gradhess!(g, h, ::Softmax, y, f)

Per-row cross-entropy gradient and diagonal Hessian against the `K-1`
reference-class logits, with integer class labels `y` in `1:K`.
"""
function gradhess!(g::AbstractVector{V}, h::AbstractVector{V}, l::Softmax, y::AbstractVector, f::AbstractVector{V}) where {Km1,T,V<:SVector{Km1,T}}
    for i in eachindex(g, h, y, f)
        p = probs(l, f[i])
        pk = SVector{Km1,T}(ntuple(k -> p[k], Km1))
        onehot = SVector{Km1,T}(ntuple(k -> T(y[i] == k), Km1))
        g[i] = pk .- onehot
        h[i] = max.(pk .* (1 .- pk), T(HMIN))
    end
    return g
end

"""
    irls_weights!(h, loss, y, f; ε)

IRLS pseudo-hessian for non-smooth losses: `l1weight(loss, r) / max(|r|, ε)`,
with a positive scale-aware floor `ε`. `ε` is required: every caller already
holds the node's residual scale, and recomputing it here would mean a second
weighted median over `y - f`. Uses `l1weight` rather than `gh`'s
gradient magnitude: at `r == 0` `gh` reports an exact-zero gradient (the
boundary of its case split), which would zero out the very row sitting at
the current fit and bias the step; `l1weight` gives that row its correct
one-sided weight (`τ` or `1-τ`) instead.
"""
function irls_weights!(h::AbstractVector{T}, loss::Union{Quantile,MAD}, y::AbstractVector, f::AbstractVector;
        ε) where {T}
    for i in eachindex(h, y, f)
        r = y[i] - f[i]
        h[i] = max(l1weight(loss, r) / max(abs(r), ε), T(HMIN))
    end
    return h
end

"""
    wquantile(y, w, τ) -> eltype(y)

Weighted `τ`-quantile of `y` by frequency weights `w`, ignoring rows with
`w[i] == 0`: the value the cumulative weight would cross `τ · Σw` at, in
ascending order. When it lands exactly on that target, return the mean of the
two adjacent order statistics, the tie a plain `quantile` takes on `y`
duplicated `w[i]` times per row. Integer weights therefore equal row
duplication, and `τ = 0.5` matches `Statistics.median` of the duplicated
sample -- which has zero copies of a zero-weight row's value, hence excluding
it rather than walking past it at zero weight. Throws `ArgumentError` when
`Σw` is not positive. Computed by `wquantile_select!` in expected
`O(length(y))` time, not by a full sort.
"""
wquantile(y, w, τ) = wquantile_select!(y, w, collect(eachindex(y)), τ)

"""
    wquantile_select!(y, w, o, τ) -> eltype(y)

`wquantile`'s value found by quickselect instead of a full sort: `o` (any
starting order, holding the indices `eachindex(y)`) is mutated in place by
partitioning around a value, never fully sorted, so this runs in expected
`O(length(o))` time. Same contract as a sort-then-walk: the exact-boundary
rule (`cum == target` returns the mean of the two adjacent order statistics,
an equal-valued run counted as one order statistic) and the
positive-total-weight `ArgumentError`. Pivot is median-of-three, not random:
these buffers are per-task scratch reused while sibling subtrees grow
concurrently, and a shared RNG would race across them.

A zero-weight row contributes no copies to the duplicated sample `wquantile`
means to match, so it is moved out of `o`'s active range up front. Walking it
at zero weight instead would make the exact-boundary rule depend on sort order:
with `y = [1, 2, 3]`, `w = [1, 0, 1]`, `τ = 0.5` the cumulative weight after
row 1 already hits half the total, and a walk that averages with whatever row
sorts next returns `1.5`, while the duplicated sample `[1, 3]` has median
`2.0`. With integer weights that boundary is hit whenever the total weight is
even, so the case is common. Once zero-weight rows are excluded every
remaining run has positive weight, the boundary falls between runs, never
inside one, and no row's exclusion depends on where in `o` it sits.
"""
function wquantile_select!(y::AbstractVector, w::AbstractVector, o::AbstractVector{<:Integer}, τ)
    n = length(o)
    total = sum(w)
    total > 0 || throw(ArgumentError("weights must have a positive sum"))
    target = τ * total

    # partition o[1:n] into positive-weight rows (kept, o[1:m]) and zero-weight
    # rows (dropped): a positive total weight guarantees m >= 1
    m = 0
    for k in 1:n
        if w[o[k]] > 0
            m += 1
            o[k], o[m] = o[m], o[k]
        end
    end

    lo, hi = 1, m
    base = zero(target)
    hasbound = false
    bound = zero(eltype(y))

    while lo < hi
        mid = (lo + hi) >>> 1
        v1, v2, v3 = y[o[lo]], y[o[mid]], y[o[hi]]
        pv = v1 <= v2 ? (v2 <= v3 ? v2 : max(v1, v3)) : (v1 <= v3 ? v1 : max(v2, v3))

        # 3-way (Dutch-flag) partition of o[lo:hi]: < pv, == pv, > pv, so a run
        # of duplicate values is grouped and skipped in one step, not recursed
        i = lo; lt = lo; gt = hi
        while i <= gt
            vi = y[o[i]]
            if vi < pv
                o[i], o[lt] = o[lt], o[i]
                lt += 1; i += 1
            elseif vi > pv
                o[i], o[gt] = o[gt], o[i]
                gt -= 1
            else
                i += 1
            end
        end

        WL = zero(target)
        for k in lo:(lt - 1)
            WL += w[o[k]]
        end
        WE = zero(target)
        for k in lt:gt
            WE += w[o[k]]
        end

        if base + WL > target
            hi = lt - 1
            hasbound = true
            bound = pv
        elseif base + WL == target
            # seeding with o[lo], a `== pv` element, is only reached when `lt == lo`
            # (the `< pv` group is empty), which needs base == target == 0, i.e. τ = 0;
            # there pv is the range minimum, so (pv + pv) / 2 == pv is still correct
            mx = y[o[lo]]
            for k in (lo + 1):(lt - 1)
                mx = max(mx, y[o[k]])
            end
            return (mx + pv) / 2
        elseif base + WL + WE > target
            return pv
        elseif base + WL + WE == target
            if gt < hi
                nxt = y[o[gt + 1]]
                for k in (gt + 2):hi
                    nxt = min(nxt, y[o[k]])
                end
                return (pv + nxt) / 2
            else
                return hasbound ? (pv + bound) / 2 : pv
            end
        else
            base += WL + WE
            lo = gt + 1
        end
    end

    i = o[lo]
    cum = base + w[i]
    cum == target && return hasbound ? (y[i] + bound) / 2 : y[i]
    return y[i]   # cum > target, guaranteed once total > 0
end

"""
    median_abs(r, w) -> eltype(r)
    median_abs!(buf, perm, r, w)

Weighted median of `abs.(r)` by `w`, the residual scale IRLS floors its ε on.
The two-argument form allocates; `median_abs!` writes `abs.(r)` into the
first `length(r)` slots of `buf` and an index buffer into `perm` (an `Int32`
buffer, partitioned in place by `wquantile_select!` rather than sorted), so
`irls_refit` can run it on per-worker scratch every pass without allocating.
"""
median_abs(r::AbstractVector, w::AbstractVector) = (a = abs.(r); wquantile_select!(a, w, collect(eachindex(a)), 0.5))

function median_abs!(buf::Vector{T}, perm::Vector{Int32}, r::AbstractVector{T}, w::AbstractVector) where {T<:AbstractFloat}
    n = length(r)
    a = view(buf, 1:n)
    a .= abs.(r)
    o = view(perm, 1:n)
    o .= 1:n
    return wquantile_select!(a, w, o, 0.5)
end

"""
    irls_epsilon(r, w)
    irls_epsilon!(buf, perm, r, w)

`1e-3` times the weighted median of `|r|` by `w`: the residual-scale half of
the ε floor IRLS uses everywhere it re-solves the pseudo-hessian for a
non-smooth loss. Written once here rather than copied at each call site
(`node_epsilon`, `irls_refit`); each caller still adds its own `sqrt(eps(T))`
floor scaled by `y`'s own magnitude.
"""
irls_epsilon(r::AbstractVector{T}, w::AbstractVector) where {T} = T(1e-3) * median_abs(r, w)
irls_epsilon!(buf::Vector{T}, perm::Vector{Int32}, r::AbstractVector{T}, w::AbstractVector) where {T} =
    T(1e-3) * median_abs!(buf, perm, r, w)

"""
    l1weight(loss, r)

Majorizer weight on residual `r` for the L1-type losses: flat for MAD,
`τ`/`1-τ` for the pinball loss above/below zero.
"""
l1weight(::MAD, r) = one(r)
l1weight(l::Quantile, r) = r >= 0 ? oftype(r, l.τ) : oftype(r, 1 - l.τ)

"""
    refit_node(st, n, rows, tid, masks=UInt64[]) -> Node

IRLS refinement of a non-smooth node's coefficients on its own rows, including
`CON` leaves. Runs `st.niter` iterations (the `niter` keyword on `fit_tree`);
each recomputes the pseudo-hessian at the current node prediction and
re-solves the chosen model kind. Smooth losses return `n` unchanged. Takes and
returns a `Node` value rather than a tree index, so it works the same on a
node still local to a growing subtree. `tid` selects the caller's own worker
scratch (`st.scratch[tid]`), reused as refit buffer storage.
"""
refit_node(st, n, rows, tid, masks = UInt64[]) = issmooth(st.loss) ? n : irls_refit(st, n, rows, tid, st.niter, masks)

# ---- init score ------------------------------------------------------------
wmean(y, w) = sum(w .* y) / sum(w)


initscore(::Union{MSE,Huber}, y, w) = wmean(y, w)
initscore(l::Quantile, y, w) = wquantile(y, w, l.τ)
initscore(::MAD, y, w) = wquantile(y, w, 0.5)
function initscore(::Logistic, y, w)
    m = wmean(y, w)
    lo = 1e-6
    p, q = m <= lo ? (lo, 1 - lo) : m >= 1 - lo ? (1 - lo, lo) : (m, 1 - m)
    return log(p / q)
end
initscore(::Union{Poisson,NegBin,Tweedie}, y, w) = log(max(wmean(y, w), 1e-6))
initscore(::Gamma, y, w) = log(wmean(y, w))

"""
    initscore(::Softmax, y, w)

Log-odds of each non-reference class against the reference class `K`,
from the weighted class frequencies.
"""
function initscore(l::Softmax{K}, y, w) where {K}
    T = float(eltype(w))
    tot = sum(w)
    pk = ntuple(k -> clamp(sum(w[i] for i in eachindex(y) if y[i] == k; init = zero(T)) / tot, 1e-6, 1.0), K)
    ref = log(pk[K])
    return SVector{K - 1,T}(ntuple(k -> log(pk[k]) - ref, K - 1))
end

# ---- true deviance for reporting ------------------------------------------
pointloss(::MSE, y, f) = (y - f)^2 / 2
pointloss(l::Huber, y, f) = (r = y - f; abs(r) <= l.δ ? r^2 / 2 : l.δ * (abs(r) - l.δ / 2))
pointloss(l::Quantile, y, f) = (r = y - f; r >= 0 ? l.τ * r : (l.τ - 1) * r)
pointloss(::MAD, y, f) = abs(y - f)
# stable softplus: log1p(exp(f)) overflows to Inf past f = 709, where the true value is ≈ f
pointloss(::Logistic, y, f) = (f > 0 ? f + log1p(exp(-f)) : log1p(exp(f))) - y * f
pointloss(::Poisson, y, f) = exp(f) - y * f
pointloss(::Gamma, y, f) = y * exp(-f) + f
pointloss(l::Tweedie, y, f) = (μ = exp(f); ρ = l.ρ; -y * μ^(1 - ρ) / (1 - ρ) + μ^(2 - ρ) / (2 - ρ))
pointloss(l::NegBin, y, f) = (μ = exp(f); θ = l.θ; -y * log(μ / (μ + θ)) + θ * log1p(μ / θ))

pointloss(l::Softmax, y, f::SVector) = -log(probs(l, f)[Int(y)])

"""
    deviance(loss, y, f, w)

Weighted `2 Σ w[i] pointloss(loss, y[i], f[i])`, the model's reported deviance
on the score scale `f`.
"""
deviance(loss::Loss, y, f, w) = 2 * sum(w[i] * pointloss(loss, y[i], f[i]) for i in eachindex(y))

# ---- score bounds ----------------------------------------------------------
"""
    scorebound(loss, y; truncation_factor=3)

`(lo, hi)` score-scale clamp bounds fit to the training target `y`, used when
`fit_tree`'s `truncate` is set. With half-width `B = (max(y) - min(y)) / 2`,
`truncation_factor` pads each side by `(truncation_factor - 1) * B`, so the
bounds are `[min(y) - (truncation_factor - 1) B, max(y) + (truncation_factor - 1) B]`.
"""
function scorebound(::Union{MSE,Huber,Quantile,MAD}, y; truncation_factor = 3)
    lo, hi = extrema(y)
    B = (hi - lo) / 2
    pad = (truncation_factor - 1) * B
    return (lo - pad, hi + pad)
end
scorebound(::Logistic, y; truncation_factor = 3) = (-10.0, 10.0)
function scorebound(::Union{Poisson,NegBin,Gamma,Tweedie}, y; truncation_factor = 3)
    S = log(max(maximum(y), 1)) + 3
    return (-S, S)
end
function scorebound(::Softmax{K}, y; truncation_factor = 3) where {K}
    T = float(eltype(y))
    return (fill(T(-10), SVector{K - 1}), fill(T(10), SVector{K - 1}))
end

# ---- target validation -----------------------------------------------------
"""
    validate_target(loss, y)

Throw `ArgumentError` if `y` is not finite everywhere or does not satisfy
`loss`'s domain (e.g. `{0,1}` for `Logistic`, non-negative integers for
`Poisson`); otherwise return `nothing`.
"""
function validate_target(loss::Loss, y)
    all(isfinite, y) || throw(ArgumentError("target contains NaN or Inf"))
    _validate(loss, y)
    return nothing
end
_validate(::Union{MSE,Huber,Quantile,MAD}, y) = nothing
_validate(::Logistic, y) = all(v -> v == 0 || v == 1, y) || throw(ArgumentError("Logistic needs targets in {0, 1}"))
_validate(::Union{Poisson,NegBin}, y) = all(v -> v >= 0 && isinteger(v), y) || throw(ArgumentError("count losses need non-negative integers"))
_validate(::Gamma, y) = all(>(0), y) || throw(ArgumentError("Gamma needs positive targets"))
_validate(::Tweedie, y) = all(>=(0), y) || throw(ArgumentError("Tweedie needs non-negative targets"))
function _validate(::Softmax{K}, y) where {K}
    all(v -> isinteger(v) && 1 <= v <= K, y) || throw(ArgumentError("Softmax($K) needs integer targets in 1:$K"))
    all(k -> any(==(k), y), 1:K) || throw(ArgumentError("every class in 1:$K must be present"))
end

# ---- LossFunctions.jl adapter -----------------------------------------------
using LossFunctions: SupervisedLoss, DistanceLoss, MarginLoss, L2DistLoss, L1DistLoss, QuantileLoss, PoissonLoss, deriv, deriv2

"Score-space link between the tree's raw score and the value the inner loss expects."
struct IdentityLink end

"Score-space link marking the tree's raw score as a logit, for a margin loss."
struct LogitLink end

"Score-space link marking the tree's raw score as a log mean, for `PoissonLoss`."
struct LogLink end

"""
Adapter around a `LossFunctions.SupervisedLoss`. Margin losses receive
targets in `{0,1}` and are re-expressed with `t = 2y - 1`. The pointwise loss,
gradient, and hessian are multiplied by `scale`.
"""
struct AdaptedLoss{L<:SupervisedLoss,K} <: Loss
    inner::L
    link::K
    scale::Float64
end

"Newton-step scale that lines an inner loss up with the matching native `Loss`."
canonical_scale(::L2DistLoss) = 0.5
canonical_scale(::SupervisedLoss) = 1.0

"""
    Loss(l::LossFunctions.SupervisedLoss, link = IdentityLink(); scale = canonical_scale(l))

Wrap a `LossFunctions.SupervisedLoss` as a `Loss` for `fit_tree`. `link`
declares the scale `l`'s `output` argument already lives on, and is used only
to pick `linkinv`, `initscore`, and `scorebound`; it never enters `gh` or
`pointloss` and no chain rule is applied. Allowed pairings: a `DistanceLoss`
with `IdentityLink` (output on the response scale), a `MarginLoss` with
`LogitLink` or `IdentityLink` (output is the logit), and `PoissonLoss` with
`LogLink` (output is the log mean).
"""
function Loss(l::SupervisedLoss, link = IdentityLink(); scale = canonical_scale(l))
    _check_link(l, link)
    return AdaptedLoss(l, link, Float64(scale))
end

_check_link(::DistanceLoss, ::IdentityLink) = nothing
_check_link(::MarginLoss, ::Union{LogitLink,IdentityLink}) = nothing
_check_link(::PoissonLoss, ::LogLink) = nothing
_check_link(l::SupervisedLoss, link) = throw(ArgumentError(
    "Loss($(typeof(l)), $(typeof(link))) is not a supported pairing: " *
    "DistanceLoss needs IdentityLink, MarginLoss needs LogitLink or IdentityLink, PoissonLoss needs LogLink"))

issmooth(a::AdaptedLoss) = !(a.inner isa L1DistLoss || a.inner isa QuantileLoss)

linkinv(::AdaptedLoss{<:Any,IdentityLink}, s) = s
linkinv(::AdaptedLoss{<:Any,LogitLink}, s) = 1 / (1 + exp(-s))
linkinv(::AdaptedLoss{<:Any,LogLink}, s) = exp(s)

_target(::DistanceLoss, y) = y
_target(::MarginLoss, y) = 2y - 1
_target(::PoissonLoss, y) = y

# `QuantileLoss`'s own `deriv` returns the one-sided `-τ` at an exact tie
# (f == t); native `Quantile`'s hand-written `gh` returns 0 there instead
# (loss.jl's `gh(::Quantile, ...)`), and `initscore` for the quantile adapter
# is a data quantile, so ties happen on the very first refresh at the root.
# Match the native boundary value here so the two agree bit-for-bit.
_deriv(l::QuantileLoss, f, t) = f == t ? zero(f) : deriv(l, f, t)
_deriv(l, f, t) = deriv(l, f, t)

# `deriv`/`deriv2` take (loss, output, target); a distance loss reads target
# as-is, a margin loss reads it as ±1. Both conventions checked against
# LossFunctions' own definitions and against a finite-difference probe.
@inline function gh(a::AdaptedLoss, y, f)
    t = _target(a.inner, y)
    return a.scale * _deriv(a.inner, f, t), a.scale * deriv2(a.inner, f, t)
end
pointloss(a::AdaptedLoss, y, f) = a.scale * a.inner(f, _target(a.inner, y))

_irls_inner(a::AdaptedLoss) = a.inner isa L1DistLoss ? MAD() : Quantile(a.inner.τ)

function irls_weights!(h::AbstractVector{T}, a::AdaptedLoss, y::AbstractVector, f::AbstractVector; ε) where {T}
    irls_weights!(h, _irls_inner(a), y, f; ε)
    h .*= a.scale
    return h
end

l1weight(a::AdaptedLoss, r) = l1weight(_irls_inner(a), r)

initscore(a::AdaptedLoss{<:Any,IdentityLink}, y, w) = a.inner isa QuantileLoss ? wquantile(y, w, a.inner.τ) :
    a.inner isa L1DistLoss ? wquantile(y, w, 0.5) : wmean(y, w)
initscore(::AdaptedLoss{<:Any,LogitLink}, y, w) = initscore(Logistic(), y, w)
initscore(::AdaptedLoss{<:Any,LogLink}, y, w) = initscore(Poisson(), y, w)

scorebound(::AdaptedLoss{<:Any,IdentityLink}, y; truncation_factor = 3) = scorebound(MSE(), y; truncation_factor)
scorebound(::AdaptedLoss{<:Any,LogitLink}, y; kw...) = scorebound(Logistic(), y; kw...)
scorebound(::AdaptedLoss{<:Any,LogLink}, y; kw...) = scorebound(Poisson(), y; kw...)

_validate(::AdaptedLoss{<:DistanceLoss}, y) = nothing
_validate(::AdaptedLoss{<:MarginLoss}, y) = _validate(Logistic(), y)
_validate(::AdaptedLoss{<:PoissonLoss}, y) = _validate(Poisson(), y)
