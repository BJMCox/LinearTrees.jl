using StaticArrays

"""
A twice-differentiable or IRLS-approximated loss. Implement `gradhess!`,
`linkinv`, `initscore`, `deviance`, `scorebound`, and `validate_target`.
"""
abstract type Loss end

"Floor applied to every unweighted row hessian before frequency weighting."
const HMIN = 1e-6

struct MSE <: Loss end
struct Huber <: Loss
    δ::Float64
    Huber(δ::Real) = (δ > 0 || throw(ArgumentError("δ must be positive")); new(Float64(δ)))
end
struct Quantile <: Loss
    τ::Float64
    Quantile(τ::Real) = (0 < τ < 1 || throw(ArgumentError("τ must lie in (0, 1)")); new(Float64(τ)))
end
struct MAD <: Loss end
struct Logistic <: Loss end
struct Poisson <: Loss end
struct NegBin <: Loss
    θ::Float64
    NegBin(θ::Real) = (θ > 0 || throw(ArgumentError("θ must be positive")); new(Float64(θ)))
end
struct Gamma <: Loss end
struct Tweedie <: Loss
    ρ::Float64
    Tweedie(ρ::Real) = (1 < ρ < 2 || throw(ArgumentError("ρ must lie in (1, 2)")); new(Float64(ρ)))
end

"K-class softmax with class K as the reference at logit zero. Diagonal Hessian."
struct Softmax <: Loss
    K::Int
    Softmax(K) = (K >= 2 || throw(ArgumentError("K must be at least 2")); new(Int(K)))
end

issmooth(::Loss) = true
issmooth(::Union{Quantile,MAD}) = false

"""
    coeftype(loss, T)

Coefficient type for `loss` given feature type `T`. `T` for every scalar
loss, `SVector{K-1,T}` for `Softmax(K)`.
"""
coeftype(::Loss, ::Type{T}) where {T} = T
coeftype(l::Softmax, ::Type{T}) where {T} = SVector{l.K - 1,T}

# ---- links -----------------------------------------------------------------
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
    abs(r) <= l.δ ? (r, one(r)) : (copysign(l.δ, r), zero(r))
end
@inline gh(l::Quantile, y, f) = (y < f ? 1 - l.τ : y > f ? -l.τ : zero(f), zero(f))
@inline gh(::MAD, y, f) = (sign(f - y), zero(f))
@inline function gh(::Logistic, y, f)
    p = 1 / (1 + exp(-f))
    (p - y, p * (1 - p))
end
@inline gh(::Poisson, y, f) = (μ = exp(f); (μ - y, μ))
@inline gh(::Gamma, y, f) = (μ = exp(f); (1 - y / μ, y / μ))
@inline function gh(l::Tweedie, y, f)
    μ = exp(f); ρ = l.ρ
    (μ^(2 - ρ) - y * μ^(1 - ρ), (2 - ρ) * μ^(2 - ρ) - (1 - ρ) * y * μ^(1 - ρ))
end
@inline function gh(l::NegBin, y, f)
    μ = exp(f); θ = l.θ
    (θ * (μ - y) / (μ + θ), θ * μ * (θ + y) / (μ + θ)^2)
end

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
    irls_weights!(h, loss, y, f)

IRLS pseudo-hessian for non-smooth losses: `l1weight(loss, r) / max(|r|, ε)`,
with a positive scale-aware floor `ε`. Uses `l1weight` rather than `gh`'s
gradient magnitude: at `r == 0` `gh` reports an exact-zero gradient (the
boundary of its case split), which would zero out the very row sitting at
the current fit and bias the step; `l1weight` gives that row its correct
one-sided weight (`τ` or `1-τ`) instead.
"""
function irls_weights!(h::AbstractVector{T}, loss::Union{Quantile,MAD}, y::AbstractVector, f::AbstractVector;
        ε = max(T(1e-3) * median_abs(y .- f), sqrt(eps(T)) * max(maximum(abs, y), one(T)))) where {T}
    r = y .- f
    for i in eachindex(h)
        h[i] = max(l1weight(loss, r[i]) / max(abs(r[i]), ε), T(HMIN))
    end
    return h
end

median_abs(r) = (a = sort(abs.(r)); m = length(a); isodd(m) ? a[(m + 1) ÷ 2] : (a[m ÷ 2] + a[m ÷ 2 + 1]) / 2)

"""
    l1weight(loss, r)

Majorizer weight on residual `r` for the L1-type losses: flat for MAD,
`τ`/`1-τ` for the pinball loss above/below zero.
"""
l1weight(::MAD, r) = one(r)
l1weight(l::Quantile, r) = r >= 0 ? oftype(r, l.τ) : oftype(r, 1 - l.τ)

"""
    refit_node(st, n, rows, masks=UInt64[], niter=5) -> Node

IRLS refinement of a non-smooth node's coefficients on its own rows, including
`CON` leaves. Each iteration recomputes the pseudo-hessian at the current node
prediction and re-solves the chosen model kind. Smooth losses return `n`
unchanged. Takes and returns a `Node` value rather than a tree index, so it
works the same on a node still local to a growing subtree.
"""
refit_node(st, n, rows, masks = UInt64[], niter = 5) = issmooth(st.loss) ? n : irls_refit(st, n, rows, niter, masks)

# ---- init score ------------------------------------------------------------
wmean(y, w) = sum(w .* y) / sum(w)

function wquantile(y, w, τ)
    o = sortperm(y)
    cum = zero(eltype(w)); tot = sum(w)
    for i in o
        cum += w[i]
        cum >= τ * tot && return y[i]
    end
    return y[o[end]]
end

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
function initscore(l::Softmax, y, w)
    T = float(eltype(w))
    tot = sum(w)
    pk = ntuple(k -> clamp(sum(w[i] for i in eachindex(y) if y[i] == k; init = zero(T)) / tot, 1e-6, 1.0), l.K)
    ref = log(pk[l.K])
    return SVector{l.K - 1,T}(ntuple(k -> log(pk[k]) - ref, l.K - 1))
end

# ---- true deviance for reporting ------------------------------------------
pointloss(::MSE, y, f) = (y - f)^2 / 2
pointloss(l::Huber, y, f) = (r = y - f; abs(r) <= l.δ ? r^2 / 2 : l.δ * (abs(r) - l.δ / 2))
pointloss(l::Quantile, y, f) = (r = y - f; r >= 0 ? l.τ * r : (l.τ - 1) * r)
pointloss(::MAD, y, f) = abs(y - f)
pointloss(::Logistic, y, f) = log1p(exp(f)) - y * f
pointloss(::Poisson, y, f) = exp(f) - y * f
pointloss(::Gamma, y, f) = y * exp(-f) + f
pointloss(l::Tweedie, y, f) = (μ = exp(f); ρ = l.ρ; -y * μ^(1 - ρ) / (1 - ρ) + μ^(2 - ρ) / (2 - ρ))
pointloss(l::NegBin, y, f) = (μ = exp(f); θ = l.θ; -y * log(μ / (μ + θ)) + θ * log1p(μ / θ))

pointloss(l::Softmax, y, f::SVector) = -log(probs(l, f)[Int(y)])

deviance(loss::Loss, y, f, w) = 2 * sum(w[i] * pointloss(loss, y[i], f[i]) for i in eachindex(y))

# ---- score bounds ----------------------------------------------------------
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
function scorebound(l::Softmax, y; truncation_factor = 3)
    T = float(eltype(y))
    return (fill(T(-10), SVector{l.K - 1}), fill(T(10), SVector{l.K - 1}))
end

# ---- target validation -----------------------------------------------------
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
function _validate(l::Softmax, y)
    all(v -> isinteger(v) && 1 <= v <= l.K, y) || throw(ArgumentError("Softmax($(l.K)) needs integer targets in 1:$(l.K)"))
    all(k -> any(==(k), y), 1:l.K) || throw(ArgumentError("every class in 1:$(l.K) must be present"))
end

# ---- LossFunctions.jl adapter -----------------------------------------------
using LossFunctions: SupervisedLoss, DistanceLoss, MarginLoss, L2DistLoss, L1DistLoss, QuantileLoss, deriv, deriv2

"Score-space link between the tree's raw score and the value the inner loss expects."
struct IdentityLink end
struct LogitLink end
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

Wrap a `LossFunctions.SupervisedLoss` as a `Loss` for `fit_tree`. `link` maps
the tree's raw score to the value `l` is defined on.
"""
Loss(l::SupervisedLoss, link = IdentityLink(); scale = canonical_scale(l)) = AdaptedLoss(l, link, Float64(scale))

issmooth(a::AdaptedLoss) = !(a.inner isa L1DistLoss || a.inner isa QuantileLoss)

linkinv(::AdaptedLoss{<:Any,IdentityLink}, s) = s
linkinv(::AdaptedLoss{<:Any,LogitLink}, s) = 1 / (1 + exp(-s))
linkinv(::AdaptedLoss{<:Any,LogLink}, s) = exp(s)

_target(::DistanceLoss, y) = y
_target(::MarginLoss, y) = 2y - 1

# `deriv`/`deriv2` take (loss, output, target); a distance loss reads target
# as-is, a margin loss reads it as ±1. Both conventions checked against
# LossFunctions' own definitions and against a finite-difference probe.
@inline function gh(a::AdaptedLoss, y, f)
    t = _target(a.inner, y)
    return a.scale * deriv(a.inner, f, t), a.scale * deriv2(a.inner, f, t)
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
