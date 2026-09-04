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

issmooth(::Loss) = true
issmooth(::Union{Quantile,MAD}) = false

# ---- links -----------------------------------------------------------------
linkinv(::Union{MSE,Huber,Quantile,MAD}, s) = s
linkinv(::Logistic, s) = 1 / (1 + exp(-s))
linkinv(::Union{Poisson,NegBin,Gamma,Tweedie}, s) = exp(s)

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
    irls_weights!(h, loss, y, f)

IRLS pseudo-hessian for non-smooth losses: `|g| / max(|r|, ε)` for quantile,
`1 / max(|r|, ε)` for MAD, with a positive scale-aware floor `ε`.
"""
function irls_weights!(h::AbstractVector{T}, loss::Union{Quantile,MAD}, y::AbstractVector, f::AbstractVector;
        ε = max(T(1e-3) * median_abs(y .- f), sqrt(eps(T)) * max(maximum(abs, y), one(T)))) where {T}
    r = y .- f
    for i in eachindex(h)
        num = loss isa MAD ? one(T) : abs(first(gh(loss, y[i], f[i])))
        h[i] = max(num / max(abs(r[i]), ε), T(HMIN))
    end
    return h
end

median_abs(r) = (a = sort(abs.(r)); m = length(a); isodd(m) ? a[(m + 1) ÷ 2] : (a[m ÷ 2] + a[m ÷ 2 + 1]) / 2)

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
