using LinearAlgebra: det
using StaticArrays

"""
Six weighted sums over a row set. Fields are `V`, the coefficient type, so
the softmax case carries one sum per fitted logit.
"""
struct MomentSums{V}
    sw::V
    sx::V
    sxx::V
    sz::V
    sxz::V
    szz::V
end

Base.zero(::Type{MomentSums{V}}) where {V} = MomentSums{V}(zero(V), zero(V), zero(V), zero(V), zero(V), zero(V))
Base.:+(a::MomentSums{V}, b::MomentSums{V}) where {V} =
    MomentSums{V}(a.sw + b.sw, a.sx + b.sx, a.sxx + b.sxx, a.sz + b.sz, a.sxz + b.sxz, a.szz + b.szz)
Base.:-(a::MomentSums{V}, b::MomentSums{V}) where {V} =
    MomentSums{V}(a.sw - b.sw, a.sx - b.sx, a.sxx - b.sxx, a.sz - b.sz, a.sxz - b.sxz, a.szz - b.szz)

"""
Stands in for a row hessian that is exactly one. `addrow` and `subrow` have
methods for it that drop the multiplication by one; the value it stands for
is `one(eltype(V))` in every coordinate. Multiplying by exactly one is exact,
so a scan over these sums is bit-identical to the same scan over a
`Vector{V}` of ones.
"""
struct OneHessian{V} end

"A read-only `hs` vector of `OneHessian{V}`, the fast path's stand-in for `ones(V, len)`."
struct UnitHessians{V} <: AbstractVector{OneHessian{V}}
    len::Int
end
Base.size(h::UnitHessians) = (h.len,)
Base.@propagate_inbounds function Base.getindex(h::UnitHessians{V}, i::Int) where {V}
    @boundscheck checkbounds(h, i)
    return OneHessian{V}()
end

"True when every scalar or vector Hessian is exactly one in every coordinate."
@inline unit_hessian_value(h::Number) = isone(h)
@inline unit_hessian_value(h) = all(isone, h)
@inline function unit_hessians(hs::AbstractVector{V}) where {V}
    V === Union{} && return false
    return all(unit_hessian_value, hs)
end

"""
Add one row. `h` and `z` have type `V`, `x` has the feature type. `h * x` and
`hx * x` scale a `V` by the scalar feature value, which `*` already does for
`SVector`; `h * z` and `z * z` multiply two `V`s elementwise, which needs `.*`
since `SVector` has no vector-times-vector `*`.
"""
@inline function addrow(s::MomentSums{V}, x, z, h) where {V}
    hx = h * x
    MomentSums{V}(s.sw + h, s.sx + hx, s.sxx + hx * x, s.sz + h .* z, s.sxz + hx .* z, s.szz + h .* z .* z)
end
@inline function subrow(s::MomentSums{V}, x, z, h) where {V}
    hx = h * x
    MomentSums{V}(s.sw - h, s.sx - hx, s.sxx - hx * x, s.sz - h .* z, s.sxz - hx .* z, s.szz - h .* z .* z)
end

# `h ≡ 1`: the same expressions with every `h *` dropped. Dotted, and with
# `one(eltype(V))` rather than `one(V)`, because `x` is the scalar feature
# value while the sums carry the coefficient type `V`, an `SVector` for a
# softmax fit.
@inline function addrow(s::MomentSums{V}, x, z, ::OneHessian{V}) where {V}
    o = one(eltype(V))
    MomentSums{V}(s.sw .+ o, s.sx .+ x, s.sxx .+ x * x, s.sz .+ z, s.sxz .+ x .* z, s.szz .+ z .* z)
end
@inline function subrow(s::MomentSums{V}, x, z, ::OneHessian{V}) where {V}
    o = one(eltype(V))
    MomentSums{V}(s.sw .- o, s.sx .- x, s.sxx .- x * x, s.sz .- z, s.sxz .- x .* z, s.szz .- z .* z)
end

const SINGULAR_TOL = 1e-12

"""
Constant fit: intercept and surrogate deviance. Dotted so `V` may be a
scalar or an `SVector` (independent per-coordinate fits). A coordinate with
zero total mass takes the minimum-norm intercept zero.
"""
@inline function fit_con(s::MomentSums)
    b = ifelse.(iszero.(s.sw), zero(s.sz), s.sz ./ s.sw)
    return b, s.szz .- s.sz .* b
end

"""
Simple linear fit `a x + b`. Returns `nothing` when the Gram determinant is
below `tol · sw · sxx` for any coordinate, which covers a constant feature.
A zero-mass coordinate takes zero coefficients and does not reject other
coordinates. Dotted so `V` may be a scalar or an `SVector` (independent
per-coordinate fits).
"""
@inline function fit_lin(s::MomentSums; tol = SINGULAR_TOL)
    d = s.sw .* s.sxx .- s.sx .* s.sx
    massless = iszero.(s.sw)
    any((.!massless) .& (d .<= tol .* s.sw .* s.sxx)) && return nothing
    a = ifelse.(massless, zero(s.sxz), (s.sw .* s.sxz .- s.sx .* s.sz) ./ d)
    b = ifelse.(massless, zero(s.sz), (s.sz .- a .* s.sx) ./ s.sw)
    rss = s.szz .- a .* s.sxz .- b .* s.sz
    return a, b, rss
end

"""
Constant fit with intercept ridge `λb`: `b = sz / (sw + λb)`, regularised
deviance `szz − b·sz`. Dotted so `V` may be a scalar or an `SVector`. `λb` is
converted to the coordinate type so `Float32` sums stay `Float32`.
"""
@inline function fit_con(s::MomentSums, λb::Real)
    sww = s.sw .+ convert(eltype(s.sw), λb)
    b = ifelse.(iszero.(sww), zero(s.sz), s.sz ./ sww)
    return b, s.szz .- s.sz .* b
end

"""
Simple linear fit `a x + b` with slope ridge `λw` and intercept ridge `λb`
(Guryanov 2019, eq. 10). Returns `nothing` when the ridged Gram determinant
is below `tol · (sw + λb)(sxx + λw)` for any coordinate, which covers a
constant feature at zero ridge. The returned deviance is the minimum of
`Σ h (z − a x − b)² + λw a² + λb b²`, which at the optimum equals
`szz − a·sxz − b·sz` exactly as in the unregularised case. A zero-mass
coordinate takes zero coefficients.
"""
@inline function fit_lin(s::MomentSums, λw::Real, λb::Real; tol = SINGULAR_TOL)
    E = eltype(s.sw)
    sww = s.sw .+ convert(E, λb)
    sxxw = s.sxx .+ convert(E, λw)
    d = sww .* sxxw .- s.sx .* s.sx
    massless = iszero.(s.sw)
    any((.!massless) .& (d .<= tol .* sww .* sxxw)) && return nothing
    a = ifelse.(massless, zero(s.sxz), (sww .* s.sxz .- s.sx .* s.sz) ./ d)
    b = ifelse.(massless, zero(s.sz), (s.sz .- a .* s.sx) ./ sww)
    rss = s.szz .- a .* s.sxz .- b .* s.sz
    return a, b, rss
end

"""
Broken linear fit with knot `t`: basis `[x, 1, max(x - t, 0)]`. Hinge sums
come from the right-child sums. Returns left and right pieces and the
surrogate deviance, or `nothing` when the `3×3` system is singular.

The `2×2`-plus-Schur-update form of this solve was measured and rejected: see
`bench/PROFILE.md`, "not worth changing".
"""
@inline function fit_blin(sl::MomentSums{T}, sr::MomentSums{T}, t; tol = SINGULAR_TOL) where {T<:Real}
    s = sl + sr
    su  = sr.sx - t * sr.sw
    suu = sr.sxx - 2t * sr.sx + t * t * sr.sw
    suz = sr.sxz - t * sr.sz
    sxu = sr.sxx - t * sr.sx
    G = @SMatrix [s.sxx  s.sx  sxu;
                  s.sx   s.sw  su;
                  sxu    su    suu]
    m = @SVector [s.sxz, s.sz, suz]
    d = det(G)
    abs(d) <= tol * s.sxx * s.sw * max(suu, eps(T) * s.sxx) && return nothing
    β = G \ m
    a, b, c = β[1], β[2], β[3]
    rss = s.szz - a * s.sxz - b * s.sz - c * suz
    return a, b, a + c, b - c * t, rss
end

"Coordinate `k` of a vector `MomentSums`, as a scalar one."
@inline coordsums(s::MomentSums{V}, k) where {T,V<:SVector{<:Any,T}} =
    MomentSums{T}(s.sw[k], s.sx[k], s.sxx[k], s.sz[k], s.sxz[k], s.szz[k])

"""
`fit_blin` for `SVector` coefficients. The softmax coordinates are fitted
independently, so this is the scalar method once per coordinate, singular
guard included; `nothing` if any coordinate is singular.
"""
@inline function fit_blin(sl::MomentSums{V}, sr::MomentSums{V}, t; tol = SINGULAR_TOL) where {Km1,T,V<:SVector{Km1,T}}
    al = zero(MVector{Km1,T}); bl = zero(MVector{Km1,T})
    ar = zero(MVector{Km1,T}); br = zero(MVector{Km1,T}); rss = zero(MVector{Km1,T})
    for k in 1:Km1
        r = fit_blin(coordsums(sl, k), coordsums(sr, k), t; tol)
        r === nothing && return nothing
        al[k], bl[k], ar[k], br[k], rss[k] = r
    end
    return SVector(al), SVector(bl), SVector(ar), SVector(br), SVector(rss)
end
