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
Base.isapprox(a::MomentSums, b::MomentSums; kw...) =
    all(isapprox(getfield(a, k), getfield(b, k); kw...) for k in fieldnames(MomentSums))

"Add one row. `h` and `z` have type `V`, `x` has the feature type."
@inline function addrow(s::MomentSums{V}, x, z, h) where {V}
    hx = h * x
    MomentSums{V}(s.sw + h, s.sx + hx, s.sxx + hx * x, s.sz + h * z, s.sxz + hx * z, s.szz + h * z * z)
end
@inline function subrow(s::MomentSums{V}, x, z, h) where {V}
    hx = h * x
    MomentSums{V}(s.sw - h, s.sx - hx, s.sxx - hx * x, s.sz - h * z, s.sxz - hx * z, s.szz - h * z * z)
end

const SINGULAR_TOL = 1e-12

"Constant fit: intercept and surrogate deviance."
@inline function fit_con(s::MomentSums)
    b = s.sz / s.sw
    return b, s.szz - s.sz * b
end

"""
Simple linear fit `a x + b`. Returns `nothing` when the Gram determinant is
below `tol · sw · sxx`, which covers a constant feature.
"""
@inline function fit_lin(s::MomentSums; tol = SINGULAR_TOL)
    det = s.sw * s.sxx - s.sx * s.sx
    det <= tol * s.sw * s.sxx && return nothing
    a = (s.sw * s.sxz - s.sx * s.sz) / det
    b = (s.sz - a * s.sx) / s.sw
    rss = s.szz - a * s.sxz - b * s.sz
    return a, b, rss
end

"""
Broken linear fit with knot `t`: basis `[x, 1, max(x - t, 0)]`. Hinge sums
come from the right-child sums. Returns left and right pieces and the
surrogate deviance, or `nothing` when the `3×3` system is singular.
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
    abs(d) <= tol * s.sxx * s.sw * max(suu, eps(T)) && return nothing
    β = G \ m
    a, b, c = β[1], β[2], β[3]
    rss = s.szz - a * s.sxz - b * s.sz - c * suz
    return a, b, a + c, b - c * t, rss
end
