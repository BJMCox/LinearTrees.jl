using LinearAlgebra, StableRNGs, StaticArrays

function wls(D, z, h)
    W = Diagonal(h)
    β = (D' * W * D) \ (D' * W * z)
    r = z .- D * β
    return β, sum(h .* r .^ 2)
end

@testset "moment sums closed forms" begin
    rng = StableRNG(2)
    x = sort(randn(rng, 12)); z = 0.5 .* x .+ 0.1 .* randn(rng, 12); h = rand(rng, 12) .+ 0.5
    s = zero(LinearTrees.MomentSums{Float64})
    for i in eachindex(x); s = LinearTrees.addrow(s, x[i], z[i], h[i]); end
    @test s.sw ≈ sum(h) && s.sxz ≈ sum(h .* x .* z)

    b, rss = LinearTrees.fit_con(s)
    β, rss0 = wls(ones(12, 1), z, h)
    @test b ≈ β[1] && rss ≈ rss0

    a, b, rss = LinearTrees.fit_lin(s)
    β, rss0 = wls([x ones(12)], z, h)
    @test a ≈ β[1] && b ≈ β[2] && rss ≈ rss0

    # blin with knot t between rows 5 and 6
    t = (x[5] + x[6]) / 2
    sl = zero(LinearTrees.MomentSums{Float64}); sr = zero(LinearTrees.MomentSums{Float64})
    for i in 1:5; sl = LinearTrees.addrow(sl, x[i], z[i], h[i]); end
    for i in 6:12; sr = LinearTrees.addrow(sr, x[i], z[i], h[i]); end
    @test all(getfield(sl + sr, k) ≈ getfield(s, k) for k in fieldnames(LinearTrees.MomentSums))
    al, bl, ar, br, rss = LinearTrees.fit_blin(sl, sr, t)
    u = max.(x .- t, 0)
    β, rss0 = wls([x ones(12) u], z, h)
    @test al ≈ β[1] && bl ≈ β[2] && ar ≈ β[1] + β[3] && br ≈ β[2] - β[3] * t && rss ≈ rss0
    @test al * t + bl ≈ ar * t + br                    # continuity at the knot

    # subtraction undoes addition
    s2 = LinearTrees.subrow(s, x[1], z[1], h[1])
    @test s2.sxx ≈ sum(h[2:end] .* x[2:end] .^ 2)

    # singular guard: constant feature
    sc = zero(LinearTrees.MomentSums{Float64})
    for i in 1:6; sc = LinearTrees.addrow(sc, 1.0, z[i], h[i]); end
    @test LinearTrees.fit_lin(sc) === nothing
end

@testset "fit_blin guard is scale invariant" begin
    x = collect(range(-1, 1, length = 12)); z = 0.5 .* x .+ 0.01 .* sin.(7 .* x); t = 0.05
    function sums(hscale)
        sl = zero(LinearTrees.MomentSums{Float64}); sr = zero(LinearTrees.MomentSums{Float64})
        for i in 1:6; sl = LinearTrees.addrow(sl, x[i], z[i], hscale); end
        for i in 7:12; sr = LinearTrees.addrow(sr, x[i], z[i], hscale); end
        return sl, sr
    end
    r1 = LinearTrees.fit_blin(sums(1.0)..., t)
    r30 = LinearTrees.fit_blin(sums(1e-30)..., t)
    @test r30 !== nothing
    @test all(isapprox.(r1[1:4], r30[1:4]; rtol = 1e-6))
    # empty right child is singular
    @test LinearTrees.fit_blin(sums(1.0)[1], zero(LinearTrees.MomentSums{Float64}), t) === nothing
end

@testset "the unit-hessian row methods equal a hessian of exactly one" begin
    # `addrow`/`subrow` take a `OneHessian` path that drops the multiply by
    # one, so the whole MSE scan can run without it. Multiplying by one is
    # exact, so the sums have to agree to the bit and not approximately: one
    # ULP moves a split point. Fails if the unit method drops or reorders a
    # term of the general one.
    fields = fieldnames(LinearTrees.MomentSums)
    xs = [-1.75, 0.0, 0.5, 3.25]        # a negative, a zero and two positives
    zs = [2.5, -0.125, 7.0, -3.0]
    for V in (Float64, SVector{2,Float64})
        one_v = V === Float64 ? 1.0 : SVector(1.0, 1.0)
        g = zero(LinearTrees.MomentSums{V}); u = zero(LinearTrees.MomentSums{V})
        rows = [(x, V === Float64 ? z : V(z, -2z)) for (x, z) in zip(xs, zs)]
        for (x, z) in rows
            g = LinearTrees.addrow(g, x, z, one_v)
            u = LinearTrees.addrow(u, x, z, LinearTrees.OneHessian{V}())
            @test all(getfield(g, f) === getfield(u, f) for f in fields)
        end
        for (x, z) in rows
            g = LinearTrees.subrow(g, x, z, one_v)
            u = LinearTrees.subrow(u, x, z, LinearTrees.OneHessian{V}())
            @test all(getfield(g, f) === getfield(u, f) for f in fields)
        end
    end
    # the vector form reads as ones and carries its length
    h = LinearTrees.UnitHessians{Float64}(7)
    @test length(h) == 7 && h[1] === LinearTrees.OneHessian{Float64}()
end
