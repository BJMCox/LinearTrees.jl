using LinearAlgebra, StableRNGs

function wls(D, z, h)
    W = Diagonal(h)
    β = (D' * W * D) \ (D' * W * z)
    r = z .- D * β
    return β, sum(h .* r .^ 2)
end

@testset "moment sums closed forms" begin
    rng = StableRNG(2)
    x = sort(randn(rng, 12)); z = 0.5 .* x .+ 0.1 .* randn(rng, 12); h = rand(rng, 12) .+ 0.5
    s = zero(MomentSums{Float64})
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
    sl = zero(MomentSums{Float64}); sr = zero(MomentSums{Float64})
    for i in 1:5; sl = LinearTrees.addrow(sl, x[i], z[i], h[i]); end
    for i in 6:12; sr = LinearTrees.addrow(sr, x[i], z[i], h[i]); end
    @test sl + sr ≈ s
    al, bl, ar, br, rss = LinearTrees.fit_blin(sl, sr, t)
    u = max.(x .- t, 0)
    β, rss0 = wls([x ones(12) u], z, h)
    @test al ≈ β[1] && bl ≈ β[2] && ar ≈ β[1] + β[3] && br ≈ β[2] - β[3] * t && rss ≈ rss0
    @test al * t + bl ≈ ar * t + br                    # continuity at the knot

    # subtraction undoes addition
    s2 = LinearTrees.subrow(s, x[1], z[1], h[1])
    @test s2.sxx ≈ sum(h[2:end] .* x[2:end] .^ 2)

    # singular guard: constant feature
    sc = zero(MomentSums{Float64})
    for i in 1:6; sc = LinearTrees.addrow(sc, 1.0, z[i], h[i]); end
    @test LinearTrees.fit_lin(sc) === nothing
end
