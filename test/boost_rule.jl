using StableRNGs, StaticArrays

@testset "GainRule rejects invalid penalties" begin
    for name in (:lambda_slope, :lambda_intercept, :gamma), value in (-1.0, Inf)
        @test_throws ArgumentError GainRule(; NamedTuple{(name,)}((value,))...)
    end
end

@testset "ridge closed forms match the explicit normal equations" begin
    rng = StableRNG(101)
    n = 40
    x = randn(rng, n); z = 2 .* x .+ 1 .+ 0.3 .* randn(rng, n); h = 0.5 .+ rand(rng, n)
    s = zero(LinearTrees.MomentSums{Float64})
    for i in 1:n
        s = LinearTrees.addrow(s, x[i], z[i], h[i])
    end
    λw, λb = 0.7, 1.3
    a, b, dev = LinearTrees.fit_lin(s, λw, λb)
    G = [s.sxx + λw  s.sx; s.sx  s.sw + λb]
    ab = G \ [s.sxz, s.sz]
    @test a ≈ ab[1] atol = 1e-12
    @test b ≈ ab[2] atol = 1e-12
    @test dev ≈ sum(h .* (z .- a .* x .- b) .^ 2) + λw * a^2 + λb * b^2 atol = 1e-9
    bc, devc = LinearTrees.fit_con(s, λb)
    @test bc ≈ s.sz / (s.sw + λb)
    @test devc ≈ sum(h .* (z .- bc) .^ 2) + λb * bc^2 atol = 1e-9
    # a rule with no ridge reaches the unridged closed form itself, bit for bit
    @test LinearTrees.fit_lin(s, BIC()) === LinearTrees.fit_lin(s)
    @test LinearTrees.fit_con(s, BIC()) === LinearTrees.fit_con(s)
    @test LinearTrees.fit_con(s, LinearTrees.PconOnly(BIC())) === LinearTrees.fit_con(s)
    @test LinearTrees.fit_lin(s, GainRule(lambda_slope = λw, lambda_intercept = λb)) === (a, b, dev)
    @test LinearTrees.fit_con(s, LinearTrees.PconOnly(GainRule(lambda_intercept = λb))) === (bc, devc)
end

@testset "the ridge keeps the coordinate type and fits coordinates independently" begin
    # Fails if `convert(eltype(s.sw), λ)` is dropped: a Float64 ridge would
    # promote a Float32 fit, and the scalar/vector agreement is what says the
    # ridge enters each softmax coordinate separately.
    s32 = zero(LinearTrees.MomentSums{Float32})
    for (x, z) in ((1.0f0, 2.0f0), (2.0f0, 3.0f0), (3.0f0, 5.0f0))
        s32 = LinearTrees.addrow(s32, x, z, 1.0f0)
    end
    a32, b32, dev32 = LinearTrees.fit_lin(s32, 0.5, 0.5)
    @test (typeof(a32), typeof(b32), typeof(dev32)) == (Float32, Float32, Float32)

    sv = zero(LinearTrees.MomentSums{SVector{2,Float64}})
    for (x, z) in ((1.0, SVector(1.0, 2.0)), (2.0, SVector(2.0, 3.5)), (3.0, SVector(3.5, 5.0)))
        sv = LinearTrees.addrow(sv, x, z, SVector(1.0, 2.0))
    end
    av, bv, devv = LinearTrees.fit_lin(sv, 0.5, 0.5)
    sk = zero(LinearTrees.MomentSums{Float64})
    for (x, z) in ((1.0, 2.0), (2.0, 3.5), (3.0, 5.0))
        sk = LinearTrees.addrow(sk, x, z, 2.0)
    end
    ak, bk, devk = LinearTrees.fit_lin(sk, 0.5, 0.5)
    @test (av[2], bv[2], devv[2]) === (ak, bk, devk)
end

@testset "GainRule allows con, pcon, plin and scores gain against gamma" begin
    r = GainRule(gamma = 2.0)
    @test LinearTrees.allowed(r, CON) && LinearTrees.allowed(r, PCON) && LinearTrees.allowed(r, PLIN)
    @test !LinearTrees.allowed(r, LIN) && !LinearTrees.allowed(r, BLIN)
    @test LinearTrees.ridge(r) == (1.0, 1.0)
    @test LinearTrees.ridge(BIC()) == (0.0, 0.0)
    @test LinearTrees.selection_score(r, CON, 10.0, 100.0, 0.0) == 10.0
    @test LinearTrees.selection_score(r, PLIN, 7.0, 100.0, 0.0) == 9.0        # 7 + gamma
    @test LinearTrees.selection_score(r, PLIN, 7.0, 100.0, 0.0, 2) == 11.0    # gamma per coordinate
    @test LinearTrees.selection_score(r, LIN, 1.0, 100.0, 0.0) == Inf
    @test LinearTrees.selection_score(r, PLIN, NaN, 100.0, 0.0) == Inf
end

@testset "GainRule's devkey scores the same as the raw deviance" begin
    # `scan_feature` carries the lowest devkey per kind and scores it once at
    # the end, so a rule whose key does not reproduce its own score, to the
    # bit, picks a different split. Fails if GainRule's devkey stops being the
    # identity on finite deviances or stops mapping a non-finite one to Inf.
    r = GainRule(gamma = 0.5)
    for s in (-Inf, 0.0, 1e-9, 0.5, 50.0, Inf, NaN)
        @test LinearTrees.selection_score(r, PLIN, LinearTrees.devkey(r, s, 1e-3), 108.0, 1e-3) ===
              LinearTrees.selection_score(r, PLIN, s, 108.0, 1e-3)
    end
    @test LinearTrees.score_logn(r, 100.0) == 0.0
end

"""
Brute-force depth-1 GainRule tree: checks the search over every feature and
every split between distinct values, given the package's own ridged closed
forms (`fit_con`/`fit_lin`), which the first testset verifies against the
explicit normal equations.
"""
function brute_depth1(X, z, h, w, λw, λb, γ; min_leaf = 5)
    n, p = size(X)
    hw = h .* w
    sums(rows, j) = begin
        s = zero(LinearTrees.MomentSums{Float64})
        for i in rows
            s = LinearTrees.addrow(s, X[i, j], z[i], hw[i])
        end
        s
    end
    stotal = sums(1:n, 1)
    bestscore = LinearTrees.fit_con(stotal, λb)[2]        # con: no gamma
    best = (CON, 0, NaN)
    for j in 1:p
        o = sortperm(X[:, j])
        xs = X[o, j]
        for i in 1:(n - 1)
            xs[i] < xs[i + 1] || continue
            L = o[1:i]; R = o[(i + 1):n]
            (sum(w[L]) >= min_leaf && sum(w[R]) >= min_leaf) || continue
            sl = sums(L, j); sr = sums(R, j)
            pc = LinearTrees.fit_con(sl, λb)[2] + LinearTrees.fit_con(sr, λb)[2] + γ
            if pc < bestscore
                bestscore = pc; best = (PCON, j, xs[i])
            end
            if length(unique(xs[1:i])) >= 5 && length(unique(xs[(i + 1):n])) >= 5
                fl = LinearTrees.fit_lin(sl, λw, λb); fr = LinearTrees.fit_lin(sr, λw, λb)
                if fl !== nothing && fr !== nothing
                    pl = fl[3] + fr[3] + γ
                    if pl < bestscore
                        bestscore = pl; best = (PLIN, j, xs[i])
                    end
                end
            end
        end
    end
    return best
end

@testset "depth-1 GainRule tree matches the brute force" begin
    rng = StableRNG(102)
    n = 60
    X = rand(rng, n, 3)
    y = 3 .* X[:, 2] .+ (X[:, 1] .> 0.5) .* 2 .+ 0.2 .* randn(rng, n)
    # two cases only: ridge off and ridge on. A third (λw, λb, γ) triple walks
    # the same code with different constants.
    for (λw, λb, γ) in ((0.0, 0.0, 0.0), (0.3, 2.0, 0.5))
        t = fit_tree(X, y; rule = GainRule(lambda_slope = λw, lambda_intercept = λb, gamma = γ),
            max_depth = 1, min_leaf = 5, min_fit = 10, truncate = false)
        root = t.nodes[1]
        # MSE: z = y - f0 with h = 1, so the brute force runs on centred y
        f0 = LinearTrees.initscore(MSE(), y, ones(n))
        kind, j, thr = brute_depth1(X, y .- f0, ones(n), ones(n), λw, λb, γ)
        @test root.model == kind
        if kind != CON
            @test root.feature == j
            @test root.threshold == thr
        end
    end
    # a large gamma forbids every split
    t = fit_tree(X, y; rule = GainRule(gamma = 1e6), max_depth = 3, truncate = false)
    @test length(t.nodes) == 1 && t.nodes[1].model == CON
end

@testset "GainRule's intercept ridge reaches a categorical PCON split under weights" begin
    # locks PconOnly's fit_con forwarding: scan_categorical always wraps the
    # rule in PconOnly, so a categorical split under GainRule is the only path
    # that reaches it. Fails (off by ~0.02 here) if the ridge forward is
    # dropped and PconOnly falls back to the unridged closed form.
    rng = StableRNG(21)
    n = 300
    lvl = rand(rng, 1:6, n)
    means = [0.0, 5.0, 0.0, 5.0, 5.0, 0.0]         # levels 2, 4, 5 are high
    y = means[lvl] .+ 0.1 .* randn(rng, n)
    X = Float64.(reshape(lvl, n, 1))
    w = Float64.(rand(rng, 1:3, n))                # frequency weights, not all one
    λb = 2.0
    t = fit_tree(X, y; categorical = [1], weights = w, rule = GainRule(lambda_intercept = λb),
        max_depth = 1, truncate = false)
    root = t.nodes[1]
    @test root.model == PCON && root.feature == 1
    f0 = LinearTrees.initscore(MSE(), y, w)
    z = y .- f0                                    # h = 1 for MSE
    left = [LinearTrees.category_is_left(t, root, l) for l in lvl]
    target(rows) = f0 + sum(w[rows] .* z[rows]) / (sum(w[rows]) + λb)
    @test root.lintercept ≈ target(left) atol = 1e-12
    @test root.rintercept ≈ target(.!left) atol = 1e-12
end
