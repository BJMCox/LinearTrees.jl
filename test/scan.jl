using StableRNGs

@testset "scan_feature picks the kind its design calls for" begin
    # One design per model kind, since each kind is a distinct closed form in
    # the sweep. `fit_tree` shows the winning kind too (test/fit.jl), but not
    # the candidate's own coefficients or split point.
    x = collect(range(-1, 1, length = 40)); z = 2 .* x .+ 1
    c = LinearTrees.scan_feature(x, z, ones(40), ones(40), BIC(), 5, 1e-12)
    @test c.kind == LIN && c.lcoef ≈ 2 && c.lintercept ≈ 1
    @test c.surrogate ≈ 0 atol = 1e-10

    x = collect(1.0:40.0); z = [i <= 20 ? 0.0 : 5.0 for i in 1:40]
    c = LinearTrees.scan_feature(x, z, ones(40), ones(40), BIC(), 5, 1e-12)
    @test c.kind == PCON && c.threshold == 20.0
    @test c.lintercept ≈ 0 && c.rintercept ≈ 5

    # kink at 0 lands exactly on a grid point: the split point is the largest left value
    # (PILOT parity), so blin's knot only lands on the true kink when 0 is itself a candidate
    x = collect(-2.0:0.05:2.0); z = max.(x, 0)
    c = LinearTrees.scan_feature(x, z, ones(81), ones(81), BIC(), 5, 1e-12)
    @test c.kind == BLIN
    @test c.threshold ≈ 0 atol = 1e-8
    @test c.lcoef ≈ 0 atol = 1e-8
    @test c.rcoef ≈ 1 atol = 1e-8

    rng = StableRNG(3)
    x = sort(randn(rng, 30)); z = randn(rng, 30)
    @test LinearTrees.scan_feature(x, z, ones(30), ones(30), BIC(), 5, 1e-12).kind == CON

    # a rule that disallows con returns a split even on noise
    x = collect(1.0:10.0); z = randn(StableRNG(4), 10)
    @test LinearTrees.scan_feature(x, z, ones(10), ones(10), MinDeviance((PCON,)), 1, 1e-12).kind == PCON

    # min_leaf = 15 leaves exactly one legal split position: at x = 15
    z2 = [i <= 15 ? 0.0 : 5.0 for i in 1:30]
    c2 = LinearTrees.scan_feature(collect(1.0:30.0), z2, ones(30), ones(30), MinDeviance((PCON,)), 15, 1e-12)
    @test c2.threshold == 15.0
end

@testset "duplicate xs values are never split between equal values" begin
    # a wrong tie-skip (comparing the wrong index, or dropping the `xs[i] <
    # xs[i + 1]` guard) would let a threshold fall strictly inside a block of
    # equal xs, splitting rows that share a feature value between children.
    x = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]
    z = [xi <= 3.0 ? 0.0 : 5.0 for xi in x]
    n = length(x)
    c = LinearTrees.scan_feature(x, z, ones(n), ones(n), BIC(), 1, 1e-12)
    @test c.kind == PCON
    @test c.threshold in unique(x)           # threshold is always a distinct value, never between two equal ones
    @test c.lintercept ≈ 0.0 atol = 1e-12    # every x == 3.0 row landed left; none split off to the right
    @test c.rintercept ≈ 5.0 atol = 1e-12
end

@testset "nocandidate when the rule allows nothing or min_leaf is too big" begin
    x = collect(1.0:10.0); z = randn(StableRNG(5), 10)

    # allowed(rule, ::ModelKind) is false for every kind, including con
    c1 = LinearTrees.scan_feature(x, z, ones(10), ones(10), MinDeviance(()), 1, 1e-12)
    @test c1.score == Inf && c1.kind == CON

    # con isn't offered by this rule, and min_leaf (100) exceeds Σw (10), so no
    # split position qualifies either
    c2 = LinearTrees.scan_feature(x, z, ones(10), ones(10), MinDeviance((PCON,)), 100, 1e-12)
    @test c2.score == Inf && c2.kind == CON
end

# ---- score once per kind per feature ---------------------------------------

"""
Wraps `inner` so only `kinds` are offered, the way `LinearTrees.PconOnly`
does for a categorical scan. Lets a test isolate one kind's own best split
under the real `BIC` score.
"""
struct OnlyKinds{R<:LinearTrees.SelectionRule} <: LinearTrees.SelectionRule
    inner::R
    kinds::Tuple{Vararg{ModelKind}}
end
LinearTrees.allowed(r::OnlyKinds, k::ModelKind) = any(==(k), r.kinds) && LinearTrees.allowed(r.inner, k)
LinearTrees.score_logn(r::OnlyKinds, n) = LinearTrees.score_logn(r.inner, n)
LinearTrees.devkey(r::OnlyKinds, dev, dmin) = LinearTrees.devkey(r.inner, dev, dmin)
LinearTrees.selection_score(r::OnlyKinds, k::ModelKind, s, n, dmin, nc::Integer = 1,
    logn = LinearTrees.score_logn(r, n)) =
    LinearTrees.allowed(r, k) ? LinearTrees.selection_score(r.inner, k, s, n, dmin, nc, logn) : Inf

"Counts `selection_score` calls and otherwise scores as `inner`. Serial use only."
mutable struct CountingRule{R<:LinearTrees.SelectionRule} <: LinearTrees.SelectionRule
    inner::R
    calls::Int
end
LinearTrees.allowed(r::CountingRule, k::ModelKind) = LinearTrees.allowed(r.inner, k)
LinearTrees.score_logn(r::CountingRule, n) = LinearTrees.score_logn(r.inner, n)
LinearTrees.devkey(r::CountingRule, dev, dmin) = LinearTrees.devkey(r.inner, dev, dmin)
function LinearTrees.selection_score(r::CountingRule, k::ModelKind, s, n, dmin, nc::Integer = 1,
        logn = LinearTrees.score_logn(r, n))
    r.calls += 1
    return LinearTrees.selection_score(r.inner, k, s, n, dmin, nc, logn)
end

@testset "selection_score runs once per kind, not once per split point" begin
    # The sweep carries the lowest `devkey` per kind and scores each kind once,
    # so a 400-row column costs at most five calls rather than three per split
    # point. Fails if `scan_feature` scores inside the split loop again.
    x = collect(1.0:400.0)
    z = [xi <= 200 ? 0.0 : 5.0 for xi in x] .+ 0.01 .* sin.(x)
    r = CountingRule(BIC(), 0)
    c = LinearTrees.scan_feature(x, z, ones(400), ones(400), r, 5, 1e-12)
    @test r.calls <= 5
    # and the winner is the one the plain rule finds, to the bit
    c2 = LinearTrees.scan_feature(x, z, ones(400), ones(400), BIC(), 5, 1e-12)
    @test c.kind == c2.kind && c.threshold === c2.threshold && c.score === c2.score
    @test c2.kind == PCON && c2.threshold == 200.0
end

@testset "within a kind the lowest devkey wins, earliest split point on a tie" begin
    # `dmin` is the BIC log floor (normally `eps · Σ h z²`); a large one makes
    # several split points score identically, which is what pins the rule.
    # blin's deviance on this design is 0.959 at t = 6, 0.918 at t = 7 and
    # 0.941 at t = 8, so with dmin = 1.0 all three floor to 1.0 and tie: the
    # earliest, t = 6, wins. Fails if the sweep carries the raw deviance
    # instead of the `dmin`-floored one, which returns t = 7.
    x = collect(1.0:20.0)
    z = [(xi > 10) + 0.1 * max(xi - 10, 0) for xi in x]
    c = LinearTrees.scan_feature(x, z, ones(20), ones(20), OnlyKinds(BIC(), (BLIN,)), 2, 1.0)
    @test c.kind == BLIN && c.threshold == 6.0
    # `MinDeviance` scores the raw deviance, so its own key must not be
    # floored: the same design under it picks the true minimiser, t = 7.
    c2 = LinearTrees.scan_feature(x, z, ones(20), ones(20), MinDeviance((BLIN,)), 2, 1.0)
    @test c2.kind == BLIN && c2.threshold == 7.0
end

@testset "an exact cross-kind score tie goes to the kind that comes first" begin
    # Same design and floor: blin's best split (t = 6) and pcon's (t = 10)
    # both floor to dmin, and `dof` is 5 for both kinds, so their scores are
    # bit-equal. The documented rule breaks the tie in the order
    # (con, lin, pcon, blin, plin), so pcon wins. Fails if the sweep resolves
    # the tie by split point again, which returns blin at t = 6.
    x = collect(1.0:20.0)
    z = [(xi > 10) + 0.1 * max(xi - 10, 0) for xi in x]
    bl = LinearTrees.scan_feature(x, z, ones(20), ones(20), OnlyKinds(BIC(), (BLIN,)), 2, 1.0)
    pc = LinearTrees.scan_feature(x, z, ones(20), ones(20), OnlyKinds(BIC(), (PCON,)), 2, 1.0)
    @test bl.threshold == 6.0 && pc.threshold == 10.0
    @test bl.score === pc.score              # the tie is exact, not close
    c = LinearTrees.scan_feature(x, z, ones(20), ones(20), BIC(), 2, 1.0)
    @test c.kind == PCON && c.threshold == 10.0
end

@testset "a UnitHessians scan equals a scan over ones, to the bit" begin
    # `fit_tree` hands `scan_feature` a `UnitHessians` instead of the gathered
    # `hs` when the loss is `MSE` and every weight is one, which drops the
    # multiply from `addrow`/`subrow`. The candidate must be identical, field
    # by field: it is the same arithmetic with a factor of exactly one taken
    # out. Fails if the unit path changes any sum.
    rng = StableRNG(31)
    fields = fieldnames(LinearTrees.Candidate)
    x = sort(randn(rng, 60)); m = 60
    # one target per winning kind: a step (pcon), a noisy line (lin/plin) and a hinge (blin)
    for z in (Float64[xi <= x[m ÷ 2] ? 0.0 : 3.0 for xi in x], 2 .* x .+ randn(rng, m),
            max.(x .- x[m ÷ 3], 0.0))
        for rule in (BIC(), MinDeviance((PCON, BLIN, PLIN)))
            a = LinearTrees.scan_feature(x, z, ones(m), ones(m), rule, 5, 1e-12)
            b = LinearTrees.scan_feature(x, z, LinearTrees.UnitHessians{Float64}(m), ones(m), rule, 5, 1e-12)
            @test all(isequal(getfield(a, f), getfield(b, f)) for f in fields)
        end
    end
    # the fast path is taken when `unit_hessian(loss)` holds and every weight is
    # one. `MSE`'s row hessian is `one(f)` and the `HMIN` floor leaves it; every
    # other loss has a hessian that varies with the fit, so a `true` there would
    # silently replace it with ones.
    @test LinearTrees.unit_hessian(MSE())
    @test !LinearTrees.unit_hessian(MAD())
end
