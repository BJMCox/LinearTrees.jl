using JSON3

"Per-fixture fit overrides. Every fixture but 'deep' uses the base
(max_depth=6, min_leaf=5, min_fit=10, truncation_factor=3, no categorical
column); 'deep' is fit with max_depth=3 so the cap actually binds, and
'categorical' declares column 1 (feature 0 in the reference) as categorical."
const FIXTURE_KW = Dict(
    "deep" => (max_depth = 3, categorical = Int[]),
    "categorical" => (max_depth = 6, categorical = [1]),
)
default_kw(name) = get(FIXTURE_KW, name, (max_depth = 6, categorical = Int[]))

@testset "PILOT reference fixtures" begin
    for name in ("linear", "piecewise", "hinge", "interaction", "step", "twoslope", "deep", "categorical")
        fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "pilot", "$name.json"), String))
        X = permutedims(reduce(hcat, [Float64.(r) for r in fx.X]))   # rows in the JSON are observations
        y = Float64.(fx.y)
        kw = default_kw(name)
        t = fit_tree(X, y; max_depth = kw.max_depth, min_leaf = 5, min_fit = 10,
            truncation_factor = 3, categorical = kw.categorical)
        @test LinearTrees.predict(t, X) ≈ Float64.(fx.pred) atol = 1e-8

        # Coefficients ride along in the sort key (`by` below) so equal
        # (kind, feature, sortkey) entries -- e.g. two lin nodes on the same
        # feature, where sortkey is NaN on both sides -- still pair
        # deterministically instead of by incidental emission order.
        # For a categorical (reference "pconc") node, threshold is meaningless on both
        # sides (NaN on ours, a stale placeholder on the reference's), so the surrogate
        # sort key there is the smallest left-level code instead, and the real
        # left-level-set check happens separately below.
        ours = map(filter(!LinearTrees.isleaf, t.nodes)) do n
            cat = LinearTrees.iscategorical(n)
            leftset = cat ? Set(c for c in 1:(n.catwords * 64) if LinearTrees.category_is_left(t, n, c)) : nothing
            sortkey = cat ? Float64(minimum(leftset)) : n.threshold
            (kind = lowercase(string(n.model)), feature = n.feature, sortkey = sortkey,
                lcoef = n.lcoef, lintercept = n.lintercept, rcoef = n.rcoef, rintercept = n.rintercept,
                cat = cat, leftset = leftset)
        end
        theirs = map(filter(n -> n.kind != "con", fx.nodes)) do n
            iscat = n.kind == "pconc"
            leftset = iscat ? Set(Int.(n.pivot_c)) : nothing
            sortkey = n.kind == "lin" ? NaN : (iscat ? Float64(minimum(leftset)) : Float64(n.threshold))
            (kind = iscat ? "pcon" : String(n.kind), feature = Int(n.feature) + 1, sortkey = sortkey,
                lcoef = Float64(n.lm_l[1]), lintercept = Float64(n.lm_l[2]),
                rcoef = Float64(n.lm_r[1]), rintercept = Float64(n.lm_r[2]),
                cat = iscat, leftset = leftset)
        end
        @test length(ours) == length(theirs)
        # Coefficients ride along in the sort key so that equal (kind, feature,
        # sortkey) entries pair deterministically rather than by emission order
        by = x -> (x.kind, x.feature, x.sortkey, x.lcoef, x.lintercept, x.rcoef, x.rintercept)
        sorted_ours, sorted_theirs = sort(ours; by), sort(theirs; by)
        for (o, r) in zip(sorted_ours, sorted_theirs)
            @test o.kind == r.kind && o.feature == r.feature && o.cat == r.cat
            if r.cat
                @test o.leftset == r.leftset
            elseif r.kind != "lin"
                @test o.sortkey ≈ r.sortkey atol = 1e-8
            end
            @test isapprox(o.lcoef, r.lcoef; atol = 1e-8) && isapprox(o.lintercept, r.lintercept; atol = 1e-8)
            r.kind == "lin" || @test isapprox(o.rcoef, r.rcoef; atol = 1e-8) && isapprox(o.rintercept, r.rintercept; atol = 1e-8)   # lin's lm_r is a placeholder
        end
    end
end
