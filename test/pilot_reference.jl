using JSON3

const PILOT_KINDS = Dict("con" => CON, "lin" => LIN, "pcon" => PCON, "blin" => BLIN, "plin" => PLIN)

@testset "PILOT reference fixtures" begin
    for name in ("linear", "piecewise", "hinge", "interaction", "step")
        fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "pilot", "$name.json"), String))
        X = permutedims(reduce(hcat, [Float64.(r) for r in fx.X]))   # rows in the JSON are observations
        y = Float64.(fx.y)
        t = fit_tree(X, y; max_depth = 6, min_leaf = 5, min_fit = 10, truncation_factor = 3)
        @test LinearTrees.predict(t, X) ≈ Float64.(fx.pred) atol = 1e-8
        # coefficients ride along in the sort key so equal (kind, feature, threshold)
        # entries (e.g. two lin nodes on the same feature) still pair deterministically
        ours = [(lowercase(string(n.model)), n.feature, n.threshold, n.lcoef, n.lintercept, n.rcoef, n.rintercept)
                for n in t.nodes if !LinearTrees.isleaf(n)]
        theirs = [(String(n.kind), Int(n.feature) + 1, n.kind == "lin" ? NaN : Float64(n.threshold),
                   Float64(n.lm_l[1]), Float64(n.lm_l[2]), Float64(n.lm_r[1]), Float64(n.lm_r[2]))
                  for n in fx.nodes if n.kind != "con"]
        @test length(ours) == length(theirs)
        for (o, r) in zip(sort(ours), sort(theirs))
            @test o[1] == r[1] && o[2] == r[2]
            r[1] == "lin" || @test o[3] ≈ r[3] atol = 1e-8
            @test isapprox(o[4], r[4]; atol = 1e-8) && isapprox(o[5], r[5]; atol = 1e-8)
            r[1] == "lin" || @test isapprox(o[6], r[6]; atol = 1e-8) && isapprox(o[7], r[7]; atol = 1e-8)   # lin's lm_r is a placeholder
        end
    end
end
