using JSON3

const PILOT_KINDS = Dict("con" => CON, "lin" => LIN, "pcon" => PCON, "blin" => BLIN, "plin" => PLIN)

@testset "PILOT reference fixtures" begin
    for name in ("linear", "piecewise", "hinge", "interaction", "step")
        fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "pilot", "$name.json"), String))
        X = permutedims(reduce(hcat, [Float64.(r) for r in fx.X]))   # rows in the JSON are observations
        y = Float64.(fx.y)
        t = fit_tree(X, y; max_depth = 6, min_leaf = 5, min_fit = 10, truncation_factor = 3)
        @test LinearTrees.predict(t, X) ≈ Float64.(fx.pred) atol = 1e-8
        ours = [(lowercase(string(n.model)), n.feature, n.threshold) for n in t.nodes if !LinearTrees.isleaf(n)]
        theirs = [(String(n.kind), Int(n.feature) + 1, Float64(n.threshold)) for n in fx.nodes if n.kind != "con"]
        @test length(ours) == length(theirs)
        for (o, r) in zip(sort(ours), sort(theirs))
            @test o[1] == r[1] && o[2] == r[2]
            r[1] in ("con", "lin") || @test o[3] ≈ r[3] atol = 1e-8
        end
    end
end
