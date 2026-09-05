using StableRNGs, Serialization, JSON3

@testset "to_dict round trip" begin
    rng = StableRNG(33)
    X = rand(rng, 200, 3); y = sin.(3 .* X[:, 1]) .+ X[:, 2]
    for t in (fit_tree(X, y), fit_tree(X, [X[i, 1] > 0.5 ? 1 : X[i, 3] > 0.5 ? 2 : 3 for i in 1:200], Softmax(3)))
        d = to_dict(t)
        @test d["nodes"] isa Vector && d["loss"] isa Dict
        t2 = from_dict(d)
        @test t2.nodes == t.nodes && t2.catmasks == t.catmasks && t2.loss == t.loss
        @test t2.base == t.base   # I6: tree.base is the SHAP empty-coalition value, round-trips exactly
        @test LinearTrees.predict(t2, X) == LinearTrees.predict(t, X)
        io = IOBuffer(); serialize(io, t); seekstart(io)
        @test LinearTrees.predict(deserialize(io), X) == LinearTrees.predict(t, X)
    end
end

@testset "to_dict round trip: categorical, MAD, LIN-root" begin
    rng = StableRNG(41)
    X = rand(rng, 150, 2); lev = rand(rng, 1:4, 150)
    y = [lev[i] in (1, 2) ? X[i, 1] : -X[i, 1] for i in 1:150] .+ 0.01 .* randn(rng, 150)
    Xc = hcat(X, Float64.(lev))
    tcat = fit_tree(Xc, y; categorical = [3])
    dcat = to_dict(tcat); t2 = from_dict(dcat)
    @test t2.nodes == tcat.nodes && t2.catmasks == tcat.catmasks && t2.loss == tcat.loss
    @test LinearTrees.predict(t2, Xc) == LinearTrees.predict(tcat, Xc)

    ymad = 3 .* X[:, 1] .+ 2 .+ 0.01 .* randn(rng, 150)
    tmad = fit_tree(X, ymad, MAD())
    dmad = to_dict(tmad); t3 = from_dict(dmad)
    @test t3.nodes == tmad.nodes && t3.loss == tmad.loss
    @test LinearTrees.predict(t3, X) == LinearTrees.predict(tmad, X)

    Xlin = reshape(collect(range(0, 1, length = 50)), 50, 1)
    ylin = 3 .* Xlin[:, 1] .+ 2
    tlin = fit_tree(Xlin, ylin)
    @test tlin.nodes[1].model == LIN && isnan(tlin.nodes[1].threshold)
    dlin = to_dict(tlin)
    @test dlin["nodes"][1]["threshold"] == "NaN"
    t4 = from_dict(dlin)
    @test t4.nodes == tlin.nodes
    @test LinearTrees.predict(t4, Xlin) == LinearTrees.predict(tlin, Xlin)
end

@testset "from_dict rejects an unrecognised element type" begin
    # fails if T falls back to Float64 instead of throwing on a bad "T" string
    rng = StableRNG(44)
    X = rand(rng, 60, 2); y = X[:, 1]
    d = to_dict(fit_tree(X, y))
    d["T"] = "Int8"
    @test_throws ArgumentError from_dict(d)
end

@testset "to_dict survives a real JSON3 round trip" begin
    rng = StableRNG(42)
    X = rand(rng, 200, 3)
    lev = rand(rng, 1:4, 200)
    Xc = hcat(X, Float64.(lev))
    yreg = X[:, 1] .+ 0.01 .* randn(rng, 200)
    ycls = [X[i, 1] > 0.5 ? 1 : X[i, 3] > 0.5 ? 2 : 3 for i in 1:200]

    tsoftmax = fit_tree(X, ycls, Softmax(3))
    tcat = fit_tree(Xc, yreg; categorical = [4])
    tuntrunc = fit_tree(X, yreg; truncate = false)

    for (t, Xt) in ((tsoftmax, X), (tcat, Xc), (tuntrunc, X))
        s = JSON3.write(to_dict(t))
        t2 = from_dict(JSON3.read(s, Dict{String,Any}))
        @test all(all(isequal(getfield(n1, f), getfield(n2, f)) for f in fieldnames(LinearTrees.Node))
                  for (n1, n2) in zip(t2.nodes, t.nodes))
        @test isequal(t2.catmasks, t.catmasks) && isequal(t2.loss, t.loss)
        @test LinearTrees.predict(t2, Xt) == LinearTrees.predict(t, Xt)
    end
end
