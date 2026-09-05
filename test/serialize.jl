using StableRNGs, Serialization, JSON3

"One tree per distinct serialization branch: a plain MSE tree, a vector-valued
Softmax tree, a categorical tree (mask words), a loss that carries parameters,
and a LIN root (a NaN threshold, which plain JSON cannot hold)."
function serialize_trees()
    rng = StableRNG(33)
    X = rand(rng, 200, 3)
    y = sin.(3 .* X[:, 1]) .+ X[:, 2]
    ycls = [X[i, 1] > 0.5 ? 1 : X[i, 3] > 0.5 ? 2 : 3 for i in 1:200]
    lev = Float64.(rand(rng, 1:4, 200))
    Xc = hcat(X, lev)
    Xlin = reshape(collect(range(0, 1, length = 50)), 50, 1)
    return [("mse", fit_tree(X, y), X),
        ("untruncated", fit_tree(X, y; truncate = false), X),
        ("softmax", fit_tree(X, ycls, Softmax(3)), X),
        ("categorical", fit_tree(Xc, y; categorical = [4]), Xc),
        ("mad", fit_tree(X, 3 .* X[:, 1] .+ 2 .+ 0.01 .* randn(rng, 200), MAD()), X),
        ("lin root", fit_tree(Xlin, 3 .* Xlin[:, 1] .+ 2), Xlin)]
end

@testset "to_dict round trip" begin
    trees = serialize_trees()
    for (_, t, X) in trees
        t2 = from_dict(to_dict(t))
        @test t2.nodes == t.nodes && t2.catmasks == t.catmasks && t2.loss == t.loss
        @test t2.base == t.base   # tree.base is the SHAP empty-coalition value, and round-trips exactly
        @test LinearTrees.predict(t2, X) == LinearTrees.predict(t, X)

        # through real JSON text, where a non-finite field has to survive as a string
        t3 = from_dict(JSON3.read(JSON3.write(to_dict(t)), Dict{String,Any}))
        @test all(all(isequal(getfield(n1, f), getfield(n2, f)) for f in fieldnames(LinearTrees.Node))
                  for (n1, n2) in zip(t3.nodes, t.nodes))
        @test isequal(t3.catmasks, t.catmasks) && isequal(t3.loss, t.loss)
        @test LinearTrees.predict(t3, X) == LinearTrees.predict(t, X)

        io = IOBuffer(); serialize(io, t); seekstart(io)
        @test LinearTrees.predict(deserialize(io), X) == LinearTrees.predict(t, X)
    end

    # the LIN root's NaN threshold is the field JSON cannot hold as a number
    tlin = last(trees)[2]
    @test tlin.nodes[1].model == LIN && isnan(tlin.nodes[1].threshold)
    @test to_dict(tlin)["nodes"][1]["threshold"] == "NaN"
end

@testset "from_dict rejects an unrecognised element type" begin
    # fails if T falls back to Float64 instead of throwing on a bad "T" string
    rng = StableRNG(44)
    X = rand(rng, 60, 2); y = X[:, 1]
    d = to_dict(fit_tree(X, y))
    d["T"] = "Int8"
    @test_throws ArgumentError from_dict(d)
end
