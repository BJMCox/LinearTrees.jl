using AbstractTrees, JLD2, StableRNGs

@testset "LinearBoost JLD2 round trip is exact" begin
    rng = StableRNG(130)
    n = 300
    lvl = Float64.(rand(rng, 1:3, n))
    X = hcat(lvl, rand(rng, n, 2))
    for (loss, y) in ((Huber(0.8), 2 .* X[:, 2] .+ (lvl .== 1) .+ 0.1 .* randn(rng, n)),
                      (Softmax(3), [X[i, 2] > 0.6 ? 1 : lvl[i] == 1 ? 2 : 3 for i in 1:n]))
        b = fit_boost(X, y, loss; nrounds = 4, eta = 0.3, max_depth = 2, categorical = [1])
        mktempdir() do dir
            path = joinpath(dir, "boost.jld2")
            JLD2.jldopen(path, "w") do file
                file["model"] = b
            end
            b2 = JLD2.load(path, "model")
            @test typeof(b2) == typeof(b)
            @test length(b2.trees) == length(b.trees)
            @test all(t.nodes == t2.nodes && t.catmasks == t2.catmasks &&
                (t.lo, t.hi, t.base, t.nfeatures, t.truncate) ==
                (t2.lo, t2.hi, t2.base, t2.nfeatures, t2.truncate)
                for (t, t2) in zip(b.trees, b2.trees))
            @test (b2.loss, b2.f0, b2.eta, b2.lo, b2.hi, b2.nfeatures,
                b2.truncate, b2.validated, b2.history) ==
                (b.loss, b.f0, b.eta, b.lo, b.hi, b.nfeatures,
                b.truncate, b.validated, b.history)
            @test predict(b2, X) == predict(b, X)
        end
    end
end

@testset "LinearBoost show and TreeView" begin
    rng = StableRNG(131)
    X = rand(rng, 200, 2); y = X[:, 1] .+ 0.1 .* randn(rng, 200)
    b = fit_boost(X, y; nrounds = 3, max_depth = 2)
    s = sprint(show, MIME"text/plain"(), b)
    @test occursin("LinearBoost", s) && occursin("3 trees", s) && occursin("MSE", s)
    final_value = string(round(b.history[end]; sigdigits = 3))
    @test occursin("final deviance = $final_value", s)
    @test !occursin("best validation deviance", s)
    Xv = rand(rng, 50, 2); yv = Xv[:, 1]
    bv = fit_boost(X, y; nrounds = 3, Xval = Xv, yval = yv)
    sv = sprint(show, MIME"text/plain"(), bv)
    validation_value = string(round(bv.history[end]; sigdigits = 3))
    @test occursin("best validation deviance = $validation_value", sv)
    @test !occursin("final deviance", sv)
    empty = typeof(b)(b.trees, b.loss, b.f0, b.eta, b.lo, b.hi, b.nfeatures, b.truncate, true, Float64[])
    @test !occursin("deviance", sprint(show, MIME"text/plain"(), empty))
    view = TreeView(b, 2)
    @test view.tree === b.trees[2]
    io = IOBuffer()
    print_tree(io, view)
    @test occursin("LinearTree", String(take!(io)))
end
