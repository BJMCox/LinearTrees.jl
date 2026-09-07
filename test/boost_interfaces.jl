using AbstractTrees, JLD2, StableRNGs
using StatsAPI, DataFrames, CategoricalArrays, StaticArrays, Tables

struct SchemaUnknownTable
    a::AbstractVector{Float64}
    b::AbstractVector{Float64}
    c::AbstractVector
end
Tables.istable(::Type{SchemaUnknownTable}) = true
Tables.columnaccess(::Type{SchemaUnknownTable}) = true
Tables.columns(t::SchemaUnknownTable) = t
Tables.columnnames(::SchemaUnknownTable) = (:a, :b, :c)
Tables.getcolumn(t::SchemaUnknownTable, nm::Symbol) = getproperty(t, nm)
Tables.schema(::SchemaUnknownTable) = nothing

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

@testset "StatsAPI boosting regressor on a table with a categorical column" begin
    rng = StableRNG(132)
    n = 500
    df = DataFrame(a = randn(rng, n), b = randn(rng, n), c = categorical(rand(rng, ["u", "v", "w"], n)))
    y = 2 .* df.a .+ (df.c .== "u") .+ 0.1 .* randn(rng, n)
    m = fit(LinearBoostRegressorFit, df, y; nrounds = 10, eta = 0.3, max_depth = 3)
    @test nobs(m) == n && weights(m) == ones(n)
    @test predict(m, df) == predict(m.boost, m.X)
    @test residuals(m) ≈ y .- predict(m, df)
    @test deviance(m) ≈ deviance(MSE(), y, score(m.boost, m.X), ones(n))
    b0, a = coeftable(m, df[1, :])
    @test b0 + a' * m.X[1, :] ≈ score(m.boost, m.X[1:1, :]; clip = false)[1]
    br, ar = coeftable(m, select(df, [:b, :c, :a])[1, :])
    @test br + ar' * m.X[1, :] ≈ score(m.boost, m.X[1:1, :]; clip = false)[1]
    @test feature_importance(m) == feature_importance(m.boost)
    @test dof(fit(LinearBoostRegressorFit, df, y; nrounds = 4, max_depth = 0)) == 4
    dfv = DataFrame(b = df.b[1:100], c = df.c[1:100], a = df.a[1:100])
    yv = y[1:100]
    wv = Float64.(1:100)
    mv = fit(LinearBoostRegressorFit, df, y; nrounds = 1, Xval = dfv, yval = yv, wval = wv)
    @test mv.boost.validated
    @test only(mv.boost.history) ≈ sum(wv .* (yv .- predict(mv, dfv)).^2)
    dfu = DataFrame(a = [0.0], b = [0.0], c = categorical(["zzz"]))
    @test_throws ArgumentError predict(m, dfu)
    mr = fit(LinearBoostRegressorFit, df, y; nrounds = 2, unseen = :right)
    @test length(predict(mr, dfu)) == 1
    unknown = SchemaUnknownTable(dfv.a, dfv.b, dfv.c)
    mu = fit(LinearBoostRegressorFit, df, y; nrounds = 1,
        Xval = unknown, yval = yv, wval = vcat(ones(99), 0.0))
    @test mu.boost.validated
    @test_throws ArgumentError fit(LinearBoostRegressorFit, df, Int.(df.a .> 0); loss = Poisson(), nrounds = 1,
        Xval = df[1:2, :], yval = [1.0, -1.0], wval = [1.0, 0.0])
    tiny = parse(BigFloat, "1e-1000")
    dfbig = DataFrame(a = [0.0, NaN], b = [0.0, 0.0], c = categorical(["u", "u"]))
    @test_throws ArgumentError fit(LinearBoostRegressorFit, df, BigFloat.(y); nrounds = 1,
        Xval = dfbig, yval = BigFloat[0, 0], wval = BigFloat[1, tiny])
end

@testset "StatsAPI boosting classifier: binary and multiclass" begin
    rng = StableRNG(133)
    n = 600
    X = rand(rng, n, 3)
    y2 = [X[i, 1] > 0.5 ? "yes" : "no" for i in 1:n]
    m2 = fit(LinearBoostClassifierFit, X, y2; nrounds = 8, eta = 0.5, max_depth = 2)
    P2 = predict(m2, X)
    @test size(P2) == (n, 2) && all(≈(1.0), sum(P2; dims = 2))
    @test m2.classes == ["no", "yes"]
    @test P2[:, 2] == 1 .- P2[:, 1]
    f2 = log.(P2[:, 1] ./ P2[:, 2])
    @test deviance(m2) ≈ deviance(Logistic(), Float64.(y2 .== "no"), f2, ones(n))
    y3 = categorical([X[i, 1] > 0.6 ? "p" : X[i, 2] > 0.5 ? "q" : "r" for i in 1:n])
    m3 = fit(LinearBoostClassifierFit, X, y3; nrounds = 8, eta = 0.5, max_depth = 2)
    P3 = predict(m3, X)
    @test size(P3) == (n, 3) && all(≈(1.0), sum(P3; dims = 2))
    f3 = [SVector{2}(log(P3[i, 1] / P3[i, 3]), log(P3[i, 2] / P3[i, 3])) for i in axes(P3, 1)]
    y3code = [v == "p" ? 1 : v == "q" ? 2 : 3 for v in y3]
    @test deviance(m3) ≈ deviance(Softmax(3), y3code, f3, ones(n))
    yv = categorical(String.(y3[1:100]))
    levels!(yv, ["r", "p", "q"])
    wv = Float64.(1:100)
    mv = fit(LinearBoostClassifierFit, X, y3; nrounds = 1, Xval = X[1:100, :], yval = yv, wval = wv)
    @test mv.boost.validated
    Pv = predict(mv, X[1:100, :])
    fv = [SVector{2}(log(Pv[i, 1] / Pv[i, 3]), log(Pv[i, 2] / Pv[i, 3])) for i in axes(Pv, 1)]
    yvcode = [v == "p" ? 1 : v == "q" ? 2 : 3 for v in yv]
    @test only(mv.boost.history) ≈ deviance(Softmax(3), yvcode, fv, wv)
    Xbad = vcat(X[1:1, :], fill(NaN, 1, 3))
    mzero = fit(LinearBoostClassifierFit, X, y2; nrounds = 1,
        Xval = Xbad, yval = ["no", "yes"], wval = [1.0, 0.0])
    @test mzero.boost.validated
    @test_throws ArgumentError fit(LinearBoostClassifierFit, X, y2; nrounds = 1,
        Xval = Xbad, yval = ["no", "ghost"], wval = [1.0, 0.0])
    @test_throws DimensionMismatch fit(LinearBoostClassifierFit, X, y2; nrounds = 1,
        Xval = X[1:2, :], yval = ["no"], wval = [1.0])
    @test_throws DimensionMismatch fit(LinearBoostClassifierFit, X, y2; nrounds = 1,
        Xval = X[1:1, :], yval = ["no", "yes"], wval = [1.0, 1.0])
    @test_throws DimensionMismatch fit(LinearBoostClassifierFit, X, y2; nrounds = 1,
        Xval = X[1:1, :], yval = ["no"], wval = [1.0, 1.0])
    @test_throws ArgumentError fit(LinearBoostClassifierFit, X, y2; nrounds = 2,
        Xval = X[1:10, :], yval = fill("maybe", 10))
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
