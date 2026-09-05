# Type-stability pass: JET `report_opt` on every exported entry point, plus
# `@code_warntype` dumps of the hot inner functions.
#
#     julia --project=bench bench/typecheck.jl
#
# Writes `bench/profiles/jet.txt` (one section per entry point, with the site
# count and the frames the sites sit in) and `bench/profiles/warntype.txt`.

using JET, InteractiveUtils, StaticArrays, Printf
using LinearTrees, StableRNGs
using CategoricalArrays, DataFrames
import MLJModelInterface as MMI
import StatsAPI

const OUT = joinpath(@__DIR__, "profiles")
mkpath(OUT)

# Small inputs: `report_opt` walks the call graph, so data size is irrelevant.
rng = StableRNG(11)
n, p = 400, 6
X = rand(rng, n, p)
X[:, p] = Float64.(rand(rng, 1:4, n))
ylin = X[:, 1] .+ 0.1 .* randn(rng, n)
ycount = Float64.(rand(rng, 0:5, n))
ypos = 1.0 .+ rand(rng, n)
ybin = Float64.(rand(rng, 0:1, n))
ycls = Float64.(rand(rng, 1:3, n))
df = DataFrame(hcat(X[:, 1:5]), [Symbol("x", j) for j in 1:5])
df.c = categorical(rand(rng, ["a", "b", "c"], n))
ystr = rand(rng, ["u", "v", "w"], n)

tree = fit_tree(X, ylin; categorical = [p], max_depth = 4)
streetree = fit_tree(X, ycls, Softmax(3); categorical = [p], max_depth = 4)
regfit = StatsAPI.fit(LinearTreeRegressorFit, df, ylin; max_depth = 4)
clsfit = StatsAPI.fit(LinearTreeClassifierFit, df, ystr; max_depth = 4)
mreg = LinearTreeRegressor(; max_depth = 4)
mcls = LinearTreeClassifier(; max_depth = 4)
_, _, _ = MMI.fit(mreg, 0, df, ylin)
regmach, _, _ = MMI.fit(mreg, 0, df, ylin)
clsmach, _, _ = MMI.fit(mcls, 0, df, ystr)
d = to_dict(tree)

"""
Run `report_opt` on one call, append it to `io`, and return the site count.
`frames` counts how many sites sit in each source file, which is what tells a
real dynamic dispatch in this package apart from `report_opt` noise inside a
dependency.
"""
function section(io, name, f, args)
    r = try
        JET.report_opt(f, args; target_modules = (LinearTrees,))
    catch e
        println(io, "\n### $name\nFAILED: ", sprint(showerror, e))
        return -1
    end
    txt = sprint(io2 -> show(IOContext(io2, :displaysize => (10_000, 200)), r))
    sites = JET.get_reports(r)
    byfile = Dict{String,Int}()
    for s in sites
        fr = s.vst[end]
        key = "$(basename(string(fr.file))):$(fr.line)"
        byfile[key] = get(byfile, key, 0) + 1
    end
    println(io, "\n### $name  -- $(length(sites)) site(s)")
    for (k, c) in sort!(collect(byfile); by = kv -> -kv[2])
        @printf(io, "  %5d  %s\n", c, k)
    end
    if !isempty(sites)
        println(io, "--- full report ---")
        println(io, txt)
    end
    return length(sites)
end

counts = Pair{String,Int}[]
open(joinpath(OUT, "jet.txt"), "w") do io
    println(io, "JET.report_opt, target_modules = (LinearTrees,), Julia ", VERSION)
    for (nm, l, yy) in (("MSE", MSE(), ylin), ("Huber", Huber(1.0), ylin), ("Quantile", Quantile(0.5), ylin),
            ("MAD", MAD(), ylin), ("Logistic", Logistic(), ybin), ("Poisson", Poisson(), ycount),
            ("NegBin", NegBin(1.0), ycount), ("Gamma", Gamma(), ypos), ("Tweedie", Tweedie(1.5), ycount),
            ("Softmax", Softmax(3), ycls))
        push!(counts, "fit_tree/$nm" => section(io, "fit_tree $nm", fit_tree, (Matrix{Float64}, Vector{Float64}, typeof(l))))
    end
    push!(counts, "fit_tree/Softmax+cat(kw)" =>
        section(io, "fit_tree Softmax categorical", (Xa, ya) -> fit_tree(Xa, ya, Softmax(3); categorical = [6], max_depth = 4),
            (Matrix{Float64}, Vector{Float64})))
    push!(counts, "predict" => section(io, "predict", LinearTrees.predict, (typeof(tree), Matrix{Float64})))
    push!(counts, "predict!" => section(io, "predict!", LinearTrees.predict!, (Vector{Float64}, typeof(tree), Matrix{Float64})))
    push!(counts, "predict/Softmax" => section(io, "predict Softmax", LinearTrees.predict, (typeof(streetree), Matrix{Float64})))
    push!(counts, "score" => section(io, "score", score, (typeof(tree), Matrix{Float64})))
    push!(counts, "shap" => section(io, "shap", shap, (typeof(tree), Matrix{Float64})))
    push!(counts, "shap!" => section(io, "shap!", shap!,
        (Matrix{Float64}, Vector{Bool}, typeof(tree), Matrix{Float64})))
    push!(counts, "shap/Softmax" => section(io, "shap Softmax", shap, (typeof(streetree), Matrix{Float64})))
    push!(counts, "feature_importance" => section(io, "feature_importance", feature_importance, (typeof(tree),)))
    push!(counts, "coeftable" => section(io, "coeftable", coeftable, (typeof(tree), Vector{Float64})))
    push!(counts, "expected_score" => section(io, "expected_score", expected_score, (typeof(tree),)))
    push!(counts, "to_dict" => section(io, "to_dict", to_dict, (typeof(tree),)))
    push!(counts, "from_dict" => section(io, "from_dict", from_dict, (typeof(d),)))
    push!(counts, "fit/RegressorFit" => section(io, "fit LinearTreeRegressorFit",
        (Xa, ya) -> StatsAPI.fit(LinearTreeRegressorFit, Xa, ya; max_depth = 4), (typeof(df), Vector{Float64})))
    push!(counts, "fit/ClassifierFit" => section(io, "fit LinearTreeClassifierFit",
        (Xa, ya) -> StatsAPI.fit(LinearTreeClassifierFit, Xa, ya; max_depth = 4), (typeof(df), Vector{String})))
    push!(counts, "predict/RegressorFit" => section(io, "predict LinearTreeRegressorFit",
        StatsAPI.predict, (typeof(regfit), typeof(df))))
    push!(counts, "predict/ClassifierFit" => section(io, "predict LinearTreeClassifierFit",
        StatsAPI.predict, (typeof(clsfit), typeof(df))))
    push!(counts, "MMI.fit/Regressor" => section(io, "MMI.fit LinearTreeRegressor",
        MMI.fit, (typeof(mreg), Int, typeof(df), Vector{Float64})))
    push!(counts, "MMI.fit/Classifier" => section(io, "MMI.fit LinearTreeClassifier",
        MMI.fit, (typeof(mcls), Int, typeof(df), Vector{String})))
    push!(counts, "MMI.predict/Regressor" => section(io, "MMI.predict LinearTreeRegressor",
        MMI.predict, (typeof(mreg), typeof(regmach), typeof(df))))
    push!(counts, "MMI.predict/Classifier" => section(io, "MMI.predict LinearTreeClassifier",
        MMI.predict, (typeof(mcls), typeof(clsmach), typeof(df))))
end

# ---- @code_warntype dumps -------------------------------------------------

const T = Float64
const VS = SVector{2,Float64}
const FSs = LinearTrees.FitState{T,T,MSE,BIC}
const FSv = LinearTrees.FitState{T,VS,Softmax,BIC}
const SCs = LinearTrees.Scratch{T,T}
const SCv = LinearTrees.Scratch{T,VS}
const COL = SubArray{Float64,1,Matrix{Float64},Tuple{UnitRange{Int},Int},true}
const ICOL = SubArray{Int32,1,Matrix{Int32},Tuple{Base.Slice{Base.OneTo{Int}},Int},true}
const TREEs = typeof(tree)
const TREEv = typeof(streetree)

open(joinpath(OUT, "warntype.txt"), "w") do io
    ctx = IOContext(io, :color => false, :displaysize => (10_000, 200))
    for (nm, f, tt) in (
            ("scan_feature scalar", LinearTrees.scan_feature, Tuple{COL,COL,COL,COL,BIC,T,T}),
            ("scan_feature softmax", LinearTrees.scan_feature,
                Tuple{COL,SubArray{VS,1,Vector{VS},Tuple{UnitRange{Int}},true},
                    SubArray{VS,1,Vector{VS},Tuple{UnitRange{Int}},true},COL,BIC,T,T}),
            ("gather! scalar", LinearTrees.gather!, Tuple{FSs,SCs,UnitRange{Int},Int}),
            ("gather! softmax", LinearTrees.gather!, Tuple{FSv,SCv,UnitRange{Int},Int}),
            ("partition_column!", LinearTrees.partition_column!,
                Tuple{ICOL,UnitRange{Int},Vector{Bool},Vector{Int32}}),
            ("score_row scalar", LinearTrees.score_row, Tuple{TREEs,Matrix{Float64},Int,Bool}),
            ("score_row softmax", LinearTrees.score_row, Tuple{TREEv,Matrix{Float64},Int,Bool}),
            ("visit! scalar", LinearTrees.visit!,
                Tuple{Matrix{Float64},TREEs,SubArray{Float64,1,Matrix{Float64},Tuple{Int,Base.Slice{Base.OneTo{Int}}},true},
                    Int,Int,Vector{LinearTrees.PathElem},LinearTrees.PathPool,Int}),
            ("addrow scalar", LinearTrees.addrow, Tuple{LinearTrees.MomentSums{T},T,T,T}),
            ("addrow softmax", LinearTrees.addrow, Tuple{LinearTrees.MomentSums{VS},T,VS,VS}),
            ("fit_lin scalar", LinearTrees.fit_lin, Tuple{LinearTrees.MomentSums{T}}),
            ("fit_lin softmax", LinearTrees.fit_lin, Tuple{LinearTrees.MomentSums{VS}}),
            ("fit_blin scalar", LinearTrees.fit_blin,
                Tuple{LinearTrees.MomentSums{T},LinearTrees.MomentSums{T},T}),
            ("fit_blin softmax", LinearTrees.fit_blin,
                Tuple{LinearTrees.MomentSums{VS},LinearTrees.MomentSums{VS},T}),
        )
        println(ctx, "\n", "="^78, "\n== ", nm, "\n", "="^78)
        try
            code_warntype(ctx, f, tt)
        catch e
            println(ctx, "FAILED: ", sprint(showerror, e))
        end
    end
end

println("JET site counts:")
for (k, v) in counts
    @printf("  %-28s %6d\n", k, v)
end
println("\nwrote ", OUT, "/jet.txt and ", OUT, "/warntype.txt")
