# Targeted JET `report_opt` pass for tree, boosting, and continuous calls,
# plus adapter diagnostics and `@code_warntype` dumps of hot inner functions.
#
#     julia --project=bench bench/typecheck.jl
#
# Writes `jet.txt` (one section per call, with source locations for reported
# sites) and `warntype.txt` in `LT_PROFILE_OUT` or `bench/profiles/`.
# Exits unsuccessfully if a direct model call gains an inference report.

using JET, InteractiveUtils, StaticArrays, Printf
using LinearTrees, StableRNGs
using CategoricalArrays, DataFrames
import MLJModelInterface as MMI
import StatsAPI
include(joinpath(@__DIR__, "continuous_typecheck.jl"))

const TARGET_MODULES = (LinearTrees, LinearTrees.Continuous)

const OUT = get(ENV, "LT_PROFILE_OUT", joinpath(@__DIR__, "profiles"))
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
boost = fit_boost(X, ylin; nrounds = 3, max_depth = 2, nthreads = 1)
refitted = refit_leaves(tree, X, ylin; features = [1, 2], max_features = 2)
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
Keep the complete report so a caller can distinguish package frames from
dependency or compiler reports when Julia or JET changes.
"""
function section(io, name, f, args)
    r = try
        JET.report_opt(f, args; target_modules = TARGET_MODULES)
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
    println(io, "JET.report_opt, target_modules = ", TARGET_MODULES, ", Julia ", VERSION)
    for (nm, l, yy) in (("MSE", MSE(), ylin), ("Huber", Huber(1.0), ylin), ("Quantile", Quantile(0.5), ylin),
            ("MAD", MAD(), ylin), ("Logistic", Logistic(), ybin), ("Poisson", Poisson(), ycount),
            ("NegBin", NegBin(1.0), ycount), ("Gamma", Gamma(), ypos), ("Tweedie", Tweedie(1.5), ycount),
            ("Softmax", Softmax(3), ycls))
        push!(counts, "fit_tree/$nm" => section(io, "fit_tree $nm", fit_tree, (Matrix{Float64}, Vector{Float64}, typeof(l))))
    end
    push!(counts, "fit_tree/Softmax+cat(kw)" =>
        section(io, "fit_tree Softmax categorical", (Xa, ya) -> fit_tree(Xa, ya, Softmax(3); categorical = [6], max_depth = 4),
            (Matrix{Float64}, Vector{Float64})))
    for (name, f) in (
            ("MSE/exact", (Xa, ya) -> fit_boost(Xa, ya; nrounds = 3, nthreads = 1)),
            ("MSE/hybrid", (Xa, ya) -> fit_boost(Xa, ya; nrounds = 3, nthreads = 1,
                split_search = HybridSearch(nbins = 32))),
            ("Logistic", (Xa, ya) -> fit_boost(Xa, ya, Logistic(); nrounds = 3, nthreads = 1)),
            ("Softmax", (Xa, ya) -> fit_boost(Xa, ya, Softmax(3); nrounds = 3, nthreads = 1)),
        )
        push!(counts, "fit_boost/$name" => section(io, "fit_boost $name", f,
            (Matrix{Float64}, Vector{Float64})))
    end
    push!(counts, "predict" => section(io, "predict", LinearTrees.predict, (typeof(tree), Matrix{Float64})))
    push!(counts, "predict!" => section(io, "predict!", LinearTrees.predict!, (Vector{Float64}, typeof(tree), Matrix{Float64})))
    push!(counts, "predict/Softmax" => section(io, "predict Softmax", LinearTrees.predict, (typeof(streetree), Matrix{Float64})))
    push!(counts, "predict/boost" => section(io, "predict boost", LinearTrees.predict,
        (typeof(boost), Matrix{Float64})))
    push!(counts, "score" => section(io, "score", score, (typeof(tree), Matrix{Float64})))
    push!(counts, "score/boost" => section(io, "score boost", score, (typeof(boost), Matrix{Float64})))
    push!(counts, "refit_leaves" => section(io, "refit_leaves", refit_leaves,
        (typeof(tree), Matrix{Float64}, Vector{Float64})))
    push!(counts, "predict/RefitTree" => section(io, "predict RefitTree", predict,
        (typeof(refitted), Matrix{Float64})))
    push!(counts, "prune_refit" => section(io, "prune_refit", prune_refit,
        (typeof(refitted), Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64})))
    push!(counts, "fit_model_tree" => section(io, "fit_model_tree", fit_model_tree,
        (Matrix{Float64}, Vector{Float64})))
    push!(counts, "shap" => section(io, "shap", shap, (typeof(tree), Matrix{Float64})))
    push!(counts, "shap!" => section(io, "shap!", shap!,
        (Matrix{Float64}, Vector{Bool}, typeof(tree), Matrix{Float64})))
    push!(counts, "shap/Softmax" => section(io, "shap Softmax", shap, (typeof(streetree), Matrix{Float64})))
    push!(counts, "shap/boost" => section(io, "shap boost", shap, (typeof(boost), Matrix{Float64})))
    for (name, f, types) in continuous_cases()
        push!(counts, "continuous/$name" => section(io, "continuous $name", f, types))
    end
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

# These data-driven adapter/serialization calls already report dynamic sites
# from Any-valued dictionaries or Tables.jl interfaces. Keep their reports
# visible, but require zero reports from every direct model call above.
const DIAGNOSTIC_CASES = Set((
    "from_dict", "fit/RegressorFit", "fit/ClassifierFit",
    "predict/RegressorFit", "predict/ClassifierFit",
    "MMI.fit/Regressor", "MMI.fit/Classifier",
    "MMI.predict/Regressor", "MMI.predict/Classifier",
))

# ---- @code_warntype dumps -------------------------------------------------

const T = Float64
const VS = SVector{2,Float64}
# ExactSearch carries no search buffer. Use complete concrete state types here:
# a partially specified FitState is a UnionAll and does not describe a hot call.
const FSs = LinearTrees.FitState{T,T,T,MSE,BIC,ExactSearch,Nothing}
const FSv = LinearTrees.FitState{T,VS,T,Softmax{3},BIC,ExactSearch,Nothing}
const SCs = LinearTrees.Scratch{T,T,Nothing}
const SCv = LinearTrees.Scratch{T,VS,Nothing}
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
                    Int,Int,Vector{LinearTrees.PathElem},LinearTrees.PathPool,Int,T}),
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
    @printf("  %-28s %6d%s\n", k, v, k in DIAGNOSTIC_CASES ? "  (adapter diagnostic)" : "")
end
println("\nwrote ", OUT, "/jet.txt and ", OUT, "/warntype.txt")
unexpected = filter(kv -> last(kv) < 0 || (!(first(kv) in DIAGNOSTIC_CASES) && last(kv) != 0), counts)
isempty(unexpected) || error("Unresolved JET reports or analysis failures: $(unexpected). See $(joinpath(OUT, "jet.txt"))")
