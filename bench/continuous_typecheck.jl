# Continuous cases shared by the standalone checker and bench/typecheck.jl.
# Run the standalone checker with the bench environment.
using LinearTrees, JET, StableRNGs

function continuous_cases()
    X = rand(StableRNG(81), 80, 3)
    y = abs.(X[:, 1] .- 0.5)
    Y = hcat(y, 2y .+ X[:, 2])
    scalar = fit_continuous_tree(X, y; max_splits=1)
    multiple = fit_continuous_tree(X, Y; max_splits=1)
    root = fit_continuous_tree(X, y; max_splits=0)
    paired = fit_continuous_tree(X, y; max_splits=0, pairs=[(1, 2)])
    geometries = [root, paired]
    ensemble = fit_continuous_ensemble(X, y, geometries)
    multi_ensemble = fit_continuous_ensemble(X, Y, geometries)
    return (
        ("fit/vector", fit_continuous_tree, (Matrix{Float64}, Vector{Float64})),
        ("fit/matrix", fit_continuous_tree, (Matrix{Float64}, Matrix{Float64})),
        ("fit/pairs+pruning", (X, y) -> fit_continuous_tree(X, y;
            pairs=[(1, 2)], candidate_search=:graph_pruned, n_thresholds=nothing),
            (Matrix{Float64}, Vector{Float64})),
        ("predict/vector", predict, (typeof(scalar), Matrix{Float64})),
        ("predict/matrix", predict, (typeof(multiple), Matrix{Float64})),
        ("predict!/vector", predict!, (Vector{Float64}, typeof(scalar), Matrix{Float64})),
        ("predict!/matrix", predict!, (Matrix{Float64}, typeof(multiple), Matrix{Float64})),
        ("predictive/vector", predictive, (typeof(scalar), Matrix{Float64})),
        ("predictive/matrix", predictive, (typeof(multiple), Matrix{Float64})),
        ("ensemble/fit/vector", fit_continuous_ensemble,
            (Matrix{Float64}, Vector{Float64}, typeof(geometries))),
        ("ensemble/fit/matrix", fit_continuous_ensemble,
            (Matrix{Float64}, Matrix{Float64}, typeof(geometries))),
        ("ensemble/predict/vector", predict, (typeof(ensemble), Matrix{Float64})),
        ("ensemble/predict/matrix", predict, (typeof(multi_ensemble), Matrix{Float64})),
        ("ensemble/predict!/vector", predict!,
            (Vector{Float64}, typeof(ensemble), Matrix{Float64})),
        ("ensemble/predictive/vector", predictive, (typeof(ensemble), Matrix{Float64})),
        ("ensemble/predictive/matrix", predictive,
            (typeof(multi_ensemble), Matrix{Float64})),
    )
end

function continuous_typecheck(io=stdout)
    counts = Pair{String,Int}[]
    for (name, f, types) in continuous_cases()
        report = JET.report_opt(f, types;
            target_modules=(LinearTrees, LinearTrees.Continuous))
        count = length(JET.get_reports(report))
        push!(counts, name => count)
        println(io, name, ": ", count, " inference reports")
        if count != 0
            show(io, report)
            println(io)
        end
    end
    return counts
end

if abspath(PROGRAM_FILE) == @__FILE__
    counts = continuous_typecheck()
    all(iszero ∘ last, counts) || error("Continuous-tree inference checks failed")
end
