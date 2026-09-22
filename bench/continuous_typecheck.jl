# Run with the bench environment. Reports include the internal solver module.
using LinearTrees, JET, StableRNGs

function continuous_typecheck(io=stdout)
    X = rand(StableRNG(81), 80, 3)
    y = abs.(X[:, 1] .- 0.5)
    Y = hcat(y, 2y .+ X[:, 2])
    scalar = fit_continuous_tree(X, y; max_splits=1)
    multiple = fit_continuous_tree(X, Y; max_splits=1)
    cases = (
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
    )
    counts = Pair{String,Int}[]
    for (name, f, types) in cases
        report = JET.report_opt(f, types;
            target_modules=(LinearTrees, LinearTrees.Continuous))
        count = length(JET.get_reports(report))
        push!(counts, name => count)
        println(io, name, ": ", count, " inference reports")
        count == 0 || show(io, report)
    end
    return counts
end

if abspath(PROGRAM_FILE) == @__FILE__
    counts = continuous_typecheck()
    all(iszero ∘ last, counts) || error("Continuous-tree inference checks failed")
end
