# From the bench environment:
# include("bench/continuous_scaling.jl")
# ContinuousScaling.measure(800, 6, :all)
module ContinuousScaling

using BenchmarkTools
using LinearTrees
using StableRNGs
using Statistics

"Nested row/feature subsets with three active predictors and fixed noise."
function fixture(n, p; seed=93023)
    rng = StableRNG(seed)
    predictors = 2rand(rng, 3200, 12) .- 1
    noise = 0.12randn(rng, 3200)
    x1, x2, x3 = eachcol(predictors[:, 1:3])
    target = @. 1 + 0.7x1 - 0.4x2 + 0.3x3 + 1.4abs(x1) +
        0.9max(x1, 0) * max(x2, 0) + noise
    return predictors[1:n, 1:p], target[1:n]
end

function fit_case(f, X, y, pairs)
    return f(X, y; pairs, candidate_search=:full, max_splits=6,
        max_depth=4, min_leaf=8, n_thresholds=3)
end

"Measure one warmed fit with BenchmarkTools; fixture construction is excluded."
function measure(n, p, pairs; samples=5, fit=fit_continuous_tree)
    X, y = fixture(n, p)
    fitfun = () -> fit_case(fit, X, y, pairs)
    model = fitfun()
    trial = run(@benchmarkable($fitfun(), evals=1); samples, seconds=600)
    estimate = median(trial)
    return (; n, p, pairs, time_ms=estimate.time / 1e6,
        bytes=estimate.memory, allocations=estimate.allocs,
        evaluations=model.fit.evaluations, dimension=size(model.fit.N, 2),
        splits=sum(last, model.fit.moves; init=0),
        moves=join(("$(kind):$count" for (kind, count) in model.fit.moves), '|'),
        skipped_dimension=model.fit.skipped_dimension,
        cache_builds=model.fit.cache_builds, cache_hits=model.fit.cache_hits)
end

"Append one result so long studies retain completed measurements."
function record(path, row)
    mkpath(dirname(abspath(path)))
    header = !isfile(path)
    open(path, "a") do io
        header && println(io, join(keys(row), ','))
        println(io, join(values(row), ','))
    end
    return row
end

end
