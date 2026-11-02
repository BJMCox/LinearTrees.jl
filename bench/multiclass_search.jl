# Run through Kaimon with --project=bench and one Julia/BLAS thread.
using BenchmarkTools, LinearTrees, Random, Statistics, LinearAlgebra

function multiclass_fixture(seed, classes, ntrain, ntest)
    rng = Xoshiro(seed)
    X = randn(rng, ntrain + ntest, 6)
    coefficients = randn(rng, 6, classes) ./ sqrt(6)
    logits = X * coefficients
    for k = 1:classes, i in axes(X, 1)
        logits[i, k] += 1.5sin(k) * (X[i, 1] > 0.2) + cos(k) * max(X[i, 2], 0)
    end
    y = Vector{Int}(undef, size(X, 1))
    for i in axes(X, 1)
        row = view(logits, i, :)
        probabilities = exp.(row .- maximum(row))
        draw = rand(rng) * sum(probabilities)
        y[i] = classes
        cumulative = 0.0
        for k = 1:classes
            cumulative += probabilities[k]
            if draw < cumulative
                y[i] = k
                break
            end
        end
    end
    return X[1:ntrain, :], y[1:ntrain], X[(ntrain+1):end, :], y[(ntrain+1):end]
end

function multiclass_metrics(probabilities, y)
    n, classes = size(probabilities)
    logloss = -sum(i -> log(max(probabilities[i, y[i]], eps(Float64))), 1:n) / n
    brier = sum((probabilities[i, k] - (y[i] == k))^2 for i = 1:n, k = 1:classes) / n
    return (; logloss, brier)
end

function multiclass_trial(X, y, Xtest, ytest, loss, search, family, seed, samples)
    fitfun =
        family === :tree ?
        (() -> fit_tree(X, y, loss; split_search = search, max_depth = 5, nthreads = 1)) :
        (
            () -> fit_boost(
                X,
                y,
                loss;
                split_search = search,
                nrounds = 8,
                max_depth = 3,
                nthreads = 1,
                rng = Xoshiro(seed),
            )
        )
    model = fitfun()
    trial = run(@benchmarkable($fitfun(), evals = 1); samples, seconds = 300)
    estimate = median(trial)
    metrics = multiclass_metrics(predict(model, Xtest; nthreads = 1), ytest)
    return (;
        seconds = estimate.time / 1e9,
        q25_seconds = quantile(trial.times, 0.25) / 1e9,
        q75_seconds = quantile(trial.times, 0.75) / 1e9,
        bytes = estimate.memory,
        allocs = estimate.allocs,
        metrics...,
    )
end

"""
    bench_multiclass_search(; seeds=1103:1105, classes=(3, 8, 16), ntrain=6000,
                           ntest=4000, samples=7, methods=..., on_result=identity)

Compare complete serial tree and boosting fits on paired synthetic data with
piecewise linear class logits. Keep each seed's train/test data fixed across
methods. The multiclass Brier score sums squared errors over classes per row.
No test labels select model settings. Returns timing quartiles, allocation
counts/bytes, held-out log loss, and Brier score for each fit.
"""
function bench_multiclass_search(;
    seeds = 1103:1105,
    classes = (3, 8, 16),
    ntrain = 6000,
    ntest = 4000,
    samples = 7,
    methods = (
        (:exact, ExactSearch()),
        (:binned, BinnedSearch(nbins = 64, refine = false)),
        (:refined, BinnedSearch(nbins = 64)),
    ),
    on_result = identity,
)
    rows = NamedTuple[]
    for seed in seeds, count in classes
        X, y, Xtest, ytest = multiclass_fixture(seed, count, ntrain, ntest)
        loss = Softmax(count)
        for family in (:tree, :boost)
            ordered = circshift(collect(methods), mod(seed, length(methods)))
            for (method, search) in ordered
                result = multiclass_trial(
                    X,
                    y,
                    Xtest,
                    ytest,
                    loss,
                    search,
                    family,
                    seed,
                    samples,
                )
                push!(
                    rows,
                    (;
                        seed,
                        classes = count,
                        family,
                        method,
                        ntrain,
                        ntest,
                        samples,
                        result...,
                    ),
                )
                on_result(rows)
            end
        end
    end
    return rows
end

PROGRAM_FILE == (@__FILE__) && display(bench_multiclass_search())
