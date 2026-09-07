# Load in the bench environment. Compare complete fits on paired train/test data.
using BenchmarkTools, LinearTrees, Random, Statistics

function bench_search(; ntrain = 20_000, ntest = 10_000, seeds = 1:3, samples = 7, nthreads = 1)
    rows = NamedTuple[]
    for seed in seeds
        rng = Xoshiro(seed)
        X = rand(rng, ntrain + ntest, 10)
        y = sum(floor.(5 .* X[:, j]) for j in 1:6) .+ X[:, 7] .* X[:, 8] .+
            0.1 .* randn(rng, size(X, 1))
        Xt = X[(ntrain + 1):end, :]; yt = y[(ntrain + 1):end]
        X = X[1:ntrain, :]; y = y[1:ntrain]
        methods = circshift([ExactSearch(), BinnedSearch()], seed - 1)
        for split_search in methods
            fitfun = () -> fit_tree(X, y; split_search, max_depth = 10, nthreads)
            model = fitfun()
            trial = run(@benchmarkable($fitfun(), evals = 1); samples, seconds = 120)
            med = median(trial)
            rmse = sqrt(mean(abs2, predict(model, Xt; nthreads) .- yt))
            push!(rows, (; seed, method = nameof(typeof(split_search)),
                seconds = med.time / 1e9, bytes = med.memory, allocs = med.allocs, rmse))
        end
    end
    return rows
end
