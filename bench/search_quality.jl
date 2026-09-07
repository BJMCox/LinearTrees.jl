# Load with the benchmark project, for example:
# julia --project=bench -t1 -e 'include("bench/search_quality.jl"); rows = bench_search_quality()'
using BenchmarkTools, LinearTrees, Random, Statistics

const SEARCH_METHODS = (ExactSearch(), BinnedSearch(), HybridSearch())

logistic(x) = inv(1 + exp(-x))

function regression_fixture(rng, case, n)
    X = rand(rng, n, 10)
    noise = randn(rng, n)
    signal = if case === :linear
        3X[:, 1] .- 2X[:, 2] .+ X[:, 3]
    elseif case === :smooth
        sin.(2pi .* X[:, 1]) .+ 2 .* (X[:, 2] .- 0.5) .^ 2 .+ X[:, 3] .* X[:, 4]
    elseif case === :steps
        2 .* (X[:, 1] .> 0.35) .- 3 .* (X[:, 2] .> 0.7) .+ (X[:, 3] .> X[:, 4])
    elseif case === :heteroscedastic
        2X[:, 1] .- X[:, 2] .+ (0.05 .+ 1.5 .* X[:, 3]) .* noise
    else
        error("unknown regression case: $case")
    end
    y = case === :heteroscedastic ? signal : signal .+ 0.2 .* noise
    return X, y
end

function binary_fixture(rng, case, n)
    X = rand(rng, n, 10)
    eta = if case === :balanced_nonlinear
        2.5 .* sin.(2pi .* X[:, 1]) .+ 2 .* (X[:, 2] .- 0.5) .* (X[:, 3] .- 0.5)
    elseif case === :rare_event
        -4.5 .+ 5 .* (X[:, 1] .> 0.85) .+ 2 .* X[:, 2] .* X[:, 3]
    elseif case === :linear_logit
        -0.25 .+ 3 .* (X[:, 1] .- 0.5) .- 2 .* (X[:, 2] .- 0.5) .+ X[:, 3]
    else
        error("unknown binary case: $case")
    end
    y = Float64.(rand(rng, n) .< logistic.(eta))
    return X, y
end

function regression_metrics(pred, y)
    return (; rmse = sqrt(mean(abs2, pred .- y)))
end

function binary_metrics(pred, y)
    p = clamp.(pred, eps(Float64), 1 - eps(Float64))
    return (; logloss = -mean(y .* log.(p) .+ (1 .- y) .* log1p.(-p)),
        brier = mean(abs2, pred .- y), error = mean((pred .>= 0.5) .!= y))
end

function benchmark_fit(fitfun, samples)
    model = fitfun() # compile before timing and retain one model for evaluation
    trial = run(@benchmarkable($fitfun(), evals = 1); samples, seconds = 300)
    estimate = median(trial)
    return model, (; seconds = estimate.time / 1e9, bytes = estimate.memory,
        allocs = estimate.allocs)
end

"""
    bench_search_quality(; seeds=901:905, ntrain=5000, ntest=5000, samples=3)

Compare complete serial fits for exact, node-local binned, and hybrid searches on
paired synthetic train/test fixtures. Returns one named tuple per fit.
"""
function bench_search_quality(; seeds = 901:905, ntrain = 5_000, ntest = 5_000,
        samples = 3, on_result = identity)
    rows = NamedTuple[]
    regression_cases = (:linear, :smooth, :steps, :heteroscedastic)
    binary_cases = (:balanced_nonlinear, :rare_event, :linear_logit)
    models = ((:tree, 1), (:boost, 10))

    for seed in seeds, task in (:regression, :binary)
        cases = task === :regression ? regression_cases : binary_cases
        for (case_index, case) in pairs(cases)
            rng = Xoshiro(seed)
            fixture = task === :regression ? regression_fixture : binary_fixture
            Xtrain, ytrain = fixture(rng, case, ntrain)
            Xtest, ytest = fixture(rng, case, ntest)
            prevalence = task === :binary ? mean(ytest) : missing
            baseline = fill(mean(ytrain), ntest)
            baseline_metrics = task === :regression ? regression_metrics(baseline, ytest) :
                binary_metrics(baseline, ytest)

            for (model_kind, rounds) in models
                methods = circshift(collect(SEARCH_METHODS), mod(seed + case_index + rounds, 3))
                for split_search in methods
                    loss = task === :regression ? MSE() : Logistic()
                    fitfun = model_kind === :tree ?
                        (() -> fit_tree(Xtrain, ytrain, loss; split_search, max_depth = 8,
                            nthreads = 1)) :
                        (() -> fit_boost(Xtrain, ytrain, loss; split_search, nrounds = 10,
                            max_depth = 3, rng = Xoshiro(seed), nthreads = 1))
                    model, performance = benchmark_fit(fitfun, samples)
                    pred = predict(model, Xtest; nthreads = 1)
                    metrics = task === :regression ? regression_metrics(pred, ytest) :
                        binary_metrics(pred, ytest)
                    push!(rows, (; seed, task, case, model = model_kind,
                        method = nameof(typeof(split_search)), ntrain, ntest, prevalence,
                        performance...,
                        rmse = get(metrics, :rmse, missing),
                        logloss = get(metrics, :logloss, missing),
                        brier = get(metrics, :brier, missing),
                        error = get(metrics, :error, missing),
                        baseline_rmse = get(baseline_metrics, :rmse, missing),
                        baseline_logloss = get(baseline_metrics, :logloss, missing),
                        baseline_brier = get(baseline_metrics, :brier, missing),
                        baseline_error = get(baseline_metrics, :error, missing)))
                    on_result(rows)
                end
            end
        end
    end
    return rows
end

PROGRAM_FILE == (@__FILE__) && display(bench_search_quality())
