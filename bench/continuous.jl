module ContinuousBench

# Real-data attribution and license
#
# Airfoil Self-Noise, UCI Machine Learning Repository:
# Brooks, Pope and Marcolini (1989), https://doi.org/10.24432/C5VW2C
# https://archive.ics.uci.edu/dataset/291/airfoil+self-noise
#
# Yacht Hydrodynamics, UCI Machine Learning Repository:
# Gerritsma, Onnink and Versluis (1981), https://doi.org/10.24432/C5XG7R
# https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics
#
# Both datasets are distributed under CC BY 4.0:
# https://creativecommons.org/licenses/by/4.0/

using BenchmarkTools
using LinearTrees
using StableRNGs
using Random
using Statistics
using LinearAlgebra
using Downloads
using DelimitedFiles
using SHA
using Profile
using PProf

export dataset, realdata, reuse_benchmark, profile_fit

const DATASETS = (
    airfoil = (
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        sha256 = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3",
        filename = "airfoil_self_noise.dat",
        rows = 1503,
        features = 5,
    ),
    yacht = (
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        sha256 = "00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1",
        filename = "yacht_hydrodynamics.data",
        rows = 308,
        features = 6,
    ),
)

function _dataset_spec(name)
    key = name isa Symbol ? name : Symbol(name)
    hasproperty(DATASETS, key) ||
        throw(ArgumentError("unknown dataset $name; expected :airfoil or :yacht"))
    return key, getproperty(DATASETS, key)
end

"""
    dataset(name; directory=joinpath(tempdir(), "lineartrees-continuous-data"))

Load a verified UCI dataset as `(name, X, y)`. The official source is
downloaded only when its local file is absent. Existing and downloaded files
must match the recorded SHA-256 digest.
"""
function dataset(name;
        directory = joinpath(tempdir(), "lineartrees-continuous-data"))
    key, spec = _dataset_spec(name)
    path = joinpath(directory, spec.filename)
    if !isfile(path)
        mkpath(directory)
        Downloads.download(spec.url, path)
    end
    digest = bytes2hex(open(SHA.sha256, path))
    digest == spec.sha256 ||
        throw(ArgumentError("SHA-256 mismatch for $path: expected $(spec.sha256), got $digest"))
    values = readdlm(path, Float64)
    size(values) == (spec.rows, spec.features + 1) ||
        throw(DimensionMismatch(
            "$(spec.filename) has size $(size(values)); " *
            "expected $(spec.rows) × $(spec.features + 1)"))
    return (; name = key, X = values[:, 1:spec.features], y = values[:, end])
end

function _benchmark(f, samples; seconds = 120)
    trial = run(@benchmarkable($f(), evals = 1); samples = samples, seconds = seconds)
    estimate = median(trial)
    return (; time = estimate.time, bytes = estimate.memory, allocs = estimate.allocs)
end

_leaf_count(model::LinearTree) = count(node -> node.feature == 0, model.nodes)
_leaf_count(model::ContinuousTree) = length(model.fit.leaves)

function _old_fit(X, y)
    return fit_tree(X, y, MSE();
        weights = nothing,
        categorical = Int[],
        rule = BIC(),
        split_search = ExactSearch(),
        max_depth = 4,
        min_fit = 16,
        min_leaf = 8,
        min_sum_hessian = 1.0,
        max_lin_chain = 10,
        truncate = true,
        truncation_factor = 3,
        features = axes(X, 2),
        presort = nothing,
        nthreads = 1,
        niter = 5,
    )
end

function _continuous_fit(X, y, pairs, candidate_search)
    return fit_continuous_tree(X, y;
        pairs = pairs,
        candidate_search = candidate_search,
        max_splits = 6,
        max_depth = 4,
        min_leaf = 8,
        n_thresholds = 3,
        split_penalty = 2.0,
        coefficient_precision = 0.01,
        noise_shape = 2.0,
        noise_rate = 1.0,
    )
end

function _methods(X, y)
    return (
        (:fit_tree, () -> _old_fit(X, y)),
        (:continuous_none_full,
            () -> _continuous_fit(X, y, :none, :full)),
        (:continuous_none_graph_pruned,
            () -> _continuous_fit(X, y, :none, :graph_pruned)),
        (:continuous_all_full,
            () -> _continuous_fit(X, y, :all, :full)),
    )
end

"""
    realdata(; seeds=(11, 22, 33), samples=3, datasets=(:airfoil, :yacht))

Benchmark fixed tree fits and predictions on deterministic 75/25 splits of
the UCI datasets. Splitting occurs on raw rows; all transformations used by a
model are learned by that model from its training rows. Returns one named
tuple per dataset, seed, and method.
"""
function realdata(; seeds = (11, 22, 33), samples = 3,
        datasets = (:airfoil, :yacht),
        directory = joinpath(tempdir(), "lineartrees-continuous-data"))
    rows = NamedTuple[]
    for requested in datasets
        data = dataset(requested; directory)
        n = length(data.y)
        ntrain = fld(3n, 4)
        ntest = n - ntrain
        for seed in seeds
            permutation = randperm(StableRNG(seed), n)
            train = permutation[1:ntrain]
            test = permutation[ntrain+1:end]
            Xtrain = data.X[train, :]
            ytrain = data.y[train]
            Xtest = data.X[test, :]
            ytest = data.y[test]

            for (method, fitfun) in _methods(Xtrain, ytrain)
                model = fitfun()
                prediction = predict(model, Xtest)
                fit_performance = _benchmark(fitfun, samples)
                predictfun = () -> predict(model, Xtest)
                predictfun()
                predict_performance = _benchmark(predictfun, samples)
                push!(rows, (;
                    dataset = data.name,
                    seed,
                    method,
                    ntrain,
                    ntest,
                    rmse = sqrt(mean(abs2, prediction .- ytest)),
                    fit_ms = fit_performance.time / 1e6,
                    bytes = fit_performance.bytes,
                    allocs = fit_performance.allocs,
                    predict_us = predict_performance.time / 1e3,
                    leaves = _leaf_count(model),
                ))
            end
        end
    end
    return rows
end

function _reuse_fixture()
    rng = StableRNG(4107)
    X = 2 .* rand(rng, 400, 3) .- 1
    signal = max.(0.0, min.(X[:, 1], X[:, 2])) .+ 0.4 .* X[:, 3]
    y = signal .+ 0.08 .* randn(rng, size(X, 1))

    center = vec((minimum(X; dims = 1) .+ maximum(X; dims = 1)) ./ 2)
    scale = vec((maximum(X; dims = 1) .- minimum(X; dims = 1)) ./ 2)
    Z = (X .- transpose(center)) ./ transpose(scale)
    ycenter = mean(y)
    yscale = norm(y .- ycenter) / sqrt(length(y))
    Y = reshape((y .- ycenter) ./ yscale, :, 1)
    thresholds = [Float64[-0.5, 0.0, 0.5] for _ in axes(Z, 2)]
    return Z, Y, thresholds
end

function _engine_fit(X, Y, thresholds, reuse_constraints)
    return LinearTrees.Continuous.fit(X, Y;
        pairs = [(1, 2)],
        candidate_search = :full,
        max_splits = 6,
        max_depth = 4,
        min_leaf = 8,
        thresholds = thresholds,
        split_penalty = 2.0,
        coefficient_precision = 0.01,
        noise_shape = 2.0,
        noise_rate = 1.0,
        reuse_constraints = reuse_constraints,
    )
end

"""
    reuse_benchmark(; samples=5)

Compare full constraint reconstruction with incremental unchanged-basis reuse
on one deterministic normalized interaction fixture. Search decisions and
predictions are asserted equal before measurement.
"""
function reuse_benchmark(; samples = 5)
    X, Y, thresholds = _reuse_fixture()
    fullfun = () -> _engine_fit(X, Y, thresholds, false)
    reusefun = () -> _engine_fit(X, Y, thresholds, true)
    full = fullfun()
    reused = reusefun()
    full_prediction = LinearTrees.Continuous.predict(full, X)
    reused_prediction = LinearTrees.Continuous.predict(reused, X)
    @assert full.moves == reused.moves
    @assert full.evaluations == reused.evaluations
    @assert isapprox(full.history, reused.history; atol = 1e-8, rtol = 1e-8)
    @assert isapprox(full_prediction, reused_prediction; atol = 1e-8, rtol = 1e-8)

    full_performance = _benchmark(fullfun, samples)
    reuse_performance = _benchmark(reusefun, samples)
    return (;
        full = (fit_ms = full_performance.time / 1e6,
            bytes = full_performance.bytes, allocs = full_performance.allocs),
        reuse = (fit_ms = reuse_performance.time / 1e6,
            bytes = reuse_performance.bytes, allocs = reuse_performance.allocs),
        speedup = full_performance.time / reuse_performance.time,
        byte_ratio = full_performance.bytes / reuse_performance.bytes,
        moves = reused.moves,
        evaluations = reused.evaluations,
        maximum_prediction_error = maximum(abs, full_prediction .- reused_prediction),
    )
end

"""
    profile_fit(path; repetitions=10, allocation_path=nothing)

Write a headless PProf CPU profile for repeated incremental fits of the reuse
fixture. If `allocation_path` is supplied, also write a separate allocation
profile sampled at 1%. Returns the absolute CPU-profile path.
"""
function profile_fit(path; repetitions = 10, allocation_path = nothing)
    X, Y, thresholds = _reuse_fixture()
    fitfun = () -> _engine_fit(X, Y, thresholds, true)
    fitfun()
    profilepath = abspath(path)
    mkpath(dirname(profilepath))
    Profile.clear()
    Profile.@profile for _ in 1:repetitions
        fitfun()
    end
    PProf.pprof(; web = false, out = profilepath)

    if allocation_path !== nothing
        allocpath = abspath(allocation_path)
        mkpath(dirname(allocpath))
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate = 0.01 begin
            for _ in 1:repetitions
                fitfun()
            end
        end
        result = Profile.Allocs.fetch()
        PProf.Allocs.pprof(result; web = false, out = allocpath)
    end
    return profilepath
end

end
