module ValidationBench

using BenchmarkTools, LinearAlgebra, LinearTrees, Random, StableRNGs, Statistics
import Distributions
include("realdata.jl")
const RD = RealDataBench

export bench_robustness, bench_calibration, bench_prior_calibration, bench_scaling,
    calibration_summary

function measure(f; samples)
    value = f()
    trial = run(@benchmarkable($f(), evals=1); samples, seconds=300)
    result = median(trial)
    return value, (;seconds=result.time / 1e9, bytes=result.memory, allocs=result.allocs)
end

function record!(rows, row, output)
    push!(rows, row)
    output === nothing || RD.write_rows(output, rows)
    return rows
end

function robust_data(rng, n, case)
    X = 2rand(rng, n, 4) .- 1
    truth = 1 .+ 2X[:, 1] .- 0.7X[:, 2] .+ 1.4max.(X[:, 1], 0)
    noise = case === :heavy_tail ?
        0.2 .* rand(rng, Distributions.TDist(3), n) ./ sqrt(3) : 0.2randn(rng, n)
    if case === :outliers
        contaminated = rand(rng, n) .< 0.1
        noise .+= contaminated .* (8 .* ifelse.(rand(rng, n) .< 0.5, -1.0, 1.0))
    end
    return (;X, y=truth + noise, truth)
end

function robust_fit(method, train, validation, seed)
    tree(loss) = fit_tree(train.X, train.y, loss; max_depth=6, min_leaf=20,
        min_fit=40, nthreads=1)
    method === :mse && return (tree(MSE()), NaN)
    method === :huber && return (tree(Huber(1.0)), 1.0)
    if method === :huber_selected
        deltas = (0.5, 1.0, 2.0)
        models = [tree(Huber(delta)) for delta in deltas]
        index = argmin([mean(abs, predict(model, validation.X) - validation.y) for model in models])
        return models[index], deltas[index]
    end
    method === :huber_boost || error("unknown robust method")
    model = fit_boost(train.X, train.y, Huber(1.0); nrounds=30, eta=0.1,
        max_depth=2, min_leaf=20, min_fit=40, nthreads=1, rng=StableRNG(seed))
    return model, 1.0
end

"Paired clean, contaminated, and heavy-tail fits; MAE validation alone selects delta."
function bench_robustness(; seeds=501:510, samples=3, output=nothing)
    rows = NamedTuple[]
    for case in (:clean, :outliers, :heavy_tail), seed in seeds
        rng = StableRNG(seed)
        train, validation, test = (robust_data(rng, n, case) for n in (512, 256, 512))
        for method in (:mse, :huber, :huber_selected, :huber_boost)
            (model, delta), cost = measure(() -> robust_fit(method, train, validation, seed); samples)
            prediction = predict(model, test.X)
            record!(rows, (;case, seed, method, delta, cost...,
                rmse=sqrt(mean(abs2, prediction - test.y)),
                mae=mean(abs, prediction - test.y),
                signal_rmse=sqrt(mean(abs2, prediction - test.truth)),
                RD.model_size(model)...), output)
        end
    end
    return rows
end

function response(rng, x, case)
    truth = 1 .+ 0.7x
    case === :kink && (truth .+= 1.5max.(x .- 0.5, 0))
    case === :curve && (truth .+= 1.5x.^2)
    σ = case === :heteroskedastic ? 0.1 .+ 0.25 .* (x .+ 1) : fill(0.25, length(x))
    noise = case === :heavy_tail ?
        rand(rng, Distributions.TDist(3), length(x)) ./ sqrt(3) : randn(rng, length(x))
    return truth + σ .* noise, truth
end

function geometries(X, y)
    candidates = [fit_continuous_tree(X, y; max_splits=k, max_depth=2,
        n_thresholds=3, min_leaf=8, split_penalty=2.0) for k in 0:2]
    return RD.distinct_geometries(candidates)
end

function marginal(p, i)
    hasproperty(p, :components) || return RD.student_t(p, i)
    active = findall(>(0), p.weights)
    parts = [RD.student_t(p.components[k], i) for k in active]
    length(parts) == 1 && return only(parts)
    return Distributions.MixtureModel(parts, p.weights[active])
end

function density_score(model, X, y)
    p = predictive(model, X)
    return mean(Distributions.logpdf(marginal(p, i), y[i]) for i in eachindex(y))
end

interval_quantile(d, p) = Distributions.quantile(d, p)
function interval_quantile(d::Distributions.MixtureModel, p)
    # Component p-quantiles can round to a bracket whose CDF misses p. Wider
    # component probabilities provide a strict bracket for the library solver.
    parts = Distributions.components(d)
    lo = minimum(Distributions.quantile(part, p / 2) for part in parts)
    hi = maximum(Distributions.quantile(part, (1 + p) / 2) for part in parts)
    return Distributions.quantile_bisect(d, p, lo, hi)
end

"One fitted replicate is one Monte Carlo cluster, including its query rows."
function predictive_metrics(model, X, y, truth)
    p = predictive(model, X)
    latent = predictive(model, X; observation=false)
    pits, latent_pits, densities, widths = Float64[], Float64[], Float64[], Float64[]
    for i in eachindex(y)
        d = marginal(p, i)
        push!(pits, Distributions.cdf(d, y[i]))
        push!(latent_pits, Distributions.cdf(marginal(latent, i), truth[i]))
        push!(densities, Distributions.logpdf(d, y[i]))
        push!(widths, interval_quantile(d, 0.95) - interval_quantile(d, 0.05))
    end
    coverage(level) = mean((1 - level) / 2 .<= pits .<= (1 + level) / 2)
    bins = ntuple(k -> mean(((k - 1) / 10 .<= pits) .&
        (k == 10 ? pits .<= 1 : pits .< k / 10)), 10)
    return (;rmse=sqrt(mean(abs2, p.location - y)), mean_log_density=mean(densities),
        coverage50=coverage(0.5), coverage80=coverage(0.8),
        coverage90=coverage(0.9), coverage95=coverage(0.95),
        latent_coverage90=mean(0.05 .<= latent_pits .<= 0.95), width90=mean(widths),
        NamedTuple{ntuple(k -> Symbol("pit_bin", k), 10)}(bins)...,
        RD.model_size(model)...)
end

"Empirical public-fit coverage with separate structure, fit, validation, and test responses."
function bench_calibration(; replicates=500, seed=7100, output=nothing)
    rows = NamedTuple[]
    x = collect(range(-1.0, 1.0; length=64))
    query = collect(range(-0.98, 0.98; length=32))
    X, Q = reshape(x, :, 1), reshape(query, :, 1)
    for case in (:affine, :kink, :curve, :heteroskedastic, :heavy_tail), rep in 1:replicates
        rng = StableRNG(seed + rep)
        discovery, _ = response(rng, x, case)
        fitting, _ = response(rng, x, case)
        validation, _ = response(rng, query, case)
        testing, truth = response(rng, query, case)
        proposed = geometries(X, discovery)
        independent = fit_continuous_ensemble(X, fitting, proposed)
        alternative = fit_continuous_ensemble(X, fitting, proposed; coefficient_precision=1.0)
        choices = (independent, alternative)
        selected = argmax(map(model -> density_score(model, Q, validation), choices))
        adaptive = fit_continuous_ensemble(X, fitting, geometries(X, fitting))
        policies = ((:root, independent.components[1]), (:independent, independent),
            (:adaptive, adaptive), (:validated, choices[selected]))
        for (method, model) in policies
            λ = method === :validated && selected == 2 ? 1.0 : 0.01
            record!(rows, (;case, rep, method, coefficient_precision=λ,
                predictive_metrics(model, Q, testing, truth)...), nothing)
        end
        output === nothing || rep % 25 != 0 || RD.write_rows(output, rows)
    end
    output === nothing || RD.write_rows(output, rows)
    return rows
end

"Frozen-transform prior-predictive diagnostic, separate from the public plug-in procedure."
function bench_prior_calibration(; replicates=500, seed=8100, output=nothing)
    C = LinearTrees.Continuous
    x = collect(range(-1.0, 1.0; length=32))
    q = collect(range(-0.97, 0.97; length=32))
    X, Q = reshape(x, :, 1), reshape(q, :, 1)
    t, λ, a0, b0 = 1 / 3, 0.2, 3.0, 1.0
    c = [1.0, t, -1.0, -t]
    P = Matrix{Float64}(I, 4, 4) - c * c' / dot(c, c)
    routed(u) = u <= t ? [1.0, u, 0.0, 0.0] : [0.0, 0.0, 1.0, u]
    D = (hcat(ones(length(x)), x), reduce(vcat, transpose.(routed.(x))))
    H = (hcat(ones(length(q)), q), reduce(vcat, transpose.(routed.(q))))
    projections = (Matrix{Float64}(I, 2, 2), P)
    nodes = (C.root(1), C.grow(C.root(1), 1, 1, t))
    rows = NamedTuple[]
    for rep in 1:replicates
        rng = StableRNG(seed + rep)
        member = rand(rng, 1:2)
        variance = rand(rng, Distributions.InverseGamma(a0, b0))
        coef = sqrt(variance / λ) .* (projections[member] * randn(rng, size(D[member], 2)))
        fitting = D[member] * coef + sqrt(variance) .* randn(rng, length(x))
        truth = H[member] * coef
        testing = truth + sqrt(variance) .* randn(rng, length(q))
        components = [ContinuousTree{true}(C.fit_fixed(deepcopy(n), X, reshape(fitting, :, 1);
            pairs=Tuple{Int,Int}[], coefficient_precision=λ, noise_shape=a0, noise_rate=b0),
            [0.0], [1.0], [0.0], [1.0]) for n in nodes]
        logmass = [part.fit.post.score for part in components]
        weights = exp.(logmass .- maximum(logmass))
        weights ./= sum(weights)
        model = ContinuousEnsemble{true}(components, weights, zeros(2))
        record!(rows, (;case=:prior_predictive, rep, method=:frozen_mixture,
            predictive_metrics(model, Q, testing, truth)...), nothing)
    end
    output === nothing || RD.write_rows(output, rows)
    return rows
end

function calibration_summary(rows)
    metrics = (:coverage50, :coverage80, :coverage90, :coverage95, :latent_coverage90,
        :width90, :mean_log_density, :rmse, :effective_components,
        ntuple(k -> Symbol("pit_bin", k), 10)...)
    summaries = NamedTuple[]
    for (case, method) in unique([(row.case, row.method) for row in rows]), metric in metrics
        values = [getproperty(row, metric) for row in rows if row.case === case && row.method === method]
        push!(summaries, (;case, method, metric, estimate=mean(values),
            mcse=length(values) > 1 ? std(values) / sqrt(length(values)) : NaN,
            replicates=length(values)))
    end
    return summaries
end

function wide_fit(method, X, y, seed)
    tree(search=ExactSearch()) = fit_tree(X, y; max_depth=6, min_leaf=20,
        min_fit=40, nthreads=1, split_search=search)
    method === :pilot_exact && return tree()
    method === :pilot_binned && return tree(BinnedSearch(nbins=32, refine=false))
    method === :pilot_hybrid && return tree(HybridSearch(nbins=32))
    method === :ridge_refit && return refit_leaves(tree(), X, y;
        max_features=size(X, 2), lambda=1.0)
    if method in (:model_exact, :model_binned, :model_refined)
        search = method === :model_exact ? ExactSearch() :
            BinnedSearch(nbins=32, refine=method === :model_refined)
        return fit_model_tree(X, y; max_features=size(X, 2), max_depth=3,
            min_leaf=40, lambda=1.0, split_penalty=1.0, split_search=search)
    end
    method === :boost && return fit_boost(X, y; nrounds=30, eta=0.1,
        max_depth=2, min_leaf=20, min_fit=40, nthreads=1, rng=StableRNG(seed))
    method === :continuous || error("unknown scaling method")
    return fit_continuous_tree(X, y; max_splits=2, max_depth=2, min_leaf=40,
        n_thresholds=1, pairs=:none, candidate_search=:graph_pruned)
end

"Nested row/column subsets, paired methods, and complete fits including ridge refitting."
function bench_scaling(; dimensions=((1000, 8), (1000, 32), (8000, 8), (8000, 32), (32000, 8)),
        seeds=601:602, samples=3, output=nothing,
        methods=(:pilot_exact, :pilot_binned, :pilot_hybrid, :ridge_refit,
            :model_exact, :model_binned, :model_refined, :boost, :continuous))
    rows = NamedTuple[]
    nmax, pmax = maximum(first, dimensions), maximum(last, dimensions)
    signal(X) = 1 .+ 2X[:, 1] .- X[:, 2] .+ 1.5max.(X[:, 1], 0) .+
        0.8 .* (X[:, 3] .> 0) .* X[:, 4]
    for seed in seeds
        rng = StableRNG(seed)
        full = 2rand(rng, nmax, pmax) .- 1
        query = 2rand(rng, 2048, pmax) .- 1
        target = signal(full) + 0.2randn(rng, nmax)
        testing = signal(query) + 0.2randn(rng, 2048)
        for (n, p) in dimensions
            X, y, Q = full[1:n, 1:p], target[1:n], query[:, 1:p]
            for method in methods
                model, cost = measure(() -> wide_fit(method, X, y, seed); samples)
                record!(rows, (;n, p, seed, method, cost...,
                    rmse=sqrt(mean(abs2, predict(model, Q) - testing)),
                    RD.model_size(model)...), output)
            end
        end
    end
    return rows
end

end
