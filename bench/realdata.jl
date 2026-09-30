module RealDataBench

using BenchmarkTools, DelimitedFiles, Distributions, Downloads, LinearTrees
using Random, SHA, StableRNGs, Statistics

export bench_realdata, bench_realdata_scaling, load_realdata, realdata_split

const DATASETS = (
    (; name=:airfoil, file="airfoil_self_noise.dat", n=1503, p=5,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        sha256="74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"),
    (; name=:abalone, file="abalone.data", n=4177, p=9,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        sha256="de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f"),
    (; name=:casp, file="CASP.csv", n=45730, p=9,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        sha256="4277cfcb4e91a181746cbc654f001b57951c9e6a80f4f795fdb5c807e0848f40"),
)

const METHODS = (:global_ridge, :pilot, :pilot_huber, :boost, :ridge_refit,
    :model_exact, :model_binned, :model_refined, :model_pruned, :continuous, :finite_ensemble)
const SCALING_METHODS = (:global_ridge, :pilot, :boost, :ridge_refit,
    :model_exact, :model_binned, :model_refined, :model_pruned)

function load_realdata(name; cache_dir=joinpath(tempdir(), "lineartrees-realdata-cache"))
    spec = only(filter(spec -> spec.name == name, DATASETS))
    mkpath(cache_dir)
    path = joinpath(cache_dir, spec.file)
    if !isfile(path)
        temporary = path * ".download"
        Downloads.download(spec.url, temporary)
        bytes2hex(open(sha256, temporary)) == spec.sha256 || error("download hash mismatch: $name")
        mv(temporary, path)
    end
    bytes2hex(open(sha256, path)) == spec.sha256 || error("cached data hash mismatch: $name")
    if name === :airfoil
        data = readdlm(path, Float64)
        X, y = data[:, 1:5], data[:, 6]
    elseif name === :abalone
        data = readdlm(path, ',', Any)
        # Fixed source vocabulary: infant is the reference category.
        all(sex -> sex in ("M", "F", "I"), data[:, 1]) || error("unexpected abalone sex")
        X = hcat(Float64.(data[:, 2:8]), Float64.(data[:, 1] .== "M"),
            Float64.(data[:, 1] .== "F"))
        y = Float64.(data[:, 9])
    else
        data = readdlm(path, ',', Float64; skipstart=1)
        X, y = data[:, 2:10], data[:, 1]
    end
    size(X) == (spec.n, spec.p) || error("unexpected dataset shape: $name")
    all(isfinite, X) && all(isfinite, y) || error("nonfinite source data: $name")
    return (; X, y, spec)
end

function predictor_groups(X)
    lookup = Dict{Tuple,Int}()
    groups = Vector{Int}[]
    for i in axes(X, 1)
        key = Tuple(view(X, i, :))
        index = get(lookup, key, 0)
        if iszero(index)
            push!(groups, Int[])
            index = length(groups)
            lookup[key] = index
        end
        push!(groups[index], i)
    end
    return groups
end

function capped_groups(groups, order, cap)
    rows = Int[]
    for index in order
        group = groups[index]
        length(rows) + length(group) <= cap || break
        append!(rows, group)
    end
    isempty(rows) && error("training cap smaller than every predictor group")
    return rows
end

"Split predictor groups 60/20/20, cap training groups, then fit all scales on training only."
function realdata_split(data, seed; train_cap=2000)
    groups = predictor_groups(data.X)
    order = randperm(StableRNG(seed), length(groups))
    train_groups, validation, test = Int[], Int[], Int[]
    ntrain = 0
    for index in order
        if ntrain < floor(Int, 0.6 * length(data.y))
            push!(train_groups, index)
            ntrain += length(groups[index])
        elseif ntrain + length(validation) < floor(Int, 0.8 * length(data.y))
            append!(validation, groups[index])
        else
            append!(test, groups[index])
        end
    end
    train = capped_groups(groups, train_groups, train_cap)
    Xtrain = data.X[train, :]
    xcenter = mean(Xtrain; dims=1)
    xscale = std(Xtrain; dims=1, corrected=false)
    xscale[iszero.(xscale)] .= 1
    ycenter = mean(data.y[train])
    yscale = std(data.y[train]; corrected=false)
    iszero(yscale) && (yscale = 1.0)
    return (; Xtrain=(Xtrain .- xcenter) ./ xscale,
        ytrain=(data.y[train] .- ycenter) ./ yscale,
        Xval=(data.X[validation, :] .- xcenter) ./ xscale,
        yval=(data.y[validation] .- ycenter) ./ yscale,
        Xtest=(data.X[test, :] .- xcenter) ./ xscale,
        ytest=(data.y[test] .- ycenter) ./ yscale,
        train, validation, test, ycenter, yscale, ntrain_available=ntrain,
        split_hash=bytes2hex(sha256(join((join(train, ','), join(validation, ','),
            join(test, ',')), ';'))))
end

function continuous_fit(split; max_splits=3, pairs=:none)
    return fit_continuous_tree(split.Xtrain, split.ytrain; pairs,
        candidate_search=:graph_pruned, max_splits, max_depth=3, min_leaf=40,
        n_thresholds=3, split_penalty=2.0)
end

function finite_ensemble_fit(split)
    geometries = [continuous_fit(split; max_splits=0), continuous_fit(split),
        continuous_fit(split; pairs=[(1, 2)])]
    # The API rejects duplicate model space entries. Equal leaf boxes and basis
    # define the same model even when a different split order produced them.
    keys = [(sort([(Tuple(node.lo), Tuple(node.hi)) for node in model.fit.nodes
        if node.feature == 0]), sort(Tuple.(model.fit.terms))) for model in geometries]
    keep = [i for i in eachindex(keys) if !any(isequal(keys[i]), keys[1:i-1])]
    return fit_continuous_ensemble(split.Xtrain, split.ytrain, geometries[keep])
end

function fit_method(method, split, seed)
    X, y = split.Xtrain, split.ytrain
    features = collect(axes(X, 2))
    tree() = fit_tree(X, y; max_depth=6, min_leaf=20, min_fit=40, nthreads=1)
    modeltree(search) = fit_model_tree(X, y; features, max_features=length(features),
        lambda=1.0, max_depth=3, min_leaf=40, split_penalty=1.0, split_search=search)
    if method === :global_ridge
        return fit_model_tree(X, y; features, max_features=length(features),
            lambda=1.0, max_depth=0, truncate=false)
    elseif method === :pilot
        return tree()
    elseif method === :pilot_huber
        return fit_tree(X, y, Huber(1.0); max_depth=6, min_leaf=20, min_fit=40, nthreads=1)
    elseif method === :boost
        return fit_boost(X, y; nrounds=30, max_depth=3, min_leaf=20, min_fit=40,
            eta=0.1, subsample=0.8, rng=StableRNG(seed), nthreads=1,
            Xval=split.Xval, yval=split.yval, patience=5)
    elseif method === :ridge_refit
        return refit_leaves(tree(), X, y; features, max_features=length(features), lambda=1.0)
    elseif method === :model_exact
        return modeltree(ExactSearch())
    elseif method === :model_binned
        return modeltree(BinnedSearch(; nbins=32, refine=false))
    elseif method === :model_refined
        return modeltree(BinnedSearch(; nbins=32))
    elseif method === :model_pruned
        return prune_refit(modeltree(ExactSearch()), X, y, split.Xval, split.yval;
            features, max_features=length(features), lambda=1.0)
    elseif method === :continuous
        return continuous_fit(split)
    elseif method === :finite_ensemble
        return finite_ensemble_fit(split)
    end
    error("unknown method: $method")
end

serial_predict(model, X) = predict(model, X; nthreads=1)
serial_predict(model::Union{ContinuousTree,ContinuousEnsemble}, X) = predict(model, X)

function time_operation(operation, samples; warm=true)
    warm && operation() # Compile and warm the complete operation before measuring.
    trial = run(@benchmarkable($operation(), evals=1); samples, seconds=300)
    estimate = median(trial)
    return (; seconds=estimate.time / 1e9, bytes=estimate.memory, allocs=estimate.allocs)
end

function model_size(model::LinearTree)
    return (; nodes=length(model.nodes), leaves=count(node -> node.feature == 0, model.nodes),
        coefficients=sum(node -> node.model == CON ? 1 :
            node.model in (LIN, PCON) ? 2 : node.model == BLIN ? 3 : 4, model.nodes),
        components=1, effective_components=1.0, largest_weight=1.0, component_weights="1")
end

function model_size(model::RefitTree)
    return (; model_size(model.routing)..., leaves=length(model.leaves),
        coefficients=sum(leaf -> 1 + length(leaf.slopes), model.leaves))
end

function model_size(model::LinearBoost)
    return (; nodes=sum(tree -> length(tree.nodes), model.trees; init=0),
        leaves=sum(tree -> count(node -> node.feature == 0, tree.nodes), model.trees; init=0),
        coefficients=1 + sum(tree -> model_size(tree).coefficients, model.trees; init=0),
        components=length(model.trees), effective_components=missing,
        largest_weight=missing, component_weights="")
end

function model_size(model::ContinuousTree)
    return (; nodes=length(model.fit.nodes), leaves=length(model.fit.leaves),
        coefficients=length(model.fit.post.coef), components=1,
        effective_components=1.0, largest_weight=1.0, component_weights="1")
end

function model_size(model::ContinuousEnsemble)
    return (; nodes=sum(component -> length(component.fit.nodes), model.components),
        leaves=sum(component -> length(component.fit.leaves), model.components),
        coefficients=sum(component -> length(component.fit.post.coef), model.components),
        components=length(model.components), effective_components=inv(sum(abs2, model.weights)),
        largest_weight=maximum(model.weights), component_weights=join(model.weights, ';'))
end

const NO_CALIBRATION = (; mean_log_density=missing, coverage50=missing, coverage80=missing,
    coverage90=missing, coverage95=missing, pit_ks=missing, mean_predictive_sd=missing)

calibration(model, split) = NO_CALIBRATION

student_t(component, i) = LocationScale(component.location[i], sqrt(component.scale2[i]),
    TDist(component.dof))

function calibration(model::Union{ContinuousTree,ContinuousEnsemble}, split)
    distribution = predictive(model, split.Xtest; observation=true)
    pits, logdensities, deviations = Float64[], Float64[], Float64[]
    for i in eachindex(split.ytest)
        marginal = model isa ContinuousTree ? student_t(distribution, i) :
            MixtureModel([student_t(component, i) for component in distribution.components],
                distribution.weights)
        push!(pits, cdf(marginal, split.ytest[i]))
        push!(logdensities, logpdf(marginal, split.ytest[i]) - log(split.yscale))
        push!(deviations, std(marginal) * split.yscale)
    end
    sorted = sort(pits)
    n = length(sorted)
    ks = max(maximum((1:n) ./ n .- sorted), maximum(sorted .- (0:n-1) ./ n))
    coverage(level) = mean(p -> (1 - level) / 2 <= p <= (1 + level) / 2, pits)
    return (; mean_log_density=mean(logdensities), coverage50=coverage(0.5),
        coverage80=coverage(0.8), coverage90=coverage(0.9), coverage95=coverage(0.95),
        pit_ks=ks, mean_predictive_sd=mean(deviations))
end

function write_rows(path, rows)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join((value === missing ? "" : string(value) for value in values(row)), ','))
        end
    end
    mv(temporary, path; force=true)
end

function evaluate_method(method, split, seed; samples=3)
    fit_operation = () -> fit_method(method, split, seed)
    model = fit_operation()
    fit_timing = time_operation(fit_operation, samples; warm=false)
    prediction_operation = () -> serial_predict(model, split.Xtest)
    predictions = prediction_operation()
    all(isfinite, predictions) || error("nonfinite predictions: $method")
    prediction_timing = time_operation(prediction_operation, samples)
    uncertainty_timing = model isa Union{ContinuousTree,ContinuousEnsemble} ?
        time_operation(() -> predictive(model, split.Xtest; observation=true), samples) :
        (; seconds=missing, bytes=missing, allocs=missing)
    residuals = predictions .- split.ytest
    nrmse = sqrt(mean(abs2, residuals))
    return (; method, rmse=nrmse * split.yscale, nrmse,
        mae=mean(abs, residuals) * split.yscale,
        r2=1 - sum(abs2, residuals) / sum(abs2, split.ytest .- mean(split.ytest)),
        baseline_rmse=sqrt(mean(abs2, split.ytest)) * split.yscale,
        fit_seconds=fit_timing.seconds, fit_bytes=fit_timing.bytes, fit_allocs=fit_timing.allocs,
        predict_seconds=prediction_timing.seconds, predict_bytes=prediction_timing.bytes,
        predict_allocs=prediction_timing.allocs, predictive_seconds=uncertainty_timing.seconds,
        predictive_bytes=uncertainty_timing.bytes, predictive_allocs=uncertainty_timing.allocs,
        model_bytes=Base.summarysize(model), model_size(model)..., calibration(model, split)...)
end

function run_cohort!(rows, data, seed, train_cap, methods, samples, output_path, on_result, cohort)
    split = realdata_split(data, seed; train_cap)
    # Rotate method order across seeds to reduce fixed-order timing bias.
    for method in circshift(collect(methods), mod(seed, length(methods)))
        metrics = evaluate_method(method, split, seed; samples)
        row = (; cohort, dataset=data.spec.name, source_sha256=data.spec.sha256, seed,
            train_cap, ntrain=length(split.train), ntrain_available=split.ntrain_available,
            nval=length(split.validation), ntest=length(split.test), p=size(split.Xtrain, 2),
            split_hash=split.split_hash, target_center=split.ycenter, target_scale=split.yscale,
            metrics...)
        push!(rows, row)
        output_path === nothing || write_rows(output_path, rows)
        on_result(row)
    end
    return rows
end

"""
    bench_realdata(; seeds=301:303, datasets=(:airfoil,:abalone,:casp), train_cap=2000,
        methods=METHODS, samples=3, output_path=nothing, on_result=identity)

Fixed paired real-data comparison. Each fit includes all geometry search/refit/
pruning/validation selection costs. Test data is used only for final evaluation.
"""
function bench_realdata(; seeds=301:303, datasets=(:airfoil, :abalone, :casp),
        train_cap=2000, methods=METHODS, samples=3, output_path=nothing,
        on_result=identity, cache_dir=joinpath(tempdir(), "lineartrees-realdata-cache"))
    rows = NamedTuple[]
    for dataset in datasets
        data = load_realdata(dataset; cache_dir)
        for seed in seeds
            run_cohort!(rows, data, seed, train_cap, methods, samples, output_path, on_result, :paired)
        end
    end
    return rows
end

"CASP scaling on nested training group prefixes; probabilistic models remain in the paired cohort."
function bench_realdata_scaling(; seed=301, caps=(2000, 8000, typemax(Int)),
        methods=SCALING_METHODS, samples=3, output_path=nothing, on_result=identity,
        cache_dir=joinpath(tempdir(), "lineartrees-realdata-cache"))
    rows = NamedTuple[]
    data = load_realdata(:casp; cache_dir)
    for train_cap in caps
        run_cohort!(rows, data, seed, train_cap, methods, samples, output_path, on_result, :scaling)
    end
    return rows
end

end # module

using .RealDataBench: bench_realdata, bench_realdata_scaling
