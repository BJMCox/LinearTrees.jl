using BenchmarkTools
using EvoTrees
using LinearTrees
using StableRNGs

const LT_CONFIG = (
    nrounds = 100,
    eta = 0.1,
    max_depth = 5,
    min_fit = 10,
    min_leaf = 5,
    min_sum_hessian = 1.0,
    lambda_slope = 1.0,
    lambda_intercept = 1.0,
    gamma = 0.0,
    subsample = 1.0,
    colsample = 1.0,
    truncate = false,
    nthreads = Threads.nthreads(),
)

const EV_CONFIG = EvoTrees.EvoTreeRegressor(
    loss = :mse,
    metric = :mse,
    nrounds = 100,
    bagging_size = 1,
    early_stopping_rounds = typemax(Int),
    early_stopping_tolerance = 0.0,
    L2 = 1.0,
    lambda = 0.0,
    gamma = 0.0,
    eta = 0.1,
    max_depth = 5,
    min_weight = 1.0,
    rowsample = 1.0,
    colsample = 1.0,
    nbins = 64,
    alpha = 0.5,
    alphas = [0.1, 0.5, 0.9],
    monotone_constraints = Dict{Int,Int}(),
    tree_type = :binary,
    seed = 1,
    device = :cpu,
)

function boost_data()
    rng = StableRNG(1)
    n, p = 100_000, 20
    X = rand(rng, n, p)
    ylin = X * randn(rng, p) .+ 0.1 .* randn(rng, n)
    # Preserve bench/run.jl's RNG stream before constructing the step target.
    _ = [X[i, 1] > 0.5 ? 2X[i, 2] : -X[i, 3] for i in 1:n] .+ 0.1 .* randn(rng, n)
    ystep = sum(floor.(5 .* X[:, j]) for j in 1:6) .+ X[:, 7] .* X[:, 8] .+ 0.1 .* randn(rng, n)
    return X, ylin, ystep
end

rmse(yhat, y) = sqrt(sum(abs2, yhat .- y) / length(y))

function report_trial(name, model, trial)
    med = median(trial)
    best = minimum(trial)
    println("RESULT\t", name, '\t', model, "\tthreads=", Threads.nthreads(),
        "\tmedian_ns=", med.time, "\tmin_ns=", best.time,
        "\tmemory_bytes=", med.memory, "\tallocs=", med.allocs)
    return nothing
end

function bench_boost(name, X, y)
    ntrain = size(X, 1) ÷ 2
    Xtr, Xte = X[1:ntrain, :], X[(ntrain + 1):end, :]
    ytr, yte = y[1:ntrain], y[(ntrain + 1):end]
    @assert size(Xtr) == (ntrain, size(X, 2))
    @assert size(Xte) == (length(yte), size(X, 2))

    ltfit = () -> fit_boost(Xtr, ytr; LT_CONFIG..., rng = StableRNG(1))
    evfit = () -> EvoTrees.fit(EV_CONFIG; x_train = Xtr, y_train = ytr, verbosity = 0)

    # Compile and verify both models before timing. For MSE, LinearTrees'
    # direct prediction equals its raw additive score because truncate=false.
    b = ltfit()
    ev = evfit()
    pred_lt = LinearTrees.predict(b, Xte)
    pred_ev = EvoTrees.predict(ev, Xte)
    @assert size(pred_lt) == size(yte)
    @assert size(pred_ev) == size(yte)
    @assert pred_lt == LinearTrees.score(b, Xte; clip = false)
    @assert all(isfinite, pred_lt) && all(isfinite, pred_ev)
    rmse_lt, rmse_ev = rmse(pred_lt, yte), rmse(pred_ev, yte)

    println("== boost ", name, " (", ntrain, " train / ", length(yte), " test) ==")
    println("LinearTrees config: ", LT_CONFIG)
    println("EvoTrees config: ", EV_CONFIG)
    println("LinearTrees.fit_boost:")
    lttrial = @benchmark $ltfit() samples = 3 evals = 1 seconds = 300
    display(lttrial)
    report_trial(name, "LinearTrees.fit_boost", lttrial)
    println()
    println("EvoTrees.fit:")
    evtrial = @benchmark $evfit() samples = 3 evals = 1 seconds = 300
    display(evtrial)
    report_trial(name, "EvoTrees.fit", evtrial)
    println("RESULT\t", name, "\tRMSE\tLinearTrees=", rmse_lt, "\tEvoTrees=", rmse_ev)
    println()
    return nothing
end

println("Julia: ", VERSION)
println("EvoTrees: ", Base.pkgversion(EvoTrees))
println("CPU: ", Sys.cpu_info()[1].model)
println("CPU_THREADS: ", Sys.CPU_THREADS)
println("Threads.nthreads(): ", Threads.nthreads())
println("seed: StableRNG(1); n = 100_000; p = 20; train/test = 50_000/50_000; samples = 3; evals = 1")
println()

X, ylin, ystep = boost_data()
bench_boost("linear", X, ylin)
bench_boost("step", X, ystep)
