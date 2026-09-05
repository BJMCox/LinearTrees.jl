# Deterministic data sets shared by make_fixtures.jl and test/partition.jl.
# Each tuple: (name, X, y, loss, fit_tree keyword arguments). Column 6 of `X`
# holds rounded values, so ties are common and the partition must stay stable.
function partition_cases()
    rng = StableRNG(2026)
    n = 4_000
    X = rand(rng, n, 6)
    X[:, 6] .= round.(X[:, 6]; digits = 1)
    y = sin.(4 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .- 2 .* (X[:, 4] .> 0.6) .+ X[:, 6] .+ 0.1 .* randn(rng, n)
    lvl = Float64.(rand(rng, 1:8, n))
    Xc = hcat(lvl, X[:, 1:4])
    yk = [X[i, 1] > 0.5 ? 1 : lvl[i] <= 4 ? 2 : 3 for i in 1:n]
    yq = y .+ 0.5 .* randn(rng, n) .^ 3
    w = Float64.(rand(rng, 0:3, n))
    force = MinDeviance((LIN, PCON, BLIN, PLIN))   # never stops on con, so depth and min_leaf bind
    return [
        ("mse_bic", X, y, MSE(), (max_depth = 12,)),
        ("mse_forced", X, y, MSE(), (max_depth = 9, rule = force, min_leaf = 5, min_fit = 10)),
        ("softmax_cat", Xc, yk, Softmax(3), (max_depth = 7, rule = force, categorical = [1], min_leaf = 20, min_fit = 40)),
        ("quantile_weighted", X, yq, Quantile(0.3), (max_depth = 8, rule = force, weights = w, min_leaf = 20, min_fit = 40)),
    ]
end
