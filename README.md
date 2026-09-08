# LinearTrees.jl

[![CI](https://github.com/BJMCox/LinearTrees.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/LinearTrees.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/BJMCox/LinearTrees.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BJMCox/LinearTrees.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://bjmcox.github.io/LinearTrees.jl/dev/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Decision trees with linear models along their paths, and gradient-boosted ensembles of those trees.
Supports regression, classification, categorical features, weighted fitting, and exact or approximate split search.

```julia
using LinearTrees, Random

# Combine a smooth trend with a threshold effect.
rng = Xoshiro(42)
X = rand(rng, 400, 3)             # observations × features
y = 2 .* X[:, 1] .- X[:, 2] .+ 3 .* (X[:, 3] .> 0.6) .+
    0.1 .* randn(rng, 400)
train, test = 1:300, 301:400

# Fit one tree, then predict held-out rows.
tree = fit_tree(X[train, :], y[train]; max_depth = 4)
yhat = predict(tree, X[test, :])
size(yhat)                       # (100,)

# Boost shallow trees using the same matrix interface.
boost = fit_boost(X[train, :], y[train]; nrounds = 50, max_depth = 3)
boosted_predictions = predict(boost, X[test, :])
```

A prediction sums the linear models along its tree path.
Pass `Logistic()` as the third fitting argument for binary probabilities.

[Documentation](https://bjmcox.github.io/LinearTrees.jl/dev/) · [Apache 2.0 license](LICENSE)
