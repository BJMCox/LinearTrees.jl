# LinearTrees

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://BJMCox.github.io/LinearTrees.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://BJMCox.github.io/LinearTrees.jl/dev/)
[![Build Status](https://github.com/BJMCox/LinearTrees.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/LinearTrees.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/BJMCox/LinearTrees.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BJMCox/LinearTrees.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

LinearTrees fits PILOT-style linear model trees, generalised to any
twice-differentiable loss (or an IRLS-approximated one: `Quantile`, `MAD`).
Each node is a constant, a line, a broken line, or a pair of lines, chosen by
a BIC selection rule; the package covers regression, binary and multiclass
classification, count and rate targets, categorical features, sample
weights, split-gain feature importance, and a local linear model
(`coeftable`) at any point. SHAP is path-dependent TreeSHAP on the unclipped
score, exact for this tree type rather than sampling-based. The package also
fits second-order gradient-boosted ensembles of linear trees with validation
early stopping. Pruning, histogram-binned splits, and predictive distributions
beyond the conditional mean are not implemented.

## Quick start

```julia
using LinearTrees, Random

rng = Xoshiro(42)
X = rand(rng, 2000, 5)
y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 2000)
train, valid = 401:2000, 1:400

b = fit_boost(X[train, :], y[train]; nrounds = 200, eta = 0.05,
    max_depth = 4, Xval = X[valid, :], yval = y[valid], patience = 20)
yhat = predict(b, X)
r = shap(b, X)
```

See the [documentation](https://BJMCox.github.io/LinearTrees.jl/dev/) for the
full guide, the [boosting guide](https://BJMCox.github.io/LinearTrees.jl/dev/boosting/),
the loss table, and the API reference.
