```@meta
CurrentModule = LinearTrees
```

# LinearTrees.jl

LinearTrees fits decision trees with linear models along their paths. A tree
can represent both smooth trends and abrupt changes. Gradient boosting combines
these trees into an ensemble.

Use the package for regression, classification, count prediction, and quantile
regression on tabular data. It supports frequency weights, categorical features,
exact and approximate split search, local coefficients, and SHAP values.

## Installation

LinearTrees requires Julia 1.10 or later. Install the development version from
GitHub in your project environment:

```julia
using Pkg
Pkg.add(url = "https://github.com/BJMCox/LinearTrees.jl")
```

## A first model

Rows are observations and columns are features. Fit with [`fit_tree`](@ref)
and evaluate new rows with [`predict`](@ref):

```@example home
using LinearTrees, Random

rng = Xoshiro(42)
X = rand(rng, 300, 3)
y = 2 .* X[:, 1] .- X[:, 2] .+ 3 .* (X[:, 3] .> 0.6)

tree = fit_tree(X, y; max_depth = 4)
Xnew = rand(rng, 5, 3)
round.(predict(tree, Xnew); digits = 3)
```

The default loss is squared error. Choose another [loss](losses.md) for
probabilities, counts, positive responses, or conditional quantiles.
Use [`fit_boost`](@ref) for an ensemble.

## Learn the package

| Goal | Page |
|:--|:--|
| Fit and assess your first model | [Getting started](guide.md) |
| Understand a tree and control its size | [Tree fitting](trees.md) |
| Train an ensemble with validation | [Boosting](boosting.md) |
| Match the loss to your target | [Loss functions](losses.md) |
| Explain fitted predictions | [Interpretation](interpretation.md) |
| Use tables, MLJ, or saved models | [Interfaces and persistence](interfaces.md) |
| Choose split search and threading | [Performance](performance.md) |
| Look up a function or type | [API reference](api.md) |

## Model scope

Tree growth follows the [PILOT approach](https://doi.org/10.1007/s10994-024-06590-3), extended here to several losses and
boosted ensembles. Linear terms are fitted one feature at a time. A prediction
can involve several features because it sums terms along a path.

The package returns point predictions, class probabilities, or conditional
quantiles according to the loss. It does not provide Bayesian posterior
distributions, predictive intervals, missing-value imputation, or pruning.
Class probabilities do not carry a calibration guarantee.

The original algorithm is described by Raymaekers, Rousseeuw, Verdonck, and Yao
(2024), *Fast linear model trees by PILOT*, Machine Learning 113, 6561–6610.

## License and support

LinearTrees uses the [Apache License 2.0](https://github.com/BJMCox/LinearTrees.jl/blob/main/LICENSE).
Copyright 2026 Benjamin Cox.

Report bugs and request features through the
[issue tracker](https://github.com/BJMCox/LinearTrees.jl/issues).
