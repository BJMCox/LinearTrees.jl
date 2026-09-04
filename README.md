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
score, exact for this tree type rather than sampling-based. Not implemented
in this sub-project: pruning, gradient boosting, histogram-binned splits, and
a predictive distribution beyond the conditional mean; see
`docs/superpowers/specs/2026-09-03-lineartrees-design.md` for the full scope.

## Quick start

```julia
using LinearTrees

X = rand(1000, 4)
y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ 0.05 .* randn(1000)

tree = fit_tree(X, y; max_depth = 4)
ŷ = predict(tree, X)

print_tree(TreeView(tree))

φ = shap(tree, X)          # φ.values[i, j] is feature j's SHAP value for row i
```

See the [documentation](https://BJMCox.github.io/LinearTrees.jl/dev/) for the
full guide, the loss table, and the API reference.
