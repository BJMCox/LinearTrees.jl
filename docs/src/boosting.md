```@meta
CurrentModule = LinearTrees
```

# Boosting

[`fit_boost`](@ref) fits a gradient-boosted ensemble of PILOT-style linear
model trees. At each round it evaluates the base loss at the current ensemble
score, freezes its gradient and Hessian, and fits a weighted least-squares
[`Frozen`](@ref) tree to the resulting Newton working response. The new raw
tree score is multiplied by `eta` and added to the ensemble. This is the
second-order piecewise-linear boosting objective of [Guryanov
(2019)](https://doi.org/10.1007/978-3-030-37334-4_4) and [Shi, Li, and Li
(2019)](https://doi.org/10.24963/ijcai.2019/476).

## Quick start

Keep validation rows separate from fitting rows. This example uses the first
40 rows only for validation and the remaining rows only for fitting.

```@example boosting
using LinearTrees, Random

rng = Xoshiro(42)
X = rand(rng, 160, 3)
y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 160)
train, valid = 41:160, 1:40

b = fit_boost(X[train, :], y[train]; nrounds = 20, eta = 0.05,
    max_depth = 3, Xval = X[valid, :], yval = y[valid], patience = 5)
yhat = predict(b, X)
r = shap(b, X)
nothing
```

`predict` applies the base loss link. [`score`](@ref) returns the score scale;
use `score(b, X; clip = false)` to inspect the raw ensemble sum.

## Base-tree selection and sampling

Each base tree uses [`GainRule`](@ref), which permits constant, piecewise
constant, and piecewise linear nodes. `lambda_slope` and
`lambda_intercept` add L2 penalties to the slope and intercept closed forms.
`gamma` is the minimum regularised deviance gain per fitted score coordinate
needed to accept a split. These are the ridge and gain-floor conventions used
in [Chen and Guestrin (2016)](https://doi.org/10.1145/2939672.2939785).

`subsample` samples `max(1, round(Int, subsample * n))` fitting rows without
replacement for each tree. `colsample` samples `ceil(Int, colsample * p)`
features. Both draws use `rng`; pass a seeded `Xoshiro` to reproduce a fit.
Sampled-out rows receive zero weight for that tree, while every row receives
its fitted score update.

## Validation and truncation

Pass `Xval` and `yval` together to record validation deviance after each
round. Fitting stops after `patience` rounds without an improvement and keeps
only trees through the best validation round. Then `history` stores the kept
validation deviances and `validated` is true. Without validation data,
`history` stores training deviance for every fitted tree and no early stopping
runs.

The ensemble forms `f0 + eta * sum(tree scores)` before it clamps the score
once to the training loss bounds. This score clamp applies only when
`truncate = true`; `predict` then applies the loss link. Feature truncation
inside each tree also follows `truncate`; see [Truncation](@ref).

## Interpretation and non-smooth losses

[`shap`](@ref) is path-dependent TreeSHAP over the raw ensemble score. Its
base is `f0 + eta * sum(expected_score(tree))`, and each row's SHAP values
sum to its unclipped score minus that base. `ShapResult.clipped` marks rows
whose final ensemble clamp changed the score. [`feature_importance`](@ref)
sums split gains across trees. [`coeftable`](@ref) returns the local slope and
intercept of the unclipped ensemble score.

For [`Quantile`](@ref) and [`MAD`](@ref), a boosting round freezes the IRLS
weight at the current ensemble score. It is one IRLS boosting step, not an
exact L1 boosting step. A single non-smooth tree instead performs its own
five-pass node refit; see [IRLS for non-smooth losses](@ref).

## Printing and persistence

Use `TreeView(b, t)` to print base tree `t` with `AbstractTrees.print_tree`.
The package writes no boosted-model schema. Persist a `LinearBoost` directly
with JLD2, then load the same object graph:

```julia
using JLD2
JLD2.jldsave("boost.jld2"; boost = b)
b = JLD2.load("boost.jld2", "boost")
```

## Interfaces

For StatsAPI and Tables.jl inputs, use [`LinearBoostRegressorFit`](@ref) or
[`LinearBoostClassifierFit`](@ref) through `fit`. The classifier chooses
`Logistic` for two classes and `Softmax` otherwise. For MLJ, use
[`LinearBoostRegressor`](@ref) or [`LinearBoostClassifier`](@ref). MLJ treats
`nrounds` as its iteration parameter and reports `history`; use MLJ's
`IteratedModel` when you need validation-based stopping.
