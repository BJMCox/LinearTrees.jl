```@meta
CurrentModule = LinearTrees
```

# Getting started

This tutorial fits a regression tree, measures its error on unseen rows, and
then fits a binary classifier. It uses only LinearTrees and Julia standard libraries.

## Prepare the data

The matrix interface takes a numeric matrix `X` and a target vector `y`.
Each row corresponds to one target. Keep the feature order the same when predicting.

The example combines a linear trend, a threshold effect, and noise:

```@example tutorial
using LinearTrees, Random, Statistics

rng = Xoshiro(42)
X = rand(rng, 400, 3)
y = 2 .* X[:, 1] .- X[:, 2] .+ 3 .* (X[:, 3] .> 0.6) .+
    0.1 .* randn(rng, 400)

train, test = 1:300, 301:400
Xtrain, ytrain = X[train, :], y[train]
Xtest, ytest = X[test, :], y[test]
(size(Xtrain), length(ytrain), size(Xtest))
```

The rows were sampled independently, so this leaves an independent test sample.
For time series or grouped observations, split according to that structure.
Fit preprocessing on training rows alone.

Feature values must be finite and numeric at this interface. Encode categorical
features as described in [Tree fitting](trees.md), or use the
[table interface](interfaces.md). Handle missing values before fitting.

## Fit a regression tree

```@example tutorial
tree = fit_tree(Xtrain, ytrain; max_depth = 4)
yhat = predict(tree, Xtest)
round.(yhat[1:5]; digits = 3)
```

[`fit_tree`](@ref) uses [`MSE`](@ref) and [`BIC`](@ref) by default. BIC balances
fit against node complexity. `max_depth` caps branching depth. A tree may stop
earlier when another split does not improve its selection score.

## Measure error on unseen rows

Compare the test error with a constant baseline fitted on the same training rows:

```@example tutorial
rmse = sqrt(mean(abs2, yhat .- ytest))
baseline_rmse = sqrt(mean(abs2, mean(ytrain) .- ytest))
@assert rmse < baseline_rmse # hide
(tree_rmse = round(rmse; digits = 3),
    baseline_rmse = round(baseline_rmse; digits = 3))
```

Use validation data or cross-validation to choose settings such as `max_depth`
and `min_leaf`. Reserve the test data for the final assessment.

## Fit binary probabilities

The direct binary interface uses targets `0` and `1`. [`Logistic`](@ref)
returns the probability of target `1`:

```@example tutorial
probability = inv.(1 .+ exp.(-4 .* (X[:, 1] .- X[:, 2])))
ybinary = Float64.(rand(rng, 400) .< probability)

classifier = fit_tree(Xtrain, ybinary[train], Logistic(); max_depth = 3)
p = predict(classifier, Xtest)
@assert all(0 .< p .< 1) # hide
round.(p[1:5]; digits = 3)
```

Choose a threshold when a decision is needed. A threshold of `0.5` is a
starting point. Different costs for false positives and false negatives can
justify another value.

```@example tutorial
labels = p .>= 0.5
mean(labels .== ybinary[test])
```

For named labels and probability columns with a stored class order, use the
[classifier wrapper](interfaces.md). For more classes, see [Loss functions](losses.md).

## Continue from here

[Tree fitting](trees.md) explains node models, weights, categorical features,
and extrapolation. [Boosting](boosting.md) builds an ensemble with early stopping.
[Interpretation](interpretation.md) shows how to inspect a fitted model.
