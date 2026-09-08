```@meta
CurrentModule = LinearTrees
```

# Boosted trees

[`fit_boost`](@ref) builds an additive ensemble of linear model trees.
Each round fits one tree to the current loss gradient and Hessian.
The learning rate `eta` scales that tree before adding it to the ensemble.

Use boosting when one tree cannot capture enough structure.
Start with shallow trees and a small learning rate.
Use separate validation data to choose the retained number of rounds.

## Fit an ensemble

The matrix interface accepts observations in rows and features in columns.
Choose the loss explicitly when the default [`MSE`](@ref) does not match the target.

```@example boosting
using LinearTrees
using Random

rng = Xoshiro(42)
X = rand(rng, 160, 3)
y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.05 .* randn(rng, 160)

model = fit_boost(X, y;
    nrounds = 40,
    eta = 0.1,
    max_depth = 3,
    rng = Xoshiro(7),
)

yhat = predict(model, X)
(nrounds(model), model.history[end], yhat[1:3])
```

[`predict`](@ref) returns values on the response scale.
[`score`](@ref) returns values on the loss score scale.
These scales match for identity-link losses such as `MSE`.
They differ for losses such as [`Logistic`](@ref), [`Poisson`](@ref), and [`Softmax`](@ref).

## Use held-out validation

Pass `Xval` and `yval` together.
Pass `wval` when validation observations have weights.
The fitter records validation deviance after each round.
It stops after `patience` rounds without improvement.
It then retains the trees through the best validation round.

```@example boosting-validation
using LinearTrees
using Random

rng = Xoshiro(19)
X = rand(rng, 180, 3)
y = 2 .* X[:, 1] .+ sin.(5 .* X[:, 2]) .+ 0.2 .* randn(rng, 180)

train = 1:140
valid = 141:180
model = fit_boost(X[train, :], y[train];
    Xval = X[valid, :],
    yval = y[valid],
    nrounds = 100,
    patience = 8,
    eta = 0.1,
    max_depth = 3,
)

(model.validated, nrounds(model), length(model.history))
```

`model.history[t]` contains the retained validation deviance after round `t`.
Without validation data, it contains training deviance for every fitted round.
Early stopping only runs when validation data is present.
[`nrounds`](@ref) reports the retained tree count.

Keep validation rows independent from fitting rows.
Repeated tuning against one validation set can still overfit that set.

## Control the ensemble

`nrounds` sets the maximum tree count.
`eta` sets every tree's contribution.
Smaller `eta` usually needs more rounds.
`max_depth`, `min_fit`, `min_leaf`, and `min_sum_hessian` limit tree growth.

Each base tree uses [`GainRule`](@ref).
`lambda_slope` penalizes fitted slopes.
`lambda_intercept` penalizes fitted intercept updates.
`gamma` requires more gain before accepting a split.
All three values must be finite and nonnegative.

`subsample` selects a fraction of training rows for each tree.
`colsample` selects a fraction of features for each tree.
Both values lie in `(0, 1]`.
Sampling occurs without replacement.
Sampled-out rows still receive the fitted tree's update.

Pass a freshly seeded `AbstractRNG` to reproduce a sampled fit.
The fitter advances the supplied generator.
Thread count does not change a fit made from the same random stream.

```@example boosting-options
using LinearTrees
using Random

rng = Xoshiro(31)
X = rand(rng, 120, 5)
y = X[:, 1] .- 2 .* X[:, 3] .+ 0.1 .* randn(rng, 120)

model = fit_boost(X, y;
    nrounds = 25,
    eta = 0.05,
    max_depth = 2,
    lambda_slope = 2.0,
    lambda_intercept = 1.0,
    gamma = 0.1,
    subsample = 0.8,
    colsample = 0.6,
    rng = Xoshiro(11),
)

nrounds(model)
```

Choose a split search as described in [Performance](performance.md).
Approximate searches have extra loss and feature restrictions.

## Understand scores and truncation

The raw ensemble score is `f0 + eta * sum(tree scores)`.
Call `score(model, X; clip = false)` to obtain that sum.
With `truncate = true`, [`score`](@ref) clamps the sum to bounds derived from the training target.
[`predict`](@ref) applies the loss link after this clamp.

Truncation also bounds numeric features inside each base tree.
Set `truncate = false` to disable both forms of truncation.
Inspect [Interpreting models](interpretation.md) before comparing explanations with clipped predictions.

For [`Quantile`](@ref) and [`MAD`](@ref), each round freezes one set of IRLS weights.
Thus, each boosted round performs one IRLS step.
See [Losses](losses.md) for loss-specific score and response behavior.
