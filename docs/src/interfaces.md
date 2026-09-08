```@meta
CurrentModule = LinearTrees
```

# Tables and model interfaces

Use the direct API when data is already a numeric matrix.
Use StatsAPI wrappers for Tables.jl inputs, stored preprocessing, and standard model statistics.
Use the MLJ models inside an MLJ workflow.

## StatsAPI regression

[`LinearTreeRegressorFit`](@ref) stores the tree, encoded training matrix,
response, weights, and table encoder.
Keywords after `X` and `y` pass through to [`fit_tree`](@ref).

```@example interfaces
using LinearTrees

X = (temperature = collect(range(8.0, 25.0; length = 60)),
     weekend = repeat([0, 1]; outer = 30))
y = 0.4 .* X.temperature .+ 2 .* X.weekend

fit = LinearTrees.fit(LinearTreeRegressorFit, X, y; max_depth = 1)
ŷ = LinearTrees.predict(fit, X)
(nobs(fit), dof(fit), maximum(abs, residuals(fit)), deviance(fit))
```

Pass `weights=...` for observation weights.
[`weights`](@ref) returns the stored weights.
[`nobs`](@ref) counts all supplied rows, including zero-weight rows.
[`dof`](@ref) counts stored coefficients.
It is not an effective degrees-of-freedom estimate.

## StatsAPI classification

[`LinearTreeClassifierFit`](@ref) accepts arbitrary sortable labels.
It fits [`Logistic`](@ref) for two observed classes and [`Softmax`](@ref) otherwise.
Predictions form an `n × K` probability matrix.
The columns follow `fit.classes`.

```@example interfaces
X = (x1 = repeat([-2.0, -1.0, 1.0, 2.0, 0.0, 0.5]; outer = 10),
     x2 = repeat([0.0, 1.0, 0.0, 1.0, 2.0, 2.5]; outer = 10))
y = repeat(["low", "low", "high", "high", "middle", "middle"]; outer = 10)

fit = LinearTrees.fit(LinearTreeClassifierFit, X, y; max_depth = 1)
P = LinearTrees.predict(fit, X)
(fit.classes, size(P), P[1:3, :])
```

For a binary fit, `fit.classes[1]` is the positive class.
The first probability column is its probability.

The boosting wrappers are [`LinearBoostRegressorFit`](@ref) and [`LinearBoostClassifierFit`](@ref).
They follow the same table and class conventions.
See [Boosting](boosting.md).

## Table schema and categorical columns

The wrappers accept matrices and Tables.jl tables, including named tuples and DataFrames.
Table column names must match the fitted encoder during prediction.
Columns are retrieved by name, so their order may change.
Numeric values convert to `Float64`.
Missing, non-finite, and non-numeric numeric-column values are unsupported.

A column is categorical when it provides a `DataAPI.refpool`, as CategoricalArrays does.
Training levels are sorted by their string form and stored with the model.
Prediction uses this stored map.
Changing a categorical pool's internal order does not change its encoding.

The default `unseen=:error` rejects a categorical level absent during training.
Set `unseen=:right` at fit time to route unseen levels right at categorical splits.

```julia
using LinearTrees
using CategoricalArrays
using DataFrames

X = DataFrame(colour = categorical(repeat(["red", "blue", "green"]; outer = 20)),
              size = collect(range(1.0, 4.0; length = 60)))
y = repeat([1.0, 2.0, 3.0]; outer = 20) .+ 0.2 .* X.size

fit = LinearTrees.fit(LinearTreeRegressorFit, X, y; unseen = :right)
Xnew = DataFrame(colour = categorical(["purple"]), size = [2.5])
LinearTrees.predict(fit, Xnew)
```

This example needs CategoricalArrays.jl and DataFrames.jl.

## MLJ

LinearTrees provides four MLJ models:

- [`LinearTreeRegressor`](@ref)
- [`LinearTreeClassifier`](@ref)
- [`LinearBoostRegressor`](@ref)
- [`LinearBoostClassifier`](@ref)

Construct the model directly, then use the normal MLJ machine workflow.

```julia
using LinearTrees
using MLJBase

X = (x1 = collect(1.0:60.0), x2 = repeat([0, 1]; outer = 30))
y = 2 .* X.x1 .+ X.x2

model = LinearTreeRegressor(max_depth = 2)
mach = machine(model, X, y)
fit!(mach)
ŷ = MLJBase.predict(mach, X)
```

This example needs MLJBase.jl, which provides the machine interface.
The full MLJ.jl package re-exports this workflow and adds other learning tools.
Qualify prediction as `MLJBase.predict` here, or `MLJ.predict` when using MLJ.jl.
LinearTrees also exports `predict` for directly fitted trees and wrappers.

Classifier machines return MLJ `UnivariateFinite` distributions over the training levels.
MLJ reports feature importances for all four models.
Boosting models also report training loss history.
They expose `nrounds` as their iteration parameter.

## Persistence

Use JLD2 for Julia-native persistence of trees, fitted wrappers, and boosting models.
It preserves the complete Julia object graph.

```julia
using JLD2
using LinearTrees

JLD2.jldsave("model.jld2"; model = fit)
fit = JLD2.load("model.jld2", "model")
```

This example needs JLD2.jl.
JLD2 is optional and is not installed with LinearTrees.

Use [`to_dict`](@ref) and [`from_dict`](@ref) for one [`LinearTree`](@ref).
The returned dictionary is compatible with JSON encoders.

```@example interfaces
X = reshape(collect(1.0:60.0), :, 1)
y = 2 .* vec(X)
tree = fit_tree(X, y; max_depth = 1)
restored = from_dict(to_dict(tree))
LinearTrees.predict(restored, X) == LinearTrees.predict(tree, X)
```

The dictionary stores nodes, categorical masks, loss, score bounds, base value,
feature count, and truncation.
It supports `Float32` and `Float64` trees with built-in losses.
It does not serialize StatsAPI wrappers, MLJ machines, boosting models, or [`AdaptedLoss`](@ref).
Use a JSON package to encode the dictionary when text interchange is needed.
