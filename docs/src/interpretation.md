```@meta
CurrentModule = LinearTrees
```

# Interpreting models

LinearTrees provides three complementary views of a fitted tree or ensemble.
[`feature_importance`](@ref) summarizes split gains across the model.
[`coeftable`](@ref) describes the linear score near one observation.
[`shap`](@ref) attributes one prediction across features.

All three methods describe the fitted model.
They do not estimate causal effects.

## Rank features by split gain

[`feature_importance`](@ref) sums each feature's positive split gains.
It then normalizes the values to sum to one.
It returns all zeros when the model has no positive split gain.

```@example interpretation
using LinearTrees

x1 = collect(range(-1, 1; length = 80))
x2 = reverse(x1)
X = [x1 x2]
y = 3 .* x1 .+ 0.2 .* x2 .^ 2
tree = fit_tree(X, y; max_depth = 3)

feature_importance(tree)
```

For an ensemble, gains are summed across all retained trees before normalization.
Gain importance can favor features offering many useful split points.
Correlated features can divide or exchange importance.
The result gives no direction for a feature's effect.

## Inspect the local linear score

[`coeftable`](@ref) returns `(intercept, slopes)` at one feature vector `x`.
The selected path determines the returned linear piece.
For scalar scores, `intercept + sum(slopes .* x)` equals the unclipped score.

```@example interpretation
x = X[20, :]
intercept, slopes = coeftable(tree, x)
local_score = intercept + sum(slopes .* x)
raw_score = only(score(tree, reshape(x, 1, :); clip = false))
@assert isapprox(local_score, raw_score) # hide

(intercept, slopes, local_score, raw_score)
```

Categorical features contribute constants and therefore have zero slopes.
With truncation enabled, an out-of-range numeric feature can also have zero slope.
Its clamped contribution moves into the intercept.

The coefficients describe only the path selected at `x`.
Crossing a split can change the intercept, slope, or both.
The coefficients remain on the score scale.
Apply the loss link before comparing them with response-scale predictions.

For [`Softmax`](@ref), the intercept and each slope are vectors with `K - 1` coordinates.
Their sum reconstructs the corresponding unclipped score vector.

## Explain predictions with SHAP values

[`shap`](@ref) returns a [`ShapResult`](@ref) with `values`, `base`, and `clipped`.
For scalar-score models, `values` has shape `n × p`.
Each row reconstructs its raw score by adding `base` and its feature values.

```@example interpretation
result = shap(tree, X[1:4, :])
reconstructed = result.base .+ vec(sum(result.values; dims = 2))
raw = score(tree, X[1:4, :]; clip = false)
@assert reconstructed ≈ raw # hide

(size(result.values), result.base, reconstructed, raw, result.clipped)
```

For `Softmax(K)`, `values` has shape `n × p × (K - 1)`.
`base` has `K - 1` coordinates.
Each final-axis slice explains one score coordinate.
[`predict`](@ref) converts those scores into `K` response probabilities.

```@example interpretation-softmax
using LinearTrees

x1 = repeat([-1.0, 0.0, 1.0], 30)
x2 = repeat(collect(range(-1, 1; length = 30)), inner = 3)
X = [x1 x2]
y = [a > 0 ? 1 : b > 0 ? 2 : 3 for (a, b) in zip(x1, x2)]
tree = fit_tree(X, y, Softmax(3); max_depth = 2)
result = shap(tree, X[1:5, :])

(size(result.values), length(result.base), size(score(tree, X[1:5, :])), size(predict(tree, X[1:5, :])))
```

These are path-dependent TreeSHAP values.
Node cover fractions weight branches when a feature is absent.
They do not integrate over an independent background dataset.
Correlated features can therefore receive attribution according to the learned paths.

## Account for clipping

SHAP and local coefficients reconstruct the unclipped score.
They do not reconstruct a clipped score or linked response directly.
`result.clipped[i]` is true when score truncation changed row `i`.

Compare these quantities when exact reconstruction matters:

```julia
raw = score(model, X; clip = false)
bounded = score(model, X)
response = predict(model, X)
explanation = shap(model, X)
```

When `explanation.clipped[i]` is true, the SHAP sum still equals `raw[i]`.
The bounded score and response no longer equal that additive decomposition.
Use `truncate = false` during fitting only when unbounded behavior suits the application.

Use [`shap!`](@ref) when reusing preallocated output arrays matters.
See [Boosted trees](boosting.md) for ensemble fitting and retained rounds.
