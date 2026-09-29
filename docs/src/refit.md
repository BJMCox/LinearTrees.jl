```@meta
CurrentModule = LinearTrees
```

# Ridge leaf refits

[`refit_leaves`](@ref) keeps an MSE tree's routing regions and fits one
multivariate affine predictor in each terminal region. The new predictor fits
the full response. It replaces the path's accumulated node score rather than
adding to it. The source tree and its nodes remain unchanged; the returned
[`RefitTree`](@ref) holds its own routing snapshot.

```julia
using LinearTrees

tree = fit_tree(X, y; max_depth = 3)
refit = refit_leaves(tree, X, y; features = :path, max_features = 8, lambda = 1.0)
predict(refit, Xnew)
```

This first interface accepts numeric matrices and scalar `MSE` trees. Each
terminal region must receive positive training weight. `:path` selects the
first distinct numeric features encountered along that route, up to
`max_features`. Columns used by categorical split nodes determine routing but
do not enter the affine predictor. Pass a vector of feature indices to use
the same regressors in every region; its length must not exceed
`max_features`. The tree records categorical status only for columns it
actually split on. If an unused column contains numeric category codes,
exclude it yourself from an explicit feature vector.

For each region, the fit centers predictors and response at their weighted
means and divides each predictor by its weighted root mean square deviation.
The scale calculation avoids squaring raw deviations, including for very large
or very small finite predictor values.
It minimizes weighted squared error plus `lambda` times the sum of squared
standardized slopes. The intercept is unpenalized. `lambda` must be positive.
Zero-weight predictor rows do not affect the refit or feature bounds. Every
target must still be finite, including targets on zero-weight rows. Changing
the overall weight scale changes the effective penalty because weights are
not normalized.

`score(refit, Xnew; clip=false)` returns the new affine score before final
score clipping. `predict` applies the identity link and, by default, clips to
the source tree's final score bounds. With `truncate=true`, each used leaf
feature is also clamped to its positive-weight training range. This feature
clamp remains in effect when `clip=false`. Pass `truncate=false` to disable
both limits. If the source tree was fitted without truncation, its inherited
final score bounds are infinite.

[`coeftable`](@ref)`(refit, x)` returns the intercept and slopes of the local
unclipped score at one row. A clamped feature has zero local slope; its fixed
contribution is folded into the intercept. The refit does not add SHAP, MLJ,
table encoding, or dictionary serialization interfaces. Julia-native JLD2
storage works through ordinary object persistence.
