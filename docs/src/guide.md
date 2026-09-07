```@meta
CurrentModule = LinearTrees
```

# Guide

## Model kinds

Every node fits one of five [`ModelKind`](@ref)s on its own rows, using the
working response `z` and weight `h` from the current loss (see [Losses](@ref)):

- `CON`: a constant. Always available, and the only kind offered when a node
  is too small or too flat to split further.
- `LIN`: one line across the whole node, `a x + b`. Needs at least 5 unique
  values of the feature in the node. A `LIN` node has a single child and does
  not add tree depth.
- `PCON`: two constants, one per side of a threshold.
- `BLIN`: a broken line, continuous at the threshold (basis `[x, 1, max(x - t, 0)]`).
  Needs at least 5 unique feature values in the node.
- `PLIN`: two independent lines, one per side of a threshold. Needs at least 5
  unique feature values on each side.

At each node, `fit_tree` scans every feature and every legal split point,
evaluates every model kind the selection rule allows there, and keeps the
lowest-scoring candidate.

## Selection: BIC

The default rule is [`BIC`](@ref), the PILOT selection rule:

```
score = n * log(surrogate_deviance / n) + dof(kind) * ncoord * log(n)
```

`n = Σ w` over the node, `ncoord` is `K-1` for a `Softmax(K)` fit and 1
otherwise, and `dof` is `(1, 2, 5, 5, 7)` for `(con, lin, pcon, blin, plin)`
by default. Lower `score` wins. For `MSE` the surrogate deviance is the
residual sum of squares; for every other loss it is the second-order Taylor
expansion of the deviance around the node's current score, which keeps each
candidate `O(1)` to score. `dof` is a field of the `BIC` instance, so a
custom tuple can be passed: `BIC(dof = (1.0, 2.0, 5.0, 5.0, 7.0))`.

Within one kind the score is monotone non-decreasing in the surrogate
deviance, so the scan carries the lowest deviance each kind reaches and scores
the kind once per feature rather than once per split point. The winning
candidate is the one with the lowest score; ties resolve in this order:

- **within a kind**, the split point with the lowest surrogate deviance wins,
  and the earliest split point (lowest threshold) on an exact tie between
  deviances. Because `n log(dev / n)` is monotone but not injective in
  `Float64`, several split points of one kind can share one score; the lowest
  deviance among them is the one kept;
- **across kinds**, an exact score tie goes to the kind that comes first in
  the fixed order `(con, lin, pcon, blin, plin)`. That is a fixed order, not
  an ordering by parameter count: with a custom `dof` the winning kind may
  carry more parameters than the kind it beat;
- **across features**, a tie goes to the lowest feature index.

A cross-kind tie is degenerate for continuous data: it needs two kinds to
reach the same score to the last bit, which in practice means the same
deviance or two deviances that both fall under the score's `eps`-scaled log
floor.

A node stops splitting (becomes a leaf) when `CON` wins, when its total weight
falls below `min_fit`, when it reaches `max_depth`, when its summed Hessian
falls below `min_sum_hessian`, or when it has reached `max_lin_chain`
consecutive `LIN` fits along its root-to-node path. There is no pruning pass.
The `max_lin_chain` guard exists because a `LIN` node adds no depth, so
without it a chain of near-collinear features could recurse indefinitely.

## Truncation

Two independent clamps, both on by default (`truncate = true`):

1. **Score truncation.** The raw score is clamped to `(lo, hi) =
   scorebound(loss, y; truncation_factor)`, computed once from the training
   target. `predict` applies this clamp; `score(tree, X; clip = false)` and
   `shap` do not.
2. **Feature truncation.** At predict time, each node clamps the feature
   value it splits or fits on to `[xmin, xmax]`, the range observed in that
   node's training rows. This clamp is always applied when `truncate = true`,
   independent of the score clamp.

`truncate = false` at fit time disables both, for callers who want raw
extrapolation past the training range.

## Categorical features

Pass 1-based integer codes and list their column indices in `categorical`:

```julia
tree = fit_tree(X, y; categorical = [2])
```

A categorical column only ever offers `PCON`: the fitter orders its levels by
their mean working response in the node, then scans a `PCON` split over that
order, exactly as CART and PILOT do. The chosen split is stored as a bit mask
over the original level codes, not the temporary order, so prediction tests
set membership directly. A level absent from the mask (an unseen level at
predict time) routes right.

Fitting from a Tables.jl table or a `CategoricalArray` column infers
`categorical` and the level codes automatically; see
[`LinearTreeRegressorFit`](@ref) / [`LinearTreeClassifierFit`](@ref).

## Weights

`weights` are frequency weights: a row with weight 2 counts as if it appeared
twice. They multiply the loss gradient and Hessian, and the effective sample
size `n` used by `BIC` is `Σ w` over the node, not the row count. `min_fit`
and `min_leaf` also compare against `Σ w`. Weights must be finite and
non-negative with a positive total; rows with zero weight are dropped before
fitting.

## Softmax and the reference class

`Softmax(K)` fits a `K`-class tree by treating class `K` as the reference
class fixed at logit zero, and fitting `K - 1` logits for the other classes.
Every coefficient in the tree becomes an `SVector{K-1}`, one per non-reference
class, and one split is chosen per node by summing the surrogate deviance over
all `K - 1` fitted logits. This is one tree with vector-valued coefficients,
not `K - 1` separate trees.

`min_sum_hessian` is compared against the Hessian summed over all `K - 1`
fitted coordinates, so the default `1.0` is a weaker stopping gate for a
larger `K`.

`fit(LinearTreeClassifierFit, X, y)` picks the loss automatically: a
two-class target fits [`Logistic`](@ref) with `classes[1]` (the first sorted
class) as the positive outcome, and three or more classes fit `Softmax`.
`predict` always returns an `n × K` probability matrix in `classes` order,
with the reference class filled back in.

## MLJ and StatsAPI together

`LinearTrees` re-exports `StatsAPI.predict` for `LinearTree` and the
`*Fit` types. `MLJModelInterface` (loaded as `MMI` inside the package, but
`MLJ` itself exports its own `predict` for machines) defines a separate
`predict(model, fitresult, Xnew)`. Loading both `using MLJ, LinearTrees`
makes the bare name `predict` ambiguous. Qualify it:

- `LinearTrees.predict(tree, X)` or `LinearTrees.predict(fit, X)` for a tree
  or a `LinearTreeRegressorFit`/`LinearTreeClassifierFit` obtained directly
  from `fit_tree` or `StatsAPI.fit`.
- `predict(mach, Xnew)` (MLJ's own, unqualified) for an MLJ `machine`.

The same distinction applies to boosting. Call
`LinearTrees.fit(LinearBoostRegressorFit, X, y; kwargs...)` or
`LinearTrees.fit(LinearBoostClassifierFit, X, y; kwargs...)` for a StatsAPI
fit. Use `machine(LinearBoostRegressor(...), X, y)` or
`machine(LinearBoostClassifier(...), X, y)` for MLJ. The direct and StatsAPI
interfaces accept `Xval`, `yval`, and `wval` for early stopping; the MLJ
models expose `nrounds` as their iteration parameter instead. See
[Boosting](@ref) for the fitting contract.
