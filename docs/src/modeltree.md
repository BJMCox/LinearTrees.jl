```@meta
CurrentModule = LinearTrees
```

# Greedy linear-leaf model trees

[`fit_model_tree`](@ref) fits an MSE tree whose terminal regions contain
multivariate ridge linear models. Unlike [`refit_leaves`](@ref), it chooses
numeric splits while fitting those linear models. It returns a [`RefitTree`](@ref),
so `predict`, `predict!`, `score`, and `coeftable` use the same ridge-leaf
prediction contract.

```julia
using LinearTrees

model = fit_model_tree(X, y;
    features = 1:size(X, 2),
    max_features = 8,
    lambda = 1.0,
    max_depth = 4,
)
predict(model, Xnew)
```

This first model-tree interface accepts numeric matrices and a scalar response
under `MSE`. `features` lists eligible split columns. Its first
`max_features` columns also form the common linear basis in every leaf; the
limit bounds the cost of each ridge solve. Pass the columns in the desired
basis order. The intercept is unpenalized. `lambda > 0` penalizes slopes after
node-local weighted-RMS standardization, as in `refit_leaves`.

At each node, the fitter scores a candidate split by fitting ridge models on
both sides and adding their **weighted penalized objectives**. `ExactSearch()`
evaluates every distinct-value threshold with at least `min_leaf` positive
weight on each side. It chooses the lowest combined objective, then accepts
that split only when an augmented-QR refit confirms improvement over the
parent greater than `split_penalty`. `max_depth` limits the number of splits
along a route. Equal objective scores settle by feature order and then the
lowest threshold.

This is a deterministic **greedy parametric linear-model partition**. It is
not a calibrated significance test and makes no unbiased-variable-selection
claim associated with GUIDE or model-based recursive partitioning.
`BinnedSearch(nbins=64)` first scores approximately equal-count bin boundaries.
Its default `refine=true` also scores all exact boundaries in the two bins
beside each feature's best coarse cut. Binning can change the chosen split
and has no quality-error bound.
`HybridSearch` and categorical predictors are not supported here.

`nthreads=Threads.nthreads()` limits parallel feature scans for binned searches
on large nodes with sufficiently sparse bin edges and refinement. Exact
searches, dense binned searches, and small nodes stay serial: many small linear
algebra calls can contend inside BLAS when scanned concurrently. Each worker
owns its scan buffers, and tied scores retain the supplied feature order.
Tree growth remains serial. Use `nthreads=1` when an outer loop already runs
independent fits in parallel. With `t` workers, moment-scan buffers grow from
`O(n + q²)` to `O(t(n + q²))`, in addition to the shared centered predictors.
QR fallback can also allocate `O(n q + q²)` temporary storage per active worker.

The fit uses positive-weight rows only. Zero-weight rows do not affect split
selection, ridge fitting, or stored feature bounds. Scaling all weights
changes the effective ridge and split penalties; their values should be
chosen for the weight scale in use. The same numeric `split_penalty` is charged
for each accepted split, in weighted squared-error units.

`score(model, Xnew; clip=false)` returns the leaf's full-response affine
score. By default, prediction clamps selected leaf features to their training
ranges and clips the final score to bounds derived from the positive-weight
targets. Pass `truncate=false` while fitting to disable both clamps.
`coeftable(model, x)` returns the local unclipped affine coefficients, with a
zero local slope for any clamped feature. A query matrix must have the same
number of columns as the training matrix.

For a node with `n` rows, `p` split features, and `q ≤ max_features` ridge
regressors, the implementation centers the regressors once, then keeps
weighted prefix and suffix matrix moments while scanning each sorted split
feature. It accumulates the two sides independently, so a small suffix weight
is not lost by subtracting a large prefix from the total. It scores children
with their centered covariance matrices and cross moments. The ridge penalty
on raw slopes is
`lambda * diag(centered_covariance) / child_weight`, matching each child's
weighted-RMS standardization. Each small system uses library Cholesky, with
the existing augmented-QR leaf fit as a fallback for ill-conditioned systems.
For nearly affine responses, the scan accumulates residuals from the parent
linear fit and includes its slopes in the ridge penalty correction. This
change of coefficient coordinates preserves the full-response objective
while avoiding repeated QR fits caused by subtracting large, nearly equal
response moments. Ordinary responses retain the original moment scores.
Nonfinite residuals restore the original coordinates for the whole node,
and inaccurate child scores still use augmented QR.
The selected split is always refit with augmented QR; its verified penalty is
computed as a norm of weighted slope deviations to avoid overflowing a large
predictor variance before multiplying by a small slope. Per node, exact-search
sorting costs `O(p n log n)`, moment updates cost `O(p n q²)`, and scoring up
to `p(n - 1)` thresholds costs `O(p n q³)` with `q` bounded by
`max_features`. Binning reduces the number of systems solved, although it
still scans the sorted rows for their moments. Without QR fallback, a scan
stores `O(n + q²)` auxiliary values, including one right-child objective per boundary.
Numerical fallbacks can still require repeated raw child fits and quadratic
work in `n`.
