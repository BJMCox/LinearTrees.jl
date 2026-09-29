```@meta
CurrentModule = LinearTrees
```

# Validation pruning with linear replacements

[`prune_refit`](@ref) is an optional post-fit step for a [`RefitTree`](@ref).
It visits routing nodes from the leaves upward. At each node, it fits one
affine ridge predictor to all positive-weight training rows in that region.
The candidate replaces the whole subtree only if it improves weighted
squared loss on separate positive-weight validation rows. Validation targets
never enter the ridge fit.

```julia
using LinearTrees

tree = fit_tree(Xtrain, ytrain; max_depth = 4)
refit = refit_leaves(tree, Xtrain, ytrain; lambda = 1.0)
pruned = prune_refit(refit, Xtrain, ytrain, Xval, yval;
    train_weights = nothing, val_weights = nothing,
    features = :path, max_features = 8, lambda = 1.0,
    tolerance = 0.0)
```

The `features`, `max_features`, and `lambda` keywords control the proposed
replacement leaves. They are specified again because a `RefitTree` does not
store its fitting options. `:path` takes the first distinct numeric routing
features from the root through the proposed replacement node, capped by
`max_features`. An explicit vector uses the same numeric regressors at every
candidate. A categorical column is recognized as such only if the routing
tree contains a categorical split on it; exclude any unused numeric category
codes yourself when specifying features.

The comparison uses `predict` semantics: leaf feature extrapolation and the
final score are clipped when the input refit model has truncation enabled.
`tolerance` is a nonnegative absolute margin in weighted squared-loss units:
a candidate is accepted only when its validation loss plus `tolerance` is
strictly below the current subtree's loss. A node with no positive-weight
validation rows is left alone. Zero-weight predictor rows do not enter
training or validation calculations, while all targets must remain finite.

The returned model owns a compact copy of its reachable routing nodes,
categorical masks, and leaves. The input model is unchanged. Pruning can
create discontinuities across region boundaries, including boundaries that
were continuous in the source tree. Evaluate the result on independent
held-out data when selection on the validation set matters.
