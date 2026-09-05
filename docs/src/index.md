```@meta
CurrentModule = LinearTrees
```

# LinearTrees

LinearTrees fits PILOT-style linear model trees: each node is either a
constant, a simple line, a broken line, or a pair of lines, chosen by a BIC
selection rule at every split. The fitter works for any twice-differentiable
loss (or an IRLS-approximated one), covers categorical features and sample
weights, and includes split-gain feature importance and path-dependent
TreeSHAP.

See the [Guide](@ref) for model kinds, selection, and truncation, and
[Losses](@ref) for the loss table.

## Quick start

```julia
using LinearTrees
using AbstractTrees     # for print_tree

X = rand(1000, 4)
y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ 0.05 .* randn(1000)

tree = fit_tree(X, y; max_depth = 4)
ŷ = predict(tree, X)

print_tree(TreeView(tree))

φ = shap(tree, X)          # φ.values[i, j] is feature j's SHAP value for row i
```

## Performance

`fit_tree` grows sibling subtrees concurrently, each on one scratch set: four
vectors plus an `Int32` buffer. It allocates two sets per worker, so a task
blocked on a sibling never denies a runnable one its buffers, and exactly one
set at `nthreads = 1`. A set's vectors start empty and grow only to the
largest node that set is used on, which is what keeps the second set per
worker affordable: on a 200_000-row, 20-column `MSE` fit at ten threads the
live scratch measures about 80 MiB, varying with the borrow pattern, against
137 MiB if all twenty sets were sized to `n` up front and 69 MiB for one
eagerly sized set per worker. The `n × p`
`Int32` presort sits next to it. For `Softmax(K)` two of the four vectors hold
`K-1` coordinates per row, so their share grows by that factor.

## Index

```@index
```
