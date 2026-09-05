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

`fit_tree` allocates one scratch set per worker, each of size `n` (four
vectors plus an `Int32` buffer), so memory is about `nthreads × 5 × 8n` bytes
plus the `n × p` `Int32` presort. For `Softmax(K)` two of the four vectors
hold `K-1` coordinates per row, so their share grows by that factor.

## Index

```@index
```
