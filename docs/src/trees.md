```@meta
CurrentModule = LinearTrees
```

# Tree fitting

[`fit_tree`](@ref) grows one linear model tree. The loss determines the prediction
scale, the selection rule determines node choice, and split search determines
which thresholds are considered.

## How a tree predicts

A row follows one path through the tree. Each node adds a constant or a
linear function of one feature to the score. The loss link maps that score
to a response, such as a regression value or a probability.

| Kind | Contribution | Branches |
|:--|:--|:--|
| [`CON`](@ref) | A constant | None |
| [`LIN`](@ref) | One line in one feature | One continuing path |
| [`PCON`](@ref) | A constant on each side of a threshold | Two |
| [`BLIN`](@ref) | Two lines that meet at the threshold | Two |
| [`PLIN`](@ref) | Two independent lines | Two |

The default [`BIC`](@ref) rule selects among these forms using a penalized
weighted least-squares objective. For nonquadratic losses, that objective is
a local approximation. [Loss functions](losses.md) explains the fitting steps.

`LIN` nodes do not increase branching depth. Nodes along a path can combine
effects from several features, even in a shallow tree. `BLIN` enforces
continuity at its own split, but a whole tree need not be continuous.

## Control model size

Start with the defaults and tune against validation data:

| Keyword | Default | Effect |
|:--|:--|:--|
| `max_depth` | `12` | Maximum branching depth |
| `min_fit` | `10` | Minimum total row weight to consider another node model |
| `min_leaf` | `5` | Minimum total row weight on each side of a split |
| `min_sum_hessian` | `1.0` | Stop when total loss curvature is too small |
| `max_lin_chain` | `10` | Limit consecutive unsplit linear nodes |

Increasing `min_leaf` or reducing `max_depth` usually restricts complexity.
The fitted size also depends on the selection rule and the data. There is
no pruning step after growth.

```@example trees
using LinearTrees, Random

rng = Xoshiro(17)
X = rand(rng, 200, 3)
y = X[:, 1] .+ 2 .* (X[:, 2] .> 0.5) .+ 0.1 .* randn(rng, 200)
tree = fit_tree(X, y; max_depth = 4, min_leaf = 10)
round.(predict(tree, X[1:4, :]); digits = 3)
```

Use `features` to restrict fitting to selected columns. Prediction still takes
the original matrix layout:

```@example trees
restricted = fit_tree(X, y; features = [1, 2], max_depth = 4)
round.(predict(restricted, X[1:4, :]); digits = 3)
```

## Frequency weights

Pass `weights` to assign each row a frequency. Weight `2` represents two copies
of a row. Weights affect the fit, BIC's effective sample size, and the
`min_fit` and `min_leaf` gates.

```@example trees
w = ones(size(X, 1))
w[1:20] .= 2
weighted = fit_tree(X, y; weights = w, max_depth = 4)
round.(predict(weighted, X[1:4, :]); digits = 3)
```

Weights must be finite and nonnegative, with a positive total. Zero-weight rows
are excluded. Rescaling all weights can change model selection and stopping,
so weights are not normalized automatically.

## Categorical features

At the matrix interface, encode levels as positive integer codes and identify
their columns with `categorical`. Reuse the same encoding for prediction.

```@example trees
group = repeat([1.0, 2.0, 3.0, 4.0], 50)
Xcat = hcat(X[:, 1], group)
ycat = Xcat[:, 1] .+ 2 .* (group .== 3)
categorical_tree = fit_tree(Xcat, ycat; categorical = [2], max_depth = 3)
round.(predict(categorical_tree, [0.2 1.0; 0.2 3.0]); digits = 3)
```

Categorical nodes split sets of levels and fit constants on the two sides.
Within each node, levels are ordered by mean working response, then the
ordered partitions are searched. An unseen level routes right.

Use the [table interface](interfaces.md) for automatic categorical encoding.
Exact and local-bin search support categorical features. Hybrid search supports
numeric features only.

## Truncation

The default `truncate = true` applies two limits:

- Each node evaluates its linear term within the feature range seen in that node.
- The accumulated score stays within loss-specific bounds learned from the training target.

These limits restrain extrapolation. `truncate = false` disables both limits.
Linear terms can then extrapolate beyond the observed range.

[`score`](@ref) returns scores on the loss scale. `score(tree, X; clip = false)`
removes score clipping but retains feature truncation if enabled during fitting.
[SHAP values](interpretation.md) explain this unclipped score.

## Selection and search

`rule = BIC()` is the standalone default. [`GainRule`](@ref) provides ridge
penalties and a minimum split gain, and is the rule used by boosting.
[`MinDeviance`](@ref) is a low-level rule for comparisons and experiments.

The default [`ExactSearch`](@ref) checks every eligible numeric threshold in a
node. It is exact for that local search, not a global optimization over trees.
[Performance](performance.md) describes two approximate search methods.
