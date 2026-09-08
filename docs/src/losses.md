```@meta
CurrentModule = LinearTrees
```

# Choosing a loss

A loss defines the target domain, fitting scale, and transformation applied by [`predict`](@ref).
Pass the loss as the third argument to [`fit_tree`](@ref):

```@example losses
using LinearTrees

X = reshape(collect(0.0:5.0), :, 1)
y = [0, 0, 1, 2, 3, 5]
tree = fit_tree(X, y, Poisson(); max_depth = 1)
(score = score(tree, X), response = predict(tree, X))
```

[`score`](@ref) returns values on the loss scale, with score clipping enabled by default.
[`predict`](@ref) applies the inverse link.
They are equal for identity-link losses.
Logistic scores are log odds.
Log-link scores are log means.
Softmax scores are class logits.

Choose a loss from the target and the quantity you want to estimate.

| Target | Loss | Prediction |
|:--|:--|:--|
| Real-valued response | [`MSE`](@ref) | Conditional mean |
| Real-valued response with outliers | [`Huber`](@ref)`(δ)` | Robust location |
| Real-valued response | [`Quantile`](@ref)`(τ)` | Conditional quantile |
| Real-valued response | [`MAD`](@ref) | Conditional median |
| Binary value in `{0, 1}` | [`Logistic`](@ref) | Probability of `1` |
| Non-negative integer count | [`Poisson`](@ref) | Conditional mean |
| Overdispersed non-negative integer count | [`NegBin`](@ref)`(θ)` | Conditional mean |
| Positive continuous response | [`Gamma`](@ref) | Conditional mean |
| Non-negative semicontinuous response | [`Tweedie`](@ref)`(ρ)` | Conditional mean |
| Integer class code in `1:K` | [`Softmax`](@ref)`(K)` | Class probabilities |

`Huber(δ)` changes from squared to linear loss at residual magnitude `δ`.
`Quantile(τ)` requires `0 < τ < 1`.
`NegBin(θ)` requires `θ > 0`.
`Tweedie(ρ)` supports powers `1 < ρ < 2`.

[`validate_target`](@ref) checks finite values and each loss domain.
Every class from `1` through `K` must occur when fitting `Softmax(K)`.

## Classification conventions

Use numeric targets in `{0, 1}` with [`Logistic`](@ref).
A prediction is the probability that the target equals `1`.

The Tables and StatsAPI classifier accepts labels.
It sorts observed labels and stores that order in `fit.classes`.
For two classes, `fit.classes[1]` is the positive class.
Probability column `k` always represents `fit.classes[k]`.

[`Softmax`](@ref)`(K)` uses class `K` as the reference class.
[`score`](@ref) returns `K - 1` logits against that reference.
[`predict`](@ref) returns all `K` probabilities in class-code order.

```@example losses
X = reshape([-2.0, -1.0, 1.0, 2.0, 0.0, 0.5], :, 1)
y = [1, 1, 2, 2, 3, 3]
tree = fit_tree(X, y, Softmax(3); max_depth = 1)
(size(score(tree, X)), size(predict(tree, X)))
```

Softmax fitting uses the diagonal of the reference-logit Hessian.
It omits cross-class Hessian terms.
For `Softmax(2)`, this diagonal is the exact logistic Hessian.

## How losses are fitted

Smooth losses fit each candidate model from gradients and Hessians on the raw score scale.
The working response is `-g/h`, and the working weight is `h`.
The method takes one local Newton step.
It does not solve every node's full nonlinear optimization problem.

The main derivatives are:

| Loss | Gradient `g` | Hessian `h` |
|:--|:--|:--|
| `MSE` | `f - y` | `1` |
| `Logistic` | `p - y` | `p(1-p)` |
| `Poisson` | `μ - y` | `μ` |

Here `f` is the score, `p` is the logistic inverse link, and `μ = exp(f)`.
[`gradhess!`](@ref) provides the complete loss-specific definitions.
Ordinary row Hessians have a floor of `1e-6` before sample weights apply.

[`Quantile`](@ref) and [`MAD`](@ref) are non-smooth.
They use iteratively reweighted least squares instead of a true Newton step.
Split search computes one set of residual weights.
After selection, the chosen node refits for `niter` iterations, which defaults to five.
The result approximates the node's L1 objective.

Boosting freezes the non-smooth working weights once per round.
Its base tree does not run the node refit.
See [Boosting](boosting.md) for the ensemble algorithm.

## Logistic step safeguard

Small logistic Hessians can produce an excessive Newton step.
Logistic trees and binary softmax trees compare each node update with the current
weighted training loss.
They keep a non-increasing full step.
Otherwise, they halve all node coefficients together until the loss does not increase.
They use a zero update if no tested scale succeeds.

The check includes score and feature truncation.
It ensures descent for that node update within floating-point accuracy.
It does not find the exact maximum-likelihood node fit or calibrate held-out probabilities.

## Deviance and truncation

[`deviance`](@ref) reports

```math
2 \sum_i w_i\,\ell(y_i, f_i),
```

where `f` is the raw score.
This convention includes constants present in the implemented point loss.
It omits distribution constants absent from that loss.
Compare deviances fitted with the same loss and data.
Lower values indicate a better fit under that loss.

With `truncate=true`, [`scorebound`](@ref) limits fitted scores.
Identity-link losses derive bounds from the target range.
Logistic and softmax use `[-10, 10]`.
Log-link losses use symmetric bounds based on the largest target.
Set `truncate=false` when extrapolation beyond the training range is required.

## LossFunctions.jl losses

Wrap a supported LossFunctions.jl loss with [`Loss`](@ref).
Use [`IdentityLink`](@ref) for distance losses.
Use [`LogitLink`](@ref) for margin losses.
Use [`LogLink`](@ref) for `PoissonLoss`.
[`canonical_scale`](@ref) matches an equivalent built-in loss where one exists.

```julia
using LinearTrees
using LossFunctions

loss = Loss(LossFunctions.L2DistLoss(), IdentityLink())
tree = fit_tree(X, y, loss)
```

Add LossFunctions.jl to your project to use its loss types directly.
Adapted L1 and quantile losses use the same IRLS path as [`MAD`](@ref) and [`Quantile`](@ref).
Adapted `LogitMarginLoss` values use the logistic step safeguard.
