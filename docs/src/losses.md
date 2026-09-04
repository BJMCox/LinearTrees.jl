```@meta
CurrentModule = LinearTrees
```

# Losses

Every node fit is a weighted least squares on second-order statistics: one
Newton step of the loss, `z = -g/h` at weight `h`, minimising
`Σ h (z - f(x))²`. The fitter never sees the loss name, only `g`, `h`, the
link, and the score interval below. `μ = exp(f)` and `p = σ(f)` throughout,
with `f` the raw score.

| loss | `g` | `h` | link | score interval |
|------|-----|-----|------|----------------|
| [`MSE`](@ref) | `f - y` | `1` | identity | `[lo, hi]` |
| [`Huber`](@ref)`(δ)` | clipped residual | `1` or `0` outside `δ`, floored | identity | `[lo, hi]` |
| [`Quantile`](@ref)`(τ)` | `1(y<f) - τ`, `0` at `y=f` | IRLS, see below | identity | `[lo, hi]` |
| [`MAD`](@ref) | `sign(f - y)` | IRLS | identity | `[lo, hi]` |
| [`Logistic`](@ref) | `p - y` | `p(1-p)` | logit | `[-10, 10]` |
| [`Softmax`](@ref)`(K)` | `p_k - 1(y=k)`, `k < K` | `p_k(1-p_k)`, diagonal | reference-class softmax | `[-10, 10]` |
| [`Poisson`](@ref) | `μ - y` | `μ` | log | `[-S, S]` |
| [`NegBin`](@ref)`(θ)` | `θ(μ - y)/(μ + θ)` | `θμ(θ + y)/(μ + θ)²` | log | `[-S, S]` |
| [`Gamma`](@ref) | `1 - y/μ` | `y/μ` | log | `[-S, S]` |
| [`Tweedie`](@ref)`(ρ)` | `μ^(2-ρ) - y μ^(1-ρ)` | `(2-ρ)μ^(2-ρ) - (1-ρ) y μ^(1-ρ)` | log | `[-S, S]` |

`[lo, hi]` comes from [`scorebound`](@ref) applied to the training target;
`S = log(max(maximum(y), 1)) + 3` for the log-link losses. Every `h` is
floored to `1e-6` after weighting (see `HMIN` in the source) so a node's
summed Hessian stays strictly positive.

`Softmax(K)` uses the diagonal of the true reference-logit Hessian,
`diag(p) - p pᵀ` restricted to its diagonal: a per-class approximation, not
the exact joint Newton step. For `K = 2` this is `Logistic`'s own `p(1-p)`
exactly, since there is only one non-reference logit.

## IRLS for non-smooth losses

`Quantile` and `MAD` have zero Hessian almost everywhere ([`issmooth`](@ref)
is `false` for both), so they cannot use the Newton step directly:

1. **Split search** uses [`irls_weights!`](@ref): the parent residual
   `r = y - f`, `Quantile`'s one-sided weight or `MAD`'s flat weight
   (`l1weight`), divided by `max(|r|, ε)` with a strictly positive
   scale-aware floor `ε = max(1e-3 · median(|r|), sqrt(eps(T)) ·
   max(maximum(|y|), 1))`. This keeps the split search `O(np)`, exactly like
   every other loss.
2. Once a node's model kind and split are chosen, `refit_node` re-solves its
   coefficients on its own rows for 5 IRLS iterations by default, recomputing
   the pseudo-Hessian at the node's current fit each pass.
3. This is not the exact per-node minimiser for `Quantile` or `MAD`, only an
   IRLS approximation to it.

## The LossFunctions.jl adapter

[`Loss`](@ref)`(l::LossFunctions.SupervisedLoss, link; scale)` wraps a
`LossFunctions.jl` loss as a `Loss` for `fit_tree`, via [`AdaptedLoss`](@ref).
Three pairings are supported: a `DistanceLoss` with [`IdentityLink`](@ref)
(output on the response scale), a `MarginLoss` with [`LogitLink`](@ref) or
`IdentityLink` (output is the logit), and `PoissonLoss` with
[`LogLink`](@ref) (output is the log mean). [`canonical_scale`](@ref) picks a
default `scale` that lines the adapted loss up with the matching native
`Loss` — for example `L2DistLoss` uses `scale = 1/2` and matches `MSE`
exactly. `L1DistLoss` and `QuantileLoss` route through the same IRLS hooks as
`MAD` and `Quantile`.
