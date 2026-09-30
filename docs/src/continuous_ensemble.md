```@meta
CurrentModule = LinearTrees
```

# Fixed-geometry continuous ensembles

[`fit_continuous_ensemble`](@ref) combines a finite list of existing
[`ContinuousTree`](@ref) geometries. It recomputes every member's posterior on
the same predictors, responses, normalization, and hyperparameters. Each
member retains its coordinate-slice continuity and its selected main and pair
terms. The ensemble mean is also continuous because its weights do not depend
on the query row.

```julia
root = fit_continuous_tree(X, y; max_splits=0)
split = fit_continuous_tree(X, y; max_splits=1)
ensemble = fit_continuous_ensemble(X, y, [root, split];
    logprior=[log(1.0), log(1.0)])
mean_prediction = predict(ensemble, Xnew)
```

Supply each distinct terminal-box partition and pair-term basis once. A
duplicate raises an error even when the same boxes were reached through a
different split order, because repeating it would silently increase its prior
mass. Candidate normalization must agree with that computed from `X`. The
fitted ensemble copies its geometry, so later changes to a source tree do not
change it.
`logprior` gives relative log prior mass; `-Inf` excludes a member. At least
one member must have finite posterior log mass. The default gives equal prior
mass to all supplied members.

The model weights are proportional to `exp(logprior[k] + evidence[k])`,
normalized with a log-sum-exp shift. Every evidence uses the member's
conjugate linear-coefficient and inverse-gamma model. It conditions on the
provided tree and pair-term basis. The result is exact conjugate averaging in
the normalized model, treating normalization as fixed. By default, fitting
estimates response center and scale from fitting responses. Fixing
the candidate space alone therefore does not establish prior-predictive
calibration for that procedure. If candidates were selected or tuned using
those same responses, this is a restricted adaptive approximation: the weights do not
account for the selection process or for omitted geometries.

Use `response_normalization=(center=..., scale=...)` to specify one common
response transform for every refitted component. For example, identity
normalization for one output uses `(center=0.0, scale=1.0)`. Multiple outputs
require vectors with one center and positive scale per output. Values are
copied. The supplied geometries' response transforms do not affect refitting.

The prior uses these normalized response units. Choose the transform, geometry
space, and hyperparameters independently of the fitting responses for a fixed
conditional Bayesian model. A common response transform contributes the same
likelihood Jacobian to every geometry, so it cancels in model weights. The
stored comparison scores omit common constants and are not complete log
evidences in original response units.

[`predict`](@ref) returns the weighted conditional mean. [`predict!`](@ref)
writes that mean after reading all predictor rows, so its output may overlap
`X`. Both accept the `batch_size` keyword used by continuous trees.

[`predictive`](@ref) returns `location`, `variance`, `within_variance`,
`between_variance`, `weights`, and `components`. Each component has its own
Student-t `location`, `scale2`, and `dof`, in the same target rank as a single
continuous tree's `predictive` result. The mixture is generally not a
Student-t distribution. Its variance adds the weighted conditional variances
and the weighted squared differences between component means and the mixture
mean. If a member with positive weight has `dof <= 2`, its conditional and
mixture variance are infinite. These predictions condition on the supplied
model space and fitted normalization.

For a scalar response, a query row can be converted to a Distributions.jl
mixture for density or interval calculations. Distributions.jl is needed by
this example, not by LinearTrees.jl at runtime.

```julia
using Distributions

posterior = predictive(ensemble, Xnew)
row = 1
members = [component.location[row] +
    sqrt(component.scale2[row]) * TDist(component.dof)
    for component in posterior.components]
mixture = MixtureModel(members, Categorical(posterior.weights))
logdensity = logpdf(mixture, ynew[row])
interval = [quantile(mixture, p) for p in (0.05, 0.95)]
```

Held-out log density and interval coverage should be assessed separately from
training evidence, using the same candidate-generation policy intended for
future data. The current API does not generate or enumerate candidate trees.

Check latent and observation intervals separately. Observation noise can hide
mean misspecification: near-nominal observation coverage does not imply that
credible intervals cover the latent function. Check several nominal levels,
since one level can appear calibrated by coincidence. Aggregate coverage also
need not hold at each predictor value. The benchmark protocol in `bench/VALIDATION.md`
separates a fixed-transform prior-predictive check from the default procedure
that estimates response normalization.
