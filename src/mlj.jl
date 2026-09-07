# MLJModelInterface glue.

import MLJModelInterface as MMI
using Random: AbstractRNG, Xoshiro, default_rng

MMI.@mlj_model mutable struct LinearTreeRegressor <: MMI.Deterministic
    loss::Loss = MSE()
    rule::SelectionRule = BIC()
    max_depth::Int = 12::(_ >= 0)
    min_samples_split::Int = 10::(_ >= 1)
    min_samples_leaf::Int = 5::(_ >= 1)
    min_sum_hessian::Float64 = 1.0::(_ >= 0)
    max_lin_chain::Int = 10::(_ >= 1)
    truncate::Bool = true
    truncation_factor::Float64 = 3.0::(_ >= 1)
end

MMI.@mlj_model mutable struct LinearTreeClassifier <: MMI.Probabilistic
    rule::SelectionRule = BIC()
    max_depth::Int = 12::(_ >= 0)
    min_samples_split::Int = 10::(_ >= 1)
    min_samples_leaf::Int = 5::(_ >= 1)
    min_sum_hessian::Float64 = 1.0::(_ >= 0)
    max_lin_chain::Int = 10::(_ >= 1)
    truncate::Bool = true
    truncation_factor::Float64 = 3.0::(_ >= 1)
end

MMI.@mlj_model mutable struct LinearBoostRegressor <: MMI.Deterministic
    loss::Loss = MSE()::(!(_ isa Frozen) && coeftype(_, Float64) <: Number)
    nrounds::Int = 100::(_ >= 1)
    eta::Float64 = 0.1::(isfinite(_) && _ > 0)
    max_depth::Int = 5::(_ >= 0)
    min_samples_split::Int = 10::(_ >= 1)
    min_samples_leaf::Int = 5::(_ >= 1)
    min_sum_hessian::Float64 = 1.0::(isfinite(_) && _ >= 0)
    lambda_slope::Float64 = 1.0::(isfinite(_) && _ >= 0)
    lambda_intercept::Float64 = 1.0::(isfinite(_) && _ >= 0)
    gamma::Float64 = 0.0::(isfinite(_) && _ >= 0)
    subsample::Float64 = 1.0::(0 < _ <= 1)
    colsample::Float64 = 1.0::(0 < _ <= 1)
    truncate::Bool = true
    rng::Union{AbstractRNG,Integer} = default_rng()
end

MMI.@mlj_model mutable struct LinearBoostClassifier <: MMI.Probabilistic
    nrounds::Int = 100::(_ >= 1)
    eta::Float64 = 0.1::(isfinite(_) && _ > 0)
    max_depth::Int = 5::(_ >= 0)
    min_samples_split::Int = 10::(_ >= 1)
    min_samples_leaf::Int = 5::(_ >= 1)
    min_sum_hessian::Float64 = 1.0::(isfinite(_) && _ >= 0)
    lambda_slope::Float64 = 1.0::(isfinite(_) && _ >= 0)
    lambda_intercept::Float64 = 1.0::(isfinite(_) && _ >= 0)
    gamma::Float64 = 0.0::(isfinite(_) && _ >= 0)
    subsample::Float64 = 1.0::(0 < _ <= 1)
    colsample::Float64 = 1.0::(0 < _ <= 1)
    truncate::Bool = true
    rng::Union{AbstractRNG,Integer} = default_rng()
end

"Fields shared by both models, renamed to the `fit_tree` keyword contract."
corekw(m::Union{LinearTreeRegressor,LinearTreeClassifier}) = (rule = m.rule, max_depth = m.max_depth,
    min_fit = m.min_samples_split, min_leaf = m.min_samples_leaf, min_sum_hessian = m.min_sum_hessian,
    max_lin_chain = m.max_lin_chain, truncate = m.truncate, truncation_factor = m.truncation_factor)

"An integer `rng` seeds a fresh `Xoshiro`; an `AbstractRNG` advances as supplied."
mljrng(rng::AbstractRNG) = rng
mljrng(seed::Integer) = Xoshiro(seed)

boostkw(m::Union{LinearBoostRegressor,LinearBoostClassifier}) = (nrounds = m.nrounds, eta = m.eta,
    max_depth = m.max_depth, min_fit = m.min_samples_split, min_leaf = m.min_samples_leaf,
    min_sum_hessian = m.min_sum_hessian, lambda_slope = m.lambda_slope,
    lambda_intercept = m.lambda_intercept, gamma = m.gamma, subsample = m.subsample,
    colsample = m.colsample, truncate = m.truncate, rng = mljrng(m.rng))

kind_counts(tree::LinearTree) = Dict(k => count(n -> n.model == k, tree.nodes) for k in instances(ModelKind))

function MMI.fit(m::LinearTreeRegressor, verbosity, X, y, w = nothing)
    fr = StatsAPI.fit(LinearTreeRegressorFit, X, collect(y); loss = m.loss, weights = w, corekw(m)...)
    report = (feature_importances = feature_importance(fr), kind_counts = kind_counts(fr.tree))
    return fr, nothing, report
end

function MMI.fit(m::LinearTreeClassifier, verbosity, X, y, w = nothing)
    fr = StatsAPI.fit(LinearTreeClassifierFit, X, collect(y); weights = w, corekw(m)...)
    report = (feature_importances = feature_importance(fr), kind_counts = kind_counts(fr.tree))
    return fr, nothing, report
end

function MMI.fit(m::LinearBoostRegressor, verbosity, X, y, w = nothing)
    fr = StatsAPI.fit(LinearBoostRegressorFit, X, collect(y); loss = m.loss, weights = w, boostkw(m)...)
    report = (feature_importances = feature_importance(fr), nrounds = nrounds(fr.boost), history = copy(fr.boost.history))
    return fr, nothing, report
end

function MMI.fit(m::LinearBoostClassifier, verbosity, X, y, w = nothing)
    fr = StatsAPI.fit(LinearBoostClassifierFit, X, collect(y); weights = w, boostkw(m)...)
    report = (feature_importances = feature_importance(fr), nrounds = nrounds(fr.boost), history = copy(fr.boost.history))
    return fr, nothing, report
end

MMI.predict(::LinearTreeRegressor, fr, Xnew) = StatsAPI.predict(fr, Xnew)
MMI.predict(::LinearBoostRegressor, fr, Xnew) = StatsAPI.predict(fr, Xnew)

"""
`fr.classes` holds `CategoricalValue`s when the training target was a
`CategoricalArray`: each carries its own pool (so unobserved levels in that
pool still appear in the output), and `UnivariateFinite` picks it up with no
`pool` keyword. A plain-value `classes` (no training pool to speak of) uses
`pool = missing` instead. Checked by type name, not `isa`, so this file does
not need `CategoricalArrays` as a dependency.
"""
function MMI.predict(::Union{LinearTreeClassifier,LinearBoostClassifier}, fr, Xnew)
    P = StatsAPI.predict(fr, Xnew)
    cls = fr.classes
    categorical_classes = !isempty(cls) && nameof(typeof(first(cls))) === :CategoricalValue
    return categorical_classes ? MMI.UnivariateFinite(cls, P) : MMI.UnivariateFinite(cls, P; pool = missing)
end

MMI.fitted_params(::Union{LinearTreeRegressor,LinearTreeClassifier}, fr) = (tree = fr.tree,)
MMI.fitted_params(::Union{LinearBoostRegressor,LinearBoostClassifier}, fr) = (boost = fr.boost,)
MMI.iteration_parameter(::Type{<:LinearBoostRegressor}) = :nrounds
MMI.iteration_parameter(::Type{<:LinearBoostClassifier}) = :nrounds
MMI.training_losses(::Union{LinearBoostRegressor,LinearBoostClassifier}, report) = report.history

function MMI.feature_importances(::Union{LinearTreeRegressor,LinearTreeClassifier,LinearBoostRegressor,LinearBoostClassifier}, fr, report)
    return [nm => v for (nm, v) in zip(fr.encoder.names, report.feature_importances)]
end

MMI.metadata_pkg.((LinearTreeRegressor, LinearTreeClassifier, LinearBoostRegressor, LinearBoostClassifier);
    name = "LinearTrees", uuid = "8dcaa25b-be43-4b0a-960a-aaf476051ed8",
    url = "https://github.com/BJMCox/LinearTrees.jl", julia = true, license = "MIT", is_wrapper = false)

MMI.metadata_model(LinearTreeRegressor;
    input_scitype = MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    target_scitype = AbstractVector{<:Union{MMI.Continuous,MMI.Count}},
    supports_weights = true, reports_feature_importances = true,
    load_path = "LinearTrees.LinearTreeRegressor")

MMI.metadata_model(LinearTreeClassifier;
    input_scitype = MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    target_scitype = AbstractVector{<:MMI.Finite},
    supports_weights = true, reports_feature_importances = true,
    load_path = "LinearTrees.LinearTreeClassifier")

MMI.metadata_model(LinearBoostRegressor;
    input_scitype = MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    target_scitype = AbstractVector{<:Union{MMI.Continuous,MMI.Count}},
    supports_weights = true, supports_training_losses = true, reports_feature_importances = true,
    load_path = "LinearTrees.LinearBoostRegressor")

MMI.metadata_model(LinearBoostClassifier;
    input_scitype = MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    target_scitype = AbstractVector{<:MMI.Finite},
    supports_weights = true, supports_training_losses = true, reports_feature_importances = true,
    load_path = "LinearTrees.LinearBoostClassifier")

"""
$(MMI.doc_header(LinearTreeRegressor))

`LinearTreeRegressor` fits a PILOT-style linear model tree for a continuous
or count target, generalised to any twice-differentiable `loss`.

# Hyperparameters

- `loss::Loss = MSE()`: the loss to fit; see `MSE`, `Huber`, `Quantile`, `MAD`,
  `Poisson`, `NegBin`, `Gamma`, `Tweedie`.
- `rule::SelectionRule = BIC()`: model-kind selection rule at each node.
- `max_depth::Int = 12`: maximum tree depth.
- `min_samples_split::Int = 10`: minimum total weight in a node to attempt a split.
- `min_samples_leaf::Int = 5`: minimum total weight required in each child.
- `min_sum_hessian::Float64 = 1.0`: minimum summed hessian in a node to attempt a split.
- `max_lin_chain::Int = 10`: maximum consecutive `lin` fits along one root-to-node path.
- `truncate::Bool = true`: clamp predictions to the training score range.
- `truncation_factor::Float64 = 3.0`: half-range multiplier for the truncation bounds.

# Operations

- `predict(mach, Xnew)`: return the predicted response for `Xnew`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `tree`: the fitted `LinearTree`.

# Report

The fields of `report(mach)` are:

- `feature_importances`: split-gain feature importances, as in `feature_importance`.
- `kind_counts`: a count of fitted nodes by `ModelKind`.

See also [`LinearTreeClassifier`](@ref).
"""
LinearTreeRegressor

"""
$(MMI.doc_header(LinearTreeClassifier))

`LinearTreeClassifier` fits a PILOT-style linear model tree for a finite
(two- or many-class) target: a `Logistic` tree for two classes, a `Softmax`
tree for three or more.

# Hyperparameters

- `rule::SelectionRule = BIC()`: model-kind selection rule at each node.
- `max_depth::Int = 12`: maximum tree depth.
- `min_samples_split::Int = 10`: minimum total weight in a node to attempt a split.
- `min_samples_leaf::Int = 5`: minimum total weight required in each child.
- `min_sum_hessian::Float64 = 1.0`: minimum summed hessian in a node to attempt a split.
- `max_lin_chain::Int = 10`: maximum consecutive `lin` fits along one root-to-node path.
- `truncate::Bool = true`: clamp predictions to the training score range.
- `truncation_factor::Float64 = 3.0`: half-range multiplier for the truncation bounds.

# Operations

- `predict(mach, Xnew)`: return the predicted class distribution for `Xnew`,
  as a `UnivariateFinite`. Use `predict_mode(mach, Xnew)` for point predictions.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `tree`: the fitted `LinearTree`.

# Report

The fields of `report(mach)` are:

- `feature_importances`: split-gain feature importances, as in `feature_importance`.
- `kind_counts`: a count of fitted nodes by `ModelKind`.

See also [`LinearTreeRegressor`](@ref).
"""
LinearTreeClassifier

"""
$(MMI.doc_header(LinearBoostRegressor))

`LinearBoostRegressor` fits a gradient-boosted ensemble of linear model trees
for a continuous or count target. MLJ iteration controls `nrounds`; this model
does not expose validation early stopping, for which use `IteratedModel`.

# Hyperparameters

- `loss::Loss = MSE()`: scalar-response loss, such as `MSE`, `Huber`, `Quantile`, `MAD`, `Poisson`, `NegBin`, `Gamma`, or `Tweedie`.
- `nrounds::Int = 100`: number of boosting rounds.
- `eta::Float64 = 0.1`: shrinkage per round.
- `max_depth::Int = 5`: maximum base-tree depth.
- `min_samples_split::Int = 10`: minimum node weight for a split.
- `min_samples_leaf::Int = 5`: minimum child weight.
- `min_sum_hessian::Float64 = 1.0`: minimum node Hessian sum.
- `lambda_slope::Float64 = 1.0`: slope penalty.
- `lambda_intercept::Float64 = 1.0`: intercept penalty.
- `gamma::Float64 = 0.0`: split-gain penalty.
- `subsample::Float64 = 1.0`: row-sampling fraction.
- `colsample::Float64 = 1.0`: feature-sampling fraction.
- `truncate::Bool = true`: clamp predictions to the training score range.
- `rng::Union{AbstractRNG,Integer} = default_rng()`: random source; an integer starts a fresh `Xoshiro` on each fit.

# Fitted parameters and report

`fitted_params(mach).boost` is the fitted `LinearBoost`. `report(mach)` holds
`feature_importances`, `nrounds`, and per-round `history`; the history is also
available through `training_losses(mach)`.

See also [`LinearBoostClassifier`](@ref).
"""
LinearBoostRegressor

"""
$(MMI.doc_header(LinearBoostClassifier))

`LinearBoostClassifier` fits a gradient-boosted ensemble of linear model trees
for a finite target. It uses `Logistic` loss for two classes and `Softmax` for
three or more. MLJ iteration controls `nrounds`; this model does not expose
validation early stopping, for which use `IteratedModel`.

# Hyperparameters

- `nrounds::Int = 100`: number of boosting rounds.
- `eta::Float64 = 0.1`: shrinkage per round.
- `max_depth::Int = 5`: maximum base-tree depth.
- `min_samples_split::Int = 10`: minimum node weight for a split.
- `min_samples_leaf::Int = 5`: minimum child weight.
- `min_sum_hessian::Float64 = 1.0`: minimum node Hessian sum.
- `lambda_slope::Float64 = 1.0`: slope penalty.
- `lambda_intercept::Float64 = 1.0`: intercept penalty.
- `gamma::Float64 = 0.0`: split-gain penalty.
- `subsample::Float64 = 1.0`: row-sampling fraction.
- `colsample::Float64 = 1.0`: feature-sampling fraction.
- `truncate::Bool = true`: clamp predictions to the training score range.
- `rng::Union{AbstractRNG,Integer} = default_rng()`: random source; an integer starts a fresh `Xoshiro` on each fit.

# Operations

`predict(mach, Xnew)` returns a `UnivariateFinite`; use `predict_mode` for
point predictions.

# Fitted parameters and report

`fitted_params(mach).boost` is the fitted `LinearBoost`. `report(mach)` holds
`feature_importances`, `nrounds`, and per-round `history`; the history is also
available through `training_losses(mach)`.

See also [`LinearBoostRegressor`](@ref).
"""
LinearBoostClassifier
