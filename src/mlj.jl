# MLJModelInterface glue.

import MLJModelInterface as MMI

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

"Fields shared by both models, renamed to the `fit_tree` keyword contract."
corekw(m::Union{LinearTreeRegressor,LinearTreeClassifier}) = (rule = m.rule, max_depth = m.max_depth,
    min_fit = m.min_samples_split, min_leaf = m.min_samples_leaf, min_sum_hessian = m.min_sum_hessian,
    max_lin_chain = m.max_lin_chain, truncate = m.truncate, truncation_factor = m.truncation_factor)

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

MMI.predict(::LinearTreeRegressor, fr, Xnew) = StatsAPI.predict(fr, Xnew)

"""
`fr.classes` holds `CategoricalValue`s when the training target was a
`CategoricalArray`: each carries its own pool (so unobserved levels in that
pool still appear in the output), and `UnivariateFinite` picks it up with no
`pool` keyword. A plain-value `classes` (no training pool to speak of) uses
`pool = missing` instead. Checked by type name, not `isa`, so this file does
not need `CategoricalArrays` as a dependency.
"""
function MMI.predict(::LinearTreeClassifier, fr, Xnew)
    P = StatsAPI.predict(fr, Xnew)
    cls = fr.classes
    categorical_classes = !isempty(cls) && nameof(typeof(first(cls))) === :CategoricalValue
    return categorical_classes ? MMI.UnivariateFinite(cls, P) : MMI.UnivariateFinite(cls, P; pool = missing)
end

MMI.fitted_params(::Union{LinearTreeRegressor,LinearTreeClassifier}, fr) = (tree = fr.tree,)

function MMI.feature_importances(::Union{LinearTreeRegressor,LinearTreeClassifier}, fr, report)
    return [nm => v for (nm, v) in zip(fr.encoder.names, report.feature_importances)]
end

MMI.metadata_pkg.((LinearTreeRegressor, LinearTreeClassifier);
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
