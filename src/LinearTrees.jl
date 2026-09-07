module LinearTrees

import StatsAPI
import StatsAPI: fit, predict, coeftable, deviance, nobs, dof, weights, residuals

include("loss.jl")
include("node.jl")
include("accumulate.jl")
include("select.jl")
include("scan.jl")
include("search.jl")
include("fit.jl")
include("hybrid.jl")
include("predict.jl")
include("importance.jl")
include("shap.jl")
include("boost.jl")
include("statsapi.jl")
include("mlj.jl")
include("show.jl")
include("serialize.jl")

export ModelKind, CON, LIN, PCON, BLIN, PLIN, Node, LinearTree
export BIC, MinDeviance, GainRule
export SplitSearch, ExactSearch, BinnedSearch, HybridSearch
export score, predict, predict!, fit_tree
export feature_importance, coeftable
export shap, shap!, ShapResult
export expected_score
export Loss, MSE, Huber, Quantile, MAD, Logistic, Poisson, NegBin, Gamma, Tweedie, Softmax, Frozen
export AdaptedLoss, IdentityLink, LogitLink, LogLink, canonical_scale
export gradhess!, linkinv, initscore, deviance, scorebound, validate_target, issmooth
export fit_boost, LinearBoost, nrounds
export LinearTreeRegressorFit, LinearTreeClassifierFit, LinearBoostRegressorFit, LinearBoostClassifierFit
export LinearTreeRegressor, LinearTreeClassifier, LinearBoostRegressor, LinearBoostClassifier
export fit, nobs, dof, weights, residuals
export TreeView, to_dict, from_dict

end
