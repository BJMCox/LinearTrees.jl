module LinearTrees

include("loss.jl")
include("node.jl")
include("accumulate.jl")
include("select.jl")
include("scan.jl")
include("fit.jl")
include("predict.jl")
include("importance.jl")
include("shap.jl")
include("statsapi.jl")
include("mlj.jl")
include("show.jl")
include("serialize.jl")

export ModelKind, CON, LIN, PCON, BLIN, PLIN, Node, LinearTree, MomentSums
export BIC, MinDeviance, GainRule
export score, predict, predict!, fit_tree
export Loss, MSE, Huber, Quantile, MAD, Logistic, Poisson, NegBin, Gamma, Tweedie
export gradhess!, linkinv, initscore, deviance, scorebound, validate_target, issmooth, irls_weights!

end
