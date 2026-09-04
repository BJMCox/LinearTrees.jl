"""
A twice-differentiable or IRLS-approximated loss. Implement `gradhess!`,
`linkinv`, `initscore`, `deviance`, `scorebound`, and `validate_target`.
"""
abstract type Loss end

struct MSE <: Loss end
