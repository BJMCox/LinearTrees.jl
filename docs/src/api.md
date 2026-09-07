```@meta
CurrentModule = LinearTrees
```

# API reference

## Fitting and prediction

```@docs
fit_tree
predict
predict!
score
LinearTree
Node
ModelKind
CON
LIN
PCON
BLIN
PLIN
```

## Selection rules

```@docs
BIC
MinDeviance
GainRule
```

## Boosting API

```@docs
fit_boost
LinearBoost
nrounds
Frozen
```

## Loss types and functions

```@docs
Loss
MSE
Huber
Quantile
MAD
Logistic
Poisson
NegBin
Gamma
Tweedie
Softmax
gradhess!
linkinv
initscore
deviance
scorebound
validate_target
issmooth
```

## The LossFunctions.jl adapter

```@docs
AdaptedLoss
IdentityLink
LogitLink
LogLink
canonical_scale
```

## Interpretation

```@docs
feature_importance
coeftable
shap
shap!
ShapResult
expected_score
```

## StatsAPI and Tables.jl

```@docs
LinearTreeRegressorFit
LinearTreeClassifierFit
LinearBoostRegressorFit
LinearBoostClassifierFit
fit
nobs
dof
weights
residuals
```

## MLJ

```@docs
LinearTreeRegressor
LinearTreeClassifier
LinearBoostRegressor
LinearBoostClassifier
```

## Printing and serialisation

```@docs
TreeView
to_dict
from_dict
```
