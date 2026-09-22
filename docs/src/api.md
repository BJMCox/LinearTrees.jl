```@meta
CurrentModule = LinearTrees
```

# API reference

The matrix API returns a tree or ensemble. StatsAPI wrappers add table encoding
and stored training data. MLJ models integrate with machines and resampling.
Start with [Getting started](guide.md) for a complete fitting workflow.

## Fit and predict

Use `fit_tree` for one PILOT tree, `fit_boost` for an ensemble, and
`fit_continuous_tree` for continuous conjugate regression. `predict` returns
responses, while `score` returns values on the loss scale for PILOT models.

```@docs
fit_tree
fit_boost
fit_continuous_tree
predict
predict!
predictive
score
```

## Fitted models and node kinds

These types describe the stored model. Fit models through the functions above.

```@docs
LinearTree
LinearBoost
ContinuousTree
nrounds
Node
ModelKind
CON
LIN
PCON
BLIN
PLIN
```

## Model selection and split search

Selection rules compare node models. Search policies choose numeric thresholds.
See [Tree fitting](trees.md) and [Performance](performance.md) for their distinct roles.

```@docs
BIC
MinDeviance
GainRule
SplitSearch
ExactSearch
BinnedSearch
HybridSearch
```

## Loss types and functions

See [Loss functions](losses.md) for target domains and prediction scales.

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
Frozen
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

These functions describe fitted scores. See [Interpretation](interpretation.md)
for reconstruction identities and the effect of clipping.

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

## Index

```@index
```
