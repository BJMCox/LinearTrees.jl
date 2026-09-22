# Changelog

## Unreleased — initial 0.0.1

- Add PILOT-style linear model trees and boosted ensembles for regression and classification.
- Add continuous multivariate trees with pair interactions, shared output geometry, and conditional Student-t predictions.
- Speed up continuous-tree fitting with sparse QR for larger sparse constraint systems and early rejection of refinements that add no degrees of freedom.
- Add exact, binned, and hybrid split searches with threaded fitting and prediction.
- Support numeric and categorical table data, observation weights, StatsAPI, and MLJ models.
- Add built-in and LossFunctions.jl-compatible losses, including multiclass softmax.
- Add feature importance, local coefficient tables, path-dependent TreeSHAP, and tree serialization.
- Add package tests, benchmarks, versioned Documenter documentation, coverage, and CI.

The initial interface is experimental. No release has been tagged.
