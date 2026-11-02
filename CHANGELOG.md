# Changelog

## 0.0.2 — Unreleased

- Add continuous multivariate trees with pair interactions, shared output geometry, and conditional Student-t predictions.
- Add finite Bayesian averaging over continuous-tree geometries.
- Add ridge leaf refits, linear-child split selection, and validation pruning.
- Support fixed response normalization for continuous trees.
- Fix numerical edge cases in tree fitting, robust losses, and conditional Bayesian predictions.
- Speed up continuous-tree fitting with sparse QR for larger sparse constraint systems and reuse of unchanged constraint bases.
- Reduce temporary allocations in model-tree ridge solves and continuous posteriors.
- Avoid repeated QR fits for near-affine model-tree responses by scoring corrected parent residuals.
- Parallelize large binned model-tree feature scans while retaining final QR verification.
- Skip zero-game SHAP path work without changing the path-dependent coalition game.

## 0.0.1 — 2026-09-11

- Add PILOT-style linear model trees and boosted ensembles for regression and classification.
- Add exact, binned, and hybrid split searches with threaded fitting and prediction.
- Support numeric and categorical table data, observation weights, StatsAPI, and MLJ models.
- Add built-in and LossFunctions.jl-compatible losses, including multiclass softmax.
- Add feature importance, local coefficient tables, path-dependent TreeSHAP, and tree serialization.
- Add package tests, benchmarks, versioned Documenter documentation, coverage, and CI.

The interface remains experimental.
