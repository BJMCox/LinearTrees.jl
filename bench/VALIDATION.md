# Robust fitting, uncertainty, and scaling

Load `validation.jl` from the benchmark environment. Use one fitting thread and
one BLAS thread for comparisons. Keep generated results outside the checkout.

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
include("bench/validation.jl")
using .ValidationBench

robust = bench_robustness(output="/tmp/lineartrees/robust.csv")
prior = bench_prior_calibration(output="/tmp/lineartrees/prior.csv")
coverage = bench_calibration(output="/tmp/lineartrees/calibration.csv")
scaling = bench_scaling(output="/tmp/lineartrees/scaling.csv")
summary = calibration_summary(coverage)
```

Start Julia with `--project=bench --threads=1`. Data construction, compilation,
and scoring stay outside timed operations. BenchmarkTools measures warmed full
fits and records median time, allocated bytes, and allocation count. The ridge
refit measurement includes its routing-tree fit. The selected-Huber measurement
includes all validation candidates. Each repetition creates a fresh seeded RNG.

## Robust fits

Seeds 501–510 generate 512 fitting, 256 validation, and 512 test rows with four
uniform predictors. The signal is piecewise linear. Independent noise is either
Gaussian with SD 0.2, 10% symmetric response contamination of magnitude 8 added
to that noise, or Student-t with three degrees of freedom rescaled to SD 0.2.

Compare MSE trees, `Huber(1)`, validation-selected Huber, and Huber boosting.
Single trees use depth 6, minimum leaf 20, and minimum fit 40. Boosting uses
30 rounds, learning rate 0.1, and depth 2. Validation MAE chooses delta from
the fixed set `(0.5, 1, 2)`. Test responses and the known signal select nothing.
Report test RMSE, test MAE, and RMSE against the noiseless signal. Symmetric
contamination preserves the signal's mean and median. Noisy test RMSE can hide
estimation differences when irreducible contamination dominates it.

## Predictive checks

Student-t scale squared differs from its variance. Score the returned Student-t
or finite mixture directly with Distributions.jl. Use its CDF for central
50/80/90/95% observation coverage and its quantiles for 90% interval width.
Check latent 90% coverage against the noiseless function separately. Retain ten
PIT histogram bins, log density, RMSE, model size, and effective component count.

Two experiments answer different questions:

- `bench_prior_calibration` fixes transforms, priors, and two geometries before
  generating any responses. It draws the root or an off-center split with equal
  probability, inverse-gamma noise variance, and normal continuous coefficients.
  An explicit continuity projector generates split coefficients. It fits the
  public ensemble API with identity response normalization. This checks the
  frozen conjugate model through the public fixed-transform route.
  The default uses 500 fitted replicates with seed `8100 + replicate`.
- `bench_calibration` measures empirical coverage of the public fitting policy.
  It draws independent structure, fitting, validation, and test responses on
  predeclared predictor grids. Shared support endpoints preserve the exact
  predictor normalization required by the public ensemble API. Response center
  and scale still come from fitting responses and are treated as fixed.

The public tree and ensemble APIs also accept
`response_normalization=(center=..., scale=...)`. Identity response normalization
uses center zero and scale one. To check that public route under a frozen prior,
choose the response transform, geometry space, and hyperparameters before
drawing fitting responses, conditional on the observed predictor design. Giving
the API fixed values estimated from those same responses does not meet this
condition. Fixed response transforms also do not account for response-driven
geometry discovery or hyperparameter tuning.

The public-fit experiment uses 64 structure rows, 64 fitting rows, 32 validation
rows, and 32 in-support test rows. Five cases isolate affine Gaussian truth, a
continuous kink, quadratic mean misspecification, heteroskedastic Gaussian
noise, and Student-t noise. The default has 500 independent fitted replicates
per case. Seed `7100 + replicate` defines paired streams across policies.

Candidate discovery offers zero, one, or two splits, with three possible cuts,
minimum leaf 8, and split penalty 2. Duplicate geometries receive no extra prior
mass. Compare four policies: root only, candidates discovered on independent
responses, candidates discovered on fitting responses, and validation-selected
coefficient precision from `(0.01, 1)`. All other priors remain fixed. Test data
enter only the final metrics. These are restricted candidate spaces, not a
posterior over all trees.

`calibration_summary` reports the mean and Monte Carlo standard error across
fitted replicates. Each replicate is a cluster. Its query rows share posterior
parameters and must not count as independent fitted experiments. Frozen-model
coverage should match nominal coverage within Monte Carlo error. Fixed-truth
coverage under the public fitted-normalization policy need not be nominal.
Neither a successful analytic oracle nor one nominal coverage level establishes
general calibration. A short smoke run checks execution, not calibration.
Predeclare a Monte Carlo tolerance before reading results, such as four MCSEs
for the five frozen-model coverage metrics. Keep these simulation diagnostics
separate from deterministic CI assertions.

## Wider and larger fits

The declared dimension pairs are `(1000, 8)`, `(1000, 32)`, `(8000, 8)`,
`(8000, 32)`, and `(32000, 8)`, with seeds 601–602 and three timing samples.
Training rows and feature columns form nested prefixes within each seed.
The signal uses four predictors, including a kink and a slope change across a
partition. Each fit uses the same 2,048 independent test rows and noise SD 0.2.

Compare exact, coarse-bin, and hybrid PILOT; ridge-refitted PILOT; exact,
coarse-bin, and refined model trees; MSE boosting; and continuous trees.
Model trees use all predictors, depth 3, minimum leaf 40, ridge penalty 1, and
split penalty 1. Binned methods use 32 bins. Continuous fits use at most two
splits, one cut per predictor, no pair terms, and graph-pruned search.
The other tree and boost settings match the robust study.

Report each method's cost together with test error and fitted model size.
Different depth and model families impose different approximation budgets.
These fixtures measure controlled scaling, not a general quality ranking.
Preserve the exact methods and assess the cost/error tradeoff before selecting
an approximate method. Record Julia, CPU, thread counts, and the resolved
benchmark manifest with every comparison.
