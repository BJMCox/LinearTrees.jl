# Real-data regression benchmark

`realdata.jl` runs paired fixed-setting comparisons across three seeds, with a
separate CASP scaling experiment. It returns result rows and optionally writes a
CSV checkpoint after every completed method. Downloads, splitting, preprocessing,
compilation and metric calculation are outside timed operations.

## Sources

The files are downloaded directly from UCI and checked against pinned SHA-256
hashes before parsing. UCI distributes these datasets under CC BY 4.0; cite the
linked dataset records when sharing results.

| Dataset | Rows | Numeric predictors | Target | Download bytes |
| --- | ---: | ---: | --- | ---: |
| [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise), Brooks, Pope & Marcolini (1989), DOI 10.24432/C5VW2C | 1,503 | 5 | Sound pressure, dB | 59,984 |
| [Abalone](https://archive.ics.uci.edu/dataset/1/abalone), Nash et al. (1994), DOI 10.24432/C55C7W | 4,177 | 9 | Rings | 191,873 |
| [Protein Tertiary Structure / CASP](https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure), Rana (2013), DOI 10.24432/C5QW3H | 45,730 | 9 | RMSD | 3,528,710 |

Abalone's seven continuous predictors are followed by fixed male and female
indicators, with infant as the reference. No target transform, missing-value
imputation, feature selection or outlier filtering is applied.

| File | SHA-256 |
| --- | --- |
| `airfoil_self_noise.dat` | `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3` |
| `abalone.data` | `de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f` |
| `CASP.csv` | `4277cfcb4e91a181746cbc654f001b57951c9e6a80f4f795fdb5c807e0848f40` |

Exact download URLs are stored in `RealDataBench.DATASETS`. The default cache is
`joinpath(tempdir(), "lineartrees-realdata-cache")`; a supplied `cache_dir` supports
offline runs using the same verified files.

## Fixed protocol

Seeds `301:303` shuffle groups of identical raw predictor rows using `StableRNG`.
Groups remain intact and targets never enter grouping. Groups are assigned to
approximately 60% training, 20% validation and 20% test rows. The training group
prefix is capped at 2,000 rows for **every method** in the paired experiment;
validation and test use their full partitions. Each cap includes only complete
groups and stops before the first group that would exceed the cap, so the actual
size can fall slightly below the cap. Scaling uses nested prefixes of the same seed-301 CASP training groups
at 2,000, 8,000 and all available training rows.

Predictor means and population standard deviations, and the target mean and
population standard deviation, are estimated on the retained training rows only.
Constant columns use scale one. Validation and test apply those fixed transforms.
All errors and predictive density are converted back to native target units.
Each CSV row identifies the source and ordered train/validation/test membership
with SHA-256 hashes, and records the target scale and actual partition sizes.

Settings are fixed before evaluation, without a hyperparameter sweep. Validation
selects boosting rounds and pruning decisions. Test outcomes select nothing, and
models are not refitted on validation before reporting test performance.

| Method | Settings |
| --- | --- |
| `global_ridge` | `fit_model_tree`, depth 0, all predictors, ridge penalty 1, no prediction clipping |
| `pilot` | MSE PILOT, BIC, exact search, depth 6, minimum leaf 20, minimum fit 40 |
| `pilot_huber` | Same PILOT settings with `Huber(1.0)` on standardized targets |
| `boost` | MSE boosted PILOT, at most 30 rounds, depth 3, learning rate 0.1, subsample 0.8, validation patience 5, seeded RNG |
| `ridge_refit` | Full-response ridge refit of `pilot`, all predictors, ridge penalty 1 |
| `model_exact` | Parametric model tree, exact search, depth 3, minimum leaf 40, ridge penalty 1, split penalty 1 |
| `model_binned` | Same model tree with 32 bins and refinement disabled |
| `model_refined` | Same model tree with 32 bins and local exact refinement |
| `model_pruned` | Validation pruning of a freshly fitted `model_exact`, all predictors, ridge penalty 1 |
| `continuous` | No interaction pairs, graph-pruned search, at most 3 splits/depth 3, minimum leaf 40, 3 cuts per predictor, split penalty 2, default coefficient/noise priors |
| `finite_ensemble` | Equal prior mass on distinct root/no-pair, 3-split/no-pair, and 3-split/pair-(1,2) geometries; common-data posterior refits and evidence weights |

The ensemble's `(1,2)` interaction is a fixed limited basis example. Geometry
duplicates are removed before fitting. Its fit timing includes building all
three source geometries. All datasets have at most nine numeric predictors, all
of which enter multivariate leaves. This bounded experiment does not measure
continuous search with unrestricted interactions, thresholds, or tree size.

The CASP scaling experiment includes global ridge, PILOT, boosted PILOT, ridge
refit, exact/binned model trees and validation-pruned exact model trees. Continuous
and finite-mixture models remain in the common capped comparison.

## Metrics and timing

Errors are native-unit RMSE and MAE, RMSE divided by training target standard
deviation (`nrmse`), and test R². `baseline_rmse` uses the training target mean.
Lower errors are preferable; larger R² is preferable.

For continuous models only, `predictive(...; observation=true)` supplies actual
Student-t marginals. The finite ensemble is evaluated as the weighted mixture
of its Student-t components. `mean_log_density` includes the target-scale
Jacobian; larger is preferable. Central 50%, 80%, 90% and 95% predictive coverage
uses the equivalent PIT condition `(1-level)/2 ≤ F(y) ≤ (1+level)/2`. This requires
no approximation or custom mixture-quantile solver. `pit_ks` measures departure
of the empirical PIT CDF from uniformity; it is a descriptive diagnostic, without
a significance claim. `mean_predictive_sd` is in native units. Calibration fields
are empty for point-prediction models, which supply no predictive distribution.

Abalone's integer ring target is treated as continuous. Its density scores and
PIT diagnostics describe that approximation, with density distinct from count
probability mass. Exact discrete-target PIT calibration would require a discrete
predictive model and randomized PIT.

Single-tree uncertainty conditions on its selected geometry and basis. The
finite ensemble is exact only within its supplied model space; its response-
selected geometries make this an adaptive approximation. Neither captures the
entire geometry-selection uncertainty. The CSV reports posterior weights,
largest weight, and `1/sum(weights.^2)` effective components to show concentration.

BenchmarkTools reports median runtime, allocated bytes and allocation count over
three samples with one operation per evaluation after compilation. Complete fits,
mean prediction and probabilistic prediction are timed separately. Set Julia and
BLAS to one thread and avoid concurrent benchmarks. `model_bytes` is Julia's
`Base.summarysize`, including retained arrays; it is not a serialized file size.
Node/leaf counts describe stored topology. Coefficient counts sum local predictive
basis sizes (CON 1, LIN/PCON 2, BLIN 3, PLIN 4), ridge intercepts/slopes, or independent
constrained continuous coefficients. PILOT counts can include redundant terminal
increments. Boosting sums counts across its component trees plus its intercept.

## Running

Materialize the benchmark environment before running. Its local Manifest is
ignored; archive that file with results when exact package versions matter.
Use `Pkg.develop` to select this checkout, including on Julia 1.10:

```sh
julia --project=bench -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Load the script in an explicit checkout session. For a shell run after setup:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=bench -t1 -e \
  'include("bench/realdata.jl"); bench_realdata(output_path="/tmp/realdata-paired.csv", on_result=println)'
OPENBLAS_NUM_THREADS=1 julia --project=bench -t1 -e \
  'include("bench/realdata.jl"); bench_realdata_scaling(output_path="/tmp/realdata-scaling.csv", on_result=println)'
```

A smoke run can restrict `datasets=(:airfoil,)`, `seeds=301:301`, and `samples=1`.
To profile model-tree split scoring with CPU and allocation flame graphs, run
`LT_PROFILE_CASES=9 julia --project=bench -t1 bench/profile.jl`.
Case 10 preserves a near-affine regression fixture: a small step beside a large
linear signal, with exact and refined searches. Select it with `LT_PROFILE_CASES=10`.
PProf progress stays in artifact logs. `LT_PROFILE_OUT` selects the output directory.
Record Julia/package versions, CPU, Julia/BLAS thread counts, checkout revision
and source changes with every result file. Compare paired per-seed outcomes and
report ranges; three seeds do not support precise confidence intervals.

These random holdouts estimate interpolation within the supplied datasets.
Airfoil experimental configurations and CASP decoys may be correlated; the
released CASP table has no protein identifiers for protein-level splits. Exact
predictor grouping prevents identical-input leakage but does not remove those
dependencies. Results do not establish performance on unseen experiments,
proteins, distributions, or all regression tasks.
