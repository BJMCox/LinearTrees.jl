# LinearTrees.jl profiling pass (P1)

Runtime, allocation and type-stability profile of the six cases in the P1
brief, plus a ranked list of remaining targets.

Reproduce with

```
julia --project=bench -t 1    bench/profile.jl      # runtime + allocation profiles, serial
julia --project=bench -t auto bench/profile.jl      # same, threaded
julia --project=bench         bench/typecheck.jl    # JET.report_opt + @code_warntype
julia --project=bench -t 1 --track-allocation=user bench/trackalloc.jl <case>
julia --project=bench         bench/trackalloc.jl <case> report
julia --project=bench -t 1    bench/ab.jl           # medians plus a bit-level dump per case
```

## Machine and versions

| | |
|---|---|
| CPU | Apple M4 Pro, 14 logical cores (10 performance) |
| Julia | 1.12.7 |
| `Threads.nthreads()` | 1 (serial pass), 10 (`-t auto`) |
| LinearTrees | this worktree; base `918c583`, fixes `38aad5e`, `dd4e1bb`, `0b92878` |
| Bench deps | BenchmarkTools 1.6, PProf 3, JET 0.10, StableRNGs 1, CategoricalArrays 1, DataFrames 1.8, StaticArrays 1.9 |

## Cases

| # | Case | Data | Tree |
|---|---|---|---|
| 1 | `fit_tree` MSE | `n = 200_000`, `p = 20`, step target, `max_depth = 12` | 3808 nodes |
| 2 | `fit_tree` `Softmax(3)` | `n = 100_000`, `p = 10`, column 10 categorical with 8 levels | 266 nodes |
| 3 | `fit_tree` `MAD()` | `n = 100_000`, `p = 10`, `niter = 5` | 1599 nodes |
| 4 | `fit(LinearTreeRegressorFit, df, y)` | `n = 100_000`, `DataFrame`, two `CategoricalArray` columns, eight numeric | 5236 nodes |
| 5 | `predict` / `predict!` | 1_000_000 rows, depth-12 tree (3674 nodes, one categorical column) | |
| 6 | `shap` | 10_000 rows, same tree | |

Data comes from `bench/cases.jl`, all from `StableRNG` seeds.

## Results at the branch base (`918c583`)

Median of five `@benchmark` samples, allocation count and bytes from the same
run. The three fixes below move some of these; the post-fix table is at the end
of the fixes section, and `bench/profiles/summary-t*.txt` holds the numbers as
committed, which are the post-fix ones.

| Case | serial (ms) | 10 threads (ms) | speedup | allocs | bytes |
|---|---|---|---|---|---|
| 1 `fit_tree` MSE | 2627 | 1071 | 2.45x | 70_453 | 102.1 MB |
| 2 `fit_tree` Softmax+cat | 1117 | 405 | 2.76x | 9_529 | 35.1 MB |
| 3 `fit_tree` MAD | 1051 | 478 | 2.20x | 20_229 | 34.8 MB |
| 4 `fit` on a DataFrame | 630 | 164 | 3.83x | 4_113_697 | 132.2 MB |
| 5 `predict` (1e6 rows) | 87.6 | 9.08 | 9.65x | 3 | 8.0 MB |
| 5 `predict!` (1e6 rows) | 86.7 | 11.4 | 7.62x | 0 | 0 |
| 6 `shap` (1e4 rows) | 3788 | 3817 | **1.00x** | 154 | 0.85 MB |

Per-unit figures: case 1 is 690 ns per node per feature scan; case 5 is 87 ns
per row (serial) for a depth-12 walk; case 6 is 379 us per row, or 103 ns per
(row, node) pair.

The frame rankings below come from the **serial** profiles. A threaded
`Profile.print` counts every thread's stack, idle ones included (utilization
was 56% for case 1), so its percentages measure occupancy, not work; the
threaded runs are used here only for wall time and scaling.

## Top frames per case, serial

### Case 1, `fit_tree` MSE (1781 samples)

| Samples | % | Frame | What it does |
|---|---|---|---|
| 1343 | 75% | `fit.jl:488` `best_split` | the whole split search for one node |
| 1164 | 65% | `fit.jl:392` `scan_feature` call | the per-feature sweep, called `p` times per node |
| 762 | 43% | `select.jl:41` `selection_score` | the BIC formula, once per candidate per split point |
| 645 | 36% | `log.jl:261` `log(::Float64)` | the two `log` calls inside that formula |
| 432 | 24% | `fit.jl:517` LIN recursion | a `LIN` node re-enters `grow_subtree` at the same depth |
| 361 | 20% | `scan.jl:100` BLIN scoring | `selection_score` for the broken-line candidate |
| 285 | 16% | `scan.jl:109` PLIN scoring | same, for the two-line candidate |
| 179 | 10% | `fit.jl:391` `gather!` | copies the node's rows into the worker's scratch, in feature order |
| 176 | 10% | `fit.jl:125` `presort!` | the one-off `sortperm` per column at the start of the fit |
| 164 | 9% | `scan.jl:93` PCON scoring | same, for the two-constant candidate |
| 100 | 6% | `scan.jl:97` / `accumulate.jl:69` `fit_blin` | the 3x3 solve behind the broken-line candidate |

`fit_blin`, the only real linear algebra in the sweep, costs 6%. Scoring the
candidates costs 43%. The BIC score is `n log(dev/n) + ncoord dof(kind) log(n)`
and its second term does not depend on the split point at all.

### Case 2, `fit_tree` Softmax(3) with a categorical column (725 samples)

| Samples | % | Frame | What it does |
|---|---|---|---|
| 617 | 85% | `fit.jl:488` `best_split` | as case 1 |
| 545 | 75% | `fit.jl:392` `scan_feature` call | |
| 285 | 39% | `select.jl:41` `selection_score` | |
| 218 | 30% | `log.jl:261` `log(::Float64)` | |
| 162 | 22% | `scan.jl:100` BLIN scoring | |
| 110 | 15% | `scan.jl:97` / `accumulate.jl:96` `fit_blin` | the `SVector` method: one scalar solve per class coordinate |
| 96 | 13% | `accumulate.jl:69` scalar `fit_blin` | called `K-1` times from the above |
| 81 | 11% | `scan.jl:93` PCON scoring | |

The `SVector{2,Float64}` coefficient type costs about a third more per
candidate than the scalar one but introduces no dispatch: `fit_blin` for a
vector `V` is the scalar method run `K-1` times, fully inlined.

### Case 3, `fit_tree` MAD, `niter = 5` (692 samples)

| Samples | % | Frame | What it does |
|---|---|---|---|
| 553 | 80% | `fit.jl:557` / `:559` child recursion | |
| 284 | 41% | `sort.jl` `_sort!` | the sort inside `median_abs!` |
| 246 | 36% | `loss.jl:253` `irls_epsilon!` | `1e-3 x` weighted median of `abs(residual)` |
| 240 | 35% | `loss.jl:238` `median_abs!` | sorts `abs.(r)` in full to read one weighted quantile |
| 224 | 32% | `loss.jl:276` `refit_node` | the IRLS refit, `niter` passes |
| 198 | 29% | `fit.jl:611` `irls_epsilon!` in `irls_refit` | once per IRLS pass per node |
| 320 | 46% | `fit.jl:488` `best_split` | the split search, smaller share than case 1 |
| 197 | 28% | `select.jl:41` `selection_score` | |

The IRLS path costs more than the split search here, and 41% of the whole fit
is one `sort!`: `median_abs!` sorts the node's entire residual vector, `niter`
times per node, to read a single weighted median.

### Case 4, `fit` on a DataFrame (447 samples)

| Samples | % | Frame | What it does |
|---|---|---|---|
| 358 | 80% | `fit.jl:61` `fit_tree` | the growth itself |
| 283 | 63% | `fit.jl:488` `best_split` | |
| 157 | 35% | `select.jl:41` `selection_score` | |
| 67 | 15% | `statsapi.jl:170` `encode` | the table-to-matrix pass |
| 87 self | 19% | `statsapi.jl:104-108` `encode_column!` | the per-row conversion and store loops |

`encode` is 15% of the case and, on its own, accounts for **4.11 million of the
case's 4.11 million allocations** (41 per row per column): see the type
stability section.

### Case 5, `predict` on 1_000_000 rows (57 samples)

Too few samples to rank finely; 87 ns per row over a depth-12 walk. All time
is inside `score_row`, split between `predict.jl:36`/`:39` (the branch
increment `n.lcoef * x + n.lintercept`) and `predict.jl:23` (`X[i, n.feature]`,
a strided load down a column-major matrix). No allocation at all in
`predict!`; `predict`'s 8 MB is the output vector.

### Case 6, `shap` on 10_000 rows (2483 samples)

| Samples | % | Frame | What it does |
|---|---|---|---|
| 2322 | 94% | `shap.jl:231`/`:232` `visit!` recursion | every node is visited for every row |
| 1035 | 42% | `shap.jl:125` `unwound_sum` call | the `O(depth)` weight recomputation, run once per path position |
| 411 | 17% | `shap.jl:224` hot-branch `attribute_constant!` | |
| 380 | 15% | `shap.jl:113` `unwound_sum` body | |
| 315 | 13% | `shap.jl:228` own-term `attribute_constant!` | |
| 253 | 10% | `shap.jl:230` cold-branch `attribute_constant!` | |
| 496 self | 20% | `promotion.jl:637` `==` | `e.onefrac == e.zerofrac` and `onefrac != 0`, the two guards in the inner loops |
| 481 self | 19% | `essentials.jl:919-920` `getindex` | loading `PathElem`s out of `Vector{PathElem}` |
| 281 self | 11% | `essentials.jl:11` `length` | |

`attribute_constant!` is `O(depth^2)` per node visit and runs three times at
every split node, so the whole recursion is `O(nodes x depth^2)` per row. The
inner loop reads one field, `weight`, out of a 32-byte struct.

## Allocations

Sampled with `Profile.Allocs.@profile sample_rate = 0.01`
(`bench/profiles/*-t1.allocs.txt`) and confirmed line by line under
`--track-allocation=user` (`bench/profiles/mem-case<N>/top.txt`).

Cases 1 to 4 were run under the malloc log at a tenth of their row count
(`LT_TRACKALLOC_DIV`, default 10): the log runs two orders of magnitude slower
than the profiled build, and it answers *which* lines allocate, not how many
bytes one full call takes -- the per-call totals in the table come from
`@benchmark`. Cases 5 and 6 were not run under it: `predict!` and the SHAP
recursion allocate nothing per row, which `test/allocations.jl` asserts
directly and the sampling profiler confirms.

| Case | allocs / call | bytes / call | bytes / row | dominant site |
|---|---|---|---|---|
| 1 | 70_453 | 102.1 MB | 510 B | `fit.jl:535` the two child row vectors per split node |
| 2 | 9_529 | 35.1 MB | 351 B | `fit.jl:314-344` `scan_categorical`'s per-node level arrays |
| 3 | 20_229 | 34.8 MB | 348 B | `fit.jl:535` as case 1 |
| 4 | 4_113_697 | 132.2 MB | 1322 B | `statsapi.jl:94-108` `encode_column!`, one box per element |
| 5 | 0 (`predict!`) | 0 | 0 | none |
| 6 | 154 | 0.85 MB | 85 B | `shap.jl:47` the per-block `PathPool`, then nothing per row |

The malloc log ranks case 1's lines as `fit.jl:539` (`leftrows`/`rightrows`,
750 kB at n = 20_000), `fit.jl:124` (`idx`, one-off `n x p x 4` bytes),
`fit.jl:126` (`FitState`'s `g`, `h`, `z`, one-off) and `fit.jl:394`
(`leftcodes = Int[]`, an empty vector allocated once per feature per node,
239 kB). Case 4 is dominated by `statsapi.jl:94-108`, 8.5 MB for
10_000 rows x 10 columns, one box per element. Case 2 adds
`fit.jl:318`/`:347`, `scan_categorical`'s per-node level arrays.

So case 1's 102 MB per full call is almost all `grow_subtree`'s
`leftrows`/`rightrows`: two fresh `Vector{Int32}` per split node, sized
exactly but never reused, `O(n)` bytes per tree level. Case 4's 4.11 million
allocations were one per element read in `encode_column!`; fix 2 below removes
them.

## Type stability

`JET.report_opt(..., target_modules = (LinearTrees,))` over every exported
entry point (`bench/profiles/jet.txt`):

| Entry point | Sites | Verdict |
|---|---|---|
| `fit_tree`, all ten losses, with and without categorical columns | 0 | clean |
| `predict`, `predict!`, `score` (scalar and `Softmax`) | 0 | clean |
| `shap`, `shap!` (scalar and `Softmax`) | 0 | clean |
| `feature_importance`, `coeftable`, `expected_score`, `to_dict` | 0 | clean |
| `from_dict` | 18 | per call |
| `fit(LinearTreeRegressorFit, df, y)` | 22 | 11 per row, 11 per fit |
| `fit(LinearTreeClassifierFit, df, y)` | 29 | 11 per row, 18 per fit |
| `predict(::LinearTree*Fit, df)` | 17 | 11 per row, 6 per column |
| `MMI.fit` (regressor / classifier) | 23 / 25 | as the `fit` above plus 1 per fit |
| `MMI.predict` (regressor / classifier) | 17 / 17 | as `predict` above |

The brief's addendum asked whether the 193 sites `report_opt` used to report
for `fit_tree` with `Softmax(3)` and a categorical column were real. They were
not reachable dispatch: they were `report_opt` noise on `SVector` broadcasts
plus the `Softmax` field read. Both are gone on the current base (`55fd0c4`
removed the last real dispatch, `918c583` made the class count a type
parameter), and the count for that exact call is now **0**. Nothing remains to
fix there.

The 11 per-row sites shared by every table entry point are all one defect:

```
src/statsapi.jl:89   col = Tables.getcolumn(cols, nm)   # ::AbstractVector
src/statsapi.jl:94   for i in eachindex(col)            # every read, convert and
src/statsapi.jl:105     v = Float64(col[i])             # store below dispatches
src/statsapi.jl:107     out[i, j] = v                   # at run time
```

`Tables.getcolumn` is typed `AbstractVector`, so `encode_column!` runs its
per-row loops on an abstract binding. **Verdict: per row.** Fix: a function
barrier, one call per column into a helper that specialises on the column's
concrete type. Applied as fix 2 below; after it the counts are

| Entry point | Sites before | after |
|---|---|---|
| `fit(LinearTreeRegressorFit, df, y)` | 22 | 11 |
| `fit(LinearTreeClassifierFit, df, y)` | 29 | 19 |
| `predict(::LinearTree*Fit, df)` | 17 | 6 |
| `MMI.fit` (regressor / classifier) | 23 / 25 | 12 / 14 |
| `MMI.predict` (regressor / classifier) | 17 / 17 | 6 / 6 |

and the six that remain on the `predict` path are the two barrier calls, the
`any(ismissing, col)` scan, `length(col)`, the `n >= PARALLEL_MIN_ROWS`
comparison and the output-matrix allocation -- one each per column or per call.
**Verdict: per column. Not worth changing**, unless `Tables.columns` can be
narrowed once per encoder, which is a table-interface question, not a
LinearTrees one.

`from_dict`'s 18 sites are reads out of a `Dict{String,Any}` and each runs once
per call. **Verdict: per call, not worth a change.** The remaining `fit`-level
sites (`DataAPI.refpool`, `DataAPI.levels`, the class map) are one per column
or one per fit. **Verdict: per fit.**

`@code_warntype` on `scan_feature`, `gather!`, `partition_column!`,
`score_row`, `visit!`, `addrow`, `fit_lin` and `fit_blin`, scalar and
`SVector{2,Float64}` (`bench/profiles/warntype.txt`): **no red types**. The
only unions are `Union{Nothing, Tuple{...}}` returned by `fit_lin` and
`fit_blin`, which is the deliberate singular-system signal and is small enough
for the compiler to split.

## Fixes applied in this pass

Each is local, bit-identical, and measured. Bit identity is checked two ways:
`test/partition.jl` replays the four stored fixture designs
(`test/fixtures/partition/*.json`) node by node, and `bench/ab.jl` writes a
canonical bit-level dump of every node field of every case's tree and of the
case's own predictions and SHAP values, which is `diff`ed against the same
dump taken on the base.

A note on that dump: an earlier version of `bench/ab.jl` summarised each tree
with `hash` over its `Any`-typed struct fields, and that hash was not stable
across processes even for an unchanged tree. Two runs of the same code agreed;
two runs separated by an unrelated edit did not. The dump prints float bit
patterns and integers as text instead, and `diff` decides. Anything that
compares "identical" through `hash` over `Any` is worth re-checking.

### 1. `log(n)` hoisted out of the split sweep (`38aad5e`)

`selection_score` under `BIC` calls `log` twice per candidate,
`log(dev / n)` and `log(n)`. The second depends only on the node's total
weight, which is fixed before the sweep starts. `score_logn(rule, n)`
(`src/select.jl`) returns it once per feature and `scan_feature` passes it to
every `PCON`, `BLIN` and `PLIN` candidate; the penalty term itself stays
inside `selection_score`, so the expression is unchanged and so is the score.

| Case | serial before | serial after | 10 threads before | after |
|---|---|---|---|---|
| 1 MSE | 2628 ms | 2183 ms (-17%) | 1071 ms | 851 ms (-21%) |
| 2 Softmax | 1103 ms | 1053 ms (-5%) | 403 ms | 383 ms (-5%) |
| 3 MAD | 1053 ms | 931 ms (-12%) | 480 ms | 430 ms (-10%) |
| 4 table | 612 ms | 526 ms (-14%) | 163 ms | 134 ms (-18%) |

Cases 5 and 6 do not call it and did not move.

Moving the whole penalty out, rather than just its `log`, was tried first and
rejected: it changed the fitted trees. The isolated arithmetic agrees to the
bit over 200_000 random inputs, so the difference is in how LLVM contracts
`a * b + c * d * e` once one factor arrives through a function boundary. That
is a reminder that "the same formula" is not the same float unless the
expression is left standing.

### 2. Function barrier in `encode_column!` (`dd4e1bb`)

`Tables.getcolumn` is typed `AbstractVector`, so the per-row loops ran on an
abstract binding: every element read, conversion, level lookup and store
dispatched at run time and boxed its result. `encode_levels!` and
`encode_numeric!` take the column as an argument, so there is one dynamic call
per column and a fully typed body inside. This closes the 17 `report_opt`
sites shared by `fit(LinearTree*Fit, ...)`, `predict` on the fits and both MLJ
methods.

| Case 4 | before | after |
|---|---|---|
| serial | 526 ms | 414 ms (-21%) |
| 10 threads | 134 ms | 110 ms (-18%) |
| allocations | 4_113_698 | 123_918 (33x fewer) |
| bytes | 132.2 MB | 49.1 MB |

`before` is already on top of fix 1; against the branch base the serial case
is 612 -> 414 ms, a 32% cut.

### 3. `shap` threads below `PARALLEL_MIN_ROWS` (`0b92878`)

`PARALLEL_MIN_ROWS = 2^14` is calibrated on one `score_row` walk per row. One
SHAP row visits every node and does `O(depth^2)` work at each, 379 us per row
on the case-6 tree, so threading pays from a handful of rows. On the shared
gate, `shap` on 10_000 rows ran in one block whatever `nthreads` said, which
is why case 6's serial and threaded medians were identical in the table above.

`row_blocks` now takes a `minrows` keyword defaulting to the old constant, and
`shap!` passes `shap_min_rows(tree) = max(2, PARALLEL_MIN_ROWS / nodes)`.

| Case 6 | before | after |
|---|---|---|
| serial | 3763 ms | 3763 ms (unchanged) |
| 10 threads | 3787 ms | 458 ms (8.3x) |

### What the profile looks like after the three fixes

Case 1 (1470 samples, down from 1781): `selection_score` is 33% (was 43%) and
`log` 27% (was 36%). `presort!` rises to 18% of the case purely because the
rest got faster, which promotes it in the ranked list below. Case 4
(275 samples, down from 447): `encode` no longer clears the 2% floor at all.
Case 6 is unchanged serially and now scales.

### Where the six cases stand now

From `bench/profiles/summary-t1.txt` and `summary-t10.txt`, the numbers as
committed. Run-to-run spread on the fit cases is a few percent, so the
per-fix tables above and this one differ slightly where the same quantity
appears in both.

| Case | serial (ms) | 10 threads (ms) | speedup | change vs base, serial | allocs | bytes |
|---|---|---|---|---|---|---|
| 1 `fit_tree` MSE | 2101 | 850 | 2.47x | -20% | 70_453 | 102.1 MB |
| 2 `fit_tree` Softmax+cat | 1062 | 380 | 2.79x | -5% | 9_529 | 35.1 MB |
| 3 `fit_tree` MAD | 926 | 429 | 2.16x | -12% | 20_229 | 34.8 MB |
| 4 `fit` on a DataFrame | 415 | 110 | 3.76x | -34% | 123_917 | 49.1 MB |
| 5 `predict` (1e6 rows) | 87.0 | 9.20 | 9.45x | 0 | 3 | 8.0 MB |
| 5 `predict!` (1e6 rows) | 86.8 | 12.3 | 7.06x | 0 | 0 | 0 |
| 6 `shap` (1e4 rows) | 3799 | 466 | 8.15x | 0 serial, 8.1x threaded | 154 | 0.85 MB |

## Ranked targets

Ordered by measured share of runtime or memory. Everything here is larger than
a local, bit-identical edit, which is why it is a target and not a fix.

### 1. Score each model kind once per feature, not once per split point

**Where** `src/scan.jl:95-117` calling `src/select.jl:54-60`.

**Share** `selection_score` was 43% of case 1 at the base, 39% of case 2, 35%
of case 4, 28% of case 3. Fix 1 removed the loop-invariant `log(n)`; what is
left is 33% of case 1, of which `log(dev / n)` is 27%.

**Change** For a fixed kind, `n log(dev/n) + pen` is strictly increasing in
`dev`, so the sweep can carry the lowest `max(dev, dmin)` seen for each of
`PCON`, `BLIN` and `PLIN`, together with the split point and coefficients that
produced it, and call `log` three times per feature instead of three times per
split point.

**Risk** Medium, and it is not bit-identical. Today the five kinds are compared
interleaved by split point, so an exact tie between two kinds resolves to
whichever was reached first; deferring the comparison resolves ties in kind
order instead. The change needs a stated tie-break rule, regenerated
`test/fixtures/partition/*.json`, and a test over a design where two kinds
reach the same score.

### 2. Weighted median by selection, not by a full sort

**Where** `src/loss.jl:232-240`, reached from `node_epsilon`
(`src/fit.jl:204-213`) and `irls_refit` (`src/fit.jl:611`).

**Share** 41% of case 3 (`MAD`, `niter = 5`) is `sort!` inside `median_abs!`.
The whole IRLS path is about 60% of that fit.

**Change** `median_abs!` sorts the node's entire residual vector to read one
weighted quantile. A weighted-median selection (quickselect partition on
`abs.(r)` carrying `w`) is `O(m)`. When every weight in the node is equal --
the `weights = nothing` default -- `partialsort!` alone suffices.

**Risk** Medium. The `cum == target` midpoint rule and the behaviour on
duplicate values and zero weights have to be reproduced exactly. Guarded by
the `quantile_weighted` fixture plus new unit tests.

### 3. SHAP: parallel path arrays, then attribution at leaves

**Where** `src/shap.jl:63-129` and `:160-233`. Fix 3 below made SHAP scale
across threads; this is the per-thread cost, which it did not touch.

**Share** `unwound_sum` is 42% of case 6; loading `PathElem`s and the two float
guards account for 39% of the profile's self time.

**Change** Two independent steps. (a) Store the path as parallel
`Vector{Float64}`s on `PathPool`: `unwound_sum`'s inner loop reads one field,
`weight`, and today pulls a 32-byte struct for it. Estimated 15-25%. (b) Move
attribution to the leaves, the classical TreeSHAP shape: the recursion is
`O(nodes x 3 x depth^2)` per row today because `attribute_constant!` runs three
times at every split node. Estimated 2-3x.

**Risk** (a) low, mechanical. (b) high: it changes the recursion's contract and
the per-branch linear increments are exactly why the attribution sits at the
nodes. Its own stream.

**Measured** Both estimates were wrong. (a) loses: four parallel arrays cost
5.68 s serial on case 6 against 3.84 s for the single `Vector{PathElem}`, and
4.75 s with an explicit length and spare capacity to keep `resize!` out of the
hot path; splitting only `weight` out costs 20%. A path holds at most one
element per tree level, so it never leaves L1 and there is no locality to win,
while each extra array costs another bounds check per read (no `@inbounds`
here) and another `resize!`/`copyto!` in `copyinto!`, which runs three times
per split node per row. The 39% self time was the loop's one bounds-checked
load. Hoisting the loop-invariant `onefrac != 0` guard out of `unwind!` and
`unwound_sum` landed and is worth 2.5-2.7% on a paired run.
(b) works but pays 1.28x, not 2-3x: 2.97 s serial and 428 ms on 10 threads.
It halves the number of `attribute_constant!` calls, but the calls it removes
sat at shallow nodes and the ones that absorb them sit at the leaves, where
`O(depth^2)` is largest. It is also not byte-identical -- 6% of case 6's
values are unchanged, max absolute difference 3.1e-14 on values up to 2.2 --
so it did not land; see
`.superpowers/sdd/2026-09-05-tree-followups/Q3-shap-report.md`. The own-feature
terms are what is left, and 659 of case 6's 2062 internal nodes carry an own
term that is exactly zero.

Byte identity was checked throughout with `bench/ab.jl`'s dumps, which is the
only sound way to check it here: SHAP's last mantissa bits depend on the target
architecture and on `--check-bounds`, so a dump is comparable between two
commits only on one machine under one flag. A bit constant recorded in the test
suite is not portable and does not belong there.

### 4. `presort!`

**Where** `src/fit.jl:168-175`.

**Share** 10% of case 1 at the base, 18% after fix 1, once per fit, and it
grows with `p`.

**Change** `sortperm(view(X, :, j); alg = MergeSort)` sorts an index vector
through indirect comparisons and allocates a fresh permutation per column.
Sorting `(value, index)` pairs, or a 3-pass LSD radix sort over the `Float64`
bit pattern, keeps the same stable order at 2-3x the throughput. Estimated 6-8%
of a wide fit.

**Risk** Low. Stability is the only contract, and `test/partition.jl` already
exercises it through a column of rounded values with many ties.

### 5. Child row vectors in `grow_subtree`

**Where** `src/fit.jl:534-538`.

**Share** 102 MB of case 1's 102 MB and most of its 70_453 allocations; about
4% of its samples.

**Change** Two fresh `Vector{Int32}` per split node, sized exactly and then
thrown away. After `partition!` the children's rows are exactly
`idx[lspan, 1]` and `idx[rspan, 1]`, so both could be views, or spans, instead
of copies.

**Risk** Medium. `partition!` needs `leftrows` before it runs and the parent's
row order feeds every downstream sum, so the ordering contract has to be
re-derived, not assumed. Little runtime win at these sizes; the gain is GC
pressure, which matters more as thread count rises.

### 6. `scan_categorical`'s per-node level arrays

**Where** `src/fit.jl:311-363`.

**Share** the top allocation site of case 2 (`fit.jl:314`, `:321`, `:328`,
`:332`, `:343`, `:344`, `:359`) -- eight arrays per node per categorical
feature.

**Change** Hold `sz`, `sw`, `counts`, `cumoffset`, `cursor` and `rank` on
`Scratch`, sized to `maximum(nlevels)`. They are already per worker, so no new
sharing question arises.

**Risk** Low. `Scratch` grows by six fields, which is the cost.

### 7. `fit_tree`'s threaded scaling

**Where** `src/fit.jl:406-424` and `:542-556`.

**Share** case 1 reaches 2.47x on 10 threads, case 3 2.16x, against 9.45x for
`predict`.

**Change** `best_split` threads over features only above `PARALLEL_MIN_ROWS`
rows, and sibling subtrees spawn only below `SUBTREE_PARALLEL_DEPTH = 3`, which
caps concurrency at eight subtrees and, because `tids` halves at each spawn,
usually fewer. A depth-12 tree keeps most of its nodes below both gates. Gate
the spawn on `length(tids) >= 2` alone and let depth run.

**Risk** High: the disjoint-`tids` argument is the entire safety case for
concurrent growth, so this needs `test/threads.jl` extended over odd thread
counts and a re-check that threaded still equals serial bit for bit.

### 8. `score_row`'s column-major loads

**Where** `src/predict.jl:23`.

**Share** case 5 is 87 ns per row for a depth-12 walk, serial, with no
allocation and no dispatch. Nothing else is left in it.

**Change** Each level reads `X[i, n.feature]`, so a row's walk touches one
cache line per level and no two rows share one. Walking a whole block of rows
through one tree level at a time would fix the access pattern. Estimated 2-3x.

**Risk** High: it needs a per-row frontier buffer and changes `predict`'s
memory contract for a path that is already the fastest thing in the package.
Only worth it if `predict` becomes the bottleneck.

### Not worth changing

- `from_dict`'s 18 dynamic-dispatch sites: once per call, on a `Dict{String,Any}`.
- `fit_blin`'s 3x3 solve: 6% of case 1, the only real linear algebra in the sweep.
- The `Union{Nothing, Tuple}` returns of `fit_lin`/`fit_blin`: a two-member union the compiler splits.
- The `Dict` level lookup left in `encode_levels!`: about 5% of case 4 after the barrier below, against a `DataAPI.refarray` rewrite that would have to handle every table type.

## P2: the scan-speed pass

Protocol for every number in this section: Apple M4 Pro, Julia 1.12.7, eleven
`@benchmark` samples, and the two arms of each comparison run alternately from
the same source tree with only the change under test toggled. Run-to-run drift
on this machine reached 8% during the pass, so a before/after pair taken
sequentially is not trustworthy at this scale; best-of-eleven is quoted next to
the median wherever the margin is small. Reproduce with `bench/ab.jl` (case 1
is its first row) or with the case-1 data from `bench/cases.jl` alone.

### 1. Score once per kind per feature (ranked target 1, done)

`scan_feature` now carries the lowest `devkey` each of `pcon`, `blin`
and `plin` reaches through the split sweep and calls `selection_score` once per
kind at the end, so `log(dev / n)` runs three times per feature instead of
three times per split point. `devkey` is the deviance in the form the rule
compares it in -- `max(dev, dmin)` for `BIC`, the raw value for `MinDeviance`,
whose score has no floor -- so the per-kind minimiser is the one the
interleaved sweep kept, and the key hands to `selection_score` for a
bit-identical score. Case 1 drops from **2086 ms to 1429 ms serial (-31%)**
and from **826 ms to 542 ms on ten threads (-34%)**; allocations are
unchanged; best-of-eleven, 2023 -> 1363 ms serial and 801 -> 532 ms on ten
threads, gives the same picture. Only an exact cross-kind score
tie can now resolve differently (it goes to the earlier kind in
`(con, lin, pcon, blin, plin)` rather than to the lower split point), and the
fitted trees are bit-identical on all thirteen designs checked: the four
`test/fixtures/partition` cases, the eight PILOT reference fixtures and a
`Softmax(3)` design with a categorical column. The fixtures were therefore not
regenerated.

### 2. The MSE unit-hessian path (committed, above the 3% bar)

With `MSE` and no frequency weights every row hessian is exactly one, and the
sweep still multiplied every row sum by it. `unit_hessian(loss)` plus a
`all(isone, w)` check at the start of the fit sets `FitState.unith`; the split
scan then gets `UnitHessians{V}`, an `hs` vector of `OneHessian` markers whose
`addrow`/`subrow` methods are the general ones with every `h *` dropped, and
`gather!` stops writing `sc.hs` at all. Case 1, with the path switched off and
on at `FitState.unith` and everything else held fixed: **serial 1554 -> 1432 ms
(-7.9%)** median, 1457 -> 1360 ms best-of-eleven (-6.6%); **ten threads
571 -> 536 ms (-6.0%)** median, 551 -> 504 ms best (-8.5%) -- over the 3% bar
the brief set on both. The multiply dropped is by exactly one, so the trees are
bit-identical, checked on
the same thirteen designs and on the case-1 tree itself (3808 nodes, every node
field compared as a bit pattern).

### 3. The blin 3x3 solve as a Schur complement: measured, not committed

`fit_blin` builds a `3x3` `SMatrix`, takes `det` and solves with `\` at every
split point. The system is the `lin` `2x2` Gram matrix bordered by one hinge
column, so it can be solved as the `2x2` solve plus a rank-one Schur update:
with `p = [sxu, su]`, `adj2` the adjugate of the `2x2` and `d2` its
determinant, `det(G3) = suu*d2 - p' adj2 p` (the guard's own quantity),
`c = (suz*d2 - p' adj2 m2) / det(G3)` and `[a; b] = (adj2 m2 - c*adj2 p) / d2`,
which is three divisions and about twenty multiplications against the dense
solve's one division and about forty. Isolated, that form **is** faster: over a
2000-point sweep, `14.29 us -> 11.96 us` per sweep (-16%), or `11.54 us`
(-19%) with the reciprocal of `d2` taken once. It is **not committed**, for
three measured reasons.

1. **The brief's exactness gate fails.** Against the dense solve on 3 x 10^5
   random node sums (rows accumulated through `addrow`, so the sums are
   reachable ones), the coefficients agree to a median of `8e-15` relative but
   only to `1.5e-11` at the 99th percentile and `4.8e-7` at worst, with the
   knot restricted to the middle 5-95% of rows as `min_leaf` restricts it. The
   gate was `1e-12` relative. The singular guard is unaffected: it fires on
   exactly the same inputs in all 3 x 10^5 cases, zero flips.
2. **Neither form is the accurate one.** Against a `BigFloat` solve of the same
   normal equations, the Schur form's worst error over the 5-95% band is
   `5.4e-9` and the dense solve's is `1.2e-8`; the Schur form is closer to
   exact in 56% of random cases. The disagreement in (1) is the conditioning of
   the blin normal equations at `Float64`, not a defect of either ordering --
   which is also why it cannot be tuned away.
3. **No end-to-end win to weigh against that, and the trees move.** Case 1
   measured `1402 -> 1469 ms` serial and `552 -> 525 ms` on ten threads, but
   those two runs were sequential rather than alternated and the tree itself
   changes, so the work changes with it: the end-to-end comparison carries no
   weight either way, and the isolated sweep above is the only trustworthy
   speed number. The fitted trees differ on 135 of case 1's 3808
   nodes, on all 28 nodes of the `mse_bic` fixture, and the `softmax_cat`
   fixture grows from 35 to 53 nodes. Blessing that means regenerating every
   partition fixture, and (1) and (2) say the new trees would be no better
   founded than the old ones.

Worth revisiting only together with the conditioning: solving the blin system
on centred sums (`x - x̄`) would cut both the cancellation and the operation
count, but it changes what the moment sums are, so it is its own item.

### 4. `@inbounds` on the two hottest loops: measured, not committed

The project bans `@inbounds` without a cited measurement, and the measurement
does not support one. Apple M4 Pro, Julia 1.12.7, `-t 1`, `@benchmark`
medians, `n = 200_000` sorted rows, three runs of each variant with the source
toggled between them.

| Loop | plain | `@inbounds` | gain |
|---|---|---|---|
| `scan_feature` sweep, `Vector{Float64}` hs | 2.20 ms | 2.36 ms | **-7%** |
| `scan_feature` sweep, `UnitHessians` hs | 2.18 ms | 2.35 ms | **-8%** |
| `gather!`, unit-hessian path (three stores) | 0.524 ms | 0.521 ms | +0.6% |
| `gather!`, general path (four stores) | 0.917 ms | 0.814 ms | +11.2% |

`scan_feature` is consistently **slower** with the bounds checks removed: the
loop is arithmetic-bound, and dropping the checks changes LLVM's scheduling
for the worse. `gather!`'s unit path -- the one an unweighted `MSE` fit takes
-- gains 0.6%, far under the 5% bar. Only `gather!`'s general path clears the
bar, and it is worth about 1-2% of a weighted or non-`MSE` fit end to end.

Case 1 end to end, `-t 1`, eleven samples, the two variants run alternately to
control for drift: plain medians 1463 / 1443 ms against 1398 / 1352 ms with
`@inbounds` in `gather!`, and best-of-eleven 1333 ms against 1298 ms -- a 2.6%
to 5.4% spread that straddles the bar and sits inside this machine's
run-to-run noise (other work was resident during the pass). `@inbounds` in
`scan_feature` alone measured 1362 / 1375 ms, no better than plain.

So: nothing committed. If the general `gather!` path is ever revisited, the
bound argument is available -- `span ⊆ 1:n`, `st.idx` is `n x p` with
`j ∈ 1:p`, every `i = st.idx[k, j]` is a row index in `1:n` because `presort!`
fills the column with a `sortperm` and `partition!` only permutes within a
span, and `m` runs from 1 to `length(span) ≤ n` over scratch buffers of length
`n` -- but the gain does not pay for the loss of the check.
