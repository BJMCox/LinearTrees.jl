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

| Case | serial (ms) | 10 threads (ms) | speedup | change vs base, serial |
|---|---|---|---|---|
| 1 `fit_tree` MSE | 2183 | 851 | 2.57x | -17% |
| 2 `fit_tree` Softmax+cat | 1053 | 383 | 2.75x | -5% |
| 3 `fit_tree` MAD | 931 | 430 | 2.17x | -12% |
| 4 `fit` on a DataFrame | 414 | 110 | 3.76x | -32% |
| 5 `predict` (1e6 rows) | 86.8 | 9.22 | 9.41x | 0 |
| 6 `shap` (1e4 rows) | 3763 | 458 | 8.22x | 0 serial, 8.3x threaded |

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

**Share** case 1 reaches 2.45x on 10 threads, case 3 2.20x, against 9.65x for
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
