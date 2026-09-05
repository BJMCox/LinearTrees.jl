# Benchmark results

Machine: Apple M4 Pro, 10 CPU threads (`Sys.CPU_THREADS`), 64.0 GiB memory.
Julia 1.12.7. `DecisionTree.build_tree(y, X, 0, 8, 5, 10)` as the comparison
baseline; `fit_tree(X, y; max_depth = 8)` for LinearTrees. Data: `StableRNG(1)`,
`n = 100_000`, `p = 20`, from `bench/run.jl`. The plan's California Housing
benchmark is not in `bench/run.jl`: `MLDatasets` does not ship that dataset
(its supervised sets are BostonHousing, Titanic, Iris, Wine, Mutagenesis, and
SMSSpamCollection), so the item was dropped rather than left as an
unreachable branch.

Reproduce with `julia --project=bench -t 1 bench/run.jl` and
`julia --project=bench -t auto bench/run.jl` (instantiate `bench/` first).
The headline tables are from `main` at `49ca56f` (2026-09-05), after the
scratch-pool threading, the radix presort and the row-order slices; the
sections further down record each change against the code before it.

## `-t 1` (`Threads.nthreads() == 1`)

| dataset | benchmark | median | min | memory | allocs |
|---|---|---|---|---|---|
| linear | `fit_tree` | 355 ms | 353 ms | 20.44 MiB | 110 |
| linear | `DecisionTree.build_tree` | 655 ms | 653 ms | 5.57 MiB | 2288 |
| linear | `LinearTrees.predict` | 1.573 ms | 1.545 ms | 784.06 KiB | 3 |
| piecewise | `fit_tree` | 132 ms | 129 ms | 20.83 MiB | 105 |
| piecewise | `DecisionTree.build_tree` | 654 ms | 653 ms | 5.57 MiB | 2180 |
| piecewise | `LinearTrees.predict` | 887 μs | 867 μs | 784.06 KiB | 3 |

## `-t auto` (`Threads.nthreads() == 10`)

| dataset | benchmark | median | min | memory | allocs |
|---|---|---|---|---|---|
| linear | `fit_tree` | 59.9 ms | 49.6 ms | 67.35 MiB | 2203 |
| linear | `DecisionTree.build_tree` | 655 ms | 654 ms | 5.57 MiB | 2288 |
| linear | `LinearTrees.predict` | 236 μs | 192 μs | 788.58 KiB | 55 |
| piecewise | `fit_tree` | 28.1 ms | 19.6 ms | 79.30 MiB | 1482 |
| piecewise | `DecisionTree.build_tree` | 655 ms | 651 ms | 5.57 MiB | 2180 |
| piecewise | `LinearTrees.predict` | 159 μs | 117 μs | 788.58 KiB | 55 |

`DecisionTree.build_tree` does not thread, so its numbers are flat across both
runs (the small drift is noise).

## Threaded vs. serial

On this 10-thread machine, threaded `fit_tree` on the `100_000 × 20` set is
**faster** than serial, not slower: 5.9x on linear (355 ms → 59.9 ms median)
and 4.7x on piecewise (132 ms → 28.1 ms median). This is the opposite of the
regression the task brief asked to watch for; there is no threaded-slower-
than-serial finding to report here. Threaded `fit_tree` does allocate more
(67 MiB and 2203 allocations vs. 20 MiB and 110 on linear) from the
`2 * nthreads` scratch sets and the spawned subtree tasks, but the wall-clock
win dominates. The ratios are below the 7x of the deep cases further down
because these trees are small (11 and 9 nodes at `max_depth = 8`), so there
are few sibling subtrees to spawn and most of the fit is the presort and the
feature-chunked scans of the first nodes.

`LinearTrees.predict`'s median beats `DecisionTree.build_tree`'s median by
about 420x (serial linear, 655 ms / 1.573 ms) up to about 4,100x (threaded
piecewise, 655 ms / 159 μs) — roughly 2.6 to 3.6 orders of magnitude, not a
flat three. This is expected, since it is prediction, not tree growth; it is
included for scale, not as a fair apples-to-apples comparison with
`build_tree`.

## Deep trees: node-wise row partition

Before this change every node rebuilt an `n`-bit row mask and `gather!`
walked the full presorted column per feature, so growth cost
O(nodes · n · p). Now each node owns a contiguous span of the presorted
index and a stable in-place partition hands the children their spans, so
growth is O(depth · n · p) as the spec asks. Trees are bit-identical
(`test/partition.jl` checks four recorded fixtures, serial and threaded).

Same machine as above. `n = 200_000`, `p = 20`, `max_depth = 12`,
`StableRNG(7)`, target `Σ_{j≤6} floor(5 x_j) + x_7 x_8 + noise` (many step
changes, so BIC keeps splitting). "forced" uses
`MinDeviance((LIN, PCON, BLIN, PLIN))` with `min_leaf = 50`, `min_fit = 100`.
Median of 3 fits.

| case | nodes | threads | before | after | speedup |
|---|---|---|---|---|---|
| BIC step target | 3638 | 10 | 2.73 s | 1.14 s | 2.4x |
| BIC step target | 3638 | 1 | 9.08 s | 2.75 s | 3.3x |
| forced growth | 2449 | 10 | 1.45 s | 0.56 s | 2.6x |
| forced growth | 2449 | 1 | 4.23 s | 1.44 s | 2.9x |

Memory fell as well (322 → 253 MiB threaded on the BIC case) because the
per-node `BitVector` mask is gone. The remaining gap to a CART fit is the
per-node `leftrows`/`rightrows` vectors and the subtree splice copies.
`bench/run.jl` now includes both the BIC step case and the forced-growth case.

### The same deep cases at `49ca56f`

From `bench/run.jl` after the scratch-pool threading, the radix presort and
the row-order slices (sections below). The forced-growth row is the same
design as the table above; the BIC step row in `bench/run.jl` is the
`100_000 × 20` set and so is not the 3638-node case.

| case | nodes | threads | median | min | memory | allocs |
|---|---|---|---|---|---|---|
| BIC step target, n = 100_000 | 1948 | 10 | 97.9 ms | 85.6 ms | 91.04 MiB | 5084 |
| BIC step target, n = 100_000 | 1948 | 1 | 541 ms | 535 ms | 21.45 MiB | 1348 |
| forced growth | 2449 | 10 | 180 ms | 168 ms | 180.59 MiB | 8125 |
| forced growth | 2449 | 1 | 973 ms | 965 ms | 42.17 MiB | 1334 |

Forced growth went from 0.56 s to 0.180 s threaded and from 1.44 s to
0.973 s serial since the table above (3.1x and 1.5x), with the threaded
speedup over serial at 5.4x on both deep cases.

## SHAP: pooled path buffers

`visit!` used to `copy(path)` three times per split node and once per LIN
node, so a row's TreeSHAP recursion allocated one fresh `Vector{PathElem}` per
branch. A `PathPool` of buffers indexed by recursion depth now holds them: a
split node's hot and cold paths stay live across both child recursions, so
each depth owns its own pair, and the own-path is a single scratch per depth.
`shap!` sizes the pool once from `shap_depth(tree)` and builds one per
`row_blocks` block, so threads never share it. SHAP values are byte-identical
(`==`, not `≈`) on the `test/shap.jl` trees, on a `max_depth = 10` tree with a
categorical column, and on a `Softmax(3)` tree.

Same machine as above, `-t 1`. 57-node tree: `StableRNG(3)`, `n = 600`,
`p = 4`, target `sin(3x₁) + 2x₂·1[x₃ > 0.5] + x₄² + noise`,
`fit_tree(X, y; max_depth = 5)`; `@benchmark shap($t, $X)` over all 600 rows.

| | median | memory | allocs | bytes/row | allocs/row |
|---|---|---|---|---|---|
| before | 3.262 ms | 20.86 MiB | 139,205 | 35.6 KiB | 232 |
| after | 2.076 ms | 28.95 KiB | 71 | 49 B | 0.12 |

Per-row allocation is gone: what is left is the pool itself plus the result
arrays, amortised over the whole call, and the recursion is 36% faster because
it no longer runs the allocator once per branch.

## Scoring each model kind once per feature

The BIC score is monotone in the surrogate deviance within one kind, so
`scan_feature` now carries the lowest deviance each of `pcon`, `blin`
and `plin` reaches through the split sweep and calls `selection_score` once per
kind per feature instead of once per kind per split point. On the P1 profiling
case 1 (`fit_tree` MSE, `n = 200_000`, `p = 20`, `max_depth = 12`, Apple M4
Pro, Julia 1.12.7, eleven `@benchmark` samples with the two arms run
alternately) this takes the serial median from **2086 ms to 1429 ms (-31%)**
and the ten-thread median from **826 ms to 542 ms (-34%)**, with the
allocation count unchanged. The fitted
trees are bit-identical on the four stored partition designs, the eight PILOT
reference fixtures and a `Softmax(3)` design with a categorical column. Two
ties resolve differently: an exact cross-kind score tie goes to the kind that
comes first in `(con, lin, pcon, blin, plin)`, and within a kind the split
point with the lowest deviance wins where the old sweep kept the earliest split
point among those sharing a score (`n log(dev/n)` is not injective in
`Float64`). In a 596-design sweep 24 trees differ, all by the second rule.

On top of that, an unweighted `MSE` fit has a row hessian of exactly one on
every row, so the scan takes an accumulation path with the multiply by it
removed: another **-7.9%** serial (1554 -> 1432 ms on the same case) and
**-6.0%** on ten threads (571 -> 536 ms), bit-identical, since the factor
dropped is exactly one. `bench/PROFILE.md` has the protocol, the two changes
that were measured and rejected (a Schur-complement form of the `blin` solve
and `@inbounds` on the two hottest loops), and the numbers behind both.

## Weighted median: quickselect instead of a full sort

`median_abs!` sorted the node's entire `|r|` vector to read one weighted
quantile; the profile put that `sort!` at 41% of a MAD fit (`bench/PROFILE.md`
case 3). `median_abs!`, `median_abs`, and `wquantile` now share
`wquantile_select!`, an `O(m)`-expected quickselect (median-of-three pivot, a
3-way partition so a run of duplicate values is skipped in one step) that
partitions the node's index buffer instead of sorting it. Zero-weight rows
are moved out of the active range before selection starts, which is also what
keeps the exact-boundary tie rule (`cum == target` averages the two adjacent
order statistics) well-defined: randomized cases mixing integer and
fractional weights, zero weights, and heavy duplicates agree with a
sort-then-walk oracle that drops zero-weight rows the same way, and MAD and
Quantile(0.3) fits on the four `test/fixtures/partition/cases.jl` designs are
bit-identical before and after (same node count, same `predict` bit pattern).

This is a genuine behavior change against `main`'s exact code in one common
corner, not just a speed-up: `main` walks the *unfiltered* order, so an exact
weight boundary immediately followed, in ascending order, by a zero-weight
row lets that row's value get averaged into the boundary tie regardless of
whether the two rows share a value -- e.g. `y = [1, 2, 3]`, `w = [1, 0, 1]`,
`τ = 0.5` returns `1.5` on `main` and `2.0` here (`Statistics.median` of the
duplicated sample `[1, 3]`). With integer weights that trigger fires whenever
the node's total weight is even, measured at 26% of `irls_epsilon!` calls with
`{0, 1}` weights on fully distinct, continuous residuals -- common, not a
corner case requiring duplicate values. No `fit_tree` output changed in
testing (see `test/loss.jl`'s golden-hash fixture fits), because `ε`'s floor
rarely binds and the two candidate order statistics are adjacent when it does.

Same machine as above, `-t 1`.

| | median | memory | allocs |
|---|---|---|---|
| `median_abs!`, m = 1,000, before | 10.709 μs | 0 bytes | 0 |
| `median_abs!`, m = 1,000, after | 3.009 μs | 0 bytes | 0 |
| `median_abs!`, m = 100,000, before | 6.969 ms | 0 bytes | 0 |
| `median_abs!`, m = 100,000, after | 1.070 ms | 0 bytes | 0 |
| case 3 (MAD, n = 100,000, p = 10, max_depth = 12, niter = 5), before | 963.820 ms | 33.22 MiB | 20,228 |
| case 3 (MAD, n = 100,000, p = 10, max_depth = 12, niter = 5), after | 679.667 ms | 32.46 MiB | 20,224 |

`median_abs!` alone is 3.6x (m = 1,000) to 6.5x (m = 100,000) faster; the
gain narrows for the full case 3 fit (1.42x) since `sort!` was 41% of that fit,
not all of it, and `niter = 5` IRLS passes still pay for everything else the
profile found (partitioning, gradient/Hessian, the MomentSums reduction).

## SHAP: branch constants attributed once per leaf

`visit!` used to credit a split node's two branch constants and a LIN node's
constant against paths built at the node, so a row's recursion ran
`attribute_constant!` three times per split node, twice per LIN node and once
per leaf. A constant credited against a node's path equals the same constant
credited against both of its child paths, so the branch constants now ride
down the recursion in a running sum and are attributed once per leaf, with the
leaf intercept. Only the own-feature linear term stays at the node: its path
is not either child's path, so it does not telescope. `bench/PROFILE.md`
section 3 has the proof sketch and why the gain is 1.3x rather than the 2-3x
the call count suggests -- the calls that disappear sat at shallow nodes and
the ones that absorb them sit at the leaves, where `attribute_constant!`'s
`O(depth^2)` is largest.

Same machine as above, one process at `-t 10`. Case 6 of `bench/cases.jl`
(`case56_data()`: a 3674-node `max_depth = 12` tree with one categorical
column, `shap` over 10,000 query rows and 10 features), `@benchmark ...
samples = 5 evals = 1`, minimum of the five, `main` (`02a691d`) and the branch
measured alternately in two rounds.

| build | serial | 10 threads |
|---|---|---|
| `main`, round 1 | 3827 ms | 444 ms |
| `main`, round 2 | 3856 ms | 460 ms |
| branch, round 1 | **2850 ms** | **340 ms** |
| branch, round 2 | 2879 ms | 355 ms |

That is 1.34x serial and 1.31x on ten threads, well outside the +-5% this
machine drifts between processes.

The reformulation reassociates the sums rather than reproducing them bit for
bit: on case 6's 100,000 values 6,016 are unchanged, the largest absolute
difference against `main` is 3.09e-14 and the largest relative difference
8.50e-11, on values up to 2.215. The efficiency identity still holds to
4.80e-14 absolute and 1.54e-14 relative over scores in 0.324 .. 14.41, and
serial and ten-thread results remain identical to each other.
