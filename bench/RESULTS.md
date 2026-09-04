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

## `-t 1` (`Threads.nthreads() == 1`)

| dataset | benchmark | median | min | memory | allocs |
|---|---|---|---|---|---|
| linear | `fit_tree` | 966 ms | 928 ms | 73.37 MiB | 901 |
| linear | `DecisionTree.build_tree` | 636 ms | 632 ms | 5.57 MiB | 2288 |
| linear | `LinearTrees.predict` | 1.571 ms | 1.466 ms | 784.06 KiB | 3 |
| piecewise | `fit_tree` | 425 ms | 392 ms | 74.63 MiB | 745 |
| piecewise | `DecisionTree.build_tree` | 647 ms | 641 ms | 5.57 MiB | 2180 |
| piecewise | `LinearTrees.predict` | 884 μs | 840 μs | 784.06 KiB | 3 |

## `-t auto` (`Threads.nthreads() == 10`)

| dataset | benchmark | median | min | memory | allocs |
|---|---|---|---|---|---|
| linear | `fit_tree` | 130 ms | 117 ms | 101.12 MiB | 3912 |
| linear | `DecisionTree.build_tree` | 653 ms | 651 ms | 5.57 MiB | 2288 |
| linear | `LinearTrees.predict` | 227 μs | 194 μs | 788.58 KiB | 55 |
| piecewise | `fit_tree` | 66.2 ms | 59.2 ms | 102.26 MiB | 1875 |
| piecewise | `DecisionTree.build_tree` | 654 ms | 654 ms | 5.57 MiB | 2180 |
| piecewise | `LinearTrees.predict` | 149 μs | 117 μs | 788.58 KiB | 55 |

`DecisionTree.build_tree` does not thread, so its numbers are flat across both
runs (the small drift is noise).

## Threaded vs. serial

On this 10-thread machine, threaded `fit_tree` on the `100_000 × 20` set is
**faster** than serial, not slower: 7.4x on linear (966 ms → 130 ms median)
and 6.4x on piecewise (425 ms → 66.2 ms median). This is the opposite of the
regression the task brief asked to watch for; there is no threaded-slower-
than-serial finding to report here. Threaded `fit_tree` does allocate more
(3912 vs. 901 allocations on linear) from the per-task scratch buffers and
`Threads.@spawn` overhead, but the wall-clock win dominates.

`LinearTrees.predict`'s median beats `DecisionTree.build_tree`'s median by
405x (serial linear, 636 ms / 1.571 ms) up to about 4,390x (threaded
piecewise, 654 ms / 149 μs) — roughly 2.6 to 3.6 orders of magnitude, not a
flat three. This is expected, since it is prediction, not tree growth; it is
included for scale, not as a fair apples-to-apples comparison with
`build_tree`.
