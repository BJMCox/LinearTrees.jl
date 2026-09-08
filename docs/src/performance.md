```@meta
CurrentModule = LinearTrees
```

# Performance

Fit quality, runtime, and memory use depend on the search method and the data.
Measure complete fits on representative inputs and compare quality on held-out
rows before changing the search policy.

## Choose a split-search method

`fit_tree` and `fit_boost` accept the same `split_search` keyword:

| Method | Numeric thresholds | Categorical features | Softmax |
|:--|:--|:--|:--|
| [`ExactSearch`](@ref) | All eligible thresholds in each node | Supported | Supported |
| [`BinnedSearch`](@ref) | Node-local bins, with optional refinement | Exact categorical search | Unsupported |
| [`HybridSearch`](@ref) | Global bins, then local refinement | Unsupported | Unsupported |

Use exact search as the reference. Try local or hybrid search for scalar-score
fits where threshold search is expensive. Even `Softmax(2)` uses vector scores
and requires exact search.

```@example searches
using LinearTrees, Random, Statistics

rng = Xoshiro(8)
X = rand(rng, 600, 4)
y = sin.(6 .* X[:, 1]) .+ X[:, 2] .+ 0.1 .* randn(rng, 600)
train, valid = 1:450, 451:600
searches = (ExactSearch(), BinnedSearch(nbins = 64), HybridSearch(nbins = 64))

map(searches) do search
    tree = fit_tree(X[train, :], y[train]; split_search = search, max_depth = 4)
    error = predict(tree, X[valid, :]) .- y[valid]
    (search = nameof(typeof(search)), rmse = round(sqrt(mean(abs2, error)); digits = 3))
end
```

### Local bins

`BinnedSearch(nbins = 64, refine = true)` builds bins from each node's rows.
Tied values stay together. Candidate scores use weighted moments of the original
values, so binning restricts thresholds without replacing features by bin centers.

Refinement searches the two bins beside the winning coarse boundary. It does
not run when a constant or unsplit line wins. Small nodes use exact search.
Set `refine = false` to omit refinement.

### Global bins with local refinement

`HybridSearch(nbins = 64)` prepares bins from the training data once and reuses
them as the tree grows. It refines the winning boundary using raw rows in
its neighboring occupied bins. Small nodes use exact search.

For boosting, bins come from all positive-weight training rows and are reused
across rounds. Row sampling selects from those bins. Validation data never
determines bin boundaries.

Hybrid search avoids the full presort index used by exact and local-bin
search. This can reduce memory use, but need not give the fastest fit.
Its bin count must lie in `2:65535`.

Both approximate methods can change node choices and predictions. Neither has
an approximation-error bound. More bins allow more thresholds, but held-out
accuracy need not improve monotonically.

## Threads

Start Julia with multiple threads, then set how many a call may use:

```sh
julia --threads=auto --project
```

```julia
tree = fit_tree(X, y; nthreads = 4)
yhat = predict(tree, Xnew; nthreads = 4)
```

The default is `Threads.nthreads()`. Small workloads remain serial where
threading would add overhead. More threads can increase scratch-memory use.
Use `nthreads = 1` when an outer loop already parallelizes independent fits.

## Reuse prediction storage

[`predict!`](@ref) fills a caller-provided output array:

```@example searches
tree = fit_tree(X[train, :], y[train]; max_depth = 4)
Xnew = X[valid, :]
out = Vector{Float64}(undef, size(Xnew, 1))
predict!(out, tree, Xnew)
@assert out == predict(tree, Xnew) # hide
round.(out[1:4]; digits = 3)
```

For `Softmax(K)`, use an `n × K` matrix. Storage reuse does not change the
model or prediction scale.

## Measure repeatable work

Separate compilation and data construction from timed fitting. BenchmarkTools
is an optional benchmarking dependency:

```julia
using BenchmarkTools

fit_tree(X, y; nthreads = 1) # compile before measuring
@benchmark fit_tree($X, $y; nthreads = 1)
```

Inspect allocation counts and bytes as well as runtime. For boosted fits
with sampling, create a fresh seeded RNG for each repetition. Reusing an
advanced RNG changes the sampled trees between calls.
