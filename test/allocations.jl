using StableRNGs, DataFrames, CategoricalArrays

@testset "hot paths do not allocate" begin
    # @allocated on a non-const global can allocate for reasons unrelated to
    # the callee, so each measurement runs inside a local function with typed
    # locals -- the arguments below are boxed globals otherwise.
    function scan_alloc()
        x = sort(rand(1000)); z = rand(1000); h = ones(1000); w = ones(1000)
        LinearTrees.scan_feature(x, z, h, w, BIC(), 5, 1e-12)
        return @allocated LinearTrees.scan_feature(x, z, h, w, BIC(), 5, 1e-12)
    end
    @test scan_alloc() == 0

    function predict_alloc()
        X = rand(50, 2); t = fit_tree(X, X[:, 1]); out = zeros(50)
        predict!(out, t, X)
        return @allocated predict!(out, t, X)
    end
    @test predict_alloc() == 0
end

@testset "a MAD fit reuses its IRLS buffers" begin
    # fails if `irls_refit`'s residual buffer or `median_abs!`'s sort buffers
    # revert to a fresh per-call allocation: measured 664_576 bytes with the
    # buffers reused and 975_360 with a fresh residual and permutation per
    # refit call, both under `--check-bounds=yes` at one thread.
    function mad_fit_alloc()
        rng = StableRNG(43)
        n = 4_000
        X = rand(rng, n, 3); y = X[:, 1] .+ 0.2 .* randn(rng, n)
        y[1:20] .+= 5   # outliers, so IRLS solves a non-degenerate residual
        fit_tree(X, y, MAD(); nthreads = 1, max_depth = 6)
        return @allocated fit_tree(X, y, MAD(); nthreads = 1, max_depth = 6)
    end
    @test mad_fit_alloc() < 800_000
end

@testset "a serial fit allocates one scratch set" begin
    # `nthreads == 1` must build exactly one `Scratch`, not `SCRATCH_PER_THREAD`
    # of them: a second set on this design is 100_000 * 28 = 2.8 MB, which the
    # budget below excludes. Measured 24.2 MB, deterministic to the byte.
    function serial_fit_alloc()
        rng = StableRNG(42)
        n, p = 100_000, 6
        X = rand(rng, n, p)
        y = sum(floor.(4 .* X[:, j]) for j in 1:3) .+ 0.1 .* randn(rng, n)
        fit_tree(X, y; nthreads = 1, max_depth = 8)
        return @allocated fit_tree(X, y; nthreads = 1, max_depth = 8)
    end
    @test serial_fit_alloc() < 25_500_000
end

@testset "shap recursion does not allocate per row after warm-up" begin
    # Fails the moment a `copy(path)` (or any other fresh Vector{PathElem})
    # comes back into `visit!`: the first row grows the pool, so a second row
    # can only allocate if the recursion builds paths instead of reusing them.
    function shap_recurse_alloc()
        rng = StableRNG(31)
        X = rand(rng, 600, 4)
        y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ X[:, 4] .^ 2 .+
            0.05 .* randn(rng, 600)
        t = fit_tree(X, y; max_depth = 8)
        φ = zeros(2, 4)
        pool = LinearTrees.PathPool()
        x1 = view(X, 1, :); x2 = view(X, 2, :)
        LinearTrees.shap_recurse!(φ, t, x1, 1, 1, pool, 1.0, 1.0, 0)
        return @allocated LinearTrees.shap_recurse!(φ, t, x2, 2, 1, pool, 1.0, 1.0, 0)
    end
    @test shap_recurse_alloc() == 0
end

@testset "table encoding does not allocate per row" begin
    # `Tables.getcolumn` is typed `AbstractVector`, so without a function
    # barrier every element read, conversion and store inside `encode_column!`
    # dispatches at run time and boxes its result: one allocation per row per
    # column. Fails the moment the per-row loops move back into the function
    # that fetched the column.
    function encode_alloc()
        rng = StableRNG(41)
        n = 5_000
        df = DataFrame(a = rand(rng, n), b = rand(rng, n),
            c = categorical(rand(rng, ["x", "y", "z"], n)))
        enc = LinearTrees.TableEncoder(df, :error)
        LinearTrees.encode(enc, df; nthreads = 1)
        return @allocated LinearTrees.encode(enc, df; nthreads = 1)
    end
    # the output matrix, the column fetches and the level map are a fixed cost;
    # anything per row would be 5_000 allocations and megabytes on top
    @test encode_alloc() < 200_000
end

@testset "boosting reuses tree work buffers across rounds" begin
    function boost_fit_alloc()
        rng = StableRNG(144)
        X = rand(rng, 4_000, 20)
        y = sum(floor.(4 .* X[:, j]) for j in 1:3)
        fit_boost(X, y; nrounds = 8, max_depth = 3, nthreads = 1)
        return @allocated fit_boost(X, y; nrounds = 8, max_depth = 3, nthreads = 1)
    end
    # Measured 9.46 MB with fresh indices and scratch each round, 6.81 MB reused.
    @test boost_fit_alloc() < 8_000_000
end
